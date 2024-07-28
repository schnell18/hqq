import json
import os
import pandas as pd
import torch

from hqq.core.quantize import Quantizer as hQuant
from safetensors import safe_open
from timeit import default_timer as timer


def quant_hqq(tensor, nbits, group_size=64, optimize=True):
    wq, meta = hQuant.quantize(tensor, nbits=nbits, group_size=group_size, optimize=optimize)
    return hQuant.dequantize(wq, meta)


def get_tensor(matrix_name, base_dir, index_json='model.safetensors.index.json'):
    fp = os.path.join(base_dir, index_json)
    with open(fp, "r") as fh:
        index = json.load(fh)
        try:
            st_file = index["weight_map"][matrix_name]
            mp = os.path.join(base_dir, st_file)
            with safe_open(mp, framework="pt", device="cpu") as f:
                return f.get_tensor(matrix_name)
        except Exception:
            raise ValueError(f"Invalid key {matrix_name}")


def calc_fnorm(
        base_dir, prefix, layer, module, suffix,
        nbits1, gsizes1, nbits2, gsizes2):
    dikts = []
    matrix_name = f"{prefix}.{layer}.{module}.{suffix}"
    w = get_tensor(matrix_name, base_dir)
    params = w.numel()
    for nbit1 in nbits1:
        for gsize1 in gsizes1:
            for nbit2 in nbits2:
                for gsize2 in gsizes2:
                    wq_hqq = quant_hqq(w, nbits=nbit1, group_size=gsize1, optimize=True)
                    norm_hqq = torch.norm(w - wq_hqq).item()
                    bpp = nbit1 + 2 * nbit2 / gsize1 + 32 / (gsize1 * gsize2)
                    memmb = bpp * params / 8 / (1024**2)
                    dikt = {
                        'layer': layer,
                        'module': module,
                        'nbit1': nbit1,
                        'gsize1': gsize1,
                        'nbit2': nbit2,
                        'gsize2': gsize2,
                        'fnorm': norm_hqq,
                        'memmb': memmb,
                        'params': params,
                    }
                    dikts.append(dikt)
    return dikts


def calc_fnorm_for_model(model_id, base_dir, layers):
    self_attns = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    mlps = ['gate_proj', 'up_proj', 'down_proj']
    nbits1 = [2, 3, 4, 8]
    gsizes1 = [32, 64, 128]
    nbits2 = [8]
    gsizes2 = [128]
    prefix = 'model.layers'
    suffix = 'weight'
    dikts = []
    for layer in range(layers):
        for attn in self_attns:
            ds = calc_fnorm(
                base_dir, prefix, layer, f"self_attn.{attn}", suffix,
                nbits1, gsizes1, nbits2, gsizes2)
            dikts.extend(ds)
        for mlp in mlps:
            ds = calc_fnorm(
                base_dir, prefix, layer, f"mlp.{mlp}", suffix,
                nbits1, gsizes1, nbits2, gsizes2)
            dikts.extend(ds)

    df = pd.DataFrame(dikts)
    file_name = f"data/fnorm-{model_id.split('/')[1]}.csv"
    df.to_csv(
        file_name,
        columns=[
            "layer", "module", "nbit1",
            "gsize1", "nbit2", "gsize2",
            "fnorm", "memmb", "params"
        ],
        index=False
    )


def gen_calc_tasks():
    home_dir = os.environ.get("HOME", "/home/justin")
    return [
        {
            "model_id": "meta-llama/Llama-2-7b-hf",
            "layers": 32,
            "base_dir": f"{home_dir}/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/",
        },
        {
            "model_id": "meta-llama/Llama-3-8B",
            "layers": 32,
            "base_dir": f"{home_dir}/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df/"
        },
        {
            "model_id": "meta-llama/Llama-2-13b-hf",
            "layers": 40,
            "base_dir": f"{home_dir}/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1/"
        },
        {
            "model_id": "meta-llama/Llama-2-70b-hf",
            "layers": 80,
            "base_dir": "/data/hugginface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/3aba440b59558f995867ba6e1f58f21d0336b5bb/"
        },
        {
            "model_id": "meta-llama/Llama-3-70B",
            "layers": 80,
            "base_dir": "/data/hugginface/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338"
        },
        {
            "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            'layers': 126,
            'base_dir': "/data/hugginface/hub/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/e04e3022cdc89bfed0db69f5ac1d249e21ee2d30/",
        },
    ]


def main():
    calc_tasks = gen_calc_tasks()[-1:]
    for task in calc_tasks:
        model_id = task["model_id"]
        t1 = timer()
        calc_fnorm_for_model(model_id, task["base_dir"], task["layers"])
        t2 = timer()
        print(f"Finished {model_id} metrics calc in {t2 - t1} seconds")


if __name__ == "__main__":
    main()
