#!/usr/bin/env python3

import json
import os
import torch

from hqq.core.quantize import Quantizer as hQuant
from safetensors import safe_open
from safetensors.torch import save_file as safe_save
from torch import uint8

home_dir = os.environ.get("HOME", "/home/justin")
models = {
    "meta-llama/Llama-2-7b-hf": {
        'layers': 32,
        'base_dir': f"{home_dir}/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/",
    },
    "meta-llama/Llama-2-13b-hf": {
        'layers': 40,
        'base_dir': f"{home_dir}/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1/",
    },
    "meta-llama/Meta-Llama-3-8B": {
        'layers': 32,
        'base_dir': f"{home_dir}/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df/",
    },
    "meta-llama/Meta-Llama-3-70B": {
        'layers': 80,
        'base_dir': "/data/hugginface/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338/",
    },
    "meta-llama/Llama-2-70b-hf": {
        'layers': 80,
        'base_dir': "/data/hugginface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/3aba440b59558f995867ba6e1f58f21d0336b5bb/",
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        'layers': 80,
        'base_dir': "/data/hugginface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c/",
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        'layers': 126,
        'base_dir': "/data/hugginface/hub/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/e04e3022cdc89bfed0db69f5ac1d249e21ee2d30/",
    },
}

llama2_7b_base_dir = models['meta-llama/Llama-2-7b-hf']['base_dir']
llama3_8b_base_dir = models['meta-llama/Meta-Llama-3-8B']['base_dir']
llama2_13b_base_dir = models['meta-llama/Llama-2-13b-hf']['base_dir']
llama2_70b_base_dir = models['meta-llama/Llama-2-70b-hf']['base_dir']
llama3_70b_base_dir = models['meta-llama/Meta-Llama-3-70B']['base_dir']


def load_weight(matrix_name, base_dir, index_json='model.safetensors.index.json'):
    m = f"{matrix_name}.weight"
    fp = os.path.join(base_dir, index_json)
    with open(fp, "r") as fh:
        index = json.load(fh)
        try:
            st_file = index["weight_map"][m]
            mp = os.path.join(base_dir, st_file)
            with safe_open(mp, framework="pt", device="cpu") as f:
                return f.get_tensor(m)
        except Exception:
            raise ValueError(f"Invalid key {m}")


def dequantize(wq, meta):
    # Zero/Scale packed together
    if "zero_scale" in meta:
        zero_scale = meta["zero_scale"]

        if zero_scale.dtype == uint8:
            meta["zero_q"], meta["scale_q"] = zero_scale[0], zero_scale[1]
        else:
            meta["zero"], meta["scale"] = zero_scale[0], zero_scale[1]

    if meta["quant_zero"]:
        meta["zero"] = hQuant.dequantize(
            meta["zero_q"], meta["meta_zero"]
        )

    if meta["quant_scale"]:
        meta["scale"] = hQuant.dequantize(
            meta["scale_q"], meta["meta_scale"]
        )
    return hQuant.dequantize(wq, meta)


def restore_weight(matrix, state_dict):
    key = matrix
    if key in state_dict:
        m_dikt = state_dict[key]
        if 'meta' in m_dikt:
            meta_dict = m_dikt['meta']
            meta_scale_dict = meta_dict.get('meta_scale', None)
            b1 = meta_dict['nbits']
            g1 = meta_dict['group_size']
            b2 = meta_scale_dict['nbits'] if meta_scale_dict else 8
            g2 = meta_scale_dict['group_size'] if meta_scale_dict else 128
            quant_config = {
                'b1': b1,
                'g1': g1,
                'b2': b2,
                'g2': g2,
            }
            wq = dequantize(m_dikt['W_q'], meta_dict)
            return wq, quant_config
        else:
            return None, None
    else:
        return None, None


def save_compare_pair(
        base_dir, quant_base_dir, quant_cfg, model_id, layers, output_dir):
    file_path = f"{quant_base_dir}/{model_id}-{quant_cfg}-hqq/qmodel.pt"
    state_dict = torch.load(file_path, map_location='cpu')

    tensors = {}
    metadata = {}
    # walk thru the linear layers
    # for each layer
    # for each linear module
    # load the original weight
    # load the quantized weight and dequantized
    # save the two matrix into a combined safetensors
    for layer in range(layers):
        matricies = [
            f"model.layers.{layer}.mlp.down_proj",
            f"model.layers.{layer}.mlp.gate_proj",
            f"model.layers.{layer}.mlp.up_proj",
            f"model.layers.{layer}.self_attn.k_proj",
            f"model.layers.{layer}.self_attn.o_proj",
            f"model.layers.{layer}.self_attn.q_proj",
            f"model.layers.{layer}.self_attn.v_proj",
        ]
        for matrix in matricies:
            wq, quant_cfg = restore_weight(matrix, state_dict)
            if wq is None:
                # skip unquantized matrix
                continue
            wo = load_weight(matrix, base_dir)
            tensors[f"{matrix}.weight"] = wo
            tensors[f"{matrix}.qweight"] = wq
            metadata[f"{matrix}.quant_cfg.b1"] = str(quant_cfg["b1"])
            metadata[f"{matrix}.quant_cfg.b2"] = str(quant_cfg["b2"])
            metadata[f"{matrix}.quant_cfg.g1"] = str(quant_cfg["g1"])
            metadata[f"{matrix}.quant_cfg.g2"] = str(quant_cfg["g2"])

    output_fp = f"{output_dir}/{model_id}-cmp.safetensors"
    safe_save(tensors, output_fp, metadata=metadata)


if __name__ == "__main__":

    # Llama-2-7b-hf-b3g128-hqq:
    # Llama-2-7b-hf-b3g32-hqq:
    # Llama-2-7b-hf-b3g64-hqq:
    # Llama-2-7b-hf-b4g128-hqq:
    # Llama-2-7b-hf-b4g32-hqq:
    # Llama-2-7b-hf-b4g64-hqq:

    quant_cfg = "b4g64"
    quant_base_dir = '/data/gqq-eval'
    output_dir = f'{home_dir}/work/hqq/examples/llama2_benchmark/snapshots/cmp'
    model_id = "meta-llama/Llama-2-7b-hf"
    model = models[model_id]
    save_compare_pair(
        model["base_dir"],
        quant_base_dir,
        quant_cfg,
        model_id.split('/')[1],
        model["layers"],
        output_dir
    )
