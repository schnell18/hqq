#!/usr/bin/env python3

import os
import torch
from scipy.stats import kurtosis

from safetensors import safe_open

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
    }
}


def load_weight(prefix, base_dir, st_file):
    o = f"{prefix}.weight"
    q = f"{prefix}.qweight"

    try:
        mp = os.path.join(base_dir, st_file)
        with safe_open(mp, framework="pt", device="cpu") as f:
            return f.get_tensor(o), f.get_tensor(q)
    except Exception:
        raise ValueError(f"Invalid key {o}")


def compare_pair(model_id, layers, output_dir):

    st_file = f"{output_dir}/{model_id}-cmp.safetensors"
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
            wo, wq = load_weight(matrix, output_dir, st_file)
            diff = torch.norm(wo - wq).item()
            kurt_peason = kurtosis(
                wo.numpy(),
                axis=None,
                fisher=False,
                bias=True,
                nan_policy='omit'
            )
            kurt_fisher = kurtosis(
                wo.numpy(),
                axis=None,
                fisher=True,
                bias=True,
                nan_policy='omit'
            )
            #print(f"{matrix} FNorm Diff: {diff:.5f} Kurtosis: {kurt:.2f}")
            print(f"{matrix},{diff:.5f},{kurt_fisher:.3f},{kurt_peason:.3f}")


if __name__ == "__main__":
    output_dir = f'{home_dir}/work/hqq/examples/llama2_benchmark/snapshots/cmp'
    model_id = "meta-llama/Llama-2-7b-hf"
    model = models[model_id]
    compare_pair(
        model_id.split('/')[1],
        model["layers"],
        output_dir
    )
