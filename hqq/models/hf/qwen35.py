# SPDX-License-Identifier: Apache-2.0
# Support for Qwen3.5 vision-language models (Qwen3_5ForConditionalGeneration)

import copy

from tqdm import tqdm

from ..base import BasePatch, is_leaf_module
from .base import BaseHQQHFModel


class Qwen35Patch(BasePatch):
    # Linear tags cover only the text decoder's attention and MLP projections.
    # q_norm / k_norm are RMSNorm (non-linear) and are handled in
    # patch_nonlinearlayers instead.
    @classmethod
    def get_linear_tags(cls):
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        # The text decoder lives at model.model.language_model, not model.model.
        # The vision encoder (model.model.visual) is walked separately at the
        # end of this method so its leaves get materialized off the meta device
        # (init_empty_weights leaves them there until something assigns real
        # tensors / moves them).
        text_model = model.model.language_model

        model.lm_head = patch_fct(model.lm_head)
        text_model.embed_tokens = patch_fct(text_model.embed_tokens)
        text_model.norm = patch_fct(text_model.norm)

        if hasattr(text_model, "rotary_emb"):
            text_model.rotary_emb = text_model.rotary_emb.to(
                device=text_model.norm.weight.device
            )

        layers = text_model.layers
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].post_attention_layernorm = patch_fct(
                layers[i].post_attention_layernorm
            )

            # Standard attention layers carry q_norm / k_norm (QK-RMSNorm).
            # Hybrid linear-attention layers (Qwen3_5GatedDeltaNet) do not.
            if hasattr(layers[i], "self_attn"):
                attn = layers[i].self_attn
                if hasattr(attn, "rotary_emb"):
                    attn.rotary_emb = patch_fct(attn.rotary_emb)
                if hasattr(attn, "q_norm"):
                    attn.q_norm = patch_fct(attn.q_norm)
                if hasattr(attn, "k_norm"):
                    attn.k_norm = patch_fct(attn.k_norm)

            if hasattr(layers[i].mlp, "act_fn"):
                layers[i].mlp.act_fn = patch_fct(layers[i].mlp.act_fn)

        if hasattr(model.model, "visual"):
            cls._patch_subtree_leaves(model.model.visual, patch_fct, verbose)

    @classmethod
    def _patch_subtree_leaves(cls, root, patch_fct, verbose=True):
        leaf_paths = [
            name for name, module in root.named_modules()
            if name and is_leaf_module(module)
        ]
        for path in tqdm(leaf_paths, disable=not verbose):
            parts = path.split(".")
            parent = root
            for part in parts[:-1]:
                parent = parent._modules[part]
            last = parts[-1]
            parent._modules[last] = patch_fct(parent._modules[last])

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        text_model = model.model.language_model
        layers = text_model.layers
        self_attns = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlps = ["gate_proj", "up_proj", "down_proj"]

        for i in tqdm(range(len(layers)), disable=not verbose):
            # Skip hybrid linear-attention layers — they have no q/k/v/o_proj.
            if hasattr(layers[i], "self_attn"):
                attn_obj = layers[i].self_attn
                for item in self_attns:
                    module = f"self_attn.{item}"
                    quant_config = cls.get_optimal_config(
                        model, i, module, patch_params
                    )
                    setattr(
                        attn_obj,
                        item,
                        patch_fct(getattr(attn_obj, item), quant_config),
                    )

            mlp_obj = layers[i].mlp
            for item in mlps:
                module = f"mlp.{item}"
                quant_config = cls.get_optimal_config(model, i, module, patch_params)
                setattr(
                    mlp_obj,
                    item,
                    patch_fct(getattr(mlp_obj, item), quant_config),
                )

    @classmethod
    def get_optimal_config(
        cls, model, layer_no: int, module: str, global_quant_config: dict
    ) -> dict:
        config = global_quant_config[module]
        if config is None:
            return None
        quant_config = copy.deepcopy(config)
        if hasattr(model, "optimal_configs"):
            opt_tpl = model.optimal_configs.get(f"{layer_no}.{module}")
            if opt_tpl:
                quant_config["weight_quant_params"]["nbits"] = opt_tpl[0]
                quant_config["weight_quant_params"]["group_size"] = opt_tpl[1]
                quant_config["weight_quant_params"]["round_zero"] = (
                    True if opt_tpl[0] == 4 else False
                )
        return quant_config


class Qwen35HQQ(Qwen35Patch, BaseHQQHFModel):
    pass
