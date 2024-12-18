# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import copy

from tqdm import tqdm

from ..base import BasePatch
from .base import BaseHQQHFModel


# Patch LLama functions
class LLamaPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
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
        base_model = model.model
        model.lm_head = patch_fct(model.lm_head)
        base_model.embed_tokens = patch_fct(base_model.embed_tokens)
        base_model.norm = patch_fct(base_model.norm)
        if hasattr(base_model, "rotary_emb"):
            base_model.rotary_emb = base_model.rotary_emb.to(
                device=base_model.norm.weight.device
            )

        layers = base_model.layers
        for i in tqdm(range(len(base_model.layers)), disable=not verbose):
            layers[i].self_attn.rotary_emb = patch_fct(layers[i].self_attn.rotary_emb)
            layers[i].mlp.act_fn = patch_fct(layers[i].mlp.act_fn)
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].post_attention_layernorm = patch_fct(
                layers[i].post_attention_layernorm
            )

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        base_model = model.model
        layers = base_model.layers
        self_attns = ["q_proj", "v_proj", "k_proj", "o_proj"]
        mlps = ["gate_proj", "up_proj", "down_proj"]

        for i in tqdm(range(len(layers)), disable=not verbose):
            self_attn_obj = layers[i].self_attn
            for item in self_attns:
                module = f"self_attn.{item}"
                quant_config = cls.get_optimal_config(model, i, module, patch_params)
                setattr(
                    self_attn_obj,
                    item,
                    patch_fct(getattr(self_attn_obj, item), quant_config),
                )
            mlp_obj = layers[i].mlp
            for item in mlps:
                module = f"mlp.{item}"
                quant_config = cls.get_optimal_config(model, i, module, patch_params)
                setattr(mlp_obj, item, patch_fct(getattr(mlp_obj, item), quant_config))

    @classmethod
    def get_optimal_config(
        cls, model, layer_no: int, module: str, global_quant_config: dict
    ) -> dict:
        config = global_quant_config[module]
        if config is None:
            return None
        quant_config = copy.deepcopy(config)
        if hasattr(model, "optimal_configs"):
            opt_tpl = model.optimal_configs[f"{layer_no}.{module}"]
            if opt_tpl:
                quant_config["weight_quant_params"]["nbits"] = opt_tpl[0]
                quant_config["weight_quant_params"]["group_size"] = opt_tpl[1]
                quant_config["weight_quant_params"]["round_zero"] = (
                    True if opt_tpl[0] == 4 else False
                )
        return quant_config


class LlamaHQQ(LLamaPatch, BaseHQQHFModel):
    pass
