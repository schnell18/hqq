# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import copy

from ..base import BasePatch
from .base import BaseHQQHFModel
from ...core.quantize import BaseQuantizeConfig
from tqdm import tqdm


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
        self_attns = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        mlps = ['gate_proj', 'up_proj', 'down_proj']
        for i in tqdm(range(len(layers)), disable=not verbose):
            self_attn_obj = layers[i].self_attn
            for item in self_attns:
                quant_config = cls.get_optimal_config(i, patch_params[f"self_attn.{item}"])
                setattr(
                    self_attn_obj,
                    item,
                    patch_fct(
                        getattr(self_attn_obj, item),
                        cls.get_optimal_quant_config(i, patch_params, f"self_attn.{item}")
                    )
                )
            mlp_obj = layers[i].mlp
            for item in mlps:
                quant_config = cls.get_optimal_config(i, patch_params[f"mlp.{item}"])
                setattr(
                    mlp_obj,
                    item,
                    patch_fct(
                        getattr(mlp_obj, item),
                        cls.get_optimal_quant_config(i, patch_params, f"mlp.{item}")
                    )
                )

    @classmethod
    def get_optimal_quant_config(cls, layer_no: int, global_quant_config: dict, item: str) -> dict:
        config = global_quant_config[item]
        if config is None:
            quant_config = BaseQuantizeConfig()
        else:
            quant_config = copy.deepcopy(config)

        match item:
            case "self_attn.q_proj":
                quant_config['weight_quant_params']['nbits'] = 3
                quant_config['weight_quant_params']['group_size'] = 32
            case "self_attn.k_proj":
                if i > 1:
                    quant_config['weight_quant_params']['nbits'] = 3
                    quant_config['weight_quant_params']['group_size'] = 32
            case "self_attn.v_proj":
                if i < 30:
                    quant_config['weight_quant_params']['nbits'] = 3
                    quant_config['weight_quant_params']['group_size'] = 32

        return quant_config


class LlamaHQQ(LLamaPatch, BaseHQQHFModel):
    pass
