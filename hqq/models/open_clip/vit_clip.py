import copy

from tqdm import tqdm

from ..base import BasePatch
from .base import BaseHQQOpenCLIPModel


# Patch ViT functions
class VitCLIPPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type.
    # For example, if you want to give different quantization parameters
    # to different layers
    @classmethod
    def get_linear_tags(cls):
        return [
            "mlp.c_fc",
            "mlp.c_proj",
            # "attn.out_proj",
        ]

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        model.visual.conv1 = patch_fct(model.visual.conv1)
        model.visual.ln_pre = patch_fct(model.visual.ln_pre)
        model.visual.ln_post = patch_fct(model.visual.ln_post)
        model.token_embedding = patch_fct(model.token_embedding)
        model.ln_final = patch_fct(model.ln_final)

        for i in tqdm(
            range(len(model.visual.transformer.resblocks)),
            desc="VisionModal-NL",
            disable=not verbose,
        ):
            model.visual.transformer.resblocks[i].ln_1 = patch_fct(
                model.visual.transformer.resblocks[i].ln_1
            )
            model.visual.transformer.resblocks[i].ln_2 = patch_fct(
                model.visual.transformer.resblocks[i].ln_2
            )

        for i in tqdm(
            range(len(model.transformer.resblocks)),
            desc="TextModal-NL",
            disable=not verbose,
        ):
            model.transformer.resblocks[i].ln_1 = patch_fct(
                model.transformer.resblocks[i].ln_1
            )
            model.transformer.resblocks[i].ln_2 = patch_fct(
                model.transformer.resblocks[i].ln_2
            )

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        # attns = ["out_proj"]
        mlps = ["c_fc", "c_proj"]

        # patch vision model
        blocks = model.visual.transformer.resblocks
        for i in tqdm(range(len(blocks)), desc="VisionModal-L", disable=not verbose):
            # attn_obj = blocks[i].attn
            # for item in attns:
            #     module = f"attn.{item}"
            #     quant_config = cls.get_optimal_config(model, i, module, patch_params)
            #     setattr(
            #         attn_obj,
            #         item,
            #         patch_fct(getattr(attn_obj, item), quant_config),
            #     )
            mlp_obj = blocks[i].mlp
            for item in mlps:
                module = f"mlp.{item}"
                quant_config = cls.get_optimal_config(
                    model, i, "vision", module, patch_params
                )
                setattr(mlp_obj, item, patch_fct(getattr(mlp_obj, item), quant_config))

        # patch text model
        blocks = model.transformer.resblocks
        for i in tqdm(range(len(blocks)), desc="TextModal-L", disable=not verbose):
            # attn_obj = blocks[i].attn
            # for item in attns:
            #     module = f"attn.{item}"
            #     quant_config = cls.get_optimal_config(model, i, module, patch_params)
            #     setattr(
            #         attn_obj,
            #         item,
            #         patch_fct(getattr(attn_obj, item), quant_config),
            #     )
            mlp_obj = blocks[i].mlp
            for item in mlps:
                module = f"mlp.{item}"
                quant_config = cls.get_optimal_config(
                    model, i, "text", module, patch_params
                )
                setattr(mlp_obj, item, patch_fct(getattr(mlp_obj, item), quant_config))

    @classmethod
    def get_optimal_config(
        cls,
        model,
        layer_no: int,
        module_type: str,
        module: str,
        global_quant_config: dict,
    ) -> dict:
        config = global_quant_config.get(module, None)
        if config is None:
            return None
        quant_config = copy.deepcopy(config)
        if hasattr(model, "optimal_configs"):
            opt_tpl = model.optimal_configs[f"{layer_no}.{module_type}.{module}"]
            if opt_tpl:
                quant_config["weight_quant_params"]["nbits"] = opt_tpl[0]
                quant_config["weight_quant_params"]["group_size"] = opt_tpl[1]
        return quant_config


class ViTCLIPHQQ(VitCLIPPatch, BaseHQQOpenCLIPModel):
    # layers to ignore when saving the weights
    @classmethod
    def get_ignore_layers(cls, model):
        return []

    # since cls_token and pos_embed are trainable parameters
    # but are not part of any module, we need to add them manually
    # for saving
    @classmethod
    def serialize_weights(cls, model, verbose):
        weights = super().serialize_weights(model, verbose)
        # weights["cls_token"] = model.cls_token.data
        # weights["pos_embed"] = model.pos_embed.data
        return weights

    # and loading
    @classmethod
    def post_module_load(cls, model, weights):
        super().post_module_load(model, weights)
        # model.cls_token.data = weights["cls_token"]
        # model.pos_embed.data = weights["pos_embed"]
