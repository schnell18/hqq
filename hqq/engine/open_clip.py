import json

import torch

from ..models.base import BaseHQQModel
from ..models.open_clip.vit_clip import ViTCLIPHQQ
from .base import HQQWrapper

_HQQ_REGISTRY = {}
_HQQ_REGISTRY["ViT-B-16"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-plus-240"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-plus"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-quickgelu"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-SigLIP-256"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-SigLIP-384"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-SigLIP-512"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-SigLIP-i18n-256"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-16-SigLIP"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-32-256"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-32"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-32-plus-256"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-B-32-quickgelu"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-bigG-14-CLIPA-336"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-bigG-14-CLIPA"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-bigG-14"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-e-14"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-g-14"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-H-14-378-quickgelu"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-H-14-CLIPA-336"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-H-14-CLIPA"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-H-14"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-H-14-quickgelu"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-H-16"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-14-280"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-14-336"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-14-CLIPA-336"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-14-CLIPA"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-14"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-14-quickgelu"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-16-320"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-16"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-16-SigLIP-256"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-L-16-SigLIP-384"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-M-16-alt"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-M-16"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-M-32-alt"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-M-32"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-S-16-alt"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-S-16"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-S-32-alt"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-S-32"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-SO400M-14-SigLIP-384"] = ViTCLIPHQQ
_HQQ_REGISTRY["ViT-SO400M-14-SigLIP"] = ViTCLIPHQQ


class HQQOpenCLIP(HQQWrapper):
    _HQQ_REGISTRY = _HQQ_REGISTRY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _make_quantizable(cls, model, quantized: bool):
        model.hqq_quantized = quantized
        model.quantize_model = (
            lambda quant_config,
            compute_dtype=torch.float16,
            device="cuda": cls.quantize_model_(
                model=model,
                quant_config=quant_config,
                compute_dtype=compute_dtype,
                device=device,
            )
        )
        model.save_quantized = lambda save_dir: cls.save_quantized_(
            model=model, save_dir=save_dir
        )
        # model.cuda = lambda *args, **kwargs: model if (quantized) else model.cuda
        # model.to = lambda *args, **kwargs: model if (quantized) else model.to
        # model.float = lambda *args, **kwargs: model if (quantized) else model.float
        # model.half = lambda *args, **kwargs: model if (quantized) else model.half
        model.base_class = ViTCLIPHQQ

    @classmethod
    def _validate_params(cls, params: dict):
        pass

    @classmethod
    def create_model(cls, model_id, **kwargs):
        cls._validate_params(kwargs)
        # check if the first positional argument is an directory that exists
        comps = model_id.split("/")
        elems = comps[1].split("-")
        model_name = "-".join(elems[1:4])
        pretrained = "-".join(elems[4:])
        model = cls._get_hqq_class(model_name).create_model(
            model_name, model_name=model_name, pretrained=pretrained, **kwargs
        )
        cls._make_quantizable(model, quantized=False)
        return model

    @classmethod
    def wrap_model(cls, model, model_name):
        model.arch_key = model_name
        cls._make_quantizable(model, quantized=False)
        return model

    @classmethod
    def _get_arch_key_from_save_dir(cls, save_dir: str):
        with open(BaseHQQModel.get_config_file(save_dir), "r") as file:
            config = json.load(file)
        return config["architecture"]
