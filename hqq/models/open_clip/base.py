import json
import os
from typing import Union

import torch
from open_clip import create_model as oc_create_model
from torch import float16

from ...core.quantize import HQQLinear
from ...utils.optimizer import find_optimal_configs
from ..base import BaseHQQModel


class BaseHQQOpenCLIPModel(BaseHQQModel):
    @classmethod
    def create_model(cls, save_dir, **kwargs):
        # check if the first positional argument is an directory that exists
        if os.path.exists(save_dir):
            # load CLIP model name from config.json
            with open(BaseHQQModel.get_config_file(save_dir), "r") as file:
                config = json.load(file)
                model_name = config["architecture"]
            pretrained = None
        else:
            model_name = kwargs.pop("model_name")
            pretrained = kwargs.pop("pretrained")
        device = kwargs.pop("device", "cpu")

        model = oc_create_model(
            model_name,
            pretrained,
            device=device,
            **kwargs,
        )
        model.arch_key = model_name
        return model

    @classmethod
    def cache_model(cls, model, save_dir):
        # save CLIP model name into config.json
        with open(BaseHQQModel.get_config_file(save_dir), "w") as file:
            config = {"architecture": model.arch_key}
            json.dump(config, file, indent=2)

    # Main function to quantize a model. Basically goes through the linear
    # layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda",
    ):
        # Check if the model was already quantized
        if getattr(model, "hqq_quantized", False):
            print("Model was already quantized")
            return

        # Set linear tags automatically
        cls.setup_model(model)

        if "budget" in quant_config:
            budget = quant_config.pop("budget")
        if "mixed" in quant_config:
            mixed = quant_config.pop("mixed")
            if mixed:
                metrics_file = quant_config.pop("quant_metrics_file")
                weight_algo = quant_config.pop("weight_algo", None)
                boost_layers = quant_config.pop("boost_layers", None)
                decline_layers = quant_config.pop("decline_layers", None)
                boost_stop = quant_config.pop("boost_stop", None)
                decline_stop = quant_config.pop("decline_stop", None)
                top_m_layer = quant_config.pop("top_m_layer", None)
                ablation = quant_config.pop("ablation", None)
                factor = quant_config.pop("factor", None)
                kwargs = {
                    "weight_algo": weight_algo,
                    "boost_layers": boost_layers,
                    "decline_layers": decline_layers,
                    "boost_stop": boost_stop,
                    "decline_stop": decline_stop,
                    "top_m_layer": top_m_layer,
                    "ablation": ablation,
                    "factor": factor,
                }
                optimal_configs, _ = find_optimal_configs(
                    metrics_file, budget, time_limit=200, verbose=True, **kwargs
                )
                model.optimal_configs = optimal_configs

        # Use the same quantization config for all linear layers.
        # Use None to skip quantizing a specfic layer.
        if True in [(key in model.linear_tags) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags,
            # the layer is not quantized via (key, None)
            patch_params = {key: None for key in model.linear_tags}
            patch_params.update(quant_config)
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in model.linear_tags}

        # We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            if type(linear_layer) is HQQLinear:
                return linear_layer

            current_device = "cuda"

            if quant_config is not None:
                out_module = HQQLinear(
                    linear_layer,
                    quant_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            current_device = "cuda"
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        # Set base class
        model.base_class = cls

        model.hqq_quantized = True

        return model
