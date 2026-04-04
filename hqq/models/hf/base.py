# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023

from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from ..base import BaseHQQModel, BasePatch

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None


class BaseHQQHFModel(BaseHQQModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        # Update model architecture in the config
        model.config.architectures = [model.__class__.__name__]
        # Save config
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, **kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        config = AutoConfig.from_pretrained(cls.get_config_file(save_dir))
        auto_class = resolve_automodel_class(config)

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs)

        return model


# Auto class used for HF models if no architecture was manually setup
class AutoHQQHFModel(BaseHQQHFModel, BasePatch):
    pass


def resolve_automodel_class(config):
    model_type = getattr(config, "model_type", "")

    # --- Detect VLM (Vision-Language Models) ---
    # Strong signals for multimodal
    has_vision = any(
        [
            hasattr(config, "vision_config"),
            hasattr(config, "vision_tower"),
            hasattr(config, "image_token_index"),
            "vision" in model_type.lower(),
            "vl" in model_type.lower(),
        ]
    )

    if has_vision:
        # Prefer newest unified API if available
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText

        # Fallback for older transformers
        if AutoModelForVision2Seq is not None:
            return AutoModelForVision2Seq

        # Last fallback
        return AutoModel

    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
    # --- Seq2Seq (text-to-text) ---
    if is_encoder_decoder:
        return AutoModelForSeq2SeqLM

    # --- Causal LM (decoder-only) ---
    # Covers GPT, LLaMA, Qwen (text-only), etc.
    if hasattr(config, "architectures"):
        arch = " ".join(config.architectures).lower()
        if "causallm" in arch or "forcausallm" in arch:
            return AutoModelForCausalLM

    # --- Default fallback ---
    return AutoModel
