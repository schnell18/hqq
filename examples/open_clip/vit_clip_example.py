from pathlib import Path

from hqq.core.quantize import BaseQuantizeConfig
from hqq.engine.open_clip import HQQOpenCLIP

model_ids = [
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
]


def quant_models(model_ids):
    for model_id in model_ids:
        # Load model (on CPU)
        model = HQQOpenCLIP.create_model(model_id, device="cpu")

        # Quantize settings
        # quant_config = BaseQuantizeConfig(nbits=8, group_size=128)
        quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
        # quant_config = BaseQuantizeConfig(nbits=3, group_size=64)
        # quant_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_scale=True)

        # Quantize
        model.quantize_model(quant_config=quant_config)

        # Save model
        save_dir = "snapshots/" + model_id
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model.save_quantized(save_dir=save_dir)


def load_quantized(model_id):
    # Load model
    save_dir = "snapshots/" + model_id
    model = HQQOpenCLIP.from_quantized(save_dir)
    print(model)


def main():
    # quant_models(model_ids)
    load_quantized(model_ids[0])


if __name__ == "__main__":
    main()
