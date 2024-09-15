from pathlib import Path

import numpy as np
import torch

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
    return model


def normalize_images_clip(data_np_in, BCHW=True):
    # Pre-processing
    mean_clip = np.array([0.4815, 0.4578, 0.4082], "float32")
    std_clip = np.array([0.2686, 0.2613, 0.2758], "float32")

    data_t = (
        torch.from_numpy(data_np_in).float()
        if (type(data_np_in) is np.ndarray)
        else data_np_in.float()
    )
    data_t = (data_t / 255.0 - mean_clip) / std_clip
    data_t = data_t.swapaxes(2, 3).swapaxes(1, 2) if (BCHW) else data_t
    return data_t


def compare(model_id):
    model = load_quantized(model_id)
    model = model.half()
    model = model.cuda()
    model.eval()
    # Load reference model to compare with
    model_ref = HQQOpenCLIP.create_model(model_id, device="cpu")
    model_ref = model_ref.half().cuda()
    model_ref.eval()

    # Compare the compressed model with the original
    x = np.random.rand(16, 224, 224, 3)
    x = normalize_images_clip(x).half().cuda()

    # Quantized model
    with torch.no_grad():
        y_q, _, _ = model(x)
        y_q /= torch.norm(y_q, p=2, dim=-1, keepdim=True)

    # Full-precision
    with torch.no_grad():
        y_r, _, _ = model_ref(x)
        y_r /= torch.norm(y_r, p=2, dim=-1, keepdim=True)

    # We want the dot product to be as close as possible to 1
    print(
        "Average dot-product score",
        float(torch.diag(torch.matmul(y_q, y_r.t())).mean()),
    )  # ~0.998 (ViT-H-14 @4bit)


def main():
    # quant_models(model_ids)
    # load_quantized(model_ids[0])
    compare(model_ids[2])


if __name__ == "__main__":
    main()
