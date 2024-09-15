import os
from pathlib import Path

import open_clip
import torch
from open_clip import tokenizer
from PIL import Image
from skimage import data_dir

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


def compare_weights_raw(model_id):
    qnt_pt_file = "snapshots/" + model_id + "/qmodel.pt"
    ref_pt_file = "/home/justin/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin"
    qnt = torch.load(qnt_pt_file)
    ref = torch.load(ref_pt_file)
    print("*" * 72)
    print("from quantized file")
    for k in qnt:
        print(k)
    print("*" * 72)
    print("from original file")
    for k in ref:
        print(k)


def compare_weights(model_id):
    model, model_ref, _ = create_model_dual(model_id)
    dict1 = dict(model.named_parameters())
    dict2 = dict(model_ref.named_parameters())
    for name, param in dict1.items():
        param_ref = dict2.get(name, None)
        if param_ref is not None:
            if not torch.allclose(param_ref.data, param.data, rtol=0.0, equal_nan=True):
                print(f"weight differs: {name}")
                if name == "text_projection":
                    print(param)
                    print(param_ref)
            else:
                print(f"weight same: {name}")
        else:
            print(f"not in ref model: {name}")


def create_model_dual(model_id, mask=3):
    model = None
    model_ref = None
    preprocess = None

    if mask & 1:
        model = load_quantized(model_id)
        model = model.half().cuda()

    # Load reference model to compare with
    if mask & 2:
        comps = model_id.split("/")
        elems = comps[1].split("-")
        model_name = "-".join(elems[1:4])
        pretrained = "-".join(elems[4:])
        model_ref, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model_ref = model_ref.half().cuda()
    return model, model_ref, preprocess


def create_and_quant_model(model_id):
    model = HQQOpenCLIP.create_model(model_id, device="cpu")
    # Quantize settings
    # quant_config = BaseQuantizeConfig(nbits=8, group_size=128)
    quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
    # quant_config = BaseQuantizeConfig(nbits=3, group_size=64)
    # quant_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_scale=True)
    # Quantize
    model.quantize_model(quant_config=quant_config)
    model = model.half().cuda()
    return model


def compare(model_id):
    model = create_and_quant_model(model_id)
    _, model_ref, preprocess = create_model_dual(model_id, mask=2)
    model.eval()
    model_ref.eval()

    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer",
    }
    texts = descriptions.values()
    text_processed = tokenizer.tokenize(texts).cuda()

    # preprocess image and text
    img = Image.open(os.path.join(data_dir, "astronaut.png")).convert("RGB")
    img_preprocessed = preprocess(img).cuda().unsqueeze(0)

    with torch.amp.autocast("cuda"):
        img_embedding, text_embedding, _ = model_ref(img_preprocessed, text_processed)
    probs = (100 * img_embedding @ text_embedding.T).softmax(dim=-1)
    print(probs)

    with torch.amp.autocast("cuda"):
        img_embedding, text_embedding, _ = model(img_preprocessed, text_processed)
    probs = (100 * img_embedding @ text_embedding.T).softmax(dim=-1)
    print(probs)


def main():
    # quant_models(model_ids)
    # load_quantized(model_ids[0])
    compare(model_ids[2])
    # compare_weights(model_ids[2])
    # compare_weights_raw(model_ids[2])


if __name__ == "__main__":
    main()
