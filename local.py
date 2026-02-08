#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Your project imports
from model import PrototypeChooser


def build_model(args, device):
    model = PrototypeChooser(
        num_prototypes=args.num_prototypes,
        num_descriptive=args.num_descriptive,
        num_classes=args.num_classes,
        use_thresh=args.use_thresh,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        prototype_activation_function=args.prototype_activation_function,
        proto_depth=args.proto_depth,
        use_last_layer=args.last_layer,
        inat=args.inat,
    )

    # IMPORTANT: you had this override in training
    model.conv1 = nn.Conv2d(3, 64, kernel_size=128, stride=2, padding=3, bias=False)

    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # supports both {"model_state_dict": ...} and raw state_dict
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    print(f"Loaded checkpoint: {ckpt_path}" + (f" (epoch={epoch})" if epoch is not None else ""))
    return model


def preprocess_image(img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),  # safe default; remove if your data is already sized
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    x = tfm(img).unsqueeze(0)  # [1,3,224,224]
    return x


@torch.no_grad()
def run_forward(model, x, device, gumbel_scale=0):
    x = x.to(device)
    logits, min_distances, proto_presence = model(x, gumbel_scale=gumbel_scale)
    return logits, min_distances, proto_presence


def main():
    p = argparse.ArgumentParser("Minimal single-image forward for PrototypeChooser")
    p.add_argument("--checkpoint", "-c", required=True, help="Path to .pth checkpoint (must contain model_state_dict).")
    p.add_argument("--image", "-i", required=True, help="Path to an image file.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gumbel_scale", type=float, default=0.0)

    # model config (must match training)
    p.add_argument("--num_prototypes", type=int, default=200)
    p.add_argument("--num_descriptive", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--arch", type=str, default="resnet34")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--add_on_layers_type", type=str, default="log")
    p.add_argument("--prototype_activation_function", type=str, default="log")
    p.add_argument("--proto_depth", type=int, default=128)
    p.add_argument("--last_layer", action="store_true")
    p.add_argument("--inat", action="store_true")
    p.add_argument("--use_thresh", action="store_true")

    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    model = build_model(args, device)
    model = load_checkpoint(model, args.checkpoint, device)

    x = preprocess_image(args.image)
    logits, min_distances, proto_presence = run_forward(model, x, device, gumbel_scale=args.gumbel_scale)

    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1).item()

    print("logits shape:", tuple(logits.shape))
    print("pred class:", pred)
    print("top-5 probs:", torch.topk(probs[0], k=min(5, probs.shape[-1])).values.cpu().numpy())
    print("top-5 idxs :", torch.topk(probs[0], k=min(5, probs.shape[-1])).indices.cpu().numpy())

    print("min_distances shape:", tuple(min_distances.shape))   # expect [1,P] or [P]
    print("proto_presence shape:", tuple(proto_presence.shape)) # depends on your model

    # Helpful: print most activated prototypes for this image
    md = min_distances.squeeze(0) if min_distances.dim() == 2 else min_distances
    # smaller distance => stronger activation (usually)
    top_proto = torch.topk(-md, k=min(10, md.numel())).indices.cpu().numpy()
    print("top-10 prototypes (by smallest distance):", top_proto)


if __name__ == "__main__":
    main()
