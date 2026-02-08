#!/usr/bin/env python3
"""
proto_activation_single_image.py

Minimal evaluation script for ProtoPool/PrototypeChooser that:
- loads a trained checkpoint (best_model_push.pth / best_model.pth)
- reconstructs model dims from checkpoint keys (NO prototype_class_identity needed)
- runs a forward pass on ONE image
- computes prototype distance maps and converts them to activation heatmaps
- saves top-K prototype heatmaps + overlays

Usage example:

python local.py \
  --ckpt /data/pwojcik/ProtoPool/results/checkpoint/mito_descriptive-10_prototypes-50_lr-0.001_resnet18_True_log_log_warmup_ll_seed-0_2024-10-04_184840/best_model_push.pth \
  --image /data/pwojcik/mito_work/dataset_512_protopool/test/0_fl_fl/10kX_914-wt__0055.png \
  --arch resnet18 \
  --pretrained \
  --out_dir /data/pwojcik/ProtoPool/proto_vis_single \
  --topk 10
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Your project import
from model import PrototypeChooser


# -----------------------------
# Checkpoint utils (FIXED)
# -----------------------------
def load_checkpoint_state(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # Rare case: already a state_dict-like dict
        state = ckpt
    else:
        raise TypeError("Unsupported checkpoint format.")
    return state


def infer_dims_from_state(state: dict):
    """
    ProtoPool checkpoint keys (from your dump):
      - prototype_vectors: [P, D, H, W]
      - proto_presence:    [C, P, N]
      - last_layer.weight exists if use_last_layer=True
    """
    if "prototype_vectors" not in state:
        raise KeyError("Missing 'prototype_vectors' in model_state_dict.")
    if "proto_presence" not in state:
        raise KeyError("Missing 'proto_presence' in model_state_dict.")

    pv = state["prototype_vectors"]
    pp = state["proto_presence"]

    num_prototypes = int(pv.shape[0])
    proto_depth = int(pv.shape[1])

    num_classes = int(pp.shape[0])
    # proto_presence should be [C, P, N]
    num_descriptive = int(pp.shape[2]) if pp.dim() == 3 else 1

    use_last_layer = "last_layer.weight" in state
    return num_classes, num_prototypes, num_descriptive, proto_depth, use_last_layer


def build_model_from_checkpoint(
    state: dict,
    arch: str,
    pretrained: bool,
    add_on_layers_type: str,
    prototype_activation_function: str,
    proto_depth_override: int = None,
    use_thresh: bool = False,
    inat: bool = False,
    device: torch.device = torch.device("cpu"),
):
    num_classes, num_prototypes, num_descriptive, proto_depth, use_last_layer = infer_dims_from_state(state)
    if proto_depth_override is not None:
        proto_depth = int(proto_depth_override)

    model = PrototypeChooser(
        num_prototypes=num_prototypes,
        num_descriptive=num_descriptive,
        num_classes=num_classes,
        use_thresh=use_thresh,
        arch=arch,
        pretrained=pretrained,
        add_on_layers_type=add_on_layers_type,
        prototype_activation_function=prototype_activation_function,
        proto_depth=proto_depth,
        use_last_layer=use_last_layer,
        inat=inat,
    )

    # IMPORTANT: your training code overwrote conv1 like this
    # (and checkpoint contains 'conv1.weight')
    model.conv1 = nn.Conv2d(3, 64, kernel_size=128, stride=2, padding=3, bias=False)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    model.to(device).eval()

    print(f"Rebuilt model from ckpt:")
    print(f"  num_classes     = {num_classes}")
    print(f"  num_prototypes  = {num_prototypes}")
    print(f"  num_descriptive = {num_descriptive}")
    print(f"  proto_depth     = {proto_depth}")
    print(f"  use_last_layer  = {use_last_layer}")
    return model


# -----------------------------
# Image preprocessing
# -----------------------------
def load_and_preprocess_image(img_path: str, resize: int = None):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    tfm_list = []
    if resize is not None:
        tfm_list.append(transforms.Resize((resize, resize)))

    # Your training code did NOT resize (Resize was commented out),
    # so default is no resize. If you want 512->512, leave resize=None.
    tfm_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
    tfm = transforms.Compose(tfm_list)
    x = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return img, x


# -----------------------------
# Heatmap helpers
# -----------------------------
def dist_to_activation_map(dist_hw: np.ndarray, max_dist: float, epsilon: float, proto_act_fn: str) -> np.ndarray:
    """
    Convert prototype distance map [h,w] -> activation heatmap [h,w] normalized 0..1
    Mirrors your training push code:
      log:    log((d+1)/(d+eps))
      linear: max_dist - d
    """
    if proto_act_fn == "log":
        act = np.log((dist_hw + 1.0) / (dist_hw + float(epsilon)))
    elif proto_act_fn == "linear":
        act = (max_dist - dist_hw)
    else:
        # safe default: negative distance
        act = -dist_hw

    act = act - act.min()
    act = act / (act.max() + 1e-10)
    return act


def save_overlay(out_path: str, pil_img: Image.Image, heat01: np.ndarray, alpha: float = 0.45):
    """
    Saves overlay image using matplotlib colormap.
    heat01: [H,W] in [0,1]
    """
    import matplotlib.pyplot as plt

    img = np.array(pil_img).astype(np.float32) / 255.0  # [H,W,3]
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat01)[..., :3].astype(np.float32)

    overlay = (1 - alpha) * img + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.imsave(out_path, overlay)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best_model_push.pth or best_model.pth")
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--out_dir", default="./proto_vis_out", help="Where to save heatmaps/overlays")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=10, help="How many top prototypes to visualize")
    ap.add_argument("--resize", type=int, default=None, help="Optional resize (e.g. 512 or 224). Default: no resize.")

    # These should match your run folder naming:
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--pretrained", action="store_true")  # your folder shows True
    ap.add_argument("--add_on_layers_type", default="log")
    ap.add_argument("--prototype_activation_function", default="log")
    ap.add_argument("--use_thresh", action="store_true")
    ap.add_argument("--inat", action="store_true")

    # Optional override if needed
    ap.add_argument("--proto_depth_override", type=int, default=None)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    # Load state dict (FIXED: uses proto_presence, prototype_vectors, etc.)
    state = load_checkpoint_state(args.ckpt, device)

    # Build model from checkpoint (FIXED: no prototype_class_identity)
    model = build_model_from_checkpoint(
        state=state,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        prototype_activation_function=args.prototype_activation_function,
        proto_depth_override=args.proto_depth_override,
        use_thresh=args.use_thresh,
        inat=args.inat,
        device=device,
    )

    # Load image
    pil_img, x = load_and_preprocess_image(args.image, resize=args.resize)
    x = x.to(device)

    # 1) Forward pass (logits + min_distances)
    logits, min_distances, proto_presence = model(x, gumbel_scale=0)
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))

    print("logits shape:", tuple(logits.shape))
    print("min_distances shape:", tuple(min_distances.shape))
    print("proto_presence shape:", tuple(proto_presence.shape))
    print("pred class:", pred)

    # 2) Prototype distance maps for heatmaps
    distances = model.prototype_distances(x)  # [1,P,h,w]
    B, P, h, w = distances.shape
    print("distances shape:", tuple(distances.shape))

    # Rank prototypes by smallest global min distance (strongest activation)
    md = min_distances.squeeze(0) if min_distances.dim() == 2 else min_distances
    topk = min(args.topk, int(md.numel()))
    top_protos = torch.topk(-md, k=topk).indices.detach().cpu().numpy().tolist()
    print(f"Top-{topk} prototypes (strongest first):", top_protos)

    # constants for activation conversion
    proto_h = int(model.prototype_shape[2])
    proto_w = int(model.prototype_shape[3])
    max_dist = float(model.prototype_shape[1] * proto_h * proto_w)
    epsilon = float(getattr(model, "epsilon", 1e-4))

    # output folder
    out_root = Path(args.out_dir) / Path(args.image).stem
    out_root.mkdir(parents=True, exist_ok=True)

    # target image size
    H_img, W_img = pil_img.size[1], pil_img.size[0]

    import matplotlib.pyplot as plt

    for rank, pidx in enumerate(top_protos, start=1):
        dist_hw = distances[0, pidx].detach().cpu().numpy()  # [h,w]
        heat01 = dist_to_activation_map(
            dist_hw=dist_hw,
            max_dist=max_dist,
            epsilon=epsilon,
            proto_act_fn=args.prototype_activation_function,
        )

        # Upsample to image size
        heat_t = torch.tensor(heat01, dtype=torch.float32).view(1, 1, h, w)
        heat_up = F.interpolate(heat_t, size=(H_img, W_img), mode="bilinear", align_corners=False)[0, 0].numpy()
        heat_up = (heat_up - heat_up.min()) / (heat_up.max() - heat_up.min() + 1e-10)

        raw_path = out_root / f"proto_{pidx:03d}_rank{rank:02d}_heat.png"
        overlay_path = out_root / f"proto_{pidx:03d}_rank{rank:02d}_overlay.png"

        plt.imsave(str(raw_path), heat_up, cmap="jet")
        save_overlay(str(overlay_path), pil_img, heat_up, alpha=0.45)

    # Save a small text summary too
    summary_path = out_root / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"image: {args.image}\n")
        f.write(f"pred_class: {pred}\n")
        f.write(f"top_prototypes: {top_protos}\n")
        top_probs = np.argsort(probs)[-min(10, probs.shape[0]):][::-1]
        f.write(f"top_prob_indices: {top_probs.tolist()}\n")
        f.write(f"top_prob_values: {probs[top_probs].tolist()}\n")

    print(f"Saved prototype heatmaps + overlays to:\n  {out_root}")
    print(f"Summary written to:\n  {summary_path}")


if __name__ == "__main__":
    main()
