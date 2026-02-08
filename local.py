#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import PrototypeChooser


def infer_dims_from_state(state: dict):
    # Prototype vectors: [P, D, H, W]
    pv = state.get("prototype_vectors", None)
    if pv is None:
        raise KeyError("Could not find 'prototype_vectors' in checkpoint state_dict.")
    num_prototypes = pv.shape[0]
    proto_depth = pv.shape[1]

    # prototype_class_identity: [P, C]
    pci = state.get("prototype_class_identity", None)
    if pci is None:
        raise KeyError("Could not find 'prototype_class_identity' in checkpoint state_dict.")
    num_classes = pci.shape[1]

    # proto_presence: [C, P, N] (common in ProtoPool) or sometimes [C, P]
    pp = state.get("proto_presence", None)
    if pp is None:
        raise KeyError("Could not find 'proto_presence' in checkpoint state_dict.")
    if pp.dim() == 3:
        num_descriptive = pp.shape[2]
    else:
        # fallback if it's [C,P]; can't infer N -> set N=1
        num_descriptive = 1

    # infer whether last layer is used from checkpoint keys
    use_last_layer = any(k.startswith("last_layer.") for k in state.keys())

    return num_classes, num_prototypes, num_descriptive, proto_depth, use_last_layer


def build_model_from_checkpoint(state: dict, arch: str, pretrained: bool, add_on_layers_type: str,
                                proto_act_fn: str, use_thresh: bool, inat: bool, device):
    num_classes, num_prototypes, num_descriptive, proto_depth, use_last_layer = infer_dims_from_state(state)

    model = PrototypeChooser(
        num_prototypes=num_prototypes,
        num_descriptive=num_descriptive,
        num_classes=num_classes,
        use_thresh=use_thresh,
        arch=arch,
        pretrained=pretrained,
        add_on_layers_type=add_on_layers_type,
        prototype_activation_function=proto_act_fn,
        proto_depth=proto_depth,
        use_last_layer=use_last_layer,
        inat=inat,
    )

    # Match training override from your code
    model.conv1 = nn.Conv2d(3, 64, kernel_size=128, stride=2, padding=3, bias=False)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    # fallback: sometimes checkpoints are raw state_dict
    return ckpt


def preprocess_image(image_path: str):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # IMPORTANT: your training pipeline did NOT resize (Resize was commented out)
    # so we keep original size (likely 512x512) to match training.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return img, x


def distance_map_to_activation(dist_map: np.ndarray, max_dist: float, epsilon: float, mode: str):
    """
    dist_map: [h,w] numpy
    mode: 'log' or 'linear'
    """
    if mode == "log":
        act = np.log((dist_map + 1.0) / (dist_map + float(epsilon)))
    elif mode == "linear":
        act = max_dist - dist_map
    else:
        # default: negative distance
        act = -dist_map

    # normalize to 0..1 for visualization
    act = act - act.min()
    act = act / (act.max() + 1e-10)
    return act


def save_heatmap_overlay(out_path: str, pil_img: Image.Image, heat01: np.ndarray, alpha=0.45):
    """
    heat01: [H,W] values in [0,1]
    """
    import matplotlib.pyplot as plt

    img = np.array(pil_img).astype(np.float32) / 255.0  # [H,W,3]
    # matplotlib colormap
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat01)[..., :3].astype(np.float32)  # [H,W,3]

    overlay = (1 - alpha) * img + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.imsave(out_path, overlay)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Directory containing best_model.pth / best_model_push.pth")
    ap.add_argument("--ckpt_name", default="best_model_push.pth",
                    choices=["best_model_push.pth", "best_model.pth"])
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--image_name", required=True)
    ap.add_argument("--out_dir", default="./proto_vis_out")

    # These are inferred from your checkpoint folder name; override if your run differs.
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--pretrained", action="store_true")  # pass this flag if it was True in training
    ap.add_argument("--add_on_layers_type", default="log")
    ap.add_argument("--prototype_activation_function", default="log")
    ap.add_argument("--use_thresh", action="store_true")
    ap.add_argument("--inat", action="store_true")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt_path = str(Path(args.ckpt_dir) / args.ckpt_name)
    img_path = str(Path(args.image_dir) / args.image_name)

    state = load_checkpoint(ckpt_path, device)
    model = build_model_from_checkpoint(
        state=state,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        proto_act_fn=args.prototype_activation_function,
        use_thresh=args.use_thresh,
        inat=args.inat,
        device=device,
    )

    pil_img, x = preprocess_image(img_path)
    x = x.to(device)

    # --- Get prototype distance maps: [B,P,h,w]
    distances = model.prototype_distances(x)
    B, P, h, w = distances.shape
    print(f"distances: {tuple(distances.shape)}")

    # --- Global min distances per prototype for ranking: [P]
    min_dist = -F.max_pool2d(-distances, kernel_size=(h, w)).view(P)  # [P]
    # smallest distance => strongest activation
    top_protos = torch.topk(-min_dist, k=min(args.topk, P)).indices.detach().cpu().numpy().tolist()
    print("Top prototypes (strongest first):", top_protos)

    # We'll visualize the strongest one + also save top-k heatmaps
    out_root = Path(args.out_dir) / Path(args.image_name).stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Need max_dist + epsilon for the same activation used in training code
    proto_h = model.prototype_shape[2]
    proto_w = model.prototype_shape[3]
    max_dist = float(model.prototype_shape[1] * proto_h * proto_w)
    epsilon = float(getattr(model, "epsilon", 1e-4))

    # Upsample to original image size (H,W)
    H_img, W_img = pil_img.size[1], pil_img.size[0]

    for rank, pidx in enumerate(top_protos):
        dist_map = distances[0, pidx]  # [h,w]
        dist_np = dist_map.detach().cpu().numpy()

        heat01 = distance_map_to_activation(
            dist_np,
            max_dist=max_dist,
            epsilon=epsilon,
            mode=args.prototype_activation_function,
        )

        # Upsample heatmap using torch for alignment
        heat_t = torch.tensor(heat01, dtype=torch.float32).view(1, 1, h, w)
        heat_up = F.interpolate(heat_t, size=(H_img, W_img), mode="bilinear", align_corners=False)[0, 0].numpy()
        heat_up = (heat_up - heat_up.min()) / (heat_up.max() - heat_up.min() + 1e-10)

        # Save raw heat + overlay
        raw_path = out_root / f"proto_{pidx:03d}_rank{rank+1}_heat.png"
        overlay_path = out_root / f"proto_{pidx:03d}_rank{rank+1}_overlay.png"

        import matplotlib.pyplot as plt
        plt.imsave(str(raw_path), heat_up, cmap="jet")
        save_heatmap_overlay(str(overlay_path), pil_img, heat_up, alpha=0.45)

    # Also run the normal forward to show logits/pred
    logits, min_distances, proto_presence = model(x, gumbel_scale=0)
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    print("pred class:", pred)
    print("top probs:", np.sort(probs)[-min(5, probs.shape[0]):][::-1])

    print(f"Saved outputs to: {out_root}")


if __name__ == "__main__":
    main()
