import argparse
import csv
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.model_dict import get_model
from utils.config import get_config
from utils.data_us import EchoVideoDataset, JointTransform3D


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose video dataset geometry and, optionally, "
            "SharedGroundedMemSAM DINO point quality."
        )
    )
    parser.add_argument("--task", default="EchoDynamic")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--split", default=None, help="train/val/test. Defaults to config test_split.")
    parser.add_argument("--class_key", default=None)
    parser.add_argument("--frame_length", type=int, default=10)
    parser.add_argument("--encoder_input_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_csv", default="")

    parser.add_argument(
        "--mode",
        choices=["data", "grounding", "both"],
        default="data",
        help="data: dataset statistics only; grounding/both: also run model DINO grounding.",
    )

    # Model options used only for grounding/both.
    parser.add_argument("--modelname", default="SharedGroundedMemSAM")
    parser.add_argument("--load_path", default="")
    parser.add_argument("--sam_ckpt", default="/root/autodl-tmp/MemSAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--dino_config", default="/root/autodl-tmp/MemSAM/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino_weights", default="/root/autodl-tmp/MemSAM/groundingdino_swint_ogc.pth")
    parser.add_argument("--dino_use_lora", action="store_true")
    parser.add_argument("--dino_lora_weights", default="/root/autodl-tmp/MemSAM/weights/best_model.pth")
    parser.add_argument("--box_prompt_text", default="left ventricle")
    parser.add_argument("--dino_box_th", type=float, default=0.35)
    parser.add_argument("--dino_text_th", type=float, default=0.25)
    parser.add_argument("--disable_dino_prompt", action="store_true")
    parser.add_argument("--train_shared_dino", action="store_true")
    parser.add_argument("--enable_memory", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--enable_apfe", action="store_true")
    parser.add_argument("--apfe_kernel_size", type=int, default=7)
    parser.add_argument("--enable_phase_memory", action="store_true")
    parser.add_argument("--phase_memory_scale", type=float, default=0.1)
    parser.add_argument("--reset_reinforce_state_per_video", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mask_stats(mask):
    mask = as_numpy(mask) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape[-2:]
        return {
            "area": 0,
            "area_ratio": 0.0,
            "bbox_w": 0,
            "bbox_h": 0,
            "bbox_area_ratio": 0.0,
            "cx": math.nan,
            "cy": math.nan,
            "bbox_cx": math.nan,
            "bbox_cy": math.nan,
            "bbox_xyxy": None,
        }
    h, w = mask.shape[-2:]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    return {
        "area": int(mask.sum()),
        "area_ratio": float(mask.mean()),
        "bbox_w": bw,
        "bbox_h": bh,
        "bbox_area_ratio": float((bw * bh) / max(h * w, 1)),
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "bbox_cx": float((x0 + x1) * 0.5),
        "bbox_cy": float((y0 + y1) * 0.5),
        "bbox_xyxy": (x0, y0, x1, y1),
    }


def point_in_mask(mask, x, y):
    mask = as_numpy(mask) > 0
    h, w = mask.shape[-2:]
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    xi = min(max(xi, 0), w - 1)
    yi = min(max(yi, 0), h - 1)
    return bool(mask[yi, xi])


def cxcywh_to_xyxy(box, height, width):
    cx, cy, bw, bh = [float(v) for v in box]
    cx *= width
    bw *= width
    cy *= height
    bh *= height
    x0 = max(0.0, cx - bw * 0.5)
    y0 = max(0.0, cy - bh * 0.5)
    x1 = min(float(width - 1), cx + bw * 0.5)
    y1 = min(float(height - 1), cy + bh * 0.5)
    return x0, y0, x1, y1


def box_iou_xyxy(a, b):
    if a is None or b is None:
        return math.nan
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0 + 1.0), max(0.0, iy1 - iy0 + 1.0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0)
    area_b = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else math.nan


def summarize(values, name):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        print(f"{name}: n=0")
        return
    qs = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100])
    print(
        f"{name}: n={arr.size} mean={arr.mean():.6f} std={arr.std():.6f} "
        f"min={qs[0]:.6f} p5={qs[1]:.6f} p25={qs[2]:.6f} "
        f"median={qs[3]:.6f} p75={qs[4]:.6f} p95={qs[5]:.6f} max={qs[6]:.6f}"
    )


def load_checkpoint(model, load_path, device):
    if not load_path:
        return
    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_state_dict[key[7:] if key.startswith("module.") else key] = value
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[checkpoint] loaded {load_path}")
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    for key in list(unexpected)[:10]:
        print(f"  unexpected: {key}")


def build_dataset(args, opt):
    split = args.split or opt.test_split
    transform = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.encoder_input_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True,
    )
    return EchoVideoDataset(
        dataset_path=opt.data_path,
        split=split,
        joint_transform=transform,
        img_size=args.encoder_input_size,
        frame_length=args.frame_length,
        disable_point_prompt=True,
        class_key=args.class_key or getattr(opt, "data_subpath", None),
    )


def make_model_args(args, opt):
    return SimpleNamespace(
        modelname=args.modelname,
        sam_ckpt=args.sam_ckpt,
        encoder_input_size=args.encoder_input_size,
        low_image_size=args.encoder_input_size,
        batch_size=args.batch_size,
        device=args.device,
        dino_config=args.dino_config,
        dino_weights=args.dino_weights,
        dino_use_lora=args.dino_use_lora,
        dino_lora_weights=args.dino_lora_weights,
        box_prompt_text=args.box_prompt_text,
        dino_box_th=args.dino_box_th,
        dino_text_th=args.dino_text_th,
        disable_dino_prompt=args.disable_dino_prompt,
        train_shared_dino=args.train_shared_dino,
        enable_memory=args.enable_memory,
        reinforce=args.reinforce,
        enable_apfe=args.enable_apfe,
        apfe_kernel_size=args.apfe_kernel_size,
        enable_osu_prompt=False,
        osu_iters=5,
        osu_ns_type="classic",
        enable_osu_state=False,
        osu_state_iters=5,
        osu_state_alpha=0.98,
        osu_state_beta=1.0,
        osu_state_scale=2.0,
        enable_phase_memory=args.enable_phase_memory,
        phase_memory_scale=args.phase_memory_scale,
        reset_reinforce_state_per_video=args.reset_reinforce_state_per_video,
    )


def main():
    args = parse_args()
    opt = get_config(args.task)
    if args.data_path:
        opt.data_path = args.data_path
    opt.mode = "test"
    opt.visual = False
    opt.batch_size = args.batch_size

    dataset = build_dataset(args, opt)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = None
    if args.mode in ("grounding", "both"):
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model_args = make_model_args(args, opt)
        model = get_model(args.modelname, args=model_args, opt=opt).to(device)
        load_checkpoint(model, args.load_path, device)
        model.eval()
    else:
        device = torch.device("cpu")

    rows = []
    count = 0
    for datapack in loader:
        imgs = datapack["image"].float()
        masks = datapack["label"]
        names = datapack["image_name"]
        spacing = datapack.get("spacing", None)
        if spacing is not None:
            spacing_np = as_numpy(spacing)
        else:
            spacing_np = None

        grounding = None
        if model is not None:
            with torch.no_grad():
                out = model(imgs.to(device), None, None, return_prompts=True)
            grounding = out.get("grounding")

        b, t = masks.shape[:2]
        for bi in range(b):
            name = names[bi]
            h, w = masks.shape[-2:]
            ed = mask_stats(masks[bi, 0])
            es = mask_stats(masks[bi, -1])
            ed_es_dist = math.hypot(ed["cx"] - es["cx"], ed["cy"] - es["cy"])
            row = {
                "name": name,
                "height": h,
                "width": w,
                "ed_area_ratio": ed["area_ratio"],
                "es_area_ratio": es["area_ratio"],
                "ed_bbox_w_ratio": ed["bbox_w"] / max(w, 1),
                "ed_bbox_h_ratio": ed["bbox_h"] / max(h, 1),
                "es_bbox_w_ratio": es["bbox_w"] / max(w, 1),
                "es_bbox_h_ratio": es["bbox_h"] / max(h, 1),
                "es_over_ed_area": es["area"] / max(ed["area"], 1),
                "ed_es_centroid_dist_ratio": ed_es_dist / max(math.hypot(h, w), 1e-6),
            }
            if spacing_np is not None:
                sp = spacing_np[bi]
                row["spacing_y"] = float(sp[1]) if len(sp) > 1 else math.nan
                row["spacing_x"] = float(sp[0]) if len(sp) > 0 else math.nan

            if grounding is not None:
                point = as_numpy(grounding["points"])[bi, 0]
                box = as_numpy(grounding["boxes_cxcywh"])[bi]
                score = float(as_numpy(grounding["scores"])[bi])
                pred_xyxy = cxcywh_to_xyxy(box, h, w)
                gt_xyxy = ed["bbox_xyxy"]
                px, py = float(point[0]), float(point[1])
                row.update(
                    {
                        "dino_score": score,
                        "dino_x": px,
                        "dino_y": py,
                        "dino_inside_ed_mask": int(point_in_mask(masks[bi, 0], px, py)),
                        "dino_dist_to_ed_centroid_ratio": math.hypot(px - ed["cx"], py - ed["cy"])
                        / max(math.hypot(h, w), 1e-6),
                        "dino_dist_to_ed_bbox_center_ratio": math.hypot(px - ed["bbox_cx"], py - ed["bbox_cy"])
                        / max(math.hypot(h, w), 1e-6),
                        "dino_box_iou_ed": box_iou_xyxy(pred_xyxy, gt_xyxy),
                    }
                )
            rows.append(row)
            count += 1
            if args.max_samples and count >= args.max_samples:
                break
        if args.max_samples and count >= args.max_samples:
            break

    print(f"[summary] task={args.task} split={args.split or opt.test_split} samples={len(rows)}")
    for key in [
        "ed_area_ratio",
        "es_area_ratio",
        "ed_bbox_w_ratio",
        "ed_bbox_h_ratio",
        "es_bbox_w_ratio",
        "es_bbox_h_ratio",
        "es_over_ed_area",
        "ed_es_centroid_dist_ratio",
        "spacing_x",
        "spacing_y",
        "dino_score",
        "dino_inside_ed_mask",
        "dino_dist_to_ed_centroid_ratio",
        "dino_dist_to_ed_bbox_center_ratio",
        "dino_box_iou_ed",
    ]:
        if rows and key in rows[0]:
            summarize([r.get(key, math.nan) for r in rows], key)

    if rows and "dino_inside_ed_mask" in rows[0]:
        failures = [r for r in rows if r["dino_inside_ed_mask"] == 0]
        print(f"[grounding] outside_ed_mask={len(failures)}/{len(rows)}")
        for r in failures[:20]:
            print(
                "[grounding miss]",
                r["name"],
                f"score={r['dino_score']:.4f}",
                f"box_iou={r['dino_box_iou_ed']:.4f}",
                f"dist_centroid={r['dino_dist_to_ed_centroid_ratio']:.4f}",
            )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[csv] wrote {out_path}")


if __name__ == "__main__":
    main()
