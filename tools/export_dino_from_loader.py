import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as torchF
from PIL import Image
from torch.utils.data import DataLoader

from utils.data_us import EchoVideoDataset


def tight_bbox_from_mask(mask_2d: np.ndarray):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def save_rgb_png(img_chw: torch.Tensor, out_path: Path):
    x = img_chw.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Expect CHW image, got {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expect 1 or 3 channels, got {tuple(x.shape)}")

    arr = x.permute(1, 2, 0).contiguous().numpy()
    if arr.dtype != np.uint8:
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max <= 1.0 and a_min >= 0.0:
            arr = arr * 255.0
        elif a_max <= 1.0 and a_min >= -1.0:
            arr = (arr + 1.0) * 127.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def resize_frame_and_mask(frame_chw: torch.Tensor, mask_hw: torch.Tensor, size: int):
    frame = frame_chw.float().unsqueeze(0)
    mask = mask_hw.float().unsqueeze(0).unsqueeze(0)
    frame = torchF.interpolate(frame, size=(size, size), mode="bilinear", align_corners=False)[0]
    mask = torchF.interpolate(mask, size=(size, size), mode="nearest")[0, 0]
    return frame, mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export EchoVideoDataset frames and mask boxes for GroundingDINO training."
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--export_mode", choices=["FIRST", "ED_ES", "ALL"], default="ED_ES")
    parser.add_argument("--label_text", default="left ventricle")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--resize_to_img_size", action="store_true")
    parser.add_argument("--frame_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--class_key", default=None)
    parser.add_argument("--skip_empty_mask", action="store_true", default=True)
    parser.add_argument("--keep_empty_mask", action="store_false", dest="skip_empty_mask")
    return parser.parse_args()


def frame_ids_for_mode(mode: str, frame_count: int):
    if mode == "FIRST":
        return [0]
    if mode == "ED_ES":
        return [0, frame_count - 1]
    if mode == "ALL":
        return list(range(frame_count))
    raise ValueError(f"Unsupported export mode: {mode}")


def export_split(args, split: str):
    out_root = Path(args.out_root)
    out_img_dir = out_root / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_root / split / f"{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ds = EchoVideoDataset(
        dataset_path=args.dataset_path,
        split=split,
        joint_transform=None,
        img_size=args.img_size,
        frame_length=args.frame_length,
        disable_point_prompt=True,
        class_key=args.class_key,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows = []
    for i, datapack in enumerate(dl):
        imgs = datapack["image"]
        masks = datapack["label"]
        image_name = datapack["image_name"][0]
        stem = image_name.split(".")[0]
        frame_count = imgs.shape[1]

        for frame_id in frame_ids_for_mode(args.export_mode, frame_count):
            frame = imgs[0, frame_id]
            mask = masks[0, frame_id]
            if args.resize_to_img_size:
                frame, mask = resize_frame_and_mask(frame, mask, args.img_size)
            mask2d = mask.detach().cpu().numpy()
            bbox = tight_bbox_from_mask(mask2d)
            if bbox is None:
                if args.skip_empty_mask:
                    continue
                raise RuntimeError(f"Empty mask found: split={split}, image={image_name}, frame={frame_id}")

            img_name = f"{stem}_f{frame_id:03d}.png"
            save_rgb_png(frame, out_img_dir / img_name)
            x, y, w, h = bbox
            rows.append([img_name, x, y, w, h, args.label_text])

        if (i + 1) % 50 == 0:
            print(f"[{split}] processed {i + 1}/{len(ds)}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "label_name"])
        writer.writerows(rows)

    print(f"[OK] split={split} images={len(rows)} csv={out_csv}")


def main():
    args = parse_args()
    for split in args.splits:
        export_split(args, split)


if __name__ == "__main__":
    main()
