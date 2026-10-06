# /root/autodl-tmp/MemSAM/tools/export_camus_dino_from_loader.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from utils.data_us import EchoVideoDataset

# =========================
# 配置区：按需修改
# =========================
CAMUS_ROOT = "/root/autodl-tmp/MemSAM/CAMUS_public"
OUT_ROOT = Path("/root/autodl-tmp/MemSAM/camus_dino_256_rotcw90")

# 导出哪些 split
EXPORT_SPLITS = ["train", "val", "test"]

# 导出模式：
#   "FIRST" -> 只导首帧
#   "ED_ES" -> 导首尾两帧（你当前 10 帧 clip 下近似 ED/ES）
EXPORT_MODE = "ED_ES"

# DINO 的文本标签
LABEL_TEXT = "left ventricle"

# 是否顺时针旋转 90°
ROTATE_CW90 = False

# 是否跳过空 mask
SKIP_EMPTY_MASK = True

# dataloader 参数
BATCH_SIZE = 1
NUM_WORKERS = 0
IMG_SIZE = 256
FRAME_LENGTH = 10


def tight_bbox_from_mask(mask_2d: np.ndarray):
    """
    输入:
        mask_2d: (H,W), 非零表示前景
    返回:
        (x, y, w, h) 或 None
    """
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # 宽高按像素闭区间计算
    w = int(x1 - x0 + 1)
    h = int(y1 - y0 + 1)
    return int(x0), int(y0), w, h


def rotate_frame_and_mask_cw90(frame_chw: torch.Tensor, mask_hw: np.ndarray):
    """
    frame_chw: (3,H,W) torch.Tensor
    mask_hw:   (H,W) numpy.ndarray
    返回:
        rotated_frame_chw, rotated_mask_hw
    顺时针90°旋转
    """
    # torch.rot90: k=-1 等价于顺时针90°
    frame_rot = torch.rot90(frame_chw, k=-1, dims=(1, 2))
    mask_rot = np.rot90(mask_hw, k=-1)
    return frame_rot, mask_rot


def save_rgb_png(img_chw: torch.Tensor, out_path: Path):
    """
    img_chw: (3,H,W) torch tensor, dtype can be uint8/float
    保存为 uint8 RGB PNG
    """
    x = img_chw.detach().cpu()

    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expect (3,H,W), got {tuple(x.shape)}")

    x = x.permute(1, 2, 0).contiguous()  # HWC
    arr = x.numpy()

    if arr.dtype != np.uint8:
        a_min, a_max = float(arr.min()), float(arr.max())

        if a_max <= 1.0 and a_min >= 0.0:
            arr = arr * 255.0
        elif a_max <= 1.0 and a_min >= -1.0:
            arr = (arr + 1.0) * 0.5 * 255.0
        # 否则视为 float 的 0~255

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(out_path)


def export_split(split: str):
    out_img_dir = OUT_ROOT / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_ROOT / split / f"{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ds = EchoVideoDataset(
        dataset_path=CAMUS_ROOT,
        split=split,
        joint_transform=None,
        img_size=IMG_SIZE,
        prompt="click",
        class_id=1,
        one_hot_mask=0,
        frame_length=FRAME_LENGTH,
        disable_point_prompt=True,
        point_numbers=1,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    rows = []

    for i, datapack in enumerate(dl):
        imgs = datapack["image"]      # (1,T,3,256,256)
        masks = datapack["label"]     # (1,T,256,256)
        image_name = datapack["image_name"][0]

        stem = image_name.split(".")[0]
        T = imgs.shape[1]

        if EXPORT_MODE == "FIRST":
            frame_ids = [0]
        elif EXPORT_MODE == "ED_ES":
            frame_ids = [0, T - 1]
        else:
            raise ValueError(f"Unsupported EXPORT_MODE: {EXPORT_MODE}")

        for fid in frame_ids:
            frame = imgs[0, fid]  # (3,H,W)
            mask2d = masks[0, fid].detach().cpu().numpy()  # (H,W)

            # 先做方向统一，再算 bbox
            if ROTATE_CW90:
                frame, mask2d = rotate_frame_and_mask_cw90(frame, mask2d)

            bbox = tight_bbox_from_mask(mask2d)

            if bbox is None:
                if SKIP_EMPTY_MASK:
                    continue
                else:
                    raise RuntimeError(f"Empty mask found: split={split}, image={image_name}, fid={fid}")

            img_name = f"{stem}_f{fid:03d}.png"
            save_rgb_png(frame, out_img_dir / img_name)

            x, y, bw, bh = bbox
            rows.append([img_name, x, y, bw, bh, LABEL_TEXT])

        if (i + 1) % 50 == 0:
            print(f"[{split}] processed {i + 1}/{len(ds)}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "label_name",
        ])
        writer.writerows(rows)

    print(f"[OK] split={split} images={len(rows)} csv={out_csv}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sp in EXPORT_SPLITS:
        export_split(sp)


if __name__ == "__main__":
    main()