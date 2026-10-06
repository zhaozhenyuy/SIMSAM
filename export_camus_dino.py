import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


def to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    把各种可能的 frame 形状统一成 (H,W,3) uint8.
    支持：
      - (H,W)
      - (H,W,1)
      - (H,W,3)
      - (1,H,W) / (3,H,W)
      - (H,W,C) 但 C 不是 1/3（例如 256） -> 尝试判断并转置/压缩
    """
    arr = np.asarray(img)

    # 去掉长度为1的维度（但要小心别把 H/W 去掉）
    # 先只 squeeze 明确的单通道轴
    if arr.ndim == 3:
        # 常见 (1,H,W) 或 (H,W,1)
        if arr.shape[0] == 1 and arr.shape[1] > 1 and arr.shape[2] > 1:
            arr = arr[0]  # -> (H,W)
        elif arr.shape[2] == 1 and arr.shape[0] > 1 and arr.shape[1] > 1:
            arr = arr[:, :, 0]  # -> (H,W)

    # 处理 (C,H,W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[1] > 8 and arr.shape[2] > 8:
        arr = np.transpose(arr, (1, 2, 0))  # -> (H,W,C)

    # 现在希望是 (H,W) 或 (H,W,C)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)  # (H,W,3)
    elif arr.ndim == 3:
        H, W, C = arr.shape
        if C == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif C == 3:
            pass
        else:
            # 你遇到的 (1,1,256) 最终通常意味着维度顺序不对或数据被展平/压缩过
            # 这里给一个更鲁棒的兜底：如果有两个维度非常小，说明不是 HWC
            # 尝试把最大两个维度当作 H/W，把剩下当作通道或无效
            dims = list(arr.shape)
            # 选出最大的两个维度作为 H/W
            idx_sorted = np.argsort(dims)[::-1]
            h_i, w_i = idx_sorted[0], idx_sorted[1]
            # 把它们转到前两维
            perm = [h_i, w_i] + [i for i in range(3) if i not in (h_i, w_i)]
            arr2 = np.transpose(arr, perm)
            # arr2.shape = (H,W,rest)
            H2, W2, C2 = arr2.shape
            # 如果 C2 不是 1/3，取第一通道当灰度
            gray = arr2[:, :, 0]
            arr = np.stack([gray] * 3, axis=-1)

    else:
        raise ValueError(f"Unsupported frame ndim={arr.ndim}, shape={arr.shape}")

    # 归一化到 uint8
    arr = arr.astype(np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = arr * 0.0
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # 最终断言
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"to_hwc_uint8 failed, got shape={arr.shape}, dtype={arr.dtype}")
    return arr


def tight_bbox_from_mask(mask: np.ndarray):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2:
        # 有些 mask 是 (1,H,W)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        else:
            return None
    bin_m = m > 0
    if bin_m.sum() == 0:
        return None
    ys, xs = np.where(bin_m)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def resize_like_dino(img_rgb: np.ndarray, short_side=800, max_side=1333):
    H, W = img_rgb.shape[:2]
    scale = short_side / min(H, W)
    if max(H, W) * scale > max_side:
        scale = max_side / max(H, W)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    pil = Image.fromarray(img_rgb)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    out = np.array(pil)
    sx = new_w / W
    sy = new_h / H
    return out, sx, sy


def export_split(split, root_dir, out_root, resize=True):
    print(f"\n=== Exporting split: {split} ===")

    video_dir = Path(root_dir) / "videos" / split
    ann_dir = Path(root_dir) / "annotations" / split

    out_img_dir = Path(out_root) / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    video_files = sorted(video_dir.glob("*.npy"))

    for vid_path in video_files:
        stem = vid_path.stem
        npz_path = ann_dir / f"{stem}.npz"
        if not npz_path.exists():
            continue

        video = np.load(vid_path)
        ann = np.load(npz_path, allow_pickle=True)

        masks_3d = None
        fnum_mask = None

        for k in ["label", "mask", "masks", "seg", "gt"]:
            if k in ann.files:
                arr = ann[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.dtype != object:
                    masks_3d = arr
                    break

        if masks_3d is None:
            for k in ann.files:
                arr = ann[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.dtype != object:
                    masks_3d = arr
                    break

        if masks_3d is None:
            if "fnum_mask" in ann.files:
                fm = ann["fnum_mask"]
                if isinstance(fm, np.ndarray) and fm.dtype == object:
                    fm = fm.item()
                if isinstance(fm, dict):
                    fnum_mask = fm
                else:
                    raise RuntimeError(f"Unsupported fnum_mask type in {npz_path}")
            else:
                raise RuntimeError(f"No mask found in {npz_path}. keys={ann.files}")

        def get_mask(fid):
            if masks_3d is not None:
                if 0 <= fid < masks_3d.shape[0]:
                    return masks_3d[fid]
                return None
            if fid in fnum_mask:
                return fnum_mask[fid]
            if str(fid) in fnum_mask:
                return fnum_mask[str(fid)]
            return None

        T = int(video.shape[0])
        frame_ids = [0, T - 1]  # 你现在 CSV 就是 f000 / f009

        for fid in frame_ids:
            frame = video[fid]
            mask = get_mask(fid)
            if mask is None:
                continue

            bbox = tight_bbox_from_mask(mask)
            if bbox is None:
                continue

            img_rgb = to_hwc_uint8(frame)

            if resize:
                img_rgb, sx, sy = resize_like_dino(img_rgb)
                x, y, w, h = bbox
                x = int(round(x * sx))
                y = int(round(y * sy))
                w = int(round(w * sx))
                h = int(round(h * sy))
            else:
                x, y, w, h = bbox

            img_name = f"{stem}_f{fid:03d}.png"
            Image.fromarray(img_rgb).save(out_img_dir / img_name)

            rows.append([img_name, x, y, w, h, "left ventricle"])

    csv_path = Path(out_root) / split / f"{split}.csv"
    df = pd.DataFrame(rows, columns=[
        "image_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "label_name"
    ])
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(rows)} boxes to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/MemSAM/CAMUS_public")
    parser.add_argument("--out", default="/root/autodl-tmp/MemSAM/camus_dino")
    parser.add_argument("--no_resize", action="store_true")
    args = parser.parse_args()

    for sp in ["train", "val", "test"]:
        export_split(sp, root_dir=args.root, out_root=args.out, resize=not args.no_resize)


if __name__ == "__main__":
    main()