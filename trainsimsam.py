import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.model_dict import get_model
from utils.config import get_config
from utils.data_us import EchoVideoDataset, JointTransform3D
from utils.evaluation import get_eval
from utils.loss_functions.sam_loss import get_criterion


MODEL_NAME = "SharedGroundedMemSAM"


def parse_args():
    parser = argparse.ArgumentParser(description="Train SIMSAM on echocardiography videos.")
    parser.add_argument(
        "--task",
        choices=("CAMUS_Video_Full", "EchoNet_Video"),
        default="CAMUS_Video_Full",
    )
    parser.add_argument("--data_path", default=None, help="Dataset root. Uses data/<dataset> by default.")
    parser.add_argument("--output_dir", default=None, help="Checkpoint directory.")
    parser.add_argument("--sam_ckpt", default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--dino_weights", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--dino_lora_weights", default=None)
    parser.add_argument(
        "--dino_config",
        default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--epochs", type=int, default=0, help="0 uses the dataset default.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--frame_length", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=25)
    parser.add_argument("--phase_memory_scale", type=float, default=0.1)
    parser.add_argument("--es_loss_weight", type=float, default=1.2)
    parser.add_argument("--es_boundary_loss_weight", type=float, default=0.1)
    parser.add_argument("--es_area_loss_weight", type=float, default=0.05)
    parser.add_argument("--resume", default=None, help="Resume from a training checkpoint.")
    parser.add_argument("--disable_phase_memory", action="store_true", help="Ablation only.")
    parser.add_argument("--disable_apfe", action="store_true", help="Disable APFE for ablation.")
    parser.add_argument("--full_supervision", action="store_true", help="Supervise every sampled frame.")
    parser.add_argument("--no_dino_lora", action="store_true", help="Use the base GroundingDINO weights.")
    parser.add_argument("--smoke_test_samples", type=int, default=0)
    return parser.parse_args()


def apply_model_defaults(args):
    args.modelname = MODEL_NAME
    args.encoder_input_size = 256
    args.low_image_size = 256
    args.point_numbers = 1
    args.enable_memory = True
    args.disable_point_prompt = True
    args.semi = not args.full_supervision
    args.enable_phase_memory = not args.disable_phase_memory
    args.enable_apfe = not args.disable_apfe
    args.apfe_kernel_size = 7
    args.dino_use_lora = not args.no_dino_lora
    args.train_shared_dino = False
    args.box_prompt_text = "left ventricle"
    args.dino_box_th = 0.35
    args.dino_text_th = 0.25
    args.enable_box_prompt = False
    args.enable_self_prompt = False
    args.enable_es_shape_loss = True
    args.seg_threshold = 0.6
    args.compute_ef = False
    args.bidirectional_endpoint_eval = False
    args.bidirectional_fusion = "endpoint"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_inputs(args):
    required = [args.sam_ckpt, args.dino_weights]
    if args.dino_use_lora:
        required.append(args.dino_lora_weights)
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Required weight files were not found:\n  {joined}")
    if not Path(args.data_path).is_dir():
        raise FileNotFoundError(f"Dataset directory was not found: {args.data_path}")


def clean_state_dict(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_compatible_weights(model, state_dict):
    source = clean_state_dict(state_dict)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return result, len(source) - len(compatible)


def endpoint_weighted_loss(criterion, prediction, target, es_weight):
    ed_loss = criterion(prediction[:, [0], 0], target[:, [0]])
    es_loss = criterion(prediction[:, [-1], 0], target[:, [-1]])
    return (ed_loss + es_weight * es_loss) / (1.0 + es_weight)


def sobel_edges(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    kernel_x = mask.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = mask.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    edge = torch.sqrt(
        F.conv2d(mask.float(), kernel_x, padding=1).square()
        + F.conv2d(mask.float(), kernel_y, padding=1).square()
        + 1e-6
    )
    maximum = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return edge / maximum


def boundary_loss(prediction, target):
    pred_edge = sobel_edges(prediction)
    target_edge = sobel_edges(target)
    dims = (1, 2, 3)
    intersection = (pred_edge * target_edge).sum(dim=dims)
    denominator = pred_edge.square().sum(dim=dims) + target_edge.square().sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return 0.5 * dice + 0.5 * F.l1_loss(pred_edge, target_edge)


def area_ratio_loss(prediction, target):
    pred_ed = prediction[:, 0].flatten(1).sum(dim=1).detach().clamp_min(1.0)
    pred_es = prediction[:, -1].flatten(1).sum(dim=1)
    gt_ed = target[:, 0].float().flatten(1).sum(dim=1).clamp_min(1.0)
    gt_es = target[:, -1].float().flatten(1).sum(dim=1)
    return F.smooth_l1_loss(
        (pred_es / pred_ed).clamp(0.0, 2.0),
        (gt_es / gt_ed).clamp(0.0, 2.0),
    )


def training_loss(criterion, prediction, target, args):
    if args.semi:
        loss = endpoint_weighted_loss(criterion, prediction, target, args.es_loss_weight)
    else:
        loss = criterion(prediction[:, :, 0], target)

    probability = torch.sigmoid(prediction[:, :, 0])
    loss = loss + args.es_boundary_loss_weight * boundary_loss(
        probability[:, -1], target[:, -1]
    )
    loss = loss + args.es_area_loss_weight * area_ratio_loss(probability, target)
    return loss


def make_datasets(args, config):
    train_transform = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=config.img_size,
        crop=config.crop,
        p_flip=0.0,
        p_rota=0.5,
        p_scale=0.5,
        p_gaussn=0.0,
        p_contr=0.5,
        p_gama=0.5,
        p_distor=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    val_transform = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=config.img_size,
        crop=config.crop,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    common = {
        "dataset_path": args.data_path,
        "img_size": args.encoder_input_size,
        "frame_length": args.frame_length,
        "disable_point_prompt": True,
        "point_numbers": 1,
    }
    train_set = EchoVideoDataset(
        split=config.train_split,
        joint_transform=train_transform,
        **common,
    )
    val_set = EchoVideoDataset(
        split=config.val_split,
        joint_transform=val_transform,
        **common,
    )
    if args.smoke_test_samples > 0:
        train_set = Subset(train_set, range(min(args.smoke_test_samples, len(train_set))))
        val_set = Subset(val_set, range(min(args.smoke_test_samples, len(val_set))))
    return train_set, val_set


def save_checkpoint(path, model, optimizer, epoch, best_dice, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    apply_model_defaults(args)
    config = get_config(args.task)
    args.data_path = args.data_path or config.data_path
    args.dino_lora_weights = args.dino_lora_weights or (
        "weights/dino_lora_echonet.pth"
        if args.task == "EchoNet_Video"
        else "weights/dino_lora_camus.pth"
    )
    args.output_dir = args.output_dir or config.save_path
    args.epochs = args.epochs or config.epochs
    args.workers = config.workers if args.workers is None else args.workers
    config.data_path = args.data_path
    config.save_path = args.output_dir
    config.batch_size = args.batch_size
    config.workers = args.workers
    config.epochs = args.epochs
    config.eval_freq = args.eval_freq
    config.save_freq = args.save_freq
    config.semi = args.semi
    config.mode = "train"
    args.device = config.device

    validate_inputs(args)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = make_datasets(args, config)
    if not train_set or not val_set:
        raise RuntimeError(f"Empty split: train={len(train_set)}, val={len(val_set)}")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    print(f"Task: {args.task}")
    print(f"Data: {args.data_path}")
    print(f"Train/val: {len(train_set)}/{len(val_set)}")
    print(f"Phase memory: {args.enable_phase_memory}")

    model = get_model(MODEL_NAME, args=args, opt=config).to(config.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    criterion = get_criterion(modelname=MODEL_NAME, opt=config)

    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.device)
        result, skipped = load_compatible_weights(model, checkpoint["model"])
        if result.missing_keys or skipped:
            raise RuntimeError("Resume checkpoint does not exactly match this release model.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(config.device, dtype=torch.float32)
            masks = batch["label"].to(config.device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(images, None, None)
            loss = training_loss(criterion, prediction, masks, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mean_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1:03d}/{args.epochs}: train_loss={mean_loss:.6f}")

        should_evaluate = (epoch + 1) % args.eval_freq == 0
        if should_evaluate:
            model.eval()
            _, mean_dice, _, val_loss = get_eval(
                val_loader,
                model,
                criterion=criterion,
                opt=config,
                args=args,
            )
            score = float(np.asarray(mean_dice).mean())
            print(f"Epoch {epoch + 1:03d}: val_loss={float(val_loss):.6f}, dice={score:.6f}")
            if score > best_dice:
                best_dice = score
                torch.save(model.state_dict(), output_dir / "simsam_best.pth")

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), output_dir / f"simsam_epoch_{epoch + 1:03d}.pth")
        save_checkpoint(
            output_dir / "simsam_latest.pth",
            model,
            optimizer,
            epoch,
            best_dice,
            args,
        )

    print(f"Training complete. Best validation Dice: {best_dice:.6f}")


if __name__ == "__main__":
    main()
