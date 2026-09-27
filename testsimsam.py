import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.model_dict import get_model
from utils.config import get_config
from utils.data_us import EchoVideoDataset, JointTransform3D
from utils.evaluation import get_eval
from utils.loss_functions.sam_loss import get_criterion


MODEL_NAME = "SharedGroundedMemSAM"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SIMSAM.")
    parser.add_argument("--load_path", required=True, help="SIMSAM checkpoint.")
    parser.add_argument(
        "--task",
        choices=("CAMUS_Video_Full", "EchoNet_Video"),
        default="CAMUS_Video_Full",
    )
    parser.add_argument("--data_path", default=None, help="Dataset root.")
    parser.add_argument("--result_path", default=None, help="Visualization output directory.")
    parser.add_argument("--sam_ckpt", default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--dino_weights", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--dino_lora_weights", default=None)
    parser.add_argument(
        "--dino_config",
        default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--frame_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seg_threshold", type=float, default=0.6)
    parser.add_argument("--phase_memory_scale", type=float, default=0.1)
    parser.add_argument("--disable_phase_memory", action="store_true", help="Ablation only.")
    parser.add_argument("--disable_apfe", action="store_true", help="Disable APFE for ablation.")
    parser.add_argument("--no_dino_lora", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--compute_ef", action="store_true")
    parser.add_argument(
        "--eval_scope",
        choices=("auto", "full", "endpoints"),
        default="auto",
        help="Auto evaluates all CAMUS frames and EchoNet ED/ES endpoints.",
    )
    return parser.parse_args()


def apply_model_defaults(args):
    args.modelname = MODEL_NAME
    args.encoder_input_size = 256
    args.low_image_size = 256
    args.point_numbers = 1
    args.enable_memory = True
    args.disable_point_prompt = True
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
    args.es_loss_weight = 1.2
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
    required = [args.load_path, args.sam_ckpt, args.dino_weights]
    if args.dino_use_lora:
        required.append(args.dino_lora_weights)
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Required weight files were not found:\n  {joined}")
    if not Path(args.data_path).is_dir():
        raise FileNotFoundError(f"Dataset directory was not found: {args.data_path}")


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    source = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return result, len(source) - len(compatible)


def print_results(results):
    if len(results) != 8:
        print(results)
        return
    dice, iou, hd95, assd, dice_std, iou_std, hd95_std, assd_std = results
    print("\n=== Test results ===")
    print(f"Dice: {float(np.asarray(dice).mean()):.6f} +/- {float(np.asarray(dice_std).mean()):.6f}")
    print(f"IoU:  {float(np.asarray(iou).mean()):.6f} +/- {float(np.asarray(iou_std).mean()):.6f}")
    print(f"HD95: {float(np.asarray(hd95).mean()):.6f} +/- {float(np.asarray(hd95_std).mean()):.6f}")
    print(f"ASSD: {float(np.asarray(assd).mean()):.6f} +/- {float(np.asarray(assd_std).mean()):.6f}")


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
    if args.result_path:
        config.result_path = args.result_path
    if args.eval_scope == "auto":
        args.semi = args.task == "EchoNet_Video"
    else:
        args.semi = args.eval_scope == "endpoints"
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.workers = args.workers
    config.semi = args.semi
    config.visual = args.visual
    config.mode = "test"
    config.load_path = args.load_path
    args.device = config.device

    validate_inputs(args)
    seed_everything(args.seed)

    transform = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=config.img_size,
        crop=config.crop,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    dataset = EchoVideoDataset(
        dataset_path=args.data_path,
        split=config.test_split,
        joint_transform=transform,
        img_size=args.encoder_input_size,
        frame_length=args.frame_length,
        disable_point_prompt=True,
        point_numbers=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Task: {args.task}")
    print(f"Data: {args.data_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Evaluation: {'ED/ES endpoints' if args.semi else 'full sequence'}")
    print(f"Phase memory: {args.enable_phase_memory}")
    print(f"APFE: {args.enable_apfe}")

    model = get_model(MODEL_NAME, args=args, opt=config).to(config.device)
    result, skipped = load_checkpoint(model, args.load_path, config.device)
    if result.missing_keys:
        print(f"Missing checkpoint tensors: {len(result.missing_keys)}")
    if skipped:
        print(f"Ignored obsolete or incompatible tensors: {skipped}")

    criterion = get_criterion(modelname=MODEL_NAME, opt=config)
    model.eval()
    results = get_eval(loader, model, criterion=criterion, opt=config, args=args)
    print_results(results)
    if args.visual:
        print(f"Visualizations: {Path(config.result_path).resolve()}")


if __name__ == "__main__":
    main()
