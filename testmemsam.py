import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.model_dict import get_model
from utils.config import get_config
from utils.data_us import EchoVideoDataset, JointTransform3D
from utils.evaluation import get_eval
from utils.lvef_evaluation import validate_clinical_request
from utils.loss_functions.sam_loss import get_criterion
from utils.release_runtime import configure_release
from utils.release_checkpoints import load_release_checkpoint


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_terminal_logger(args, opt):
    if getattr(args, "no_test_terminal_log", False):
        return None

    log_dir = args.test_log_dir or os.path.join(opt.result_path, "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_time = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.task}_{args.modelname}_{log_time}.log"
    log_path = os.path.join(log_dir, log_name)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)
    print(f"[LOG] Test terminal output is being saved to: {log_path}")
    return log_file


def parse_args(argv=None, defaults=None):
    parser = argparse.ArgumentParser(description="Evaluate MemSAM variants.")
    parser.add_argument("--modelname", type=str, default="SharedGroundedMemSAM")
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="", help="Override dataset root from utils/config.py.")
    parser.add_argument("--task", type=str, default="CAMUS_Video_Full")
    parser.add_argument("--sam_ckpt", type=str, default="weights/sam_vit_b_01ec64.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--encoder_input_size", type=int, default=256)
    parser.add_argument("--low_image_size", type=int, default=256)
    parser.add_argument("--frame_length", type=int, default=10)
    parser.add_argument("--point_numbers", type=int, default=1)
    parser.add_argument("--vit_name", type=str, default="vit_b")

    parser.add_argument("--enable_memory", action="store_true", default=True)
    parser.add_argument("--disable_memory", action="store_false", dest="enable_memory")
    parser.add_argument("--reinforce", action="store_true", default=False)
    parser.add_argument("--disable_reinforce", action="store_false", dest="reinforce")
    parser.add_argument("--disable_point_prompt", action="store_true", default=True)
    parser.add_argument("--use_point_prompt", action="store_false", dest="disable_point_prompt")
    parser.add_argument("--semi", action="store_true", default=True)
    parser.add_argument("--full_eval", action="store_false", dest="semi")
    parser.add_argument("--visual", action="store_true", default=False)
    parser.add_argument("--test_log_dir", type=str, default="", help="Directory for test terminal logs. Defaults to <result_path>/test_logs.")
    parser.add_argument("--no_test_terminal_log", action="store_true", default=False, help="Disable tee logging of test stdout/stderr.")

    # For legacy MemSAM + external DINO prompt, keep this on only when modelname=MemSAM.
    parser.add_argument("--enable_box_prompt", action="store_true", default=False)
    parser.add_argument("--box_prompt_text", type=str, default="left ventricle")
    parser.add_argument("--dino_config", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino_weights", type=str, default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--dino_use_lora", action="store_true", default=False)
    parser.add_argument("--disable_dino_lora", action="store_false", dest="dino_use_lora")
    parser.add_argument("--dino_lora_weights", type=str, default="weights/best_model.pth")
    parser.add_argument("--disable_dino_prompt", action="store_true", default=False)
    parser.add_argument("--dino_box_th", type=float, default=0.35)
    parser.add_argument("--dino_text_th", type=float, default=0.25)
    parser.add_argument("--train_shared_dino", action="store_true", default=False)
    parser.add_argument("--enable_self_prompt", action="store_true", default=False)
    parser.add_argument("--enable_apfe", action="store_true", default=True)
    parser.add_argument("--disable_apfe", action="store_false", dest="enable_apfe")
    parser.add_argument("--apfe_kernel_size", type=int, default=7)
    parser.add_argument("--enable_phase_memory", action="store_true", default=False)
    parser.add_argument("--disable_phase_memory", action="store_false", dest="enable_phase_memory")
    parser.add_argument("--phase_memory_scale", type=float, default=0.1)
    parser.add_argument("--es_loss_weight", type=float, default=1.0)
    parser.add_argument("--enable_es_shape_loss", action="store_true", default=False)
    parser.add_argument("--es_boundary_loss_weight", type=float, default=0.2)
    parser.add_argument("--es_area_loss_weight", type=float, default=0.1)
    parser.add_argument("--seg_threshold", type=float, default=0.6)
    parser.add_argument(
        "--eval_spacing_scale",
        type=float,
        default=1.0,
        help="Scale voxel spacing for EchoNet/EchoDynamic surface metrics after resizing images.",
    )
    parser.add_argument("--debug_low_dice_threshold", type=float, default=0.05)
    parser.add_argument("--debug_high_hd_threshold", type=float, default=0.0)
    parser.add_argument("--reset_reinforce_state_per_video", action="store_true", default=False)
    parser.add_argument("--enable_osu_prompt", action="store_true", default=False)
    parser.add_argument("--osu_iters", type=int, default=5)
    parser.add_argument("--osu_ns_type", type=str, default="classic", choices=["classic", "quintic"])
    parser.add_argument("--enable_osu_state", action="store_true", default=False)
    parser.add_argument("--osu_state_iters", type=int, default=5)
    parser.add_argument("--osu_state_alpha", type=float, default=0.98)
    parser.add_argument("--osu_state_beta", type=float, default=1.0)
    parser.add_argument("--osu_state_scale", type=float, default=2.0)
    parser.add_argument(
        "--bidirectional_endpoint_eval",
        action="store_true",
        default=False,
        help="Run an extra reversed pass so ED and ES can both act as endpoint anchors during evaluation.",
    )
    parser.add_argument(
        "--bidirectional_fusion",
        type=str,
        default="endpoint",
        choices=["endpoint", "mean", "backward"],
        help="How to combine forward ED-anchored logits with reversed ES-anchored logits.",
    )
    parser.add_argument("--compute_ef", action="store_true", default=False,
                        help="CAMUS paired-view LVEF: corr, signed Bias, MAE and 95%% LoA; no PSD.")
    parser.add_argument("--clinical_output_dir", type=str, default="",
                        help="New directory for patient LVEF pairs and summary; default is a timestamped lvef directory.")
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def seed_everything(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    state_dict = {}
    for key, value in checkpoint.items():
        state_dict[key[7:] if key.startswith("module.") else key] = value

    return model.load_state_dict(state_dict, strict=False)


def main(args=None):
    if args is None:
        args = parse_args()
    opt = get_config(args.task)
    configure_release(args, opt, training=False)
    if args.data_path:
        opt.data_path = args.data_path
    opt.load_path = args.load_path
    opt.mode = "test"
    opt.semi = args.semi
    opt.visual = args.visual
    opt.batch_size = args.batch_size * args.n_gpu
    validate_clinical_request(args, opt)
    args.device = opt.device
    test_log_file = setup_terminal_logger(args, opt)

    seed_everything(args.seed)
    device = torch.device(opt.device)

    print("Current test config:")
    print(f"  modelname          = {args.modelname}")
    print(f"  checkpoint         = {opt.load_path}")
    print(f"  enable_memory      = {args.enable_memory}")
    print(f"  reinforce          = {args.reinforce}")
    print(f"  semi               = {opt.semi}")
    print(f"  visual             = {opt.visual}")
    print(f"  box_prompt_text    = {args.box_prompt_text}")
    print(f"  dino_use_lora      = {args.dino_use_lora}")
    print(f"  dino_lora_weights  = {args.dino_lora_weights}")
    print(f"  disable_dino_prompt= {args.disable_dino_prompt}")
    print(f"  enable_phase_memory= {args.enable_phase_memory}")
    print(f"  seg_threshold      = {args.seg_threshold}")
    print(f"  eval_spacing_scale = {args.eval_spacing_scale}")
    print(f"  reset_reinforce    = {args.reset_reinforce_state_per_video}")
    print(f"  bidir_endpoint     = {args.bidirectional_endpoint_eval}")
    print(f"  bidir_fusion       = {args.bidirectional_fusion}")

    tf_val = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True,
    )

    test_dataset = EchoVideoDataset(
        dataset_path=opt.data_path,
        split=opt.test_split,
        joint_transform=tf_val,
        img_size=args.encoder_input_size,
        frame_length=args.frame_length,
        disable_point_prompt=args.disable_point_prompt,
        point_numbers=args.point_numbers,
        class_key=getattr(opt, "data_subpath", None),
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=min(opt.workers, 4),
        pin_memory=True,
    )

    print("Building model...")
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)

    print("Loading checkpoint...")
    missing_keys, unexpected_keys = load_release_checkpoint(model, opt.load_path, device)
    print("Checkpoint loaded.")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        for key in missing_keys[:20]:
            print("  missing:", key)
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys[:20]:
            print("  unexpected:", key)

    if args.compute_ef and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "LVEF evaluation requires a fully matching checkpoint. "
            "Check modelname, memory/reinforce, APFE and phase switches; "
            "do not report clinical metrics from partially loaded weights."
        )

    criterion = get_criterion(modelname=args.modelname, opt=opt)
    model.eval()

    print("Start evaluation...")
    results = get_eval(testloader, model, criterion=criterion, opt=opt, args=args)

    print("\n=== Test results ===")
    if len(results) == 8:
        dice_mean, iou_mean, hd_mean, assd_mean, dice_std, iou_std, hd_std, assd_std = results
        print("Dice Mean:", dice_mean)
        print("Dice Std: ", dice_std)
        print("IoU Mean: ", iou_mean)
        print("IoU Std:  ", iou_std)
        print("HD Mean:  ", hd_mean)
        print("HD Std:   ", hd_std)
        print("ASSD Mean:", assd_mean)
        print("ASSD Std: ", assd_std)
        if hasattr(dice_mean, "__len__") and len(dice_mean) > 1:
            print(f"Foreground Dice: {np.mean(dice_mean[1:]):.4f}")
            print(f"Foreground IoU:  {np.mean(iou_mean[1:]):.4f}")
    elif len(results) == 4:
        dices, mean_dice, mean_hdis, val_losses = results
        print("Dices:", dices)
        print("Mean Dice:", mean_dice)
        print("Mean HD:", mean_hdis)
        print("Val Loss:", val_losses)
    else:
        for index, result in enumerate(results):
            print(f"Result[{index}]: {result}")


if __name__ == "__main__":
    main()
