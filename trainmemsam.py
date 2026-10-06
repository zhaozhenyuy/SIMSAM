import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import sys
from pickle import FALSE, TRUE
from statistics import mode
from easydict import EasyDict
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import EchoVideoDataset, JointTransform3D
from utils.data_us import JointTransform2D, EchoDataset
from utils.loss_functions.sam_loss import get_criterion
from utils.release_runtime import configure_release
from utils.release_checkpoints import load_release_checkpoint
from utils.generate_prompts import get_click_prompt
from groundingdino.util.inference import load_model, predict


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
    if getattr(args, "no_terminal_log", False):
        return None

    log_dir = args.log_dir or os.path.join(opt.save_path, "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_time = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.task}_{args.modelname}_{log_time}.log"
    log_path = os.path.join(log_dir, log_name)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)
    print(f"[LOG] Terminal output is being saved to: {log_path}")
    return log_file


def is_self_prompt_model(args):
    return args.modelname == "SelfPromptMemSAM" or getattr(args, "enable_self_prompt", False)


def is_shared_grounded_model(args):
    return args.modelname == "SharedGroundedMemSAM"


def masks_to_first_frame_boxes(masks):
    """Build pixel-space xyxy boxes from the first-frame foreground mask."""
    if masks.dim() == 5:
        mask0 = masks[:, 0, 0]
    elif masks.dim() == 4:
        mask0 = masks[:, 0]
    elif masks.dim() == 3:
        mask0 = masks
    else:
        raise ValueError(f"Unsupported mask shape for box supervision: {tuple(masks.shape)}")

    mask0 = mask0 > 0.5
    b, h, w = mask0.shape
    ys = torch.arange(h, dtype=torch.float32, device=mask0.device).view(1, h, 1).expand(b, h, w)
    xs = torch.arange(w, dtype=torch.float32, device=mask0.device).view(1, 1, w).expand(b, h, w)
    valid = mask0.flatten(1).any(dim=1)

    inf_x = torch.full_like(xs, float(w - 1))
    inf_y = torch.full_like(ys, float(h - 1))
    zeros_x = torch.zeros_like(xs)
    zeros_y = torch.zeros_like(ys)

    x1 = torch.where(mask0, xs, inf_x).flatten(1).min(dim=1).values
    y1 = torch.where(mask0, ys, inf_y).flatten(1).min(dim=1).values
    x2 = torch.where(mask0, xs, zeros_x).flatten(1).max(dim=1).values
    y2 = torch.where(mask0, ys, zeros_y).flatten(1).max(dim=1).values
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    fallback = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], device=mask0.device)
    boxes = torch.where(valid[:, None], boxes, fallback[None, :])
    return boxes, valid


def add_self_prompt_box_loss(train_loss, model_output, masks, weight):
    if not isinstance(model_output, dict):
        return train_loss
    prompt_boxes = model_output.get("prompt_boxes")
    if prompt_boxes is None:
        return train_loss

    target_boxes, valid = masks_to_first_frame_boxes(masks)
    if not valid.any():
        return train_loss

    norm = torch.tensor(
        [masks.shape[-1] - 1, masks.shape[-2] - 1, masks.shape[-1] - 1, masks.shape[-2] - 1],
        dtype=prompt_boxes.dtype,
        device=prompt_boxes.device,
    ).clamp_min(1.0)
    box_loss = F.smooth_l1_loss(prompt_boxes[valid] / norm, target_boxes[valid] / norm)
    return train_loss + weight * box_loss


def endpoint_weighted_loss(criterion, pred, masks, es_loss_weight):
    if abs(float(es_loss_weight) - 1.0) < 1e-6:
        return criterion(pred[:, [0, -1], 0, :, :], masks[:, [0, -1]])

    ed_loss = criterion(pred[:, [0], 0, :, :], masks[:, [0]])
    es_loss = criterion(pred[:, [-1], 0, :, :], masks[:, [-1]])
    return (ed_loss + float(es_loss_weight) * es_loss) / (1.0 + float(es_loss_weight))


def sobel_edges(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(mask, kernel_x, padding=1)
    grad_y = F.conv2d(mask, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    edge_norm = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return edge / edge_norm


def soft_boundary_loss(pred_prob, target):
    pred_edge = sobel_edges(pred_prob)
    target_edge = sobel_edges(target)
    dims = (1, 2, 3)
    smooth = 1e-5
    intersect = (pred_edge * target_edge).sum(dim=dims)
    denom = pred_edge.pow(2).sum(dim=dims) + target_edge.pow(2).sum(dim=dims)
    dice_loss = 1.0 - (2.0 * intersect + smooth) / (denom + smooth)
    l1_loss = F.l1_loss(pred_edge, target_edge)
    return 0.5 * dice_loss.mean() + 0.5 * l1_loss


def es_area_ratio_loss(pred_prob, masks):
    pred_ed = pred_prob[:, 0].flatten(1).sum(dim=1)
    pred_es = pred_prob[:, -1].flatten(1).sum(dim=1)
    gt_ed = masks[:, 0].float().flatten(1).sum(dim=1)
    gt_es = masks[:, -1].float().flatten(1).sum(dim=1)

    # Keep the ED prediction as a stable reference so the ES shape loss does
    # not improve the ratio by degrading the already-strong ED endpoint.
    pred_ratio = pred_es / pred_ed.detach().clamp_min(1.0)
    gt_ratio = gt_es / gt_ed.clamp_min(1.0)
    return F.smooth_l1_loss(pred_ratio.clamp(0.0, 2.0), gt_ratio.clamp(0.0, 2.0))


def add_es_shape_loss(train_loss, pred, masks, args):
    if not getattr(args, "enable_es_shape_loss", False):
        return train_loss
    if pred.dim() != 5 or masks.dim() != 4 or pred.shape[1] < 2 or masks.shape[1] < 2:
        return train_loss

    pred_prob = torch.sigmoid(pred[:, :, 0])
    shape_loss = pred_prob.new_tensor(0.0)

    boundary_weight = float(getattr(args, "es_boundary_loss_weight", 0.0))
    if boundary_weight > 0.0:
        boundary_loss = soft_boundary_loss(pred_prob[:, -1], masks[:, -1])
        shape_loss = shape_loss + boundary_weight * boundary_loss

    area_weight = float(getattr(args, "es_area_loss_weight", 0.0))
    if area_weight > 0.0:
        area_loss = es_area_ratio_loss(pred_prob, masks)
        shape_loss = shape_loss + area_weight * area_loss

    return train_loss + shape_loss


def parse_args(argv=None, defaults=None):

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='MemSAM', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('--low_image_size', type=int, default=256, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='CAMUS_Video_Full', help='task or dataset name: CAMUS_Video_Full or EchoNet_Video')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='weights/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--load_path', type=str, default='', help='Optional checkpoint path for fine-tuning the current model.')
    parser.add_argument('--data_path', type=str, default='', help='Override dataset root from utils/config.py.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', action="store_true", help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--keep_log', action="store_true", help='keep the loss&lr&dice during training or not')
    parser.add_argument('--log_dir', type=str, default='', help='Directory for terminal log files. Defaults to <save_path>/train_logs.')
    parser.add_argument('--no_terminal_log', action="store_true", help='Disable tee logging of stdout/stderr to a .log file.')
    parser.add_argument('--frame_length', type=int, default=10)
    parser.add_argument('--point_numbers', type=int, default=1)
    parser.add_argument('--enable_memory', action="store_true")
    parser.add_argument('--semi', action="store_true")
    parser.add_argument('--reinforce', action="store_true")
    parser.add_argument('--disable_point_prompt', action="store_true")
    parser.add_argument('--enable_self_prompt', action="store_true")
    parser.add_argument('--self_prompt_box_loss_weight', type=float, default=1.0)
    parser.add_argument('--self_prompt_warmup_epochs', type=int, default=0)
    parser.add_argument('--train_shared_dino', action="store_true")
    parser.add_argument('--enable_apfe', action="store_true", default=True)
    parser.add_argument('--disable_apfe', action="store_false", dest='enable_apfe')
    parser.add_argument('--apfe_kernel_size', type=int, default=7)
    parser.add_argument('--enable_phase_memory', action="store_true")
    parser.add_argument('--disable_phase_memory', action="store_false", dest='enable_phase_memory')
    parser.add_argument('--phase_memory_scale', type=float, default=0.1)
    parser.add_argument('--es_loss_weight', type=float, default=1.0)
    parser.add_argument('--enable_es_shape_loss', action="store_true")
    parser.add_argument('--es_boundary_loss_weight', type=float, default=0.2)
    parser.add_argument('--es_area_loss_weight', type=float, default=0.1)
    parser.add_argument('--seg_threshold', type=float, default=0.6)
    parser.add_argument('--debug_low_dice_threshold', type=float, default=0.05)
    parser.add_argument('--reset_reinforce_state_per_video', action="store_true")
    parser.add_argument('--enable_osu_prompt', action="store_true")
    parser.add_argument('--osu_iters', type=int, default=5)
    parser.add_argument('--osu_ns_type', type=str, default='classic', choices=['classic', 'quintic'])
    parser.add_argument('--enable_osu_state', action="store_true")
    parser.add_argument('--osu_state_iters', type=int, default=5)
    parser.add_argument('--osu_state_alpha', type=float, default=0.98)
    parser.add_argument('--osu_state_beta', type=float, default=1.0)
    parser.add_argument('--osu_state_scale', type=float, default=2.0)
    parser.add_argument('--grad_clip_norm', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.02)
    # ---------------- GroundingDINO box prompt ----------------
    parser.add_argument('--enable_box_prompt', action="store_true")
    parser.add_argument('--box_prompt_text', type=str, default='left ventricle')
    parser.add_argument('--dino_config', type=str, default='groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument('--dino_weights', type=str, default='weights/groundingdino_swint_ogc.pth')
    parser.add_argument('--dino_use_lora', action="store_true")
    parser.add_argument('--disable_dino_lora', action="store_false", dest='dino_use_lora')
    parser.add_argument('--dino_lora_weights', type=str, default='weights/best_model.pth')
    parser.add_argument('--disable_dino_prompt', action="store_true", help='Disable SharedGroundedMemSAM internal DINO point grounding.')
    parser.add_argument('--dino_box_th', type=float, default=0.35)
    parser.add_argument('--dino_text_th', type=float, default=0.25)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.modelname == "SelfPromptMemSAM":
        args.enable_self_prompt = True
    if args.grad_clip_norm <= 0.0 and (args.enable_osu_prompt or args.enable_osu_state):
        args.grad_clip_norm = 1.0
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    # ==================================================parameters setting==================================================

    # override_args = EasyDict(base_lr=0.0001,
    #                     batch_size=1,
    #                     encoder_input_size=256,
    #                     keep_log=True,
    #                     low_image_size=256,
    #                     frame_length=10,
    #                     modelname='XMemSAM',
    #                     n_gpu=1,
    #                     sam_ckpt='checkpoints/sam_vit_b_01ec64.pth',
    #                     task='CAMUS_Video_Full',
    #                     vit_name='vit_b',
    #                     enable_memory=True,
    #                     enable_point_prompt=True,
    #                     point_numbers=1,
    #                     warmup=False,
    #                     warmup_period=250)
    opt = get_config(args.task)
    configure_release(args, opt, training=True)
    if args.data_path:
        opt.data_path = args.data_path
    opt.semi = args.semi
    args.device = opt.device
    terminal_log_file = setup_terminal_logger(args, opt)
    print(args)

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime(
            '%m%d%H%M'
        )  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    # ==================================================set random seed==================================================
    seed_value = args.seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) 

    # ==================================================build model==================================================
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # tf_train, tf_val = None, None
    train_dataset = EchoVideoDataset(
        opt.data_path,
        opt.train_split,
        tf_train,
        img_size=args.encoder_input_size,
        frame_length=args.frame_length,
        point_numbers=args.point_numbers,
        disable_point_prompt=args.disable_point_prompt,
        class_key=getattr(opt, "data_subpath", None),
    )
    val_dataset = EchoVideoDataset(
        opt.data_path,
        opt.val_split,
        tf_val,
        img_size=args.encoder_input_size,
        frame_length=args.frame_length,
        point_numbers=args.point_numbers,
        disable_point_prompt=args.disable_point_prompt,
        class_key=getattr(opt, "data_subpath", None),
    )  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    model.to(device)
    if args.load_path:
        missing_keys, unexpected_keys = load_release_checkpoint(model, args.load_path, device)
        print(f"Fine-tuning from checkpoint: {args.load_path}")
        print(f"Missing keys: {len(missing_keys)}")
        for key in missing_keys[:20]:
            print("  missing:", key)
        print(f"Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys[:20]:
            print("  unexpected:", key)
    elif opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=not (is_self_prompt_model(args) or is_shared_grounded_model(args)))
        
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    b_lr = args.base_lr / args.warmup_period if args.warmup else args.base_lr
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=b_lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            trainable_parameters,
            lr=b_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))


    dino_model = None
    if getattr(args, "enable_box_prompt", False) and not is_self_prompt_model(args) and not is_shared_grounded_model(args):
        # 直接复用 groundingdino.util.inference.load_model 的接口（它要 model_config.config_path / weights_path / lora_weigths）
        dino_cfg = EasyDict(
            config_path=args.dino_config,
            weights_path=args.dino_weights,
            lora_weigths=args.dino_lora_weights,
        )
        dino_model = load_model(
            model_config=dino_cfg,
            use_lora=getattr(args, "dino_use_lora", False),
            device=str(opt.device),
            strict=False
        ).to(opt.device)
        dino_model.eval()


    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['label'].to(dtype = torch.float32, device=opt.device)
            if args.disable_point_prompt:
                # pt[0]: b t point_num 2
                # pt[1]: t point_num
                pt = None
            else:
                pt = get_click_prompt(datapack, opt) 
            # video to image
            # b, t, c, h, w = imgs.shape
            # ---- GroundingDINO center-point prompt (only first frame) ----
            use_self_prompt = is_self_prompt_model(args)
            pt_dino = None
            bbox_prompt = None
            return_prompts = use_self_prompt

            if use_self_prompt and epoch < args.self_prompt_warmup_epochs:
                target_boxes, _ = masks_to_first_frame_boxes(masks)
                bbox_prompt = target_boxes.detach()

            if getattr(args, "enable_box_prompt", False) and not use_self_prompt and not is_shared_grounded_model(args):
                frame0 = imgs[:, 0]  # (B,T,C,H,W) -> 首帧 (B,C,H,W)

                boxes, scores, phrases = predict(
                    model=dino_model,
                    image=frame0[0],
                    caption=args.box_prompt_text,
                    box_threshold=args.dino_box_th,
                    text_threshold=args.dino_text_th,
                    device=str(opt.device),
                )

                if boxes is not None and len(boxes) > 0:
                    from torchvision.ops import box_convert

                    # normalized cxcywh -> pixel xyxy
                    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                    xyxy = xyxy * torch.tensor([256, 256, 256, 256], dtype=xyxy.dtype, device=xyxy.device)

                    best = torch.argmax(scores)
                    best_box = xyxy[best]  # (4,)

                    x1, y1, x2, y2 = best_box
                    px = 0.5 * (x1 + x2)
                    py = 0.5 * (y1 + y2)

                    px = px.clamp(0, 255)
                    py = py.clamp(0, 255)

                    B, T = imgs.shape[0], imgs.shape[1]

                    # coords: (B,T,1,2)，labels: (B,1)
                    coords = torch.zeros((B, T, 1, 2), dtype=torch.float32, device=opt.device)
                    labels = torch.ones((B, 1), dtype=torch.int64, device=opt.device)

                    coords[:, 0, 0, 0] = px
                    coords[:, 0, 0, 1] = py

                    pt_dino = (coords, labels)

                    print(
                        "[DINO POINT][TRAIN]",
                        "boxes_n =", len(boxes),
                        "best_box =", best_box.detach().cpu().tolist(),
                        "point_xy =", [float(px.detach().cpu()), float(py.detach().cpu())],
                        "score =", float(scores[best].detach().cpu())
                    )
                else:
                    print("[DINO POINT][TRAIN] no box found, use no point prompt")

            # ---- MemSAM forward uses center-point prompt only ----
            model_output = model(imgs, pt_dino, bbox_prompt, return_prompts=return_prompts)
            pred = model_output["masks"] if isinstance(model_output, dict) else model_output
            # -------------------------------------------------------- forward --------------------------------------------------------
            # if masks.shape[1] == 10:
            #     masks = masks[:,[0,-1]]
            # semi supervised
            if opt.semi:
                train_loss = endpoint_weighted_loss(criterion, pred, masks, args.es_loss_weight)
            # full supervised
            else:
                train_loss = criterion(pred[:,:,0], masks)
            if use_self_prompt:
                train_loss = add_self_prompt_box_loss(
                    train_loss,
                    model_output,
                    masks,
                    args.self_prompt_box_loss_weight,
                )
            train_loss = add_es_shape_loss(train_loss, pred, masks, args)
            # -------------------------------------------------------- backward -------------------------------------------------------
            if not torch.isfinite(train_loss):
                print("[WARN] non-finite train loss, skip optimizer step")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            train_loss.backward()
            if args.grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    args.grad_clip_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm):
                    print("[WARN] non-finite grad norm, skip optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()
            train_losses += train_loss.item()
            print(train_loss)
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            # if args.keep_log:
            #     with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
            #         for i in range(len(loss_log)):
            #             f.write(str(loss_log[i])+'\n')
            #     with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
            #         for i in range(len(dice_log)):
            #             f.write(str(dice_log[i])+'\n')


if __name__ == '__main__':
    main()
