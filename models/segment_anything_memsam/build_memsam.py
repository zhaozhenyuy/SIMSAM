import torch

from easydict import EasyDict
from functools import partial

from groundingdino.util.inference import load_model as load_dino_model

from .modeling import (
    DINOToSAMAdapter,
    ImageEncoderViT,
    MaskDecoder,
    Mem,
    MemSAM,
    PromptEncoder,
    SharedGroundedMemSAM,
    TwoWayTransformer,
)
from torch.nn import functional as F


def build_memsam_vit_h(args, checkpoint=None):
    return _build_memsam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_memsam = build_memsam_vit_h


def build_memsam_vit_l(args, checkpoint=None):
    return _build_memsam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_memsam_vit_b(args, checkpoint=None):
    return _build_memsam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_shared_grounded_memsam(args, checkpoint=None):
    return _build_shared_grounded_memsam(args, checkpoint=checkpoint)


memsam_model_registry = {
    "default": build_memsam_vit_h,
    "vit_h": build_memsam_vit_h,
    "vit_l": build_memsam_vit_l,
    "vit_b": build_memsam_vit_b,
    "shared_grounded": build_shared_grounded_memsam,
}


def _load_partial_state_dict(model, checkpoint):
    model_dict = model.state_dict()
    trained = {
        key: value
        for key, value in checkpoint.items()
        if key in model_dict and model_dict[key].shape == value.shape
    }
    model_dict.update(trained)
    model.load_state_dict(model_dict, strict=False)
    return trained.keys()


def _build_memsam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size//32
    image_embedding_size = image_size // patch_size
    sam = MemSAM(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size= patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            batch_size=args.batch_size,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        memory=Mem(
            config={
                "key_dim": 64,
                "value_dim": 256,
                "hidden_dim": 64,
                "reinforce":args.reinforce,
                "enable_apfe": getattr(args, "enable_apfe", False),
                "apfe_kernel_size": getattr(args, "apfe_kernel_size", 7),
                "enable_osu_prompt": getattr(args, "enable_osu_prompt", False),
                "osu_iters": getattr(args, "osu_iters", 5),
                "osu_ns_type": getattr(args, "osu_ns_type", "classic"),
                "enable_osu_state": getattr(args, "enable_osu_state", False),
                "osu_state_iters": getattr(args, "osu_state_iters", 5),
                "osu_state_alpha": getattr(args, "osu_state_alpha", 0.98),
                "osu_state_beta": getattr(args, "osu_state_beta", 1.0),
                "osu_state_scale": getattr(args, "osu_state_scale", 2.0),
            },
        ) if args.enable_memory else None,
        self_prompt=getattr(args, "enable_self_prompt", False),
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(sam, state_dict, image_size, patch_size)
            sam.load_state_dict(new_state_dict)
    return sam


def _build_shared_grounded_memsam(args, checkpoint=None):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size // 32
    image_embedding_size = image_size // patch_size

    dino_cfg = EasyDict(
        config_path=args.dino_config,
        weights_path=args.dino_weights,
        lora_weigths=getattr(args, "dino_lora_weights", None),
    )
    dino_model = load_dino_model(
        model_config=dino_cfg,
        use_lora=getattr(args, "dino_use_lora", False),
        device=getattr(args, "device", "cuda"),
        strict=False,
    )

    model = SharedGroundedMemSAM(
        dino_model=dino_model,
        feature_adapter=DINOToSAMAdapter(
            in_channels=dino_model.backbone.num_channels,
            out_channels=prompt_embed_dim,
            output_size=(image_embedding_size, image_embedding_size),
            enable_apfe=getattr(args, "enable_apfe", False),
            apfe_kernel_size=getattr(args, "apfe_kernel_size", 7),
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            batch_size=args.batch_size,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        memory=Mem(
            config={
                "key_dim": 64,
                "value_dim": 256,
                "hidden_dim": 64,
                "reinforce": args.reinforce,
                # SharedGroundedMemSAM applies APFE before feature fusion so
                # key, value, query, and mask decoding share enhanced features.
                "enable_apfe": False,
                "apfe_kernel_size": getattr(args, "apfe_kernel_size", 7),
                "enable_osu_prompt": getattr(args, "enable_osu_prompt", False),
                "osu_iters": getattr(args, "osu_iters", 5),
                "osu_ns_type": getattr(args, "osu_ns_type", "classic"),
                "enable_osu_state": getattr(args, "enable_osu_state", False),
                "osu_state_iters": getattr(args, "osu_state_iters", 5),
                "osu_state_alpha": getattr(args, "osu_state_alpha", 0.98),
                "osu_state_beta": getattr(args, "osu_state_beta", 1.0),
                "osu_state_scale": getattr(args, "osu_state_scale", 2.0),
            },
        ) if args.enable_memory else None,
        caption=getattr(args, "box_prompt_text", "left ventricle"),
        box_threshold=getattr(args, "dino_box_th", 0.35),
        freeze_dino=not getattr(args, "train_shared_dino", False),
        disable_dino_prompt=getattr(args, "disable_dino_prompt", False),
        enable_phase_memory=getattr(args, "enable_phase_memory", False),
        phase_memory_scale=getattr(args, "phase_memory_scale", 0.1),
        reset_reinforce_state_per_video=getattr(args, "reset_reinforce_state_per_video", False),
    )
    model.eval()

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        _load_partial_state_dict(model, state_dict)

    return model

def load_from(samus, sam_dict, image_size, patch_size):
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    token_size = int(image_size//patch_size)
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict


def load_from2(samus, sam_dict, image_size, patch_size): # load the positional embedding
    samus_dict = samus.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    token_size = int(image_size//patch_size)
    # pos_embed = dict_trained['image_encoder.pos_embed']
    # pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
    # pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
    # pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    # dict_trained['image_encoder.pos_embed'] = pos_embed
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict
