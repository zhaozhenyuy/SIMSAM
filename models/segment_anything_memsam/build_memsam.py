from pathlib import Path

import torch
from easydict import EasyDict

from groundingdino.util.inference import load_model as load_dino_model

from .modeling import (
    DINOToSAMAdapter,
    MaskDecoder,
    Mem,
    PromptEncoder,
    SharedGroundedMemSAM,
    TwoWayTransformer,
)


def _checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_compatible_state_dict(model, checkpoint):
    """Load matching tensors and ignore obsolete experimental branches."""
    source = _checkpoint_state(checkpoint)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return compatible.keys()


def build_shared_grounded_memsam(args, checkpoint=None):
    image_size = args.encoder_input_size
    image_embedding_size = 32
    prompt_embed_dim = 256

    dino_config = EasyDict(
        config_path=args.dino_config,
        weights_path=args.dino_weights,
        lora_weigths=args.dino_lora_weights,
    )
    dino_model = load_dino_model(
        model_config=dino_config,
        use_lora=args.dino_use_lora,
        device=args.device,
        strict=False,
    )

    model = SharedGroundedMemSAM(
        dino_model=dino_model,
        feature_adapter=DINOToSAMAdapter(
            in_channels=dino_model.backbone.num_channels,
            out_channels=prompt_embed_dim,
            output_size=(image_embedding_size, image_embedding_size),
            enable_apfe=args.enable_apfe,
            apfe_kernel_size=args.apfe_kernel_size,
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
            }
        ),
        caption=args.box_prompt_text,
        box_threshold=args.dino_box_th,
        freeze_dino=not args.train_shared_dino,
        enable_phase_memory=args.enable_phase_memory,
        phase_memory_scale=args.phase_memory_scale,
    )

    if checkpoint:
        state = torch.load(Path(checkpoint), map_location="cpu")
        load_compatible_state_dict(model, state)

    return model


memsam_model_registry = {
    "shared_grounded": build_shared_grounded_memsam,
}
