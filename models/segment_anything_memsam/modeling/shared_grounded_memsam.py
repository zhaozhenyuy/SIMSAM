from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from groundingdino.util.misc import NestedTensor

from .common import LayerNorm2d
from .mask_decoder import MaskDecoder
from .mem import Mem
from .mem_modules import APFE
from .prompt_encoder import PromptEncoder


def _preprocess_caption(caption: str) -> str:
    caption = caption.lower().strip()
    if caption.endswith("."):
        return caption
    return caption + "."


class DINOToSAMAdapter(nn.Module):
    """Projects GroundingDINO/Swin multi-scale features into SAM embedding space."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        output_size: Tuple[int, int] = (32, 32),
        enable_apfe: bool = False,
        apfe_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
                    LayerNorm2d(out_channels),
                    nn.GELU(),
                )
                for channels in in_channels
            ]
        )
        self.apfe = (
            nn.ModuleList(
                [APFE(out_channels, kernel_size=apfe_kernel_size) for _ in in_channels]
            )
            if enable_apfe
            else None
        )
        self.apfe_scale = (
            nn.Parameter(torch.full((len(in_channels),), 0.1))
            if enable_apfe
            else None
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
        )

    def forward(self, features: List[NestedTensor]) -> torch.Tensor:
        projected = []
        for feature_index, (feature, projection) in enumerate(zip(features, self.proj)):
            tensor = feature.tensors if isinstance(feature, NestedTensor) else feature
            tensor = projection(tensor)
            if self.apfe is not None:
                enhanced = self.apfe[feature_index](tensor)
                scale = self.apfe_scale[feature_index].clamp(0.0, 1.0)
                tensor = tensor + scale * enhanced
            tensor = F.interpolate(
                tensor,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
            projected.append(tensor)
        fused = torch.stack(projected, dim=0).mean(dim=0)
        return self.fuse(fused)


class SharedGroundedMemSAM(nn.Module):
    """GroundingDINO text-image alignment with a shared DINO visual encoder for MemSAM."""

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        dino_model: nn.Module,
        feature_adapter: DINOToSAMAdapter,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        memory: Optional[Mem],
        caption: str = "left ventricle",
        box_threshold: float = 0.35,
        freeze_dino: bool = True,
        enable_phase_memory: bool = False,
        phase_memory_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.dino_model = dino_model
        self.feature_adapter = feature_adapter
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.memory = memory
        self.caption = _preprocess_caption(caption)
        self.box_threshold = box_threshold
        self.last_grounding: Optional[Dict[str, torch.Tensor]] = None
        self.phase_memory = (
            nn.Sequential(
                nn.Linear(1, 256),
                nn.GELU(),
                nn.Linear(256, 256),
            )
            if enable_phase_memory
            else None
        )
        self.phase_memory_scale = (
            nn.Parameter(torch.tensor(float(phase_memory_scale)))
            if enable_phase_memory
            else None
        )
        if self.phase_memory is not None:
            nn.init.zeros_(self.phase_memory[-1].weight)
            nn.init.zeros_(self.phase_memory[-1].bias)
        self.register_buffer(
            "dino_pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_dino:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_model.eval()

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    @property
    def device(self) -> Any:
        return self.prompt_encoder.point_embeddings[0].weight.device

    def train(self, mode: bool = True) -> "SharedGroundedMemSAM":
        super().train(mode)
        if not any(param.requires_grad for param in self.dino_model.parameters()):
            self.dino_model.eval()
        return self

    def _make_nested(self, images: torch.Tensor) -> NestedTensor:
        return NestedTensor(images, "auto")

    def _preprocess_dino_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.detach().max() > 2.0:
            images = images / 255.0
        return (images - self.dino_pixel_mean) / self.dino_pixel_std

    def _select_first_frame_features(
        self,
        features: List[NestedTensor],
        positions: List[torch.Tensor],
        batch_size: int,
        frame_count: int,
    ) -> Tuple[List[NestedTensor], List[torch.Tensor]]:
        indices = torch.arange(batch_size, device=features[0].tensors.device) * frame_count
        selected_features = [
            NestedTensor(
                feature.tensors.index_select(0, indices),
                feature.mask.index_select(0, indices),
            )
            for feature in features
        ]
        selected_positions = [position.index_select(0, indices) for position in positions]
        return selected_features, selected_positions

    def _ground_first_frame(
        self,
        first_frame: torch.Tensor,
        features: List[NestedTensor],
        positions: List[torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size, _, height, width = first_frame.shape
        samples = self._make_nested(self._preprocess_dino_images(first_frame))
        captions = [self.caption] * batch_size

        self.dino_model.set_image_features(features, positions)
        outputs = self.dino_model(samples, captions=captions, unset_image_tensor=True)

        logits = outputs["pred_logits"].sigmoid()
        boxes = outputs["pred_boxes"]
        scores = logits.max(dim=-1).values
        best = scores.argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=boxes.device)

        best_boxes = boxes[batch_indices, best]
        best_scores = scores[batch_indices, best]
        centers = best_boxes[:, :2].clone()
        centers[:, 0] = centers[:, 0] * float(width - 1)
        centers[:, 1] = centers[:, 1] * float(height - 1)
        centers[:, 0].clamp_(0, width - 1)
        centers[:, 1].clamp_(0, height - 1)

        point_coords = centers[:, None, :]
        point_labels = torch.ones((batch_size, 1), dtype=torch.int64, device=centers.device)

        self.last_grounding = {
            "boxes_cxcywh": best_boxes,
            "scores": best_scores,
            "points": point_coords,
        }
        return (point_coords, point_labels), self.last_grounding

    def _encode_shared_features(
        self,
        imgs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[NestedTensor], List[torch.Tensor]]:
        batch_size, frame_count = imgs.shape[:2]
        flat_imgs = imgs.flatten(start_dim=0, end_dim=1)
        samples = self._make_nested(self._preprocess_dino_images(flat_imgs))
        features, positions = self.dino_model.backbone(samples)
        image_embeddings = self.feature_adapter(features)
        image_embeddings = image_embeddings.view(batch_size, frame_count, *image_embeddings.shape[-3:])
        image_embeddings = self._add_phase_memory(image_embeddings)
        return image_embeddings, features, positions

    def _add_phase_memory(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        if self.phase_memory is None:
            return image_embeddings

        batch_size, frame_count, channels = image_embeddings.shape[:3]
        if frame_count <= 1:
            phase = torch.zeros(
                (batch_size, frame_count, 1),
                device=image_embeddings.device,
                dtype=image_embeddings.dtype,
            )
        else:
            phase = torch.linspace(
                0.0,
                1.0,
                frame_count,
                device=image_embeddings.device,
                dtype=image_embeddings.dtype,
            ).view(1, frame_count, 1)
            phase = phase.expand(batch_size, -1, -1)

        phase_bias = self.phase_memory(phase.reshape(-1, 1))
        phase_bias = phase_bias.view(batch_size, frame_count, channels, 1, 1)
        scale = self.phase_memory_scale.to(dtype=image_embeddings.dtype).clamp(-1.0, 1.0)
        return image_embeddings + scale * phase_bias

    def _project_memory_keys(
        self,
        image_embeddings: torch.Tensor,
        need_sk: bool = True,
        need_ek: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, frame_count = image_embeddings.shape[:2]
        flat_embeddings = image_embeddings.flatten(start_dim=0, end_dim=1)
        key, shrinkage, selection = self.memory.key_proj(flat_embeddings, need_sk, need_ek)
        key = key.view(batch_size, frame_count, *key.shape[-3:]).transpose(1, 2).contiguous()
        if shrinkage is not None:
            shrinkage = shrinkage.view(batch_size, frame_count, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
        if selection is not None:
            selection = selection.view(batch_size, frame_count, *selection.shape[-3:]).transpose(1, 2).contiguous()
        return key, shrinkage, selection

    def forward(
        self,
        imgs: torch.Tensor,
        pt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        bbox: Optional[torch.Tensor] = None,
        return_prompts: bool = False,
    ) -> torch.Tensor:
        self.last_grounding = None
        image_embeddings, features, positions = self._encode_shared_features(imgs)

        if pt is None:
            batch_size, frame_count = imgs.shape[:2]
            first_features, first_positions = self._select_first_frame_features(
                features,
                positions,
                batch_size,
                frame_count,
            )
            pt, _ = self._ground_first_frame(imgs[:, 0], first_features, first_positions)

        if self.memory is not None:
            pred = self._forward_with_memory_embeddings(imgs, image_embeddings, pt)
        else:
            pred = self._forward_without_memory_embeddings(imgs, image_embeddings, pt)

        if return_prompts:
            return {
                "masks": pred,
                "grounding": self.last_grounding,
            }
        return pred

    def _forward_without_memory_embeddings(
        self,
        imgs: torch.Tensor,
        image_embeddings: torch.Tensor,
        pt: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        _, frame_count, _, height, width = imgs.shape
        frames_pred = []
        points = None if pt is None else (pt[0], pt[1])

        for frame_index in range(frame_count):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            mask, _ = self.mask_decoder(
                image_embeddings=image_embeddings[:, frame_index],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
            frames_pred.append(mask)

        return torch.stack(frames_pred, dim=1)

    def _forward_with_memory_embeddings(
        self,
        imgs: torch.Tensor,
        image_embeddings: torch.Tensor,
        pt: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, frame_count, _, height, width = imgs.shape
        key, shrinkage, selection = self._project_memory_keys(image_embeddings)
        hidden = torch.zeros(
            (batch_size, 1, self.memory.hidden_dim, *key.shape[-2:]),
            device=image_embeddings.device,
        )
        points = None if pt is None else (pt[0], pt[1])

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        mask, _ = self.mask_decoder(
            image_embeddings=image_embeddings[:, 0],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
        values_0, hidden = self.memory("encode_value", imgs[:, 0], image_embeddings[:, 0], hidden, mask)
        values = values_0[:, :, :, :0]

        frames_pred = []
        for frame_index in range(frame_count):
            if frame_index == 0:
                ref_keys = key[:, :, [0]]
                ref_shrinkage = shrinkage[:, :, [0]] if shrinkage is not None else None
                ref_values = values_0
            else:
                ref_keys = key[:, :, :frame_index]
                ref_shrinkage = shrinkage[:, :, :frame_index] if shrinkage is not None else None
                ref_values = values

            frame_embedding = image_embeddings[:, frame_index]
            memory_readout = self.memory(
                "read_memory",
                key[:, :, frame_index],
                selection[:, :, frame_index] if selection is not None else None,
                ref_keys,
                ref_shrinkage,
                ref_values,
            )
            hidden, memory_embedding = self.memory("decode", frame_embedding, hidden, memory_readout)
            mask, _ = self.mask_decoder(
                image_embeddings=frame_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=None,
                dense_prompt_embeddings=memory_embedding[:, 0],
                multimask_output=False,
            )
            mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
            frames_pred.append(mask)

            if frame_index < frame_count - 1:
                is_deep_update = torch.rand((), device=image_embeddings.device).item() < 0.2
                value, hidden = self.memory(
                    "encode_value",
                    imgs[:, frame_index],
                    frame_embedding,
                    hidden,
                    mask,
                    is_deep_update=is_deep_update,
                )
                values = torch.cat([values, value], dim=3)

        return torch.stack(frames_pred, dim=1)
