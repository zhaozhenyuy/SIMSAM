import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """Weighted BCE with logits plus soft Dice loss."""

    def __init__(self, pos_weight=2.0, dice_weight=0.8):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target):
        if target.dim() == 5:
            target = target.flatten(0, 1)
            logits = logits.flatten(0, 1)
        target = target.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight.to(logits.device),
        )
        probability = torch.sigmoid(logits)
        dims = tuple(range(1, probability.dim()))
        intersection = (probability * target).sum(dim=dims)
        denominator = probability.square().sum(dim=dims) + target.square().sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice


def get_criterion(modelname="SharedGroundedMemSAM", opt=None):
    if modelname != "SharedGroundedMemSAM":
        raise ValueError(f"Unsupported model for this release: {modelname}")
    criterion = DiceBCELoss(pos_weight=2.0, dice_weight=0.8)
    return criterion.to(opt.device)
