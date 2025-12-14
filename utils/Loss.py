import torch
import torch.nn.functional as F


def DiceLoss(
    logits: torch.tensor,
    targets: torch.tensor,
    num_classes: int,
    ignore_index: int = None,
    eps: float = 1e-06,
):
    """
    Retunrs DiceLoss.

    Args:
        logits (torch.tensor): [B, C, H, W] raw outputs.
        targets (torch.tensor): [B, H, W] ground truth integer masks.
        num_classes (int): Number of classes
        ignore_index (int): Ignore class index
        eps (float): epsilon value


    Returns:
        diceloss(float)
    """
    probs = F.softmax(logits, dim=1)

    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = targets != ignore_index

    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

    targets_onehot = F.one_hot(safe_targets, num_classes=num_classes)
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
    probs = probs * valid_mask
    targets_onehot = targets_onehot * valid_mask

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)

    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    dice_loss = 1.0 - dice_score.mean()

    return dice_loss


def FocalLoss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = None,
    labels_smoothing: float = 0.1,
):
    ce_loss = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=labels_smoothing,
    )

    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    return focal_loss.mean()
