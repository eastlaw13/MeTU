import torch
import torch.nn.functional as F


def iou_component(preds, targets, num_class, ignore_idx=None):
    """
    Calcualate intersection and union value for segmentation.

    Args:
        preds (torch.Tensor): model prediction tensor(B, H, W). Each pixels in [0, num_class-1]
        targets (torch.Tensor): Ground truth mask tensor(B, H, W). Each pixels in [0, num_class-1]
        num_class (int): Number of valid classes (Except background)
        ignore_class: Ignore class when calculating IoU
    """
    if ignore_idx is not None:
        valid_mask = targets != ignore_idx
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    intersection_sum = torch.zeros(num_class, dtype=torch.long, device=preds.device)
    union_sum = torch.zeros(num_class, dtype=torch.long, device=preds.device)

    preds, targets = preds.flatten(), targets.flatten()

    for class_id in range(num_class):
        is_pred = preds == class_id
        is_target = targets == class_id

        intersection = is_pred & is_target
        union = is_pred | is_target

        intersection_sum[class_id] = intersection.sum()
        union_sum[class_id] = union.sum()

    return intersection_sum, union_sum


def iou_calculation(intersection_sum, union_sum):
    """
    Calculate the stacked intersection adn union value. When union value is 0, then the IoU is 0.

    Args:
        intersection_sum(torch.tensor): The intersection value of each class.
        union_sum(torch.tensor): The union value of each class.

    """

    intersection_f = intersection_sum.float()
    union_f = union_sum.float()

    iou_per_class = torch.where(
        union_f > 0, intersection_f / union_f, torch.tensor(0.0, device=union_f.device)
    )

    valid_classes = (union_f > 0).sum().item()

    if valid_classes == 0:
        return torch.tensor(0.0, device=union_f.device)

    m_iou = iou_per_class.sum() / valid_classes

    return m_iou, iou_per_class
