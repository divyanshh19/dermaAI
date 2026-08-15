"""
Configurable Loss Functions for Imbalanced Medical Image Classification.

Provides:
1. Standard CrossEntropyLoss
2. Weighted CrossEntropyLoss
3. Focal Loss (Lin et al.)
4. Label Smoothing CrossEntropyLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss reduces the relative loss for well-classified examples,
    putting more focus on hard, misclassified samples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(
    loss_name: str = "focal",
    class_weights: torch.Tensor = None,
    gamma: float = 2.0,
    smoothing: float = 0.1
) -> nn.Module:
    loss_name = loss_name.lower().strip()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif loss_name in ("weighted_ce", "weighted_cross_entropy"):
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)

    elif loss_name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

    else:
        raise ValueError(f"Unknown loss function '{loss_name}'. Supported: cross_entropy, weighted_ce, focal, label_smoothing")
