from typing import Optional

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: bool = True,
        batch_average: bool = True,
        ignore_index: int = 255,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.batch_average = batch_average
        self.criterion = nn.CrossEntropyLoss(size_average=size_average)

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for multi-class classification

        Args:
            logit (torch.Tensor): prediction from model (batch_size, n_classes, ...)
            target (torch.Tensor): ground truth label (batch_size, ...)
        """
        n = logit.size()[0]
        target = target.argmax(dim=1)

        logpt = -self.criterion(logit, target.long())
        pt = torch.exp(logpt)

        if self.alpha is not None:
            logpt *= self.alpha

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
