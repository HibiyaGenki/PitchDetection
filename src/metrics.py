from typing import List, Tuple

import torch


def calc_average_precision(
    pred: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[float]:
    """Calculate average precision at top-k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    aps = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        ap = correct_k.mul_(1 / (k * batch_size)).sum().item()
        aps.append(ap)

    return aps
