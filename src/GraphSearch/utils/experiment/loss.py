import torch
import torch.nn.functional as function


def bpr_loss(pred: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
    return -function.logsigmoid(torch.sum(pred * (pos - neg), dim=1)).mean()


def triplet_margin_loss(pred: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
    return function.triplet_margin_loss(pred, pos, neg, reduction='mean')
