import torch
import torch.nn.functional as function


def bpr_loss(pred: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
    return -function.logsigmoid(torch.sum(pred * (pos - neg), dim=1)).mean()


def hem_loss(pred: torch.Tensor, pos: torch.Tensor, negs: torch.Tensor):
    # Data Enhancement Form
    negs = negs.permute(1, 0, 2)
    loss = 0
    for neg in negs:
        loss += -(function.logsigmoid(torch.sum(pred * pos, dim=1)) +
                  function.logsigmoid(torch.sum(-pred * neg, dim=1))).mean()
    return loss
    # Expectation Form
    # pos = function.logsigmoid(torch.sum(pred * pos, dim=1))
    # neg = torch.sum(function.logsigmoid(-torch.einsum('bke,be->bk', negs, pred)))
    # return -(pos + neg).mean()


def triplet_margin_loss(pred: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
    return function.triplet_margin_loss(pred, pos, neg, reduction='mean')


def log_loss(pred: torch.Tensor, pos: torch.Tensor):
    return -function.logsigmoid(torch.sum(pred * pos, dim=1)).mean()
