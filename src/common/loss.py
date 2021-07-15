import torch
import torch.nn.functional as function


def nce_loss(anchor, positive_embeddings, negative_embeddings, positive_biases, negative_biases):
    pos = torch.sum(anchor * positive_embeddings, dim=1) + positive_biases
    pos = function.binary_cross_entropy_with_logits(pos, torch.ones_like(pos).cuda(), reduction='none')
    neg = anchor @ negative_embeddings.t() + negative_biases
    neg = function.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg).cuda(), reduction='none')
    neg = torch.sum(neg, dim=1)
    return pos + neg