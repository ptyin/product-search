import numpy as np

import torch

from .AmazonDataset import AmazonDataset


def mrr(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(float(index + 1))
    else:
        return 0


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_dataset, test_loader, top_k):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    for _, (user, item, neg, query) in enumerate(test_loader):
        # ---------Test---------
        assert len(user) == 1 and len(item) == 1 and len(neg) == 1 and len(query) == 1
        pred, pos = model(user, item, query, 'test')
        pred = pred.squeeze(dim=1)
        negs = test_dataset.neg_candidates(item)
        candidates = [pos]
        candidates += [model(user, negs, query, 'output_embedding')]
        candidates = torch.cat(candidates, dim=0).squeeze(dim=1)
        scores = torch.pairwise_distance(pred.repeat(100, 1), candidates)
        _, ranking_list = scores.sort(dim=-1, descending=True)
        ranking_list = ranking_list.tolist()
        top_idx = []
        while len(top_idx) < top_k:
            candidate_item = ranking_list.pop()
            top_idx.append(candidate_item)

        Mrr.append(mrr(0, top_idx))
        Hr.append(hit(0, top_idx))
        Ndcg.append(ndcg(0, top_idx))
    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
