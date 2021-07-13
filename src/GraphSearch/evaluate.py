import numpy as np

import torch

from .Model import Model
from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset


def evaluate(model: Model, test_dataset: AmazonDataset, test_loader, top_k):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    all_items_ids = torch.tensor(list(range(len(test_dataset.users),
                                            len(test_dataset.users) + len(test_dataset.item_map))),
                                 dtype=torch.long).cuda()
    all_items_embed = model(None, all_items_ids, None, mode='output_embedding')
    model.eval()
    for _, (user, item, _, query) in enumerate(test_loader):
        # ---------Test---------
        assert len(user) == 1 and len(item) == 1 and len(query) == 1
        # ---------rank all---------
        item = item.cpu()
        pred = model(user, None, query, 'test')
        scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1)
        # scores = model.predict(user, all_items_ids, query)

        # scores = torch.pairwise_distance(pred.repeat(len(all_items_ids), 1), all_items_embed)
        # _, ranking_list = scores.sort(dim=-1, descending=False)
        # ranking_list = ranking_list.tolist()
        # top_idx = []
        # while len(top_idx) < top_k:
        #     candidate_item = ranking_list.pop()
        #     top_idx.append(candidate_item)
        _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
        ranking_list = ranking_list.tolist()
        Mrr.append(mrr(item-len(test_dataset.users), ranking_list))
        Hr.append(hit(item-len(test_dataset.users), ranking_list))
        Ndcg.append(ndcg(item-len(test_dataset.users), ranking_list))

        # ---------rank 100---------
        # pred, pos = model(user, item, query, 'test')
        # pred = pred.squeeze(dim=1)
        # negs = test_dataset.neg_candidates(item)
        # candidates = [pos]
        # candidates += [model(user, negs, query, 'output_embedding')]
        # candidates = torch.cat(candidates, dim=0).squeeze(dim=1)
        # scores = torch.pairwise_distance(pred.repeat(100, 1), candidates)
        # _, ranking_list = scores.sort(dim=-1, descending=True)
        # ranking_list = ranking_list.tolist()
        # top_idx = []
        # while len(top_idx) < top_k:
        #     candidate_item = ranking_list.pop()
        #     top_idx.append(candidate_item)
        #
        # Mrr.append(mrr(0, top_idx))
        # Hr.append(hit(0, top_idx))
        # Ndcg.append(ndcg(0, top_idx))
    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
