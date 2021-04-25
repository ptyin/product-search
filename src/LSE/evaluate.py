import numpy as np

import torch

from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset


def evaluate(model, test_dataset, test_loader, top_k):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    all_items_ids = torch.tensor(list(range(0, len(test_dataset.item_map))),
                                 dtype=torch.long).cuda()
    all_items_embed = model(all_items_ids, None, mode='output_embedding')
    for _, (item, query) in enumerate(test_loader):
        # ---------Test---------
        assert len(item) == 1 and len(query) == 1
        # ---------rank all---------
        item = item.cpu()
        pred = model(None, query, 'test')

        scores = torch.pairwise_distance(pred.repeat(len(all_items_ids), 1), all_items_embed)
        _, ranking_list = scores.sort(dim=-1, descending=True)
        ranking_list = ranking_list.tolist()
        top_idx = []
        while len(top_idx) < top_k:
            candidate_item = ranking_list.pop()
            top_idx.append(candidate_item)
        Mrr.append(mrr(item, top_idx))
        Hr.append(hit(item, top_idx))
        Ndcg.append(ndcg(item, top_idx))

        # query, pos = model(item, query_words, 'test')
        # query = query.squeeze(dim=1)
        # negs = test_dataset.neg_candidates(item)
        # candidates = [pos]
        # candidates += [model(negs, query_words, 'output_embedding')]
        # candidates = torch.cat(candidates, dim=0).squeeze(dim=1)
        # # similarity function
        # scores = torch.sum(query.repeat(100, 1) * candidates, dim=1)
        #
        # _, ranking_list = scores.sort(dim=-1, descending=False)
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
