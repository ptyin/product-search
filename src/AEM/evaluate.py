import numpy as np

import torch

from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset


def evaluate(model, test_dataset: AmazonDataset, test_loader, top_k):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    all_items_ids = torch.tensor(list(range(1, len(test_dataset.item_map))),
                                 dtype=torch.long).cuda()
    all_items_embed, all_items_bias = model(None, all_items_ids, None, mode='output_embedding')

    for _, (user_bought_items, item, query) in enumerate(test_loader):
        # ---------Test---------
        assert len(user_bought_items) == 1 and len(item) == 1 and len(query) == 1
        user_bought_items = user_bought_items.cuda()
        query = query.cuda()

        # ---------rank all---------
        pred = model(user_bought_items, None, query, 'test')

        scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1) + all_items_bias
        # _, ranking_list = scores.sort(dim=-1, descending=True)
        _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
        ranking_list = ranking_list.tolist()
        Mrr.append(mrr(item-1, ranking_list))
        Hr.append(hit(item-1, ranking_list))
        Ndcg.append(ndcg(item-1, ranking_list))

    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
