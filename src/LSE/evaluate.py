import numpy as np

import torch

from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset


def evaluate_neg(model, test_dataset: AmazonDataset, test_loader, candidates, top_k):
    with torch.no_grad():
        Mrr, Hr, Ndcg = [], [], []
        loss = 0  # No effect, ignore this line
        for _, (item, query) in enumerate(test_loader):
            # ---------Test---------
            assert len(item) == 1 and len(query) == 1
            query = query.cuda()
            all_items_ids = torch.cat([item.cuda(), torch.tensor(candidates[item.item()]).cuda()], dim=0)
            all_items_embed, all_items_bias = model(all_items_ids, None, mode='output_embedding')
            # ---------rank all---------
            pred = model(None, query, 'test')
            scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1) + all_items_bias
            _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
            ranking_list = ranking_list.tolist()
            Mrr.append(mrr(0, ranking_list))
            Hr.append(hit(0, ranking_list))
            Ndcg.append(ndcg(0, ranking_list))
    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)


def evaluate(model, test_dataset, test_loader, top_k):
    with torch.no_grad():

        Mrr, Hr, Ndcg = [], [], []
        loss = 0  # No effect, ignore this line
        all_items_ids = torch.tensor(list(range(0, len(test_dataset.item_map))),
                                     dtype=torch.long).cuda()
        all_items_embed, all_items_bias = model(all_items_ids, None, mode='output_embedding')
        for _, (item, query) in enumerate(test_loader):
            # ---------Test---------
            assert len(item) == 1 and len(query) == 1
            # ---------rank all---------
            query = query.cuda()
            pred = model(None, query, 'test')

            scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1) + all_items_bias
            # _, ranking_list = scores.sort(dim=-1, descending=True)
            _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
            ranking_list = ranking_list.tolist()
            Mrr.append(mrr(item, ranking_list))
            Hr.append(hit(item, ranking_list))
            Ndcg.append(ndcg(item, ranking_list))

    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
