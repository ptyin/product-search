import numpy as np

import torch

from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset


def evaluate_neg(model, test_dataset: AmazonDataset, test_loader, candidates, top_k):
    with torch.no_grad():
        Mrr, Hr, Ndcg = [], [], []
        loss = 0  # No effect, ignore this line
        for _, (user, item, query) in enumerate(test_loader):
            # ---------Test---------
            assert len(user) == 1 and len(item) == 1 and len(query) == 1
            user = user.cuda()
            query = query.cuda()
            all_items_ids = torch.cat([item.cuda(), torch.tensor(candidates[item.item()]).cuda()], dim=0)
            all_items_embed, all_items_bias = model(None, all_items_ids, None, mode='output_embedding')
            # ---------rank all---------
            pred = model(user, None, query, 'test')
            scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1) + all_items_bias
            _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
            ranking_list = ranking_list.tolist()
            Mrr.append(mrr(0, ranking_list))
            Hr.append(hit(0, ranking_list))
            Ndcg.append(ndcg(0, ranking_list))
    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)


def evaluate(model, test_dataset: AmazonDataset, test_loader, top_k):
    with torch.no_grad():
        Mrr, Hr, Ndcg = [], [], []
        loss = 0  # No effect, ignore this line
        all_items_ids = torch.tensor(list(range(len(test_dataset.users),
                                                len(test_dataset.users) + len(test_dataset.item_map))),
                                     dtype=torch.long).cuda()
        all_items_embed, all_items_bias = model(None, all_items_ids, None, mode='output_embedding')

        for _, (user, item, query) in enumerate(test_loader):
            # ---------Test---------
            assert len(user) == 1 and len(item) == 1 and len(query) == 1
            user = user.cuda()
            query = query.cuda()
            # ---------rank all---------
            pred = model(user, None, query, 'test')

            # scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1)
            scores = torch.sum(pred.repeat(len(all_items_ids), 1) * all_items_embed, dim=1) + all_items_bias
            # scores = (pred @ all_items_embed.t()).squeeze(dim=0) + all_items_bias
            _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
            ranking_list = ranking_list.tolist()
            Mrr.append(mrr(item-len(test_dataset.users), ranking_list))
            Hr.append(hit(item-len(test_dataset.users), ranking_list))
            Ndcg.append(ndcg(item-len(test_dataset.users), ranking_list))

        # pred, pos = model(user, item, query, 'test')
        # pred = pred.squeeze(dim=1)
        # negs = test_dataset.neg_candidates(item)
        # candidates = [pos]
        # candidates += [model(user, negs, query, 'output_embedding')]
        # candidates = torch.cat(candidates, dim=0).squeeze(dim=1)
        #
        # # similarity function
        # scores = torch.sum(pred.repeat(100, 1) * candidates, dim=1)
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
