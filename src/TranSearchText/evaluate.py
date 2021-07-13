import numpy as np

import torch

from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset
from .Model import Model


def __get_all_item(test_dataset: AmazonDataset, model: Model):
    all_items_embed = []
    all_item_ids = []
    for text, asin in test_dataset.get_all_test():
        text = torch.cuda.FloatTensor(text)
        all_items_embed.append(text)
        # item_embed = model(None, None, text, None, True)
        all_item_ids.append(asin)
        # all_items_embed.append(item_embed.view(-1).data.cpu().numpy())

    all_items_embed = torch.stack(all_items_embed, dim=0)
    all_items_embed = model(None, None, all_items_embed, None, True)
    all_items_map = {i: item for i, item in enumerate(all_item_ids)}

    return all_items_embed, all_items_map


def evaluate(model, test_dataset: AmazonDataset, test_loader, top_k):
    Mrr, Hr, Ndcg = [], [], []
    model.eval()
    model.is_training = False
    all_items_embed, all_items_map = __get_all_item(test_dataset, model)
    for idx, batch_data in enumerate(test_loader):
        # ---------Test---------
        # ------------rank all------------
        user = batch_data['userID'].cuda()
        query = batch_data['query'].cuda()
        item = batch_data['item'][0]

        pred = model(user, query, None, None, False)
        scores = torch.pairwise_distance(pred.repeat(len(all_items_map), 1), all_items_embed)
        _, ranking_list = scores.topk(top_k, dim=-1, largest=False)
        ranking_list = [all_items_map[i] for i in ranking_list.tolist()]
        # top_idx = []
        # while len(top_idx) < top_k:
        #     candidate_item = ranking_list.pop()
        #     top_idx.append(candidate_item)
        Mrr.append(mrr(item, ranking_list))
        Hr.append(hit(item, ranking_list))
        Ndcg.append(ndcg(item, ranking_list))

        # ------------rank 100------------
        # user = batch_data['userID'].cuda()
        # query = batch_data['query'].cuda()
        # item = batch_data['item'][0]
        # pos_text = torch.tensor(test_dataset.text_vec[item]).unsqueeze(dim=0).cuda()
        # pred, pos = model(user, query, pos_text, None, False)
        #
        # negs = test_dataset.neg_candidates(item)
        #
        # candidates = [pos]
        # candidates += [model(None, None, negs, None, True)]
        # candidates = torch.cat(candidates, dim=0).squeeze(dim=1)
        # scores = torch.pairwise_distance(pred.repeat(len(candidates), 1), candidates)
        # ------------New Form------------
        # users = batch_data['userID'].cuda()
        # queries = batch_data['query'].cuda()
        # items = batch_data['item']
        # pos_text = torch.stack([torch.tensor(test_dataset.text_vec[item]) for item in items], dim=0).cuda()
        # scores = model(users, queries, pos_text, None)
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
