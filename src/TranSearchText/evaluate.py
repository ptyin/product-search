import numpy as np

import torch

from common.metrics import hit, mrr, ndcg
from .AmazonDataset import AmazonDataset
from .Model import Model


def evaluate(model, test_dataset: AmazonDataset, test_loader, top_k):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    model.eval()
    model.is_training = False
    all_vec = test_dataset.get_all_test()
    for idx, batch_data in enumerate(test_loader):
        # ---------Test---------
        user = batch_data['userID'].cuda()
        query = batch_data['query'].cuda()
        reviewer_id = batch_data['reviewerID'][0]
        item = batch_data['item'][0]
        query_text = batch_data['query_text']
        pos_text = torch.tensor(test_dataset.text_vec[item]).unsqueeze(dim=0).cuda()
        pred, pos = model(user, query, pos_text, None, False)

        negs = test_dataset.neg_candidates(item)
        # negs = all_vec.copy()
        # del negs[test_dataset.asin_map[item]]
        # negs = torch.tensor(negs).cuda()

        candidates = [pos]
        candidates += [model(None, None, negs, None, True)]
        candidates = torch.cat(candidates, dim=0).squeeze(dim=1)
        scores = torch.pairwise_distance(pred.repeat(len(candidates), 1), candidates)
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
