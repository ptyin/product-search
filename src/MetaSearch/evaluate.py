import numpy as np

import torch
import torch.nn.functional as F
from common.metrics import hit, mrr, ndcg


def evaluate(meta, test_dataset, test_loader, top_k, optimizer=None):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    all_items = torch.tensor(list(range(len(test_dataset.item_map)))).cuda()
    all_asin, all_review_words = test_dataset.get_all_items()
    candidates = meta(None, None, all_items, all_review_words, None, 'output_embedding')

    for _, (support_users,
            support_user_reviews_words,
            support_items, support_item_reviews_words, support_queries,
            support_negative_items, support_negative_reviews_words,

            query_users,
            query_user_reviews_words,
            query_items, query_item_reviews_words, query_queries,
            query_negative_items, query_negative_reviews_words, query_item_asin) in enumerate(test_loader):

        # ---------Local Update---------
        learner = meta.clone(allow_nograd=True)
        learner.module.set_local()
        for i in range(len(support_item_reviews_words)):
            # ---------Fast Adaptation---------

            loss = learner(support_users[:len(support_item_reviews_words[i])],
                           support_user_reviews_words[:len(support_item_reviews_words[i])],
                           support_items[i], support_item_reviews_words[i],
                           support_queries[i], 'train',
                           support_negative_items[i], support_negative_reviews_words[i])

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            learner.adapt(loss)

        # ---------Test---------
        assert len(query_item_reviews_words) == 1
        query_item = query_items[0].cpu()

        pred = learner(query_users, query_user_reviews_words,
                       query_items[0], query_item_reviews_words[0],
                       query_queries[0], 'test')

        # candidates = learner(None, None,
        #                      all_items,
        #                      # None,
        #                      all_review_words,
        #                      # None,
        #                      query_queries[0].repeat(len(all_asin), 1),
        #                      'output_embedding')
        # scores = F.pairwise_distance(pred.repeat(len(all_items), 1), candidates)
        scores = torch.sum(pred.repeat(len(all_items), 1) * candidates, dim=-1)
        _, ranking_list = scores.topk(top_k, dim=-1, largest=True)
        ranking_list = ranking_list.tolist()

        Mrr.append(mrr(query_item, ranking_list))
        Hr.append(hit(query_item, ranking_list))
        Ndcg.append(ndcg(query_item, ranking_list))
    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
