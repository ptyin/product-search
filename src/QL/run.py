import os
import json
from argparse import ArgumentParser
import time

from torch.utils.data import DataLoader

from common.metrics import display
from common.metrics import hit, mrr, ndcg
from common.data_preparation import parser_add_data_arguments

from .AmazonDataset import *


def ql(query, tf, prior_p, mu):
    # tf[query]: (word, items), prior_p[query]: (word,)
    prob = (tf[query] + mu * prior_p[query].unsqueeze(dim=-1)) / (torch.sum(tf, dim=0) + mu)
    # (word, items)
    prob = torch.sum(prob, dim=0)
    # (word, items)
    return prob


def uql(user, query, tf, prior_p, mu, _lambda):
    prob_q_i = ql(query, tf, prior_p, mu)
    prob_u_i = ql(user, tf, prior_p, mu)
    prob = _lambda * prob_q_i + (1 - _lambda) * prob_u_i
    return prob


def run(model_str, mu, lam):
    parser = ArgumentParser()
    # parser.add_argument('--mu', type=int, default=6000)
    # parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--log_dir', )

    parser_add_data_arguments(parser)
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    # ------------------------prepare for data------------------------
    config.processed_path = os.path.join(config.processed_path, config.dataset + '/')
    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))
    full_df = pd.read_csv(full_path)
    # train_df = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)
    test_df = full_df[full_df['filter'] == 'Test'].reset_index(drop=True)
    # word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))
    # word_dict = json.load(open(word_dict_path, 'r'))
    # ----------------------------------------------------------------
    # item_map = init(full_df)
    # tf, word_distribution, u_words = prior_knowledge(train_df, len(word_dict) + 1, item_map)

    # ------------------------load from disk------------------------
    item_map = json.load(open(os.path.join(config.processed_path, 'ql', 'item_map.json'), 'r'))
    tf = torch.load(os.path.join(config.processed_path, 'ql', 'tf.pt')).cuda()
    word_distribution = torch.load(os.path.join(config.processed_path, 'ql', 'word_distribution.pt')).cuda()
    u_words = json.load(open(os.path.join(config.processed_path, 'ql', 'u_words.json'), 'r'))
    # ----------------------------------------------------------------

    test_dataset = AmazonDataset(test_df, item_map)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=lambda batch: batch[0])

    top_k = 10
    Mrr, Hr, Ndcg = [], [], []
    start_time = time.time()
    for user, item, query in test_loader:
        user = torch.tensor(u_words[str(user)], dtype=torch.long).cuda()
        query = torch.tensor(query, dtype=torch.long).cuda()

        if model_str == 'ql':
            prob = ql(query, tf, word_distribution, mu)
        elif model_str == 'uql':
            prob = uql(user, query, tf, word_distribution, mu, lam)
        else:
            raise NotImplementedError

        _, ranking_list = prob.topk(top_k, dim=-1, largest=True)
        ranking_list = ranking_list.tolist()
        Hr.append(hit(item, ranking_list))
        Mrr.append(mrr(item, ranking_list))
        Ndcg.append(ndcg(item, ranking_list))

    display(0, 1, 0, np.mean(Hr), np.mean(Mrr), np.mean(Ndcg), start_time)


def run_ql():
    for mu in [2000, 6000, 10000]:
        print('++++++++++++++QL, mu: {}++++++++++++++'.format(mu))
        run('ql', mu, None)


def run_uql():
    for mu in [2000, 6000, 10000]:
        for lam in [0.2, 0.4, 0.6, 0.8, 1.0]:
            print('++++++++++++++UQL, mu: {}, lambda: {}++++++++++++++'.format(mu, lam))
            run('uql', mu, lam)

