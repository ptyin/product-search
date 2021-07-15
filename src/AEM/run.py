import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from common.metrics import display
from common.data_preparation import parser_add_data_arguments, data_preparation
from .AmazonDataset import AmazonDataset
from .Model import AEM, ZAM
from .evaluate import evaluate


def run_aem():
    run('AEM')


def run_zam():
    run('ZAM')


def run(model_name: str):
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--lr',
                        default=0.5,
                        help='learning rate')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='batch size for training')
    parser.add_argument('--neg_sample_num',
                        default=5,
                        type=int,
                        help='negative sample number')
    # parser.add_argument('--max_history_length',
    #                     default=10,
    #                     type=int,
    #                     help='max length of user bought items')

    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--embedding_size',
                        default=64,
                        type=int,
                        help='embedding size for words and entities')
    parser.add_argument('--head_num',
                        default=5,
                        type=int,
                        help='attention hidden units')
    # parser.add_argument('--regularization',
    #                     default=0.005,
    #                     type=float,
    #                     help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_df, test_df, full_df, query_dict, asin_dict, word_dict = data_preparation(config)
    users, item_map, user_bought, query_max_length = AmazonDataset.init(full_df)

    train_dataset = AmazonDataset(train_df, users, item_map, query_max_length, len(word_dict)+1, user_bought,
                                  asin_dict, 'train', config.neg_sample_num)
    test_dataset = AmazonDataset(test_df, users, item_map, query_max_length, len(word_dict) + 1, user_bought,
                                 asin_dict, 'test')
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=False, num_workers=0,
                              collate_fn=AmazonDataset.collate_fn)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)
    if model_name == 'AEM':
        model = AEM(len(word_dict) + 1, len(item_map) + 1, config.embedding_size, config.head_num)
    elif model_name == 'ZAM':
        model = ZAM(len(word_dict) + 1, len(item_map) + 1, config.embedding_size, config.head_num)
    else:
        raise NotImplementedError
    model = model.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, initial_accumulator_value=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    # ------------------------------------Train------------------------------------
    loss = 0

    for epoch in range(config.epochs):
        start_time = time.time()
        epoch_loss = 0
        step = 0
        model.train()
        for _, (user_bought_items, user_bought_masks, items, words, query_words) in enumerate(train_loader):
            user_bought_items, user_bought_masks, items, words, query_words = \
                (user_bought_items.cuda(), user_bought_masks.cuda(), items.cuda(), words.cuda(), query_words.cuda())

            neg_items = train_dataset.sample_neg_items(items)
            neg_words = train_dataset.sample_neg_words(words)
            loss = model(user_bought_items, items, query_words, 'train', user_bought_masks, words, neg_items, neg_words)
            # print("loss:{:.3f}".format(float(loss)))
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            step += 1

        Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 10)
        # Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 10) if epoch == config.epochs - 1 else (-1, -1, -1)
        display(epoch, config.epochs, epoch_loss / step, Hr, Mrr, Ndcg, start_time)

# Running Epoch 001/020 loss:12.850 Hr 0.078, Mrr 0.030, Ndcg 0.041 costs: 00: 00: 52
# Running Epoch 002/020 loss:4.754 Hr 0.137, Mrr 0.054, Ndcg 0.073 costs: 00: 00: 53
# Running Epoch 003/020 loss:3.527 Hr 0.178, Mrr 0.063, Ndcg 0.089 costs: 00: 00: 52
# Running Epoch 004/020 loss:2.909 Hr 0.217, Mrr 0.078, Ndcg 0.111 costs: 00: 00: 56
# Running Epoch 005/020 loss:3.737 Hr 0.237, Mrr 0.084, Ndcg 0.120 costs: 00: 00: 56
# Running Epoch 006/020 loss:3.330 Hr 0.261, Mrr 0.094, Ndcg 0.133 costs: 00: 00: 53
# Running Epoch 007/020 loss:2.166 Hr 0.281, Mrr 0.102, Ndcg 0.144 costs: 00: 00: 54
# Running Epoch 008/020 loss:2.899 Hr 0.291, Mrr 0.099, Ndcg 0.145 costs: 00: 00: 52
# Running Epoch 009/020 loss:8.179 Hr 0.312, Mrr 0.104, Ndcg 0.153 costs: 00: 00: 52
# Running Epoch 010/020 loss:2.712 Hr 0.320, Mrr 0.108, Ndcg 0.158 costs: 00: 00: 53
# Running Epoch 011/020 loss:3.011 Hr 0.313, Mrr 0.104, Ndcg 0.153 costs: 00: 00: 53
# Running Epoch 012/020 loss:2.871 Hr 0.325, Mrr 0.109, Ndcg 0.160 costs: 00: 00: 53
# Running Epoch 013/020 loss:2.384 Hr 0.329, Mrr 0.111, Ndcg 0.162 costs: 00: 00: 53
# Running Epoch 014/020 loss:2.492 Hr 0.343, Mrr 0.113, Ndcg 0.167 costs: 00: 00: 52
# Running Epoch 015/020 loss:3.431 Hr 0.330, Mrr 0.112, Ndcg 0.164 costs: 00: 00: 55
# Running Epoch 016/020 loss:3.676 Hr 0.336, Mrr 0.112, Ndcg 0.165 costs: 00: 00: 54
# Running Epoch 017/020 loss:2.714 Hr 0.349, Mrr 0.117, Ndcg 0.172 costs: 00: 00: 53
# Running Epoch 018/020 loss:1.995 Hr 0.347, Mrr 0.119, Ndcg 0.173 costs: 00: 00: 54
# Running Epoch 019/020 loss:2.194 Hr 0.355, Mrr 0.121, Ndcg 0.176 costs: 00: 00: 54
# Running Epoch 020/020 loss:2.294 Hr 0.349, Mrr 0.119, Ndcg 0.174 costs: 00: 00: 54
# -----------Best Result:-----------
# Hr: 0.355, Mrr: 0.121, Ndcg: 0.176
# ----------------------------------
#
# Running Epoch 001/020 loss:16.608 Hr 0.023, Mrr 0.008, Ndcg 0.011 costs: 00: 00: 52
# Running Epoch 002/020 loss:11.101 Hr 0.061, Mrr 0.020, Ndcg 0.029 costs: 00: 00: 51
# Running Epoch 003/020 loss:8.369 Hr 0.113, Mrr 0.037, Ndcg 0.055 costs: 00: 00: 51
# Running Epoch 004/020 loss:6.682 Hr 0.153, Mrr 0.055, Ndcg 0.078 costs: 00: 00: 51
# Running Epoch 005/020 loss:5.811 Hr 0.198, Mrr 0.071, Ndcg 0.100 costs: 00: 00: 53
# Running Epoch 006/020 loss:5.165 Hr 0.235, Mrr 0.085, Ndcg 0.120 costs: 00: 00: 53
# Running Epoch 007/020 loss:4.696 Hr 0.266, Mrr 0.100, Ndcg 0.138 costs: 00: 00: 52
# Running Epoch 008/020 loss:4.324 Hr 0.291, Mrr 0.114, Ndcg 0.155 costs: 00: 00: 56
# Running Epoch 009/020 loss:4.137 Hr 0.318, Mrr 0.120, Ndcg 0.166 costs: 00: 00: 51
# Running Epoch 010/020 loss:3.952 Hr 0.338, Mrr 0.132, Ndcg 0.180 costs: 00: 00: 51
# Running Epoch 011/020 loss:3.762 Hr 0.377, Mrr 0.153, Ndcg 0.205 costs: 00: 00: 54
# Running Epoch 012/020 loss:3.639 Hr 0.395, Mrr 0.159, Ndcg 0.214 costs: 00: 00: 53
# Running Epoch 013/020 loss:3.540 Hr 0.401, Mrr 0.165, Ndcg 0.220 costs: 00: 00: 53
# Running Epoch 014/020 loss:3.403 Hr 0.427, Mrr 0.180, Ndcg 0.238 costs: 00: 00: 53
# Running Epoch 015/020 loss:3.325 Hr 0.442, Mrr 0.184, Ndcg 0.244 costs: 00: 00: 51
# Running Epoch 016/020 loss:3.256 Hr 0.449, Mrr 0.195, Ndcg 0.255 costs: 00: 00: 51
# Running Epoch 017/020 loss:3.191 Hr 0.467, Mrr 0.195, Ndcg 0.258 costs: 00: 00: 51
# Running Epoch 018/020 loss:3.150 Hr 0.464, Mrr 0.199, Ndcg 0.261 costs: 00: 00: 51
# Running Epoch 019/020 loss:3.092 Hr 0.472, Mrr 0.201, Ndcg 0.265 costs: 00: 00: 51
# Running Epoch 020/020 loss:3.042 Hr 0.501, Mrr 0.213, Ndcg 0.281 costs: 00: 00: 53
# -----------Best Result:-----------
# Hr: 0.501, Mrr: 0.213, Ndcg: 0.281
# ----------------------------------
#
# Process finished with exit code 0
