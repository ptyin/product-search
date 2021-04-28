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
                        default=0.1,
                        help='learning rate')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='batch size for training')
    parser.add_argument('--neg_sample_num',
                        default=5,
                        type=int,
                        help='negative sample number')
    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--embedding_size',
                        default=128,
                        type=int,
                        help='embedding size for words and entities')
    parser.add_argument('--attention_hidden_dim',
                        default=384,
                        type=int,
                        help='dimension of attention hidden layer')
    # parser.add_argument('--regularization',
    #                     default=0.005,
    #                     type=float,
    #                     help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_df, test_df, full_df, query_dict, asin_dict, word_dict = data_preparation(config)
    users, item_map, user_bought = AmazonDataset.init(full_df)

    train_dataset = AmazonDataset(train_df, users, item_map, len(word_dict)+1, user_bought,
                                  asin_dict, 'train', config.neg_sample_num)
    test_dataset = AmazonDataset(test_df, users, item_map, len(word_dict) + 1, user_bought,
                                 asin_dict, 'test')
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              collate_fn=AmazonDataset.collate_fn)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=AmazonDataset.collate_fn)
    if model_name == 'AEM':
        model = AEM(len(word_dict) + 1, len(item_map) + 1, config.embedding_size, config.attention_hidden_dim)
    elif model_name == 'ZAM':
        model = ZAM(len(word_dict) + 1, len(item_map) + 1, config.embedding_size, config.attention_hidden_dim)
    else:
        raise NotImplementedError
    model = model.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    # ------------------------------------Train------------------------------------
    loss = 0

    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        for _, (users, items, words, neg_items, query_words) in enumerate(train_loader):
            neg_words = train_dataset.sample_neg_words(words)
            loss = model(users, items, query_words, 'train', words, neg_items, neg_words)
            # print("loss:{:.3f}".format(float(loss)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 10)
        display(epoch, config.epochs, loss, Hr, Mrr, Ndcg, start_time)

