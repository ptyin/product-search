import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import learn2learn as l2l

from common.metrics import display
from common.data_preparation import parser_add_data_arguments, data_preparation
from .AmazonDataset import AmazonDataset
from .AmazonDataset import collate_fn
from .Model import Model
from .evaluate import evaluate


# if __name__ == '__main__':
def run():
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--fast_lr',
                        default=0.5,
                        help='learning rate for fast adaptation')
    parser.add_argument('--meta_lr',
                        default=0.001,
                        help='learning rate for meta learning')
    parser.add_argument('--batch_size',
                        default=100,
                        type=int,
                        help='batch size for training')
    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--embedding_size',
                        default=128,
                        type=int,
                        help="word embedding size")
    parser.add_argument('--attention_hidden_dim',
                        default=384,
                        type=int,
                        help="dimension of attention hidden layer")
    parser.add_argument('--margin',
                        default=1.,
                        type=float,
                        help="Margin Loss margin")
    parser.add_argument('--regularization',
                        default=0.0,
                        type=float,
                        help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    train_df, test_df, full_df, query_dict, asin_dict, word_dict = data_preparation(config)

    AmazonDataset.clip_words(full_df)
    item_reviews_words = AmazonDataset.cluster_item_reviews(full_df)

    train_support = full_df[full_df["metaFilter"] == "TrainSupport"]
    train_query = full_df[full_df["metaFilter"] == "TrainQuery"]
    test_support = full_df[full_df["metaFilter"] == "TestSupport"]
    test_query = full_df[full_df["metaFilter"] == "TestQuery"]

    train_dataset = AmazonDataset(train_support, train_query, item_reviews_words, asin_dict)
    test_dataset = AmazonDataset(test_support, test_query, item_reviews_words, asin_dict)
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
    # valid_loader = DataLoader(valid_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate_fn)
    # ------------------------------------Model Construction------------------------------------
    # word_dict starts from 1
    model = Model(len(word_dict)+1, config.embedding_size, config.attention_hidden_dim)
    model = model.cuda()
    meta = l2l.algorithms.MAML(model, lr=config.fast_lr)

    criterion = nn.TripletMarginLoss(margin=config.margin, reduction='mean')
    # criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=config.meta_lr)

    # ------------------------------------Train------------------------------------
    step = 0
    loss = torch.tensor(0.).cuda()
    # Mrr, Hr, Ndcg = evaluate(meta, test_dataset, test_loader, 10, criterion)
    # display(-1, config.epochs, loss, Hr, Mrr, Ndcg, time.time())

    for epoch in range(config.epochs):
        start_time = time.time()
        epoch_loss = 0
        step = 0
        for _, (support_user_reviews_words,
                support_item_reviews_words, support_queries,
                support_negative_reviews_words,
                query_user_reviews_words,
                query_item_reviews_words, query_queries,
                query_negative_reviews_words, _) in enumerate(train_loader):

            learner = meta.clone(allow_nograd=True)
            learner.module.set_local()
            # ---------Local Update---------
            for i in range(len(support_item_reviews_words)):
                # ---------Fast Adaptation---------

                pred, pos, neg = learner(support_user_reviews_words[:len(support_item_reviews_words[i])],
                                         support_item_reviews_words[i],
                                         support_queries[i], 'train',
                                         support_negative_reviews_words[i])
                learner.adapt(criterion(pred, pos, neg))
                # learner.adapt(criterion(torch.sum(pred*pos), torch.tensor(1.).cuda()))

            # ---------Global Update---------
            loss = torch.tensor(0.).cuda()
            learner.module.set_global()
            for i in range(len(query_item_reviews_words)):
                # ---------Meta Learning---------

                pred, pos, neg = learner(query_user_reviews_words[:len(query_item_reviews_words[i])],
                                         query_item_reviews_words[i],
                                         query_queries[i], 'train',
                                         query_negative_reviews_words[i])
                loss += criterion(pred, pos, neg)

            loss /= len(query_item_reviews_words)
            loss += config.regularization * learner.module.regularization_term()
            epoch_loss += loss
            step += 1
            # print("loss:{:.3f}".format(float(loss)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Mrr, Hr, Ndcg = evaluate(meta, test_dataset, test_loader, 10, criterion)
        display(epoch, config.epochs, epoch_loss / step, Hr, Mrr, Ndcg, start_time)

