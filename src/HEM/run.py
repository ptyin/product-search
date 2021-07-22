import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn

from common.metrics import display
from common.data_preparation import parser_add_data_arguments, data_preparation
from .AmazonDataset import AmazonDataset
from .Model import Model
from .evaluate import evaluate, evaluate_neg
from common import training_progress, testing_progress
from common.cold_start import get_user_to_num, test


def run():
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
    parser.add_argument('--sub_sampling_rate',
                        default=0.0,
                        type=float,
                        help='sub-sampling rate')
    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--embedding_size',
                        default=128,
                        type=int,
                        help='embedding size for words and entities')
    parser.add_argument('--factor',
                        default=0.5,
                        type=float,
                        help='the query weight of personalized model')
    parser.add_argument('--l2',
                        default=0.005,
                        type=float,
                        help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_df, test_df, full_df, query_dict, asin_dict, word_dict, candidates = data_preparation(config)

    users, item_map, query_max_length = AmazonDataset.init(full_df)

    model = Model(len(word_dict) + 1, len(users) + len(item_map), config.embedding_size, config.factor, config.l2)
    model = model.cuda()

    if config.load:
        g = test(model, full_df, test_df, item_map, candidates, evaluate_neg, config)
        for test_df_by_num in g:
            test_dataset = AmazonDataset(test_df_by_num, users, item_map, query_max_length, len(word_dict) + 1,
                                         asin_dict, 'test', config.debug)
            g.send(test_dataset)

        return

    train_dataset = AmazonDataset(train_df, users, item_map, query_max_length, len(word_dict)+1, asin_dict, 'train',
                                  config.debug, config.neg_sample_num, config.sub_sampling_rate)
    test_dataset = AmazonDataset(test_df, users, item_map, query_max_length, len(word_dict) + 1, asin_dict, 'test',
                                 config.debug)

    # train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=True,
    #                           num_workers=0, collate_fn=AmazonDataset.collate_fn)
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.worker_num,
                              # collate_fn=AmazonDataset.collate_fn
                              )
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    # ------------------------------------Train------------------------------------
    loss = 0

    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        progress = training_progress(train_loader, epoch, config.epochs, loss, config.debug)

        epoch_loss = step = 0
        prepare_time, forward_time, step_time, temp_time = 0, 0, 0, time.time()

        for _, (users, items, words, query_words) in enumerate(progress):
            users, items, words, query_words = (users.cuda(), items.cuda(), words.cuda(), query_words.cuda())
            # temp_time = time.time()
            neg_items = train_dataset.sample_neg_items(items)
            neg_words = train_dataset.sample_neg_words(words)
            prepare_time += time.time() - temp_time

            temp_time = time.time()
            loss = model(users, items, query_words, 'train', words, neg_items, neg_words)
            # print("loss:{:.3f}".format(float(loss)))
            epoch_loss += loss
            forward_time += time.time() - temp_time

            temp_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            step += 1
            step_time += time.time() - temp_time

            temp_time = time.time()

        # Mrr, Hr, Ndcg = evaluate(model, test_dataset,
        #                          testing_progress(test_loader, epoch, config.epochs, config.debug),
        #                          10)
        Mrr, Hr, Ndcg = evaluate(model, test_dataset,
                                 testing_progress(test_loader, epoch, config.epochs, config.debug),
                                 10) if epoch == config.epochs - 1 else (-1, -1, -1)
        display(epoch, config.epochs, epoch_loss / step, Hr, Mrr, Ndcg,
                start_time, prepare_time, forward_time, step_time)
        time.sleep(0.1)

    if not config.load:
        torch.save(model.state_dict(), config.save_path)
