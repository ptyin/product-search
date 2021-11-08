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
from common.cold_start import test


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
    parser.add_argument('--window_size',
                        default=9,
                        help='n-gram, must be odd')
    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--embedding_size',
                        default=128,
                        type=int,
                        help='embedding size for words and entities')
    parser.add_argument('--regularization',
                        default=0.0,
                        type=float,
                        help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_df, test_df, full_df, query_dict, asin_dict, word_dict, candidates = data_preparation(config)

    users, item_map, query_max_length = AmazonDataset.init(full_df)

    model = Model(len(word_dict) + 1, len(item_map), config.embedding_size, config.regularization)
    model = model.cuda()

    if config.load:
        g = test(model, full_df, test_df, item_map, candidates, evaluate_neg, config)
        for test_df_by_num in g:
            test_dataset = AmazonDataset(test_df_by_num, item_map, query_max_length, len(word_dict) + 1,
                                         config.window_size, 'test', config.debug)
            g.send(test_dataset)

        return

    train_dataset = AmazonDataset(train_df, item_map, query_max_length, len(word_dict) + 1, config.window_size, 'train',
                                  config.debug, config.neg_sample_num)
    test_dataset = AmazonDataset(test_df, item_map, query_max_length, len(word_dict) + 1, config.window_size, 'test',
                                 config.debug)

    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.worker_num)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    # ------------------------------------Train------------------------------------
    loss = 0
    # Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 10)
    # display(-1, config.epochs, loss, Hr, Mrr, Ndcg, time.time())

    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        progress = training_progress(train_loader, epoch, config.epochs, loss, config.debug)

        epoch_loss = step = 0
        prepare_time, forward_time, step_time, temp_time = 0, 0, 0, time.time()

        for _, (items, words, query_words) in enumerate(progress):
            items, words, query_words = (items.cuda(), words.cuda(), query_words.cuda())
            neg_items = train_dataset.sample_neg_items(items)

            prepare_time += time.time() - temp_time

            temp_time = time.time()
            loss = model(items, query_words, 'train', words, neg_items)
            if config.debug:
                progress.set_postfix({"loss": "{:.3f}".format(float(loss))})
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

        Mrr, Hr, Ndcg = evaluate(model, test_dataset,
                                 testing_progress(test_loader, epoch, config.epochs, config.debug),
                                 10)

        Mrr, Hr, Ndcg = evaluate(model, test_dataset,
                                 testing_progress(test_loader, epoch, config.epochs, config.debug),
                                 10) if epoch == config.epochs - 1 else (-1, -1, -1)
        # display(epoch, config.epochs, epoch_loss / step, Hr, Mrr, Ndcg,
        #         start_time, prepare_time, forward_time, step_time)
        time.sleep(0.1)

        if not config.load:
            if epoch == config.epochs - 1:
                save_path = os.path.join(config.save_path, '{}.pt'.format(config.save_str))
            else:
                save_path = os.path.join(config.save_path, '{}-{}.pt'.format(config.save_str, epoch))
            torch.save(model.state_dict(), save_path)
