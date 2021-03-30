import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn

from .AmazonDataset import AmazonDataset
from .Model import Model
from .evaluate import metrics


def run():
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    # ------------------------------------Dataset Parameters------------------------------------
    parser.add_argument('--dataset',
                        default='Musical_Instruments',
                        help='name of the dataset')
    parser.add_argument('--processed_path',
                        default='/disk/yxk/processed/cold_start/ordinary/Musical_Instruments/',
                        help="preprocessed path of the raw data")
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--device',
                        default='0',
                        help="using device")
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help="training epochs")
    parser.add_argument('--lr',
                        default=0.1,
                        help='learning rate for fast adaptation')
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
                        default=8,
                        type=int,
                        help='embedding size for words and entities')
    parser.add_argument('--regularization',
                        default=0.005,
                        type=float,
                        help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_path = os.path.join(config.processed_path, "{}_train.csv".format(config.dataset))
    test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))
    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))

    query_path = os.path.join(config.processed_path, '{}_query.json'.format(config.dataset))
    asin_sample_path = config.processed_path + '{}_asin_sample.json'.format(config.dataset)
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))

    query_dict = json.load(open(query_path, 'r'))
    asin_dict = json.load(open(asin_sample_path, 'r'))
    word_dict = json.load(open(word_dict_path, 'r'))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(full_path)

    users, item_map = AmazonDataset.init(full_df)

    train_dataset = AmazonDataset(train_df, users, item_map, len(word_dict)+1, asin_dict, 'train', config.neg_sample_num)
    test_dataset = AmazonDataset(test_df, users, item_map, len(word_dict) + 1, asin_dict, 'test')

    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              collate_fn=AmazonDataset.collate_fn)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=AmazonDataset.collate_fn)

    model = Model(len(word_dict) + 1, len(users) + len(item_map), config.embedding_size, config.regularization)
    model = model.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
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

        Mrr, Hr, Ndcg = metrics(model, test_dataset, test_loader, 20)
        print(
            "Running Epoch {:03d}/{:03d}".format(epoch + 1, config.epochs),
            "loss:{:.3f}".format(float(loss)),
            "Mrr {:.3f}, Hr {:.3f}, Ndcg {:.3f}".format(Mrr, Hr, Ndcg),
            "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
