import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn

from common.data_preparation import parser_add_data_arguments, data_preparation
from common.metrics import display
from .AmazonDataset import AmazonDataset
from .Model import Model
from .evaluate import evaluate


# if __name__ == '__main__':
def run():
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
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
                        default=1000,
                        type=int,
                        help='batch size for training')
    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--word_embedding_size',
                        default=64,
                        type=int,
                        help="word embedding size")
    parser.add_argument('--entity_embedding_size',
                        default=64,
                        type=int,
                        help="entity embedding size")
    parser.add_argument('--attention_hidden_dim',
                        default=384,
                        type=int,
                        help="LSTM hidden size")
    parser.add_argument('--margin',
                        default=1.,
                        type=float,
                        help="Margin Loss margin")

    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    train_df, test_df, full_df, query_dict, asin_dict, word_dict = data_preparation(config)
    # clip words
    AmazonDataset.clip_words(full_df)
    users, item_map, query_map, graph = AmazonDataset.construct_graph(full_df, len(word_dict) + 1)
    # graph = graph.to('cuda:{}'.format(config.device))
    graph = graph.to('cuda')

    train_dataset = AmazonDataset(train_df, users, item_map, asin_dict)
    test_dataset = AmazonDataset(test_df, users, item_map, asin_dict)

    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              collate_fn=AmazonDataset.collate_fn)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)

    model = Model(graph, len(word_dict) + 1, len(query_map), len(users) + len(item_map),
                  word_embedding_size=config.word_embedding_size,
                  entity_embedding_size=config.entity_embedding_size)
    model = model.cuda()
    model.init_graph()

    criterion = nn.TripletMarginLoss(margin=config.margin, reduction='sum')
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    # ------------------------------------Train------------------------------------
    loss = 0
    # Mrr, Hr, Ndcg = metrics(model, test_dataset, test_loader, 20)

    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        for _, (users, items, negs, query_words) in enumerate(train_loader):
            pred, pos, neg = model(users, items, query_words, 'train', negs)
            loss = criterion(pred, pos, neg)
            print("loss:{:.3f}".format(float(loss)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 20)
        display(epoch, config.epochs, loss, Hr, Mrr, Ndcg, start_time)
