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


# if __name__ == '__main__':
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

        Mrr, Hr, Ndcg = metrics(model, test_dataset, test_loader, 20)
        print(
            "Running Epoch {:03d}/{:03d}".format(epoch + 1, config.epochs),
            "loss:{:.3f}".format(float(loss)),
            "Mrr {:.3f}, Hr {:.3f}, Ndcg {:.3f}".format(Mrr, Hr, Ndcg),
            "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
