import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import learn2learn as l2l


from .AmazonDataset import AmazonDataset
from .AmazonDataset import collate_fn
from .Model import Model
from .evaluate import metrics


# if __name__ == '__main__':
def run():
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    # ------------------------------------Dataset Parameters------------------------------------
    parser.add_argument('--dataset',
                        default='Automotive',
                        help='name of the dataset')
    parser.add_argument('--processed_path',
                        default='/disk/yxk/processed/cold_start/ordinary/Automotive/',
                        help="preprocessed path of the raw data")
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--device',
                        default='0',
                        help="using device")
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help="training epochs")
    parser.add_argument('--fast_lr',
                        default=0.5,
                        help='learning rate for fast adaptation')
    parser.add_argument('--meta_lr',
                        default=0.01,
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

    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_path = os.path.join(config.processed_path, "{}_train.csv".format(config.dataset))
    test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))

    asin_sample_path = config.processed_path + '{}_asin_sample.json'.format(config.dataset)
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))

    asin_dict = json.load(open(asin_sample_path, 'r'))
    word_dict = json.load(open(word_dict_path, 'r'))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df: pd.DataFrame = pd.concat([train_df, test_df], ignore_index=True)

    init = AmazonDataset.init(full_df)

    train_support = full_df[full_df["metaFilter"] == "TrainSupport"]
    train_query = full_df[full_df["metaFilter"] == "TrainQuery"]
    test_support = full_df[full_df["metaFilter"] == "TestSupport"]
    test_query = full_df[full_df["metaFilter"] == "TestQuery"]

    train_dataset = AmazonDataset(train_support, train_query, train_df, asin_dict)
    test_dataset = AmazonDataset(test_support, test_query, train_df, asin_dict)
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

    criterion = nn.TripletMarginLoss(margin=config.margin)
    # criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=config.meta_lr)

    # ------------------------------------Train------------------------------------
    step = 0
    loss = torch.tensor(0.).cuda()
    for epoch in range(config.epochs):
        start_time = time.time()
        epoch_loss = 0
        for _, (user_reviews_words,
                support_item_reviews_words, support_queries,
                support_negative_reviews_words,
                query_item_reviews_words, query_queries,
                query_negative_reviews_words, _) in enumerate(train_loader):

            learner = meta.clone(allow_nograd=True)
            learner.module.set_local()
            # ---------Local Update---------
            for i in range(len(support_item_reviews_words)):
                # ---------Fast Adaptation---------

                pred, pos, neg = learner(user_reviews_words[:len(support_item_reviews_words[i])],
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

                pred, pos, neg = learner(user_reviews_words[:len(query_item_reviews_words[i])],
                                         query_item_reviews_words[i],
                                         query_queries[i], 'train',
                                         query_negative_reviews_words[i])
                loss += criterion(pred, pos, neg)

            loss /= len(query_item_reviews_words)
            epoch_loss += loss
            # print("loss:{:.3f}".format(float(loss)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Mrr, Hr, Ndcg = metrics(meta, test_dataset, test_loader, 20, criterion)
        print(
            "Running Epoch {:03d}/{:03d}".format(epoch + 1, config.epochs),
            "loss:{:.3f}".format(float(epoch_loss / len(train_dataset))),
            "Mrr {:.3f}, Hr {:.3f}, Ndcg {:.3f}".format(Mrr, Hr, Ndcg),
            "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)), flush=True)

