import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.nn import functional as func
from torch.utils.data import DataLoader
from gensim.models.doc2vec import Doc2Vec

from common.metrics import display
from common.data_preparation import parser_add_data_arguments, data_preparation
from common.experiment import neighbor_similarity
from .AmazonDataset import AmazonDataset
from .Model import Model
from .evaluate import evaluate


def run():
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--lr',
                        default=0.001,
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
    parser.add_argument('--mode',
                        default='text',
                        type=str,
                        help='the model mode')
    parser.add_argument('--embedding_size',
                        default=64,
                        type=int,
                        help='embedding size for words and entities')
    parser.add_argument('--visual_size',
                        default=0,
                        type=int)
    parser.add_argument('--textual_size',
                        default=512,
                        type=int)
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="the dropout rate")
    parser.add_argument('--regularization',
                        default=0.0001,
                        type=float,
                        help='regularization factor')
    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    train_df, test_df, full_df, query_dict, asin_dict, word_dict = data_preparation(config)

    user_size = len(full_df.userID.unique())
    model = Model(config.visual_size, config.textual_size, config.embedding_size, user_size,
                  config.mode, config.dropout, is_training=True)
    model = model.cuda()

    if config.load:
        model.load_state_dict(torch.load(config.save_path))
        user_embeddings: torch.Tensor = model.user_embed.weight
        neighbor_similarity(config, user_embeddings)
        return

    user_bought = json.load(open(config.processed_path + '{}_user_bought.json'.format(config.dataset), 'r'))

    doc2vec_model = Doc2Vec.load(config.processed_path + '{}_doc2model'.format(config.dataset))
    train_dataset = AmazonDataset(train_df, query_dict=query_dict,
                                  user_bought=user_bought,
                                  asin_dict=asin_dict,
                                  doc2vec_model=doc2vec_model,
                                  neg_num=config.neg_sample_num, is_training=True)
    test_dataset = AmazonDataset(test_df, query_dict=query_dict,
                                 user_bought=user_bought,
                                 asin_dict=asin_dict,
                                 doc2vec_model=doc2vec_model,
                                 neg_num=config.neg_sample_num, is_training=False)
    train_dataset.sample_neg()
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_dataset.sample_neg()
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0001)

    # Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 10)
    # display(-1, config.epochs, 0, Hr, Mrr, Ndcg, time.time())
    loss = 0
    # ------------------------------------Train------------------------------------
    for epoch in range(config.epochs):
        model.is_training = True
        model.train()
        start_time = time.time()

        for i, batch_data in enumerate(train_loader):
            user = batch_data['userID'].cuda()
            query = batch_data['query'].cuda()
            # pos_vis = batch_data['pos_vis'].cuda()
            pos_text = batch_data['pos_text'].cuda()
            # neg_vis = batch_data['neg_vis'].cuda()
            neg_text = batch_data['neg_text'].cuda()

            optimizer.zero_grad()
            item_predict, pos_item, neg_items = model(user, query, pos_text, neg_text, False)
            loss = triplet_loss(item_predict, pos_item, neg_items)
            loss.backward()
            optimizer.step()

        Mrr, Hr, Ndcg = evaluate(model, test_dataset, test_loader, 10)
        display(epoch, config.epochs, loss, Hr, Mrr, Ndcg, start_time)

    if not config.load:
        torch.save(model.state_dict(), config.save_path)


def triplet_loss(anchor, positive, negatives) -> torch.Tensor:
    """
    We found that add all the negative ones together can
    yeild relatively better performance.
    """
    batch_size, neg_num, embed_size = negatives.size()
    negatives = negatives.view(neg_num, batch_size, embed_size)

    losses = torch.tensor(0.).cuda()
    for negative in negatives:
        losses += torch.mean(
            func.triplet_margin_loss(anchor, positive, negative))
    return losses / len(negatives)
