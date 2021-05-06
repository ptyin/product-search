import os
import json
import pandas as pd
from argparse import ArgumentParser
from gensim.models import Word2Vec
import numpy as np
import torch
from src.common.data_preparation import parser_add_data_arguments


def test(df: pd.DataFrame):
    word_list = df['reviewWords'].tolist()


def form_corpus(df: pd.DataFrame):
    reviews = df['reviewText'].tolist()
    queries = df['query_'].unique().tolist()
    corpus = reviews + queries

    processed = []
    for sentence in corpus:
        if len(eval(sentence)) != 0:
            processed.append(eval(sentence))
    return processed


if __name__ == '__main__':
    embedding_size = 64
    epochs = 40
    passes = epochs

    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    config = parser.parse_args()
    config.processed_path = os.path.join(config.processed_path, config.dataset + '/')

    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))
    full_df = pd.read_csv(full_path)

    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))
    word_dict = json.load(open(word_dict_path, 'r'))

    sentences = form_corpus(full_df)
    model = Word2Vec(sentences, min_count=1, workers=4, iter=epochs, size=embedding_size)
    # model.build_vocab(sentences)
    for epoch in range(passes):
        model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
        print('epochs:', epoch)

    # word_matrix = torch.zeros(len(word_dict) + 1, embedding_size)
    word_matrix = np.zeros((len(word_dict) + 1, embedding_size))

    for word in word_dict:
        word_matrix[word_dict[word]] = model.wv[word]
    # for word in model.wv.index2word:
    #     word_matrix[word_dict[word]] = torch.tensor(model.wv[word])

    # word_matrix = torch.tensor(word_matrix)
    word_matrix_path = os.path.join(config.processed_path, '{}_word_matrix.pt'.format(config.dataset))
    torch.save(torch.tensor(word_matrix), word_matrix_path)
    print('word initialized')
