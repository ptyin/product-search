import os
import json
import pandas as pd
from argparse import ArgumentParser
from src.common.data_preparation import parser_add_data_arguments, data_preparation


def generate_word_dict(full_df: pd.DataFrame):
    reviews = full_df['reviewText'].tolist()
    queries = full_df['query_'].unique().tolist()

    corpus = reviews + queries
    word_dict = {}
    for sentence in corpus:
        for word in eval(sentence):
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict


def modify(df: pd.DataFrame):
    query_words = []
    review_words = []

    for index, series in df.iterrows():
        query = eval(series['query_'])
        review = eval(series['reviewText'])
        query_words.append(list(map(lambda word: word_dict[word], query)))
        review_words.append(list(map(lambda word: word_dict[word], review)))

    df['queryWords'] = query_words
    df['reviewWords'] = review_words


if __name__ == '__main__':
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    config = parser.parse_args()

    config.processed_path = os.path.join(config.processed_path, config.dataset + '/')
    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))

    full_df = pd.read_csv(full_path)

    word_dict = generate_word_dict(full_df)
    json.dump(word_dict, open(word_dict_path, 'w'))
    modify(full_df)
    full_df.to_csv(full_path)
