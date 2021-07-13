import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


def init(full_df: pd.DataFrame):
    items = full_df['asin'].unique()
    item_map = dict(zip(items, range(len(items))))
    return item_map


def prior_knowledge(train_df: pd.DataFrame, word_num, item_map):
    users = train_df.groupby('userID')
    items = train_df.groupby('asin')
    tf = torch.zeros((word_num, len(item_map)))
    u_words = {}
    word_distribution = torch.zeros(word_num)

    def concat_words(asin):
        # assert asin not in tf
        # tf[asin] = {}
        # tf[asin] = np.zeros(word_num)
        item = item_map[asin[0]]
        for review in asin[1]['reviewWords']:
            for word in eval(review):
                word_distribution[word] += 1
                tf[word][item] += 1

    def most_frequent_words(user):
        assert user[0] not in u_words
        u_words[user[0]] = set()
        word_count = {}
        for review in user[1]['reviewWords']:
            for word in eval(review):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
                    if word_count[word] > 20 and word not in u_words[user[0]]:
                        u_words[user[0]].add(word)
        u_words[user[0]] = list(u_words[user[0]])

    for u in users:
        most_frequent_words(u)
    for i in items:
        concat_words(i)
    # users.apply(most_frequent_words)
    # items.apply(concat_words)
    word_distribution = word_distribution / word_distribution.sum()
    return tf, word_distribution, u_words


class AmazonDataset(Dataset):
    def __init__(self, test_df: pd.DataFrame, item_map):
        self.test_df = test_df
        self.item_map = item_map

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, index):
        return (self.test_df['userID'][index],
                self.item_map[self.test_df['asin'][index]],
                eval(self.test_df['queryWords'][index]))
