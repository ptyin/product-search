import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

from common import building_progress


class AmazonDataset(Dataset):
    def __init__(self, df, item_map: dict, query_max_length,
                 word_num, window_size, mode, debug, neg_sample_num=1):
        self.df = df
        self.item_map = item_map
        self.query_max_length = query_max_length
        self.word_num = word_num
        self.window_size = window_size
        self.mode = mode
        self.neg_sample_num = neg_sample_num

        self.all_items = torch.tensor(range(len(self.item_map))).cuda()
        self.item_distribution = torch.ones(len(self.all_items), dtype=torch.bool).cuda()
        self.data = []

        progress = building_progress(df, debug)
        if mode == 'train':
            for _, series in progress:
                current_words = eval(series['reviewWords'])
                padded_words = [0] * (window_size // 2) + current_words + [0] * (window_size // 2)
                current_asin = series['asin']
                current_item = self.item_map[current_asin]

                query = eval(series['queryWords'])
                current_query_words = torch.zeros(self.query_max_length, dtype=torch.long)
                current_query_words[:len(query)] = torch.tensor(query, dtype=torch.long)

                for i, word in enumerate(current_words):
                    word = torch.tensor(padded_words[i: i+self.window_size], dtype=torch.long)
                    self.data.append((current_item, current_query_words, word))
        elif mode == 'test':
            for _, series in progress:
                current_asin = series['asin']
                current_item = self.item_map[current_asin]
                query = eval(series['queryWords'])
                current_query_words = torch.zeros(self.query_max_length, dtype=torch.long)
                current_query_words[:len(query)] = torch.tensor(query, dtype=torch.long)

                self.data.append((current_item, current_query_words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train':
            item, query, word = self.data[index]
            # neg_words = torch.tensor(self.sample_neg_words(word), dtype=torch.long)
            return item, word, query
        else:
            item, query = self.data[index]
            return item, query

    def sample_neg_items(self, items):
        self.item_distribution[items] = False
        masked_all_items = self.all_items.masked_select(self.item_distribution)
        negs = np.random.randint(0, len(masked_all_items), self.neg_sample_num, dtype=np.long)
        self.item_distribution[items] = True
        return masked_all_items[negs]

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(0, len(items))))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length
