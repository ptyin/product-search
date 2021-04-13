import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np


class AmazonDataset(Dataset):
    def __init__(self, df, item_map: dict, word_num, asin_dict, window_size, mode, neg_sample_num=1):
        self.df = df
        self.item_map = item_map
        self.word_num = word_num
        self.asin_dict = asin_dict
        self.window_size = window_size
        self.mode = mode
        self.neg_sample_num = neg_sample_num

        self.corpus = set(range(word_num))
        # self.word_distribution = torch.zeros(word_num)
        self.word_distribution = np.zeros(word_num)
        self.data = []
        if mode == 'train':
            for _, series in self.df.iterrows():
                current_words = eval(series['reviewWords'])
                padded_words = [0] * (window_size // 2) + current_words + [0] * (window_size // 2)
                current_asin = series['asin']
                current_item = self.item_map[current_asin]
                current_neg_item = self.sample_neg_items(current_asin)

                current_query_words = eval(series['queryWords'])
                for i, word in enumerate(current_words):
                    self.word_distribution[word] += 1
                    word = torch.tensor(padded_words[i: i+self.window_size], dtype=torch.long)
                    self.data.append((current_item, current_neg_item, current_query_words, word))

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'train':
            item, neg_items, query, word = self.data[index]
            query_words = torch.tensor(query, dtype=torch.long)
            # neg_words = torch.tensor(self.sample_neg_words(word), dtype=torch.long)
            return item, word, neg_items, query_words
        else:
            series = self.df.loc[index]
            item = self.item_map[series['asin']]
            query = torch.tensor(eval(series['queryWords']), dtype=torch.long)
            return item, query

    def sample_neg_items(self, asin):
        """ Take the also_view or buy_after_viewing as negative samples. """
        # We tend to sample negative ones from the also_view and
        # buy_after_viewing items, if don't have enough, we then
        # randomly sample negative ones.

        # -----------sample item-----------
        sample = self.asin_dict[asin]
        all_sample = sample['positive'] + sample['negative']
        neg_asin = np.random.choice(all_sample, self.neg_sample_num, replace=False, p=sample['prob'])
        negs = torch.zeros(neg_asin.shape, dtype=torch.long)
        for i, neg in enumerate(neg_asin):
            if neg not in self.item_map:
                neg_asin[i] = np.random.choice(list(set(self.item_map.keys()) - {asin}), 1, replace=False)
            negs[i] = self.item_map[neg_asin[i]]

        return negs

    def neg_candidates(self, item: torch.LongTensor):
        a = list(range(0, item.item())) + list(range(item.item() + 1, len(self.item_map)))
        candidates = torch.tensor(np.random.choice(a, 99, replace=False), dtype=torch.long).cuda()
        return candidates

    @staticmethod
    def collate_fn(batch):
        entities = []
        query_words = []
        for sample in batch:
            entities.append(sample[:-1])
            query_words.append(sample[-1])
        entity_result = default_collate(entities)  # shape:(*, batch)
        entity_result = list(map(lambda entity: entity.cuda(), entity_result))
        query_result = pad_sequence(query_words, batch_first=True, padding_value=0).cuda()  # shape: (batch, seq)
        return (*entity_result, query_result)

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(0, len(items))))
        return users, item_map
