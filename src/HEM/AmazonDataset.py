import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from common import building_progress


class AmazonDataset(Dataset):
    def __init__(self, df, users, item_map: dict, query_max_length,
                 word_num,
                 mode, debug, neg_sample_num=1, sub_sampling_rate=0.):
        self.df = df
        self.users = users
        self.item_map = item_map
        self.query_max_length = query_max_length
        self.word_num = word_num
        self.mode = mode
        self.neg_sample_num = neg_sample_num

        self.all_items = torch.tensor(range(len(self.users), len(self.users) + len(self.item_map))).cuda()
        self.item_distribution = torch.ones(len(self.all_items), dtype=torch.bool).cuda()
        self.corpus = torch.tensor(list(range(word_num)))
        # self.word_distribution = torch.zeros(word_num)
        self.word_distribution = np.zeros(word_num)
        for words in self.df['reviewWords']:
            for word in eval(words):
                self.word_distribution[word] += 1

        self.sub_sampling_rate = np.ones(word_num)
        # self.subsample(sub_sampling_rate)
        self.data = []

        progress = building_progress(df, debug)
        if mode == 'train':
            for _, series in progress:
                current_user = series['userID']
                current_words = eval(series['reviewWords'])
                current_asin = series['asin']
                current_item = self.item_map[current_asin]
                # current_neg_item = self.sample_neg_items(current_asin)

                query = eval(series['queryWords'])
                current_query_words = torch.zeros(self.query_max_length, dtype=torch.long)
                current_query_words[:len(query)] = torch.tensor(query, dtype=torch.long)

                for word in current_words:
                    if sub_sampling_rate == 0. or np.random.random() < self.sub_sampling_rate[word]:
                        # self.data.append((current_user, current_item, current_neg_item, current_query_words, word))
                        self.data.append((current_user, current_item, current_query_words, word))
        elif mode == 'test':
            for _, series in progress:
                current_user = series['userID']
                current_asin = series['asin']
                current_item = self.item_map[current_asin]
                query = eval(series['queryWords'])
                current_query_words = torch.zeros(self.query_max_length, dtype=torch.long)
                current_query_words[:len(query)] = torch.tensor(query, dtype=torch.long)

                self.data.append((current_user, current_item, current_query_words))

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'train':
            # user, item, neg_items, query, word = self.data[index]
            user, item, query, word = self.data[index]
            # query_words = torch.tensor(query, dtype=torch.long)
            # query_words = torch.zeros(self.query_max_length, dtype=torch.long)
            # query_words[:len(query)] = torch.tensor(query, dtype=torch.long)
            # neg_words = torch.tensor(self.sample_neg_words(word), dtype=torch.long)
            # return user, item, word, neg_items, query_words
            return user, item, word, query
        else:
            # series = self.df.loc[index]
            # user = series['userID']
            # item = self.item_map[series['asin']]
            # # query = torch.tensor(eval(series['queryWords']), dtype=torch.long)
            # query = eval(series['queryWords'])
            # query_words = torch.zeros(self.query_max_length, dtype=torch.long)
            # query_words[:len(query)] = torch.tensor(query, dtype=torch.long)
            user, item, query = self.data[index]

            return user, item, query

    def subsample(self, threshold):
        if threshold == 0.:
            return
        distribution = np.clip(self.word_distribution / self.word_distribution.sum(axis=0), a_min=1e-8, a_max=None)
        self.sub_sampling_rate = np.ones(self.word_num)
        threshold = sum(distribution) * threshold
        self.sub_sampling_rate = np.clip((np.sqrt(distribution / threshold) + 1.) * threshold / distribution,
                                         a_min=None, a_max=1.)

    def sample_neg_items(self, items):

        # -----------sample item-----------

        self.item_distribution[items - len(self.users)] = False
        masked_all_items = self.all_items.masked_select(self.item_distribution)

        # items = items.cpu()
        # distribution: np.ndarray = np.concatenate([np.zeros(len(self.users)), np.ones(len(self.item_map))], axis=0)
        # distribution[items] = 0
        # distribution = distribution / distribution.sum(axis=0)
        # negs = np.random.choice(range(len(self.users) + len(self.item_map)), self.neg_sample_num, replace=True,
        #                         p=distribution)

        negs = np.random.randint(0, len(masked_all_items), self.neg_sample_num, dtype=np.long)
        self.item_distribution[items - len(self.users)] = True

        return masked_all_items[negs]
        # return torch.tensor(negs, dtype=torch.long).cuda()

    def sample_neg_words(self, words: torch.LongTensor):
        """
        :param words: (batch, )
        :return: (batch, k)
        """
        words = words.cpu()
        # a = list(self.corpus - set(words))

        temp = self.word_distribution[words]
        self.word_distribution[words] = 0

        distribution = self.word_distribution / self.word_distribution.sum()
        distribution = distribution ** 0.75  # distortion
        distribution = distribution / distribution.sum()
        # negs = np.random.choice(self.corpus, len(words) * self.neg_sample_num, replace=True, p=distribution)
        # negs = torch.tensor(negs.reshape(len(words), self.neg_sample_num), dtype=torch.long).cuda()
        negs = np.random.choice(self.corpus, self.neg_sample_num, replace=True, p=distribution)

        self.word_distribution[words] = temp
        return torch.tensor(negs, dtype=torch.long).cuda()

    def neg_candidates(self, item: torch.LongTensor):
        a = list(range(len(self.users), item.item())) + list(range(item.item() + 1, len(self.users) + len(self.item_map)))
        candidates = torch.tensor(np.random.choice(a, 99, replace=False), dtype=torch.long).cuda()
        return candidates

    @staticmethod
    def collate_fn(batch):
        # entities = []
        # query_words = []
        # for sample in batch:
        #     entities.append(sample[:-1])
        #     query_words.append(sample[-1])
        # entity_result = default_collate(entities)  # shape:(*, batch)
        # entity_result = list(map(lambda entity: entity.cuda(), entity_result))
        # query_result = pad_sequence(query_words, batch_first=True, padding_value=0).cuda()  # shape: (batch, seq)

        entities = []
        for sample in batch:
            entities.append(sample)
        entity_result = default_collate(entities)  # shape:(*, batch)
        # entity_result = list(map(lambda entity: entity.cuda(), entity_result))
        return entity_result

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(users), len(users) + len(items))))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length
