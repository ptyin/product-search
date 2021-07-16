import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from common import building_progress


class AmazonDataset(Dataset):
    def __init__(self, df, users, item_map: dict, query_max_length, user_bought_max_length,
                 word_num, history: dict, asin_dict, mode, debug, neg_sample_num=1):
        self.df = df
        self.users = users
        self.item_map = item_map
        self.query_max_length = query_max_length
        self.user_bought_max_length = user_bought_max_length

        self.word_num = word_num
        self.history = history
        self.asin_dict = asin_dict
        self.mode = mode
        self.neg_sample_num = neg_sample_num

        self.all_items = torch.tensor(range(1, len(self.item_map) + 1)).cuda()
        self.item_distribution = torch.ones(len(self.all_items), dtype=torch.bool).cuda()
        self.corpus = list(range(word_num))
        # self.word_distribution = torch.zeros(word_num)
        self.word_distribution = np.zeros(word_num)
        self.data = []

        progress = building_progress(df, debug)
        if mode == 'train':
            for _, series in progress:
                current_user = series['userID']
                current_asin = series['asin']
                current_item = self.item_map[current_asin]
                # current_neg_items = self.sample_neg_items(current_asin)
                # current_user_bought = self.user_bought[current_user]
                #
                # current_user_bought_mask = torch.ones(self.user_bought_max_length, dtype=torch.bool)
                # current_user_bought_mask[:len(current_user_bought)] = \
                #     torch.zeros(len(current_user_bought), dtype=torch.bool)
                #
                # current_user_bought_items = torch.zeros(self.user_bought_max_length, dtype=torch.long)
                # current_user_bought_items[:len(current_user_bought)] = \
                #     torch.tensor(current_user_bought, dtype=torch.long)
                current_user_bought_items, current_user_bought_mask = \
                    (x for x in self.history[current_user, current_item])

                current_item = torch.tensor(current_item, dtype=torch.long)
                query = eval(series['queryWords'])
                current_query_words = torch.zeros(self.query_max_length, dtype=torch.long)
                current_query_words[:len(query)] = torch.tensor(query, dtype=torch.long)

                current_words = torch.tensor(eval(series['reviewWords']), dtype=torch.long)

                for word in current_words:
                    self.word_distribution[word] += 1
                    self.data.append((current_user_bought_items, current_user_bought_mask,
                                      current_item, current_query_words, word))
        elif mode == 'test':
            for _, series in progress:
                current_user = series['userID']
                current_asin = series['asin']
                current_item = self.item_map[current_asin]
                # current_user_bought_items = torch.tensor(self.user_bought[current_user], dtype=torch.long)
                # current_user_bought_items = self.user_bought[current_user, current_item][0]
                current_user_bought_items, current_user_bought_mask = \
                    (x for x in self.history[current_user, current_item])
                current_item = torch.tensor(current_item, dtype=torch.long)
                query = eval(series['queryWords'])
                current_query_words = torch.zeros(self.query_max_length, dtype=torch.long)
                current_query_words[:len(query)] = torch.tensor(query, dtype=torch.long)

                self.data.append((current_user_bought_items.cuda(), current_user_bought_mask.cuda(),
                                  current_item, current_query_words.cuda()))

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'train':
            # user_bought_items, item, neg_items, query, word = self.data[index]
            user_bought_items, user_bought_mask, item, query, word = self.data[index]
            # query_words = torch.tensor(query, dtype=torch.long)
            # query_words = torch.zeros(self.query_max_length, dtype=torch.long)
            # query_words[:len(query)] = torch.tensor(query, dtype=torch.long)
            # neg_words = torch.tensor(self.sample_neg_words(word), dtype=torch.long)
            # return user_bought_items, item, word, neg_items, query_words
            return user_bought_items, user_bought_mask, item, word, query
        else:
            # series = self.df.loc[index]
            # user = series['userID']
            # user_bought_items = torch.tensor(self.user_bought[user], dtype=torch.long)
            # item = self.item_map[series['asin']]
            # # query = torch.tensor(eval(series['queryWords']), dtype=torch.long)
            # query = eval(series['queryWords'])
            # query_words = torch.zeros(self.query_max_length, dtype=torch.long)
            # query_words[:len(query)] = torch.tensor(query, dtype=torch.long)
            user_bought_items, user_bought_mask, item, query = self.data[index]

            return user_bought_items, user_bought_mask, item, query

    def sample_neg_items(self, items):

        # -----------sample item-----------

        self.item_distribution[items - 1] = False
        masked_all_items = self.all_items.masked_select(self.item_distribution)
        negs = np.random.randint(0, len(masked_all_items), self.neg_sample_num, dtype=np.long)
        self.item_distribution[items - 1] = True

        return masked_all_items[negs]

    def sample_neg_words(self, words: torch.LongTensor):
        """
        :param words: (batch, )
        :return: (batch, k)
        """
        words = words.cpu()
        # a = list(self.corpus - set(words))

        temp = self.word_distribution[words]
        self.word_distribution[words] = 0

        distribution = self.word_distribution / self.word_distribution.sum(axis=0)
        distribution = distribution ** 0.75  # distortion
        distribution = distribution / distribution.sum(axis=0)
        # negs = np.random.choice(self.corpus, len(words) * self.neg_sample_num, replace=True, p=distribution)
        # negs = torch.tensor(negs.reshape(len(words), self.neg_sample_num), dtype=torch.long).cuda()
        negs = np.random.choice(self.corpus, self.neg_sample_num, replace=True, p=distribution)

        self.word_distribution[words] = temp
        return torch.tensor(negs, dtype=torch.long).cuda()

    def neg_candidates(self, item: torch.LongTensor):
        a = list(range(1, item.item())) + list(range(item.item() + 1, len(self.item_map) + 1))
        candidates = torch.tensor(np.random.choice(a, 99, replace=False), dtype=torch.long).cuda()
        return candidates

    @staticmethod
    def collate_fn(batch):
        # user_bought = []
        # user_bought_mask = []
        # entities = []
        # # query_words = []
        # for sample in batch:
        #     user_bought.append(torch.tensor(sample[0], dtype=torch.long))
        #     user_bought_mask.append(torch.zeros(len(sample[0]), dtype=torch.bool))
        #     # entities.append(sample[1:-1])
        #     entities.append(sample[1:])
        #     # entities.append([len(sample[0])] + sample[1:])  # user bought length
        #     # query_words.append(sample[-1])
        # user_bought_result = pad_sequence(user_bought, batch_first=True, padding_value=0)
        # user_bought_mask_result = pad_sequence(user_bought_mask, batch_first=True, padding_value=True)
        # entity_result = default_collate(entities)  # shape:(*, batch)
        # # entity_result = list(map(lambda entity: entity.cuda(), entity_result))
        # # query_result = pad_sequence(query_words, batch_first=True, padding_value=0).cuda()  # shape: (batch, seq)
        # return (user_bought_result, user_bought_mask_result, *entity_result)
        entities = []
        for sample in batch:
            entities.append(sample)
        entity_result = default_collate(entities)  # shape:(*, batch)
        return entity_result

    @staticmethod
    def init(full_df: pd.DataFrame, user_bought_max_length):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(1, len(items) + 1)))
        user_bought = {}
        history = {}
        for _, series in full_df.iterrows():
            # if series['filter'] == 'Train':
            if series['userID'] not in user_bought:
                user_bought[series['userID']] = [item_map[series['asin']]]
            # elif len(user_bought[series['userID']]) < threshold:
            else:
                user_bought[series['userID']].append(item_map[series['asin']])
        user_bought_max_length = min(user_bought_max_length, max(map(lambda x: len(x), user_bought.values())))

        for user in user_bought:
            for index, item in enumerate(user_bought[user]):
                # user_bought_items = torch.zeros(user_bought_max_length, dtype=torch.long)
                # user_bought_items[:len(user_bought[user])] = torch.tensor(user_bought[user], dtype=torch.long)
                #
                # user_bought_mask = torch.ones(user_bought_max_length, 1, dtype=torch.bool)
                # user_bought_mask[:len(user_bought[user])] = torch.zeros(len(user_bought[user]), 1, dtype=torch.bool)
                #
                # user_bought[user] = (user_bought_items, user_bought_mask)
                user_bought_items = torch.zeros(user_bought_max_length, dtype=torch.long)
                user_bought_items[:user_bought_max_length if index >= user_bought_max_length else index] = \
                    torch.tensor(user_bought[user][max(index-user_bought_max_length, 0):index],
                                 dtype=torch.long)

                user_bought_mask = torch.zeros(user_bought_max_length, 1, dtype=torch.bool)
                user_bought_mask[:user_bought_max_length if index >= user_bought_max_length else index] = True

                history[user, item] = (user_bought_items, user_bought_mask)
        # for user in user_bought:
        #     if len(user_bought[user]) > user_bought_max_length:
        #         user_bought_max_length = user
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))

        return users, item_map, history, query_max_length, user_bought_max_length
