import torch
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def collate_fn(batch):
    data_list = \
        (user_reviews_words_list,
         support_item_reviews_words_list, support_queries_list,
         support_negative_reviews_words_list,
         query_item_reviews_words_list, query_queries_list,
         query_negative_reviews_words_list, query_item_asin_list) = \
        [], [], [], [], [], [], [], []

    for data in batch:
        # (user_reviews_words,
        #  support_item_reviews_words, support_queries,
        #  support_negative_reviews_words,
        #  query_item_reviews_words, query_queries,
        #  query_negative_reviews_words, query_item_asin) = data
        for i in range(len(data_list)):
            data_list[i].append(data[i])

    # pad user_review_words
    user_reviews_words_list = stack_reviews(user_reviews_words_list)

    # pad support_item_reviews_words
    support_item_reviews_words_list = stack_variant_list(support_item_reviews_words_list, 'review')
    # pad support_queries
    support_queries_list = stack_variant_list(support_queries_list, 'query')
    # pad support_negative_reviews_words_list
    support_negative_reviews_words_list = stack_variant_list(support_negative_reviews_words_list, 'review')

    # pad query data
    query_item_reviews_words_list = stack_variant_list(query_item_reviews_words_list, 'review')
    query_queries_list = stack_variant_list(query_queries_list, 'query')
    query_negative_reviews_words_list = stack_variant_list(query_negative_reviews_words_list, 'review')

    # query_item_asin_list = torch.tensor(query_item_asin_list, dtype=torch.long)

    return (user_reviews_words_list,
            support_item_reviews_words_list, support_queries_list,
            support_negative_reviews_words_list,
            query_item_reviews_words_list, query_queries_list,
            query_negative_reviews_words_list, query_item_asin_list)


def stack_reviews(review_list: list):
    max_word_num = max(review_list, key=lambda r: r.shape[1]).shape[1]
    # # -----------------Clip Words-----------------
    # max_word_num = max_word_num if max_word_num < 15 else 15
    # # --------------------------------------------

    for i, reviews in enumerate(review_list):
        padded_reviews = torch.zeros(reviews.shape[0], max_word_num, dtype=torch.long).cuda()
        # if reviews.shape[1] < max_word_num:
        padded_reviews[:, :reviews.shape[1]] = reviews
        # else:
        #     padded_reviews = reviews[:, :max_word_num]
        review_list[i] = padded_reviews
    review_list = pad_sequence(review_list, batch_first=True, padding_value=0)
    return review_list


def stack_variant_list(variant_list: list, mode):
    variant_list = sorted(variant_list, key=lambda items: len(items), reverse=True)
    stacked = []
    for i in range(len(variant_list[0])):
        stacked.append([])
        for elements in variant_list:
            if i >= len(elements):
                break
            stacked[len(stacked) - 1].append(elements[i])
        if mode == 'review':
            stacked[len(stacked) - 1] = stack_reviews(stacked[len(stacked) - 1])
        elif mode == 'query':
            stacked[len(stacked) - 1] = pad_sequence(stacked[len(stacked) - 1], batch_first=True, padding_value=0)
        else:
            raise NotImplementedError
    variant_list = stacked
    return variant_list


class AmazonDataset(Dataset):
    def __init__(self, support_df: DataFrame, query_df: DataFrame, train_df: DataFrame,
                 asin_dict: dict, pre_train=False):
        """
        Parameters
        ----------
        support_df: DataFrame
        query_df: DataFrame
        asin_dict: dict
        """
        self.support_df = support_df
        self.query_df = query_df
        self.train_df = train_df
        self.support_df = self.support_df.set_index(['userID'])
        self.query_df = self.query_df.set_index(['userID'])
        self.asin_dict = asin_dict
        self.pre_train = pre_train

        self.users = self.support_df.index.unique()

        self.user_reviews_words = {}  # record the support user-item reviews
        self.item_reviews_words = {}  # same as above

        # self.user_reviews_lengths = {}  # record the corresponding lengths of users' reviews
        # self.item_reviews_lengths = {}  # record the corresponding lengths of items' reviews

        # self.distribution = np.zeros(word_num)
        # for _, series in self.train_df.iterrows():
        #     review = eval(series['reviewWords'])
        #     for word in review:
        #         self.distribution[word] += 1
        # self.distribution = self.distribution / self.distribution.sum()
        self.cluster_reviews()

    def cluster_reviews(self):

        users = self.support_df.groupby(by="userID")  # for user, cluster the reviews from support data
        items = self.train_df.groupby(by="asin")  # for item, cluster all the reviews except for the test query data
        for user in users:
            # mask = user[1]['reviewText'].map(lambda review: True if len(eval(review)) > 0 else False)
            user_reviews_words = user[1]['reviewWords'].map(lambda review: eval(review)).to_numpy(dtype=object)
            # -----------------Clip Reviews-----------------
            user_reviews_words = np.random.choice(user_reviews_words, 20, replace=False)\
                if len(user_reviews_words) > 20 else user_reviews_words
            # -----------------Clip Words-----------------
            user_reviews_words = [torch.tensor(np.random.choice(review, 15, replace=False), dtype=torch.long)
                                  if len(review) > 15 else torch.tensor(review, dtype=torch.long)
                                  for review in user_reviews_words]
            # ----------------------------------------------
            self.user_reviews_words[user[0]] = pad_sequence(user_reviews_words, batch_first=True, padding_value=0).cuda()

        for item in items:
            # mask = item[1]['reviewText'].map(lambda review: True if len(eval(review)) > 0 else False)
            item_reviews_words = item[1]['reviewWords'].map(lambda review: eval(review)).to_numpy(dtype=object)
            # -----------------Clip Reviews-----------------
            item_reviews_words = np.random.choice(item_reviews_words, 20, replace=False)\
                if len(item_reviews_words) > 20 else item_reviews_words
            # -----------------Clip Words-----------------
            item_reviews_words = [torch.tensor(np.random.choice(review, 15, replace=False), dtype=torch.long)
                                  if len(review) > 15 else torch.tensor(review, dtype=torch.long)
                                  for review in item_reviews_words]
            # ----------------------------------------------
            self.item_reviews_words[item[0]] = pad_sequence(item_reviews_words, batch_first=True, padding_value=0).cuda()
            # shape: (batch, seq_lens)

    def __len__(self):
        return len(self.users)

    def __get_instance(self, asin,
                       item_reviews_words, negative_reviews_words):
        item_reviews_words.append(self.item_reviews_words[asin])
        negative_item = self.sample_neg(asin)
        negative_reviews_words.append(self.item_reviews_words[negative_item])

    def __getitem__(self, index):
        """
        Return
        ----------
        (user_reviews_words,
         support_item_reviews_words, support_queries,
         support_negative_reviews_words,
          query_item_reviews_words, query_queries,
          query_negative_reviews_words)
        """
        user = self.users[index]

        (support_item_reviews_words,
         support_negative_reviews_words) = [], []

        (query_item_reviews_words,
         query_negative_reviews_words) = [], []

        pd.Series(self.support_df.loc[user, 'asin'], dtype=str).apply(
            self.__get_instance, args=(support_item_reviews_words, support_negative_reviews_words))
        support_queries = pd.Series(self.support_df.loc[user, 'queryWords'], dtype=str).map(
            lambda query: torch.tensor(eval(query), dtype=torch.long).cuda()).tolist()

        query_item_asin = self.query_df.loc[user, 'asin']  # useful iff TestQuery
        pd.Series(self.query_df.loc[user, 'asin'], dtype=str).apply(
            self.__get_instance, args=(query_item_reviews_words,
                                       query_negative_reviews_words))
        query_queries = pd.Series(self.query_df.loc[user, 'queryWords'], dtype=str).map(
            lambda query: torch.tensor(eval(query), dtype=torch.long).cuda()).tolist()

        return (self.user_reviews_words[user],
                support_item_reviews_words, support_queries,
                support_negative_reviews_words,
                query_item_reviews_words, query_queries,
                query_negative_reviews_words, query_item_asin)

    def sample_neg(self, item):
        """ Take the also_view or buy_after_viewing as negative samples. """
        # We tend to sample negative ones from the also_view and
        # buy_after_viewing items, if don't have enough, we then
        # randomly sample negative ones.

        sample = self.asin_dict[item]
        all_sample = sample['positive'] + sample['negative']
        neg = np.random.choice(all_sample, 1, replace=False, p=sample['prob'])
        if neg[0] not in self.item_reviews_words:
            neg = np.random.choice(list(self.item_reviews_words.keys()), 1, replace=False)
        return neg[0]

    def neg_candidates(self, item):
        """random select 99 candidates to participate test evaluation"""
        a = list(self.item_reviews_words.keys() - {item, })
        candidates = np.random.choice(a, 99, replace=False)
        candidates_reviews_words = list(map(lambda candidate: self.item_reviews_words[candidate], candidates))
        candidates_reviews_words = stack_reviews(candidates_reviews_words)
        return candidates_reviews_words

    @staticmethod
    def init(full_df: DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        # item_map = dict(zip(map(lambda item: item[0], items), range(len(items))))
        return {'users': users, 'items': items}
