import torch
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def collate_fn(batch):
    # (user,
    #  self.user_reviews_words[user],
    #
    #  support_items,
    #  support_item_reviews_words, support_queries,
    #  support_negative_items,
    #  support_negative_reviews_words,
    #
    #  query_items,
    #  query_item_reviews_words, query_queries,
    #  query_negative_items,
    #  query_negative_reviews_words, query_item_asin)
    data_list = \
        (user_list, user_reviews_words_list,
         support_items_list, support_item_reviews_words_list, support_queries_list,
         support_negative_items_list, support_negative_reviews_words_list,
         query_items_list, query_item_reviews_words_list, query_queries_list,
         query_negative_items_list, query_negative_reviews_words_list, query_item_asin_list) = \
        tuple([] for _ in range(13))

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
    
    def process_half(items_list, item_reviews_words_list, queries_list,
                     negative_items_list, negative_reviews_words_list):
        # pad item_reviews_words
        order, item_reviews_words_list = sort_items(item_reviews_words_list)
        item_reviews_words_list = stack_variant_list(item_reviews_words_list, 'review')
        items_list = stack_variant_list(sort_by_order(items_list, order), 'id')

        # sort user_review_words according to length of support items
        processed_user_reviews_words_list = torch.tensor(sort_by_order(user_reviews_words_list.tolist(), order)).cuda()
        processed_user_list = torch.tensor(sort_by_order(user_list, order)).cuda()
        # pad queries
        queries_list = sort_by_order(queries_list, order)
        queries_list = stack_variant_list(queries_list, 'query')
        # pad negative_reviews_words_list
        negative_reviews_words_list = sort_by_order(negative_reviews_words_list, order)
        negative_reviews_words_list = stack_variant_list(negative_reviews_words_list, 'review')
        negative_items_list = stack_variant_list(sort_by_order(negative_items_list, order), 'id')
        return (processed_user_list, processed_user_reviews_words_list,
                items_list, item_reviews_words_list, queries_list,
                negative_items_list, negative_reviews_words_list)

    return (*process_half(support_items_list, support_item_reviews_words_list, support_queries_list,
                          support_negative_items_list, support_negative_reviews_words_list),
            *process_half(query_items_list, query_item_reviews_words_list, query_queries_list,
                          query_negative_items_list, query_negative_reviews_words_list),
            query_item_asin_list)


def sort_items(items):
    items = zip(range(len(items)), items)
    items = sorted(items, key=lambda x: len(x[1]), reverse=True)
    order, sorted_items = np.zeros(len(items), dtype=int), []
    for i, item in enumerate(items):
        order[item[0]] = i
        sorted_items.append(item[1])
    return order, sorted_items


def sort_by_order(target_list: list, order: np.ndarray):
    temp = zip(target_list, order)
    sorted_list = sorted(temp, key=lambda x: x[1])
    sorted_list = list(map(lambda x: x[0], sorted_list))
    return sorted_list


def stack_reviews(review_list: list) -> torch.Tensor:
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
    # variant_list = sorted(variant_list, key=lambda items: len(items), reverse=True)
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
        elif mode == 'id':
            stacked[len(stacked) - 1] = torch.tensor(stacked[len(stacked) - 1], dtype=torch.long).cuda()
        else:
            raise NotImplementedError
    variant_list = stacked
    return variant_list


class AmazonDataset(Dataset):
    def __init__(self,
                 support_df: DataFrame, query_df: DataFrame,
                 item_map,
                 item_reviews_words: dict,
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
        self.support_df = self.support_df.set_index(['userID'])
        self.query_df = self.query_df.set_index(['userID'])
        self.item_map = item_map
        self.asin_dict = asin_dict
        self.pre_train = pre_train

        self.users = self.support_df.index.unique()

        self.user_reviews_words = {}  # record the support user-item reviews
        self.item_reviews_words = item_reviews_words  # same as above

        # self.user_reviews_lengths = {}  # record the corresponding lengths of users' reviews
        # self.item_reviews_lengths = {}  # record the corresponding lengths of items' reviews

        # self.distribution = np.zeros(word_num)
        # for _, series in self.train_df.iterrows():
        #     review = eval(series['reviewWords'])
        #     for word in review:
        #         self.distribution[word] += 1
        # self.distribution = self.distribution / self.distribution.sum()
        self.cluster_user_reviews()

    def cluster_user_reviews(self):
        users = self.support_df.groupby(by="userID")  # for user, cluster the reviews from support data
        review_nums: np.ndarray = users.apply(AmazonDataset.compute_max_review_num).to_numpy()
        mode_review_num = np.argmax(np.bincount(review_nums))
        max_review_num = mode_review_num
        # max_review_num = 1
        for user in users:
            mask = user[1]['reviewWords'].map(lambda review: len(review) > 0)
            user_reviews_words = user[1]['reviewWords'][mask].to_numpy(dtype=object)
            # user_reviews_words = user[1]['reviewWords']\
            #     .map(lambda review: torch.tensor(eval(review), dtype=torch.long)).to_numpy(dtype=object)
            # -----------------Clip Reviews-----------------
            user_reviews_words = np.random.choice(user_reviews_words, max_review_num, replace=False)\
                if len(user_reviews_words) > max_review_num else user_reviews_words
            # # -----------------Clip Words-----------------
            # user_reviews_words = [torch.tensor(np.random.choice(review, 50, replace=False), dtype=torch.long)
            #                       if len(review) > 50 else torch.tensor(review, dtype=torch.long)
            #                       for review in user_reviews_words]
            # ----------------------------------------------
            self.user_reviews_words[user[0]] = pad_sequence(user_reviews_words, batch_first=True, padding_value=0).cuda()

    def __len__(self):
        return len(self.users)

    def __get_instance(self, asin, items, item_reviews_words, negative_items, negative_reviews_words):
        items.append(self.item_map[asin])
        item_reviews_words.append(self.item_reviews_words[asin])
        negative_asin = self.sample_neg(asin)
        negative_items.append(self.item_map[asin])
        negative_reviews_words.append(self.item_reviews_words[negative_asin])

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

        (support_items,
         support_item_reviews_words,
         support_negative_items,
         support_negative_reviews_words) = [], [], [], []

        (query_items,
         query_item_reviews_words,
         query_negative_items,
         query_negative_reviews_words) = [], [], [], []
        
        def process_half(df, items, item_reviews_words,
                         negative_items, negative_reviews_words):
            pd.Series(df.loc[user, 'asin'], dtype=str).apply(
                self.__get_instance, args=(items,
                                           item_reviews_words,
                                           negative_items,
                                           negative_reviews_words))
            # items = torch.tensor(items, dtype=torch.long)
            # negative_items = torch.tensor(negative_items, dtype=torch.long)

            queries = pd.Series(df.loc[user, 'queryWords'], dtype=str).map(
                lambda query: torch.tensor(eval(query), dtype=torch.long).cuda()).tolist()
            return items, item_reviews_words, negative_items, negative_reviews_words, queries

        (support_items, support_item_reviews_words,
         support_negative_items, support_negative_reviews_words,
         support_queries) = process_half(self.support_df, support_items, support_item_reviews_words,
                                         support_negative_items, support_negative_reviews_words)

        (query_items, query_item_reviews_words,
         query_negative_items, query_negative_reviews_words,
         query_queries) = process_half(self.query_df, query_items, query_item_reviews_words,
                                       query_negative_items, query_negative_reviews_words)

        query_item_asin = self.query_df.loc[user, 'asin']  # useful iff TestQuery

        return (user,
                self.user_reviews_words[user],

                support_items,
                support_item_reviews_words, support_queries,
                support_negative_items,
                support_negative_reviews_words,

                query_items,
                query_item_reviews_words, query_queries,
                query_negative_items,
                query_negative_reviews_words, query_item_asin)

    def sample_neg(self, item):

        a = list(self.item_reviews_words.keys() - {item, })
        negs = np.random.choice(a, 1, replace=False)
        return negs[0]

        # sample = self.asin_dict[item]
        # all_sample = sample['positive'] + sample['negative']
        # neg = np.random.choice(all_sample, 5, replace=False, p=sample['prob'])
        # if neg[0] not in self.item_reviews_words:
        #     neg = np.random.choice(list(self.item_reviews_words.keys()), 1, replace=False)
        # return neg[0]

    # def neg_candidates(self, item):
    #     """random select 99 candidates to participate test evaluation"""
    #     a = list(self.item_reviews_words.keys() - {item, })
    #     candidates = np.random.choice(a, 99, replace=False)
    #     candidates_reviews_words = list(map(lambda candidate: self.item_reviews_words[candidate], candidates))
    #     candidates_reviews_words = stack_reviews(candidates_reviews_words)
    #     return candidates_reviews_words

    def get_all_items(self):
        """random select 99 candidates to participate test evaluation"""
        all_asin = {}
        all_review_words = []
        for pair in self.item_reviews_words.items():
            all_asin[pair[0]] = len(all_asin)
            all_review_words.append(pair[1])
        all_review_words = stack_reviews(all_review_words)
        return all_asin, all_review_words

    @staticmethod
    def clip_words(full_df: DataFrame):
        def clip(review):
            review = eval(review)
            max_word_num = 15
            if len(review) > max_word_num:
                return torch.tensor(np.random.choice(review, max_word_num, replace=False), dtype=torch.long)
            else:
                return torch.tensor(review, dtype=torch.long)
        full_df['reviewWords'] = full_df['reviewWords'].map(clip)

    @staticmethod
    def compute_max_review_num(group: DataFrame):
        return group['reviewWords'].map(lambda review: len(review) > 0).to_numpy(dtype=np.bool).sum()

    @staticmethod
    def cluster_item_reviews(full_df: DataFrame):
        items = full_df.groupby(by="asin")  # for item, cluster all the reviews except for the test query data
        review_nums: np.ndarray = items.apply(AmazonDataset.compute_max_review_num).to_numpy()
        mode_review_num = np.argmax(np.bincount(review_nums))
        max_review_num = mode_review_num
        # max_review_num = 1

        items_reviews_words = {}
        for item in items:
            mask = item[1]['reviewWords'].map(lambda review: len(review) > 0)
            item_reviews_words = item[1]['reviewWords'][mask].to_numpy(dtype=object)
            # item_reviews_words = item[1]['reviewWords']\
            #     .map(lambda review: torch.tensor(eval(review), dtype=torch.long)).to_numpy(dtype=object)
            # -----------------Clip Reviews-----------------
            item_reviews_words = np.random.choice(item_reviews_words, max_review_num, replace=False)\
                if len(item_reviews_words) > max_review_num else item_reviews_words
            # ----------------------------------------------
            items_reviews_words[item[0]] = pad_sequence(item_reviews_words, batch_first=True, padding_value=0).cuda()
            # shape: (batch, seq_lens)

        return items_reviews_words

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        return users, item_map
