import os
import json
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from src.common.data_preparation import parser_add_data_arguments


def neg_candidates(reserved_idx: list, all_items: list, ones: np.ndarray):
    temp = ones[reserved_idx]
    ones[reserved_idx] = 0
    prob = ones / ones.sum()
    candidates = np.random.choice(all_items, 99, replace=False, p=prob).tolist()
    ones[reserved_idx] = temp
    return candidates


def __append(frame_like, to_append: dict) -> dict:
    if frame_like is None:
        frame_like = to_append
    else:
        for key in frame_like:
            frame_like[key] += to_append[key]
    return frame_like


# class __CategoryTree:
#     class __Node:
#         def __init__(self, value, father):
#             self.value = value
#             self.father = father
#
#     def __init__(self, ):
#         self.root = None
#
#     def add_note(self,):
#
#
#
# def construct_category_tree():
#     queries = full_df['query_'].unique()
#     for query in queries:


def run():
    user_bought_dict = json.load(open(config.processed_path + '{}_user_bought.json'.format(config.dataset), 'r'))

    items = full_df['asin'].unique()
    item_map = dict(zip(items, range(len(items))))
    # extended_test_df = pd.DataFrame(columns=['reviewerID', 'userID', 'query_', 'queryWords', 'asin', 'reviewText', 'reviewWords', 'gender', 'filter'])
    extended_test_df = None
    ones = np.ones(len(items))

    for i, series in test_df.iterrows():
        user_bought_items = user_bought_dict[series['reviewerID']]
        reserved_idx = list(map(lambda asin: item_map[asin], [series['asin'], *user_bought_items]))
        candidate_items = neg_candidates(reserved_idx, items, ones)

        inserted_rows = {'userID': [series['userID']] * 100,
                         'query_': [series['query_']] * 100,
                         'queryWords': [series['queryWords']] * 100,
                         'asin': [series['asin']] + candidate_items,
                         'filter': [series['filter']] * 100}
        extended_test_df = __append(extended_test_df, inserted_rows)

    extended_test_df = pd.DataFrame(extended_test_df)
    extended_test_df.to_csv(os.path.join(config.processed_path, '{}_test_new.csv'.format(config.dataset)), index=False)
    print('new test file for dataset {} generated!'.format(config.dataset))


if __name__ == '__main__':
    dataset_list = ["Automotive",
                    "Cell_Phones_and_Accessories",
                    "Clothing_Shoes_and_Jewelry",
                    "Musical_Instruments",
                    "Office_Products",
                    "Toys_and_Games"]

    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    for dataset in dataset_list:
        config = parser.parse_args(['--dataset', dataset])
        config.processed_path = os.path.join(config.processed_path, config.dataset + '/')

        full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))
        test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))

        full_df = pd.read_csv(full_path)
        test_df = pd.read_csv(test_path)
        run()
