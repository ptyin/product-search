from argparse import ArgumentParser
from argparse import Namespace
import os
import json
import pandas as pd


def parser_add_data_arguments(parser: ArgumentParser):
    # ------------------------------------Dataset Parameters------------------------------------
    parser.add_argument('--dataset',
                        default='Musical_Instruments',
                        choices=("Automotive",
                                 "Cell_Phones_and_Accessories",
                                 "Clothing_Shoes_and_Jewelry",
                                 "Musical_Instruments",
                                 "Office_Products",
                                 "Toys_and_Games"),
                        help='name of the dataset')
    parser.add_argument('--processed_path',
                        default='/disk/yxk/processed/cold_start/ordinary/Musical_Instruments/',
                        help="preprocessed path of the raw data")


def data_preparation(config: Namespace):
    train_path = os.path.join(config.processed_path, "{}_train.csv".format(config.dataset))
    test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))
    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))

    query_path = os.path.join(config.processed_path, '{}_query.json'.format(config.dataset))
    asin_sample_path = config.processed_path + '{}_asin_sample.json'.format(config.dataset)
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))

    query_dict = json.load(open(query_path, 'r'))
    asin_dict = json.load(open(asin_sample_path, 'r'))
    word_dict = json.load(open(word_dict_path, 'r'))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(full_path)

    return train_df, test_df, full_df, query_dict, asin_dict, word_dict
