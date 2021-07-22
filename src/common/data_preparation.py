from argparse import ArgumentParser
from argparse import Namespace
import os
import json
import pandas as pd
import logging


def parser_add_data_arguments(parser: ArgumentParser):
    # ------------------------------------Dataset Parameters------------------------------------
    # parser.add_argument('--dataset',
    #                     default='Musical_Instruments',
    #                     choices=("Automotive",
    #                              "Cell_Phones_and_Accessories",
    #                              "Clothing_Shoes_and_Jewelry",
    #                              "Musical_Instruments",
    #                              "Office_Products",
    #                              "Toys_and_Games"),
    #                     help='name of the dataset')
    parser.add_argument('--dataset',
                        default='Luxury_Beauty',
                        # choices=('All_Beauty', 'Appliances', 'Magazine_Subscriptions', 'Software'),
                        choices=("Digital_Music", "Luxury_Beauty", "Musical_Instruments", "Software"),
                        help='name of the dataset')
    parser.add_argument('--processed_path',
                        default='/disk/yxk/processed/cold_start/',
                        help="preprocessed path of the raw data")
    parser.add_argument('--save_path',
                        default='/disk/yxk/saved/cold_start/',
                        help="preprocessed path of the raw data")
    parser.add_argument('--save_str',
                        default='temp',
                        help='unique string to identify the saved model')
    parser.add_argument('--worker_num',
                        default=0,
                        type=int,
                        help='number of workers for data loading')
    # parser.add_argument('--test_mode',
    #                     default='product_score',
    #                     choices=('similarity_compute', 'product_score'),
    #                     help='test mode')
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--debug',
                        default=False,
                        action='store_true',
                        # type=bool,
                        help="enable debug")
    parser.add_argument('--device',
                        default='0',
                        help="using device")
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help="training epochs")
    parser.add_argument('--load',
                        default=False,
                        action='store_true',
                        # type=bool,
                        help='whether load from disk or not ')


def data_preparation(config: Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    config.processed_path = os.path.join(config.processed_path, config.dataset + '/')
    config.save_path = os.path.join(config.save_path, str(config.embedding_size), config.dataset)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    config.save_path = os.path.join(config.save_path, '{}.pt'.format(config.save_str))

    # train_path = os.path.join(config.processed_path, "{}_train.csv".format(config.dataset))
    # test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))
    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))

    query_path = os.path.join(config.processed_path, '{}_query.json'.format(config.dataset))
    asin_sample_path = config.processed_path + '{}_asin_sample.json'.format(config.dataset)
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))
    candidates_path = os.path.join(config.processed_path, '{}_candidates.json'.format(config.dataset))

    # train_df = pd.read_csv(train_path)
    # test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(full_path)
    train_df = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)
    test_df = full_df[full_df['filter'] == 'Test'].reset_index(drop=True)

    query_dict = json.load(open(query_path, 'r')) if os.path.exists(query_path) else None
    asin_dict = json.load(open(asin_sample_path, 'r')) if os.path.exists(asin_sample_path) else None
    word_dict = json.load(open(word_dict_path, 'r')) if os.path.exists(word_dict_path) else None
    candidates = json.load(open(candidates_path, 'r')) if os.path.exists(candidates_path) else None

    # word_dict = generate_word_dict(full_df)

    return train_df, test_df, full_df, query_dict, asin_dict, word_dict, candidates
