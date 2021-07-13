import os
import sys
import json
from argparse import ArgumentParser

sys.path.append('..')
from QL.AmazonDataset import *
from common.data_preparation import parser_add_data_arguments


if __name__ == '__main__':
    parser = ArgumentParser()

    parser_add_data_arguments(parser)
    config = parser.parse_args()

    # ------------------------prepare for data------------------------
    config.processed_path = os.path.join(config.processed_path, config.dataset + '/')
    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))
    full_df = pd.read_csv(full_path)
    train_df = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)
    test_df = full_df[full_df['filter'] == 'Test'].reset_index(drop=True)
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))
    word_dict = json.load(open(word_dict_path, 'r'))
    # ----------------------------------------------------------------

    item_map = init(full_df)
    tf, word_distribution, u_words = prior_knowledge(train_df, len(word_dict) + 1, item_map)
    if not os.path.exists(os.path.join(config.processed_path, 'ql')):
        os.makedirs(os.path.join(config.processed_path, 'ql'))

    json.dump(item_map, open(os.path.join(config.processed_path, 'ql', 'item_map.json'), 'w'))
    torch.save(tf, os.path.join(config.processed_path, 'ql', 'tf.pt'))
    torch.save(word_distribution, os.path.join(config.processed_path, 'ql', 'word_distribution.pt'))
    json.dump(u_words, open(os.path.join(config.processed_path, 'ql', 'u_words.json'), 'w'))
