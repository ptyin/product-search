import os
import json
import time
from argparse import ArgumentParser
import pandas as pd
from core.gz_to_df import get_df
from core.extraction import extraction, generate_queries
from core.split_data import split_data
from core.remove_review import remove_review


def reindex(df):
    """ Reindex the reviewID from 0 to total length. """
    reviewer = df['reviewerID'].unique()
    reviewer_map = {r: i for i, r in enumerate(reviewer)}

    userIDs = [reviewer_map[df['reviewerID'][i]] for i in range(len(df))]
    df['userID'] = userIDs
    return df


if __name__ == '__main__':
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument('--word_count', type=int, default=5, help="remove the words number less than count")
    parser.add_argument('--dataset', type=str,
                        # choices=('All_Beauty', 'Appliances', 'Magazine_Subscriptions', 'Software'),
                        choices=("Digital_Music", "Luxury_Beauty", "Musical_Instruments", "Software"),
                        default='Digital_Music')
    parser.add_argument('--data_path', type=str, default='/disk/yxk/data/cold_start/')
    parser.add_argument('--stop_file', type=str, default='./stopwords.txt')
    parser.add_argument('--processed_path', type=str, default='/disk/yxk/processed/cold_start/')

    parser.add_argument('--unprocessed_path', type=str, default='/disk/yxk/unprocessed/cold_start/')

    config = parser.parse_args()
    # --------------PREPARE PATHS--------------
    if not os.path.exists(config.processed_path):
        os.makedirs(config.processed_path)

    stop_path = config.stop_file
    meta_path = os.path.join(config.data_path, "meta_{}.json.gz".format(config.dataset))
    # review_path = os.path.join(config.data_path, "{}.json.gz".format(config.dataset))
    # review_df = get_df(review_path)
    # unprocessed_paths = \
    #     [os.path.join(config.unprocessed_path, config.dataset, "{}.csv".format(str(i))) for i in range(2, 11)]
    # review_dfs = [pd.read_csv(unprocessed_path) for unprocessed_path in unprocessed_paths]
    unprocessed_path = os.path.join(config.unprocessed_path, config.dataset, "origin.csv")
    review_df = pd.read_csv(unprocessed_path)
    stop_df = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_words = set(stop_df['stopword'].unique())

    # --------------PRE-EXTRACTION--------------
    categories, also_viewed = extraction(meta_path)
    df, word_dict = generate_queries(review_df, stop_words, config.word_count, categories, config.dataset.split('_'))
    df = df.drop(['reviewerName', 'reviewTime', 'verified', 'summary', 'overall', 'vote', 'image'], axis=1)
    reindex(df)
    df = split_data(df)
    # ---------------------------------Save Parameters---------------------------------
    processed_path = os.path.join(config.processed_path, config.dataset)
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    json.dump(word_dict, open(os.path.join(processed_path, '{}_word_dict.json'.format(config.dataset)), 'w'))
    df = remove_review(df, word_dict)  # remove the reviews from test set
    df.to_csv(os.path.join(processed_path, '{}_full.csv'.format(config.dataset)), index=False)

    print("The number of {} users is {:d}; items is {:d}; feedbacks is {:d}.".format(
        config.dataset, len(df.reviewerID.unique()), len(df.asin.unique()), len(df)),
        "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
