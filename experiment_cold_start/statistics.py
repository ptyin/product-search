import os
import json
import gzip
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_df(path):
    """ Apply raw data to pandas DataFrame. """
    idx = 0
    df = {}
    length = len(gzip.open(path, 'rb').readlines())
    g = gzip.open(path, 'rb')
    progress = tqdm(g, desc='transforming', total=length, leave=False, unit_scale=True)
    for line in progress:
        # if idx > 2000:
        #     break
        df[idx] = json.loads(line)
        idx += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_user_bought_num(df):
    user_bought_num = df.groupby('reviewerID').size()
    df['user_bought_num'] = df['reviewerID'].map(lambda user: user_bought_num[user])

    # user_bought = {}
    # for i in range(len(df)):
    #     user = df['reviewerID'][i]
    #     item = df['asin'][i]
    #     if user not in user_bought:
    #         user_bought[user] = []
    #     user_bought[user].append(item)
    return df


def filter_user_bought(df, num):
    if num is not None:
        # mask = df['reviewerID'].map(lambda reviewer: user_bought_num[reviewer] == num).tolist()
        # df = df[df['user_bought_num'] == num].reset_index(drop=True)
        df = df[df['user_bought_num'] // stride == num].reset_index(drop=True)
    return df


def compute_review_num(group: pd.DataFrame):
    return group['reviewText'].map(lambda x: len(eval(x)) > 0).to_numpy(dtype=np.bool).sum()


def compute_avg_review_num(groups):
    review_nums: np.ndarray = groups.apply(compute_review_num).to_numpy()
    avg_review_num = review_nums.mean()
    return avg_review_num


def summarize(df):
    user_num = len(df['reviewerID'].unique())
    item_num = len(df['asin'].unique())

    # avg_review_length = df['reviewText'].map(lambda element: len(element.split(None)) if type(element) == str else 0)\
    #     .to_numpy().mean()
    triplet_num = len(df)
    data = {'Dataset': [dataset],
            'Bought': [(bought_num * stride, (bought_num + 1) * stride)],
            'Number of users': [user_num],
            'Number of items': [item_num],
            'Number of <U,Q,I> Triplets': [triplet_num],
            # 'Average length of reviews': [avg_review_length]
            }
    return pd.DataFrame(data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--main_path', type=str, default='/disk/yxk/data/cold_start/')
    parser.add_argument('--save_dir', type=str, default='/disk/yxk/statistics/cold_start/')
    parser.add_argument('--unprocessed_dir', type=str, default='/disk/yxk/unprocessed/cold_start/')

    config = parser.parse_args()
    # datasets = ["All_Beauty",
    #             "Appliances",
    #             "Magazine_Subscriptions",
    #             "Software"]
    datasets = ["Prime_Pantry",
                "Luxury_Beauty",
                "Musical_Instruments",
                "Software"]
    # datasets = ["Software", "Magazine_Subscriptions"]
    bought_num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stride = 5
    # bought_num_list = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
    results = None
    for dataset in datasets:
        meta_path = os.path.join(config.main_path, "meta_{}.json.gz".format(dataset))
        review_path = os.path.join(config.main_path, "{}.json.gz".format(dataset))
        print('generating Data Frame: {}'.format(dataset))
        review = get_df(review_path)
        if not os.path.exists(os.path.join(config.unprocessed_dir, dataset)):
            os.makedirs(os.path.join(config.unprocessed_dir, dataset))
        review.to_csv(os.path.join(config.unprocessed_dir, dataset, 'origin.csv'), index=False)
        print('generating user bought dict: {}'.format(dataset))
        # user_bought_num = get_user_bought_num(review)
        review = get_user_bought_num(review)
        for bought_num in tqdm(bought_num_list, desc='summarizing', leave=False, unit_scale=True):
            print('filtering fixed user bought number {}'.format(bought_num))
            filtered_review = filter_user_bought(review, bought_num)
            if not os.path.exists(os.path.join(config.unprocessed_dir, dataset)):
                os.makedirs(os.path.join(config.unprocessed_dir, dataset))
            filtered_review.to_csv(os.path.join(config.unprocessed_dir, dataset,
                                                '{}_stride{}.csv'.format(bought_num, stride)), index=False)
            print("summarizing {}-{}-stride {}".format(dataset, bought_num, stride))
            result = summarize(filtered_review)
            if results is None:
                results = result
            else:
                results = results.append(result)
    print("generating final result")
    results: pd.DataFrame = results.set_index(['Dataset', 'Bought'], drop=True)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    results.to_excel(os.path.join(config.save_dir, 'data_summary.xlsx'))
