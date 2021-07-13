import os
import json
import pandas as pd
import numpy as np


def compute_review_num(group: pd.DataFrame):
    return group['reviewWords'].map(lambda review: len(eval(review)) > 0).to_numpy(dtype=np.bool).sum()


def compute_avg_review_num(groups):
    review_nums: np.ndarray = groups.apply(compute_review_num).to_numpy()
    avg_review_num = review_nums.mean()
    return avg_review_num


def summarize_one_dataset(df_path, word_dict_path, dataset):
    df = pd.read_csv(df_path)
    word_dict = json.load(open(word_dict_path, 'r'))
    word_num = len(word_dict)

    user_num = len(df['userID'].unique())
    item_num = len(df['asin'].unique())
    query_num = len(df['query_'].unique())
    users = df.groupby(by="userID")
    items = df.groupby(by='asin')
    avg_review_num_per_user = compute_avg_review_num(users)
    avg_review_num_per_item = compute_avg_review_num(items)
    assert round(avg_review_num_per_user * user_num) == round(avg_review_num_per_item * item_num)
    review_num = round(avg_review_num_per_user * user_num)
    mask = df['reviewWords'].map(lambda review: len(eval(review)) > 0)
    avg_review_length = df['reviewWords'][mask].map(lambda review: len(eval(review))).to_numpy().mean()
    avg_query_length = df['queryWords'].map(lambda query: len(eval(query))).to_numpy().mean()
    triplet_num = len(df)
    data = {'Dataset': [dataset],
            'Number of users': [user_num],
            'Number of items': [item_num],
            'Number of queries': [query_num],
            'Number of <U,Q,I> Triplets': [triplet_num],
            'Number of reviews': [review_num],
            'Number of words': [word_num],
            'Average number of reviews per user': [avg_review_num_per_user],
            'Average number of reviews per item': [avg_review_num_per_item],
            'Average length of reviews': [avg_review_length],
            'Average length of queries': [avg_query_length]}
    result = pd.DataFrame(data)
    return result


def main():
    data_path = '/disk/yxk/processed/cf/ordinary/'
    save_path = '../result/'
    datasets = ["Automotive",
                "Cell_Phones_and_Accessories",
                "Clothing_Shoes_and_Jewelry",
                "Musical_Instruments",
                "Office_Products",
                "Toys_and_Games"]

    result_df = None
    for dataset in datasets:
        processed_path = os.path.join(data_path, dataset)
        df_path = os.path.join(processed_path, '{}_full.csv'.format(dataset))
        word_dict_path = os.path.join(processed_path, '{}_word_dict.json'.format(dataset))

        result = summarize_one_dataset(df_path, word_dict_path, dataset)
        if result_df is None:
            result_df = result
        else:
            result_df = result_df.append(result)
        print('{} summarized'.format(dataset))
    result_df = result_df.set_index('Dataset', drop=True)
    result_df.to_excel(os.path.join(save_path, 'data_summary.xlsx'))


if __name__ == '__main__':
    main()
