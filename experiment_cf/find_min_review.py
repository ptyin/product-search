import os
import json
import gzip
from argparse import ArgumentParser
import pandas as pd
import numpy as np


def get_reviews(group):
    review_dict = {}
    for member in group:
        review_dict[member[0]] = set(member[1]['reviewWords'].tolist())
    return review_dict


def find_max_words(user_review: dict, reverse_word_dict: dict):
    anchor = None
    max_words = set()
    for user, reviews in user_review.items():
        all_words = set()
        for review in reviews:
            for word in eval(review):
                all_words.add(word)
        if len(all_words) > len(max_words):
            max_words = all_words
            anchor = user

    word_map = {}
    for review in user_review[anchor]:
        for word in eval(review):
            if word not in word_map:
                word_map[word] = 1
            else:
                word_map[word] += 1
    top_words = sorted(word_map, key=lambda x: word_map[x], reverse=True)
    # for i, word in enumerate(top_words):
    #     if i < 20:
    #         print('word:', reverse_word_dict[word], 'count:', word_map[word])
    print('anchor:', anchor, '; total_words:', len(max_words))
    print('----------------------------------')
    # for review in user_review[anchor]:
    #     print(anchor, ' '.join(eval(review)), sep='\t')
    return anchor, max_words


def main():
    parser = ArgumentParser()
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
                        default='/disk/yxk/processed/cf/ordinary/',
                        help="preprocessed path of the raw data")
    parser.add_argument('--transformed_path',
                        type=str,
                        default='/disk/yxk/transformed/cf/',
                        help="transformed path for ESRT")
    # parser.add_argument('--iteration',
    #                     default=10,
    #                     help='number of iterations to generate neighbors')

    # (536, 'A2NOW4U7W3F7RI'), 109

    config = parser.parse_args()
    config.processed_path = os.path.join(config.processed_path, config.dataset + '/')
    config.transformed_path = os.path.join(config.transformed_path, config.dataset, 'seq_min_count5')

    full_path = os.path.join(config.processed_path, "{}_full.csv".format(config.dataset))
    full_df = pd.read_csv(full_path)
    train_df = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)

    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))
    word_dict = json.load(open(word_dict_path, 'r'))
    reversed_word_dict = {v: k for k, v in word_dict.items()}

    esrt_user_id = []
    with gzip.open(os.path.join(config.transformed_path, 'users.txt.gz'), 'rt') as fin:
        for line in fin:
            esrt_user_id.append(line.strip())
    esrt_r_u = dict(zip(esrt_user_id, range(len(esrt_user_id))))
    print(esrt_r_u['ADH0O8UVJOT10'])

    # users = train_df.groupby(by='reviewerID')
    # ADH0O8UVJOT10
    users = train_df.groupby(by='userID')
    items = train_df.groupby(by='asin')
    user_review = get_reviews(users)
    item_review = get_reviews(items)
    anchor, max_words = find_max_words(user_review, reversed_word_dict)
    max_words = np.array(list(max_words), dtype=np.longlong)
    np.save(os.path.join(config.processed_path, 'experiments', 'anchor'), anchor)
    np.save(os.path.join(config.processed_path, 'experiments', 'words'), max_words)

    items = train_df[train_df['userID'] == anchor]['asin']
    bought_item_word = {}
    for item in items:
        for review in item_review[item]:
            review = eval(review)
            flag = False
            for word in review:
                if word not in bought_item_word:
                    bought_item_word[word] = 1
                else:
                    bought_item_word[word] += 1
                # if word in 'smooths calibration Watts afternoon contacting'.split(' '):
                if reversed_word_dict[word].lower() == 'rock':
                    flag = True
            if flag:
                print('asin:', item, 'review:', ' '.join([reversed_word_dict[word] for word in review]))
                # for query in train_df[train_df['asin'] == item]['queryWords']:
                #     print('asin:', item, 'review:',
                #           ' '.join([reversed_word_dict[word] for word in eval(query)]))

    # top_words = sorted(bought_item_word, key=lambda x: bought_item_word[x], reverse=True)
    # for i, word in enumerate(top_words):
    #     if word not in max_words:
    #         if bought_item_word[word] < 10:
    #             break
    #         print('word:', reversed_word_dict[word], 'frequency:', bought_item_word[word])
            # print(' '.join(review))


if __name__ == '__main__':
    main()
