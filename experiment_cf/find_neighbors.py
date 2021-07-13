import os
import json
import gzip
import math
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import numpy as np


def get_user_bought(group):
    user_bought = {}
    for member in group:
        user_bought[(member[0], member[1]['reviewerID'].tolist()[0])] = set(member[1]['asin'].tolist())
    return user_bought


def find_anchor_user(sequence, user_bought, esrt_r_u, num=9, step=4):
    # max_item_num = 0
    min_item_num = 5
    # max_item_num = 15
    anchors = []
    anchors_esrt = []
    all_neighbors = []
    all_neighbors_esrt = []

    for user in sequence:
        item_num = len(user_bought[user])
        # if min_item_num <= item_num <= max_item_num:
        if min_item_num <= item_num:
            anchor = user
            top_k_neighbors_count, top_k_neighbors, top_k_neighbors_esrt =\
                find_neighbors(anchor, user_bought, esrt_r_u, num, step)

            if len(top_k_neighbors_count) == num // step + 1:
                # print('anchor:', anchor[1], '; number of purchased items:', len(top_k_neighbors_count))
                # print(top_k_neighbors_count)
                anchors.append(anchor[0])
                anchors_esrt.append(esrt_r_u[anchor[1]])
                all_neighbors.append(top_k_neighbors)
                all_neighbors_esrt.append(top_k_neighbors_esrt)

    anchors = np.array(anchors)
    anchors_esrt = np.array(anchors_esrt)
    all_neighbors = np.array(all_neighbors)
    all_neighbors_esrt = np.array(all_neighbors_esrt)

    return anchors, anchors_esrt, all_neighbors, all_neighbors_esrt


# def find_top_k_neighbors(anchor, user_bought, top_k=5):
#     top_k_neighbors = []
#     top_k_neighbors_esrt = []
#     top_k_neighbors_count = {}
#     for k in range(top_k):
#         co_bought = 0
#         for user in user_bought:
#             if user[1] not in top_k_neighbors_count and user != anchor and\
#                     co_bought < len(user_bought[anchor] & user_bought[user]):
#                 co_bought = len(user_bought[anchor] & user_bought[user])
#                 k_neighbor = user


def find_neighbors(anchor, user_bought, esrt_r_u, num, step):
    u_bought = user_bought.copy()
    top_k_neighbors = []
    top_k_neighbors_esrt = []
    top_k_neighbors_count = {}
    max_co_bought = 0

    # for user in u_bought:
    #     if user != anchor and max_co_bought < len(u_bought[anchor] & u_bought[user]):
    #         max_co_bought = len(u_bought[anchor] & u_bought[user])
    # step = math.ceil(max_co_bought / 5)

    for k in range(num, 0, -step):
        k_neighbor = None
        co_bought = 0
        for user in u_bought:
            if user[1] not in top_k_neighbors_count and user != anchor and\
                    len(u_bought[anchor] & u_bought[user]) == k:
                # co_bought <= len(u_bought[anchor] & u_bought[user]) <= k:
                co_bought = len(u_bought[anchor] & u_bought[user])
                k_neighbor = user

        if k_neighbor is not None:
            top_k_neighbors_count[k_neighbor[1]] = co_bought
            top_k_neighbors.append(k_neighbor[0])
            top_k_neighbors_esrt.append(esrt_r_u[k_neighbor[1]])

    return top_k_neighbors_count, top_k_neighbors, top_k_neighbors_esrt


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        default='Toys_and_Games',
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

    esrt_user_id = []
    with gzip.open(os.path.join(config.transformed_path, 'users.txt.gz'), 'rt') as fin:
        for line in fin:
            esrt_user_id.append(line.strip())
    esrt_r_u = dict(zip(esrt_user_id, range(len(esrt_user_id))))

    users = train_df.groupby(by='userID')
    user_bought = get_user_bought(users)
    sequence = list(user_bought.keys())

    # np.random.shuffle(sequence)
    # anchor, top_k_neighbors, top_k_neighbors_esrt = find_anchor_user(sequence, user_bought, esrt_r_u)
    anchors, anchors_esrt, all_neighbors, all_neighbors_esrt = find_anchor_user(sequence, user_bought, esrt_r_u)
    # anchor = (536, 'A2NOW4U7W3F7RI')
    # top_k_neighbors, top_k_neighbors_esrt = find_neighbors(anchor, user_bought, esrt_r_u)

    if not os.path.exists(os.path.join(config.processed_path, 'experiments')):
        os.makedirs(os.path.join(config.processed_path, 'experiments'))
    np.save(os.path.join(config.processed_path, 'experiments', 'anchors_step4'), anchors)
    np.save(os.path.join(config.processed_path, 'experiments', 'anchors_esrt_step4'), anchors_esrt)
    np.save(os.path.join(config.processed_path, 'experiments', 'all_neighbors_step4'), all_neighbors)
    np.save(os.path.join(config.processed_path, 'experiments', 'all_neighbors_esrt_step4'), all_neighbors_esrt)

    # json.dump({'anchor': anchor[0], 'anchor_esrt': esrt_r_u[anchor[1]],
    #            'neighbors': top_k_neighbors, 'neighbors_esrt': top_k_neighbors_esrt},
    #           open(os.path.join(config.processed_path, 'experiments', 'all_neighbors.json'), 'w'))


if __name__ == '__main__':
    main()
