import os
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm


def all_candidates(df, sub_sampling_count):
    asin_list = df['asin'].unique()
    alternative_sets = [asin_list[i: i+sub_sampling_count] for i in range(0, len(asin_list), sub_sampling_count)]

    def select():
        for alternative_set in alternative_sets:
            for index in range(len(alternative_set)):
                a = list(range(index)) + list(range(index + 1, len(alternative_set)))
                one_candidates = np.random.choice(a, 99, replace=False)
                yield alternative_set[one_candidates].tolist()

        # for index in range(len(asin_list)):
        #     a = list(range(index)) + list(range(index + 1, len(asin_list)))
        #     one_candidates = np.random.choice(a, 99, replace=False)
        #     yield asin_list[one_candidates].tolist()
    progress = tqdm(select(), desc='sampling', total=len(asin_list), leave=False, unit_scale=True)

    candidates = dict(zip(asin_list, progress))
    return candidates


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sub_sampling_count', type=int, default=10000)
    parser.add_argument('--processed_path', type=str, default='/disk/yxk/processed/cold_start/')
    parser.add_argument('--dataset', type=str,
                        # choices=('All_Beauty', 'Appliances', 'Magazine_Subscriptions', 'Software'),
                        choices=("Prime_Pantry", "Luxury_Beauty", "Musical_Instruments", "Software"),
                        default='Prime_Pantry')
    config = parser.parse_args()
    full_path = os.path.join(config.processed_path, config.dataset, "{}_full.csv".format(config.dataset))
    full_df = pd.read_csv(full_path)

    json.dump(all_candidates(full_df, config.sub_sampling_count),
              open(os.path.join(config.processed_path, config.dataset,
                                '{}_candidates.json'.format(config.dataset)), 'w'))

