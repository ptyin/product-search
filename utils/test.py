import os
import pandas as pd
from argparse import ArgumentParser

from src.common.data_preparation import parser_add_data_arguments


def statistics(df: pd.DataFrame):
    query_item_map = {}
    groups = df.groupby(by='query_')
    for group in groups:
        for _, row in group[1].iterrows():
            if group[0] not in query_item_map:
                query_item_map[group[0]] = [(row['userID'], row['asin'])]
            else:
                query_item_map[group[0]].append((row['userID'], row['asin']))


def main():
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

        full_df = pd.read_csv(full_path)
        statistics(full_df)


if __name__ == '__main__':
    main()
