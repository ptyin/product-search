import os
import itertools
import pandas as pd
import numpy as np


def find_best_metrics(path):
    if not os.path.exists(path):
        return None

    data = None
    with open(path, 'r') as file:
        line = file.readline()
        while line != '':
            if line == '-----------Best Result:-----------\n':
                line = file.readline()
                data = eval(line.join(['{', '}']))
            line = file.readline()
    return data


def save_result(df: pd.DataFrame, path):
    # df.to_csv(path)
    df.to_excel(path)


def to_numpy(data: dict, axis: list, rows: list, columns: list):
    r0, r1, c0, c1 = len(axis[rows[0]]), len(axis[rows[1]]), len(axis[columns[0]]), len(axis[columns[1]])
    axis_dicts = [dict(zip(axis[0], range(len(axis[0])))),
                  dict(zip(axis[1], range(len(axis[1])))),
                  dict(zip(axis[2], range(len(axis[2])))),
                  dict(zip(axis[3], range(len(axis[3]))))]

    array = np.zeros((r0, r1, c0, c1), dtype=float)
    for i, element in enumerate(itertools.product(*axis)):
        array[axis_dicts[rows[0]][element[rows[0]]],
              axis_dicts[rows[1]][element[rows[1]]],
              axis_dicts[columns[0]][element[columns[0]]],
              axis_dicts[columns[1]][element[columns[1]]]] = data[element[0]][element[1]][element[2]][element[3]]
    array = array.reshape((r0 * r1, c0 * c1))
    return array


def main():
    log_path = '/disk/yxk/log/cf/'
    save_path = '../result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = ["Automotive",
                "Cell_Phones_and_Accessories",
                "Clothing_Shoes_and_Jewelry",
                "Musical_Instruments",
                "Office_Products",
                "Toys_and_Games"]
    models = ["ql", "uql", "tran_search", "graph_search"]
    embedding_sizes = [64]
    metrics = ['HR', 'MRR', 'NDCG']

    # data = dict(zip(datasets, zip(models, embedding_sizes)))
    data = {}
    default = {'HR': None, 'MRR': None, 'NDCG': None}
    for dataset, model, embedding_size in itertools.product(datasets, models, embedding_sizes):
        if dataset not in data:
            data[dataset] = {model: {embedding_size: default}}
        elif model not in data[dataset]:
            data[dataset][model] = {embedding_size: default}
        elif embedding_size not in data[dataset][model]:
            data[dataset][model][embedding_size] = default
        else:
            raise NotImplementedError

    for dataset, model, embedding_size in itertools.product(datasets, models, embedding_sizes):
        path = os.path.join(log_path, str(embedding_size), dataset, model, 'train_log.txt')
        result = find_best_metrics(path)
        data[dataset][model][embedding_size] = result if result is not None else default

    # a = list(itertools.product(datasets, models))
    # index = pd.MultiIndex.from_tuples(a, names=['dataset', 'model'])
    # b = list(itertools.product(embedding_sizes, metrics))
    # columns = pd.MultiIndex.from_tuples(b, names=['embedding_size', 'metric'])
    a = list(itertools.product(embedding_sizes, models))
    index = pd.MultiIndex.from_tuples(a, names=['embedding_size', 'model'])
    b = list(itertools.product(datasets, metrics))
    columns = pd.MultiIndex.from_tuples(b, names=['dataset', 'metric'])

    array = to_numpy(data, [datasets, models, embedding_sizes, metrics], [2, 1], [0, 3])
    df = pd.DataFrame(array, index=index, columns=columns)
    save_result(df, os.path.join(save_path, 'log.xlsx'))


if __name__ == '__main__':
    Hr, Mrr, Ndcg = 'HR', 'MRR', 'NDCG'
    main()
