#!/usr/bin/env python
# coding: utf-8

import argparse
import pathlib
import pickle
import random

import networkx as nx
import numpy as np
import tqdm.auto as tqdm
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('dir', type=pathlib.Path)
    parser.add_argument('--n_train', type=int, default=100000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=10000)
    args = parser.parse_args()

    if (args.dir / 'scalers.pkl').is_file():
        raise Exception('scalers.pkl already exists.')

    # train test split
    instances = list(args.dir.glob('*.pkl'))
    random.shuffle(instances)

    train_set = instances[:args.n_train]
    test_set = instances[args.n_train:args.n_train + args.n_test]
    val_set = instances[args.n_train + args.n_test:args.n_train + args.n_test + args.n_val]

    for data_set, file_name in zip([train_set, val_set, test_set], ['train.txt', 'val.txt', 'test.txt']):
        with open(args.dir / file_name, 'w') as data_file:
            for path in data_set:
                data_file.write(str(path.relative_to(args.dir)) + '\n')
            print(f'{file_name} contains {len(data_set)} instances.')

    scalers = {
        'features': MinMaxScaler(),
        'regret': MinMaxScaler()
    }

    for instance_path in tqdm.tqdm(train_set, total=len(train_set)):
        G = nx.read_gpickle(instance_path)

        for k in scalers:
            scalers[k].partial_fit(np.vstack([G.edges[e][k] for e in G.edges]))

    pickle.dump(scalers, open(args.dir / 'scalers.pkl', 'wb'))
