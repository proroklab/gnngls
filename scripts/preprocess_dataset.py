if __name__ == '__main__':
    import argparse
    import pathlib
    import numpy as np
    import networkx as nx
    import tqdm.auto as tqdm
    import pickle
    import random

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    if (args.dir / 'scalers.pkl').is_file():
        raise Exception('scalers.pkl already exists.')

    # train test split
    instances = list(args.dir.glob('*.pkl'))
    random.shuffle(instances)

    n_train = 100000
    n_test = 1000
    n_val = 10000

    train_set = instances[:n_train]
    test_set = instances[n_train:n_train + n_test]
    val_set = instances[n_train + n_test:n_train + n_test + n_val]

    for data_set, file_name in zip([train_set, val_set, test_set], ['train.txt', 'val.txt', 'test.txt']):
        with open(args.dir / file_name, 'w') as data_file:
            for path in data_set:
                data_file.write(str(path.relative_to(args.dir)) + '\n')
            print(f'{file_name} contains {len(data_set)} instances.')

    scalers = {
        'nodes': {
            'features': MinMaxScaler()
        },
        'edges': {
            'features': MinMaxScaler(),
            'regret': MinMaxScaler()
        }
    }

    for instance_path in tqdm.tqdm(train_set, total=len(train_set)):
        G = nx.read_gpickle(instance_path)

        for k in scalers['nodes']:
            scalers['nodes'][k].partial_fit(np.vstack([G.nodes[n][k] for n in G.nodes]))
        for k in scalers['edges']:
            scalers['edges'][k].partial_fit(np.vstack([G.edges[e][k] for e in G.edges]))

    pickle.dump(scalers, open(args.dir / 'scalers.pkl', 'wb'))
