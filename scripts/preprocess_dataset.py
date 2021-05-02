if __name__ == '__main__':
    import argparse
    import pathlib
    import numpy as np
    import networkx as nx
    import tqdm.auto as tqdm
    import pickle

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    if (args.dir / 'scalers.pkl').is_file():
        raise Exception('scalers.pkl already exists.')

    # train test split
    instances = list(args.dir.glob('*.pkl'))

    train_set, test_set = train_test_split(instances, train_size=0.854, shuffle=True)
    train_set, val_set = train_test_split(train_set, train_size=0.854, shuffle=True)

    for data_set, file in zip([train_set, val_set, test_set], ['train.txt', 'val.txt', 'test.txt']):
        with open(args.dir / file, 'w') as data_file:
            for path in data_set:
                data_file.write(str(path.relative_to(args.dir)) + '\n')

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
