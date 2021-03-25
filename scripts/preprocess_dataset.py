import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import numpy as np
import networkx as nx
import dgl

def prepare_line_graph(scalers, path):
    G = nx.read_gpickle(path)

    e2i = {}
    regret = []
    efeats = []
    for i, e in enumerate(G.edges):
        e2i[e] = i
        regret.append(G.edges[e]['regret'])
        efeats.append(G.edges[e]['features'])
    regret = scalers['edges']['regret'].transform(np.vstack(regret))
    efeats = scalers['edges']['features'].transform(np.vstack(efeats))

    n2i = {}
    nfeats = []
    for i, n in enumerate(G.nodes):
        n2i[n] = i
        nfeats.append(G.nodes[n]['features'])
    nfeats = scalers['nodes']['features'].transform(np.vstack(nfeats))


    lG = nx.line_graph(G)
    for e in lG.nodes:
        i, j = e
        lG.nodes[e]['in_solution'] = G.edges[e]['in_solution']
        lG.nodes[e]['regret'] = regret[e2i[e]]
        lG.nodes[e]['features'] = np.hstack((
            nfeats[n2i[i]],
            efeats[e2i[e]],
            nfeats[n2i[j]]
        ))

    H = dgl.from_networkx(lG, node_attrs=['features', 'regret', 'in_solution'])
    return H


def prepare_graph(scalers, G):
    raise Exception('NYI')


if __name__ == '__main__':
    import argparse
    import pathlib
    import tqdm.auto as tqdm
    import pickle
    import functools

    # need this wrapper when using DGL (ancdata error)
    # import torch.multiprocessing as mp
    # mp.set_sharing_strategy('file_system')

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('dir', type=pathlib.Path)
    parser.add_argument('--line_graph', action='store_true')
    args = parser.parse_args()

    # train test split
    if (args.dir / 'scalers.pkl').is_file():
        raise Exception('scalers.pkl already exists.')

    data_set = list(args.dir.glob('*.pkl'))

    train_set, test_set = train_test_split(data_set, train_size=0.8, shuffle=True)
    train_set, val_set = train_test_split(train_set, train_size=0.8, shuffle=True)

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

    print('Fitting scalers...')
    for instance_path in tqdm.tqdm(train_set, total=len(train_set)):
        G = nx.read_gpickle(instance_path)

        for k in scalers['nodes']:
            scalers['nodes'][k].partial_fit(np.vstack([G.nodes[n][k] for n in G.nodes]))
        for k in scalers['edges']:
            scalers['edges'][k].partial_fit(np.vstack([G.edges[e][k] for e in G.edges]))

    pickle.dump(scalers, open(args.dir / 'scalers.pkl', 'wb'))

    if args.line_graph:
        f = functools.partial(prepare_line_graph, scalers)
    else:
        f = functools.partial(prepare_graph, scalers)

    print('Applying scalers...')
    # pool = mp.Pool(processes=None)

    train_graphs = [f(G) for G in tqdm.tqdm(train_set)]
    dgl.save_graphs(str(args.dir / 'train_graphs.bin'), train_graphs)

    train_graphs = [f(G) for G in tqdm.tqdm(val_set)]
    dgl.save_graphs(str(args.dir / 'val_graphs.bin'), train_graphs)

    # pool.close()
    # pool.join()
