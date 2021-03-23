import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import torch
import dgl
import egls
import networkx as nx
import numpy as np
import itertools


def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    return in_tour


def get_regret(G, optimal_cost):
    regret = {}

    for e in G.edges:
        if G.edges[e]['in_solution']:
            regret[e] = 0.
        else:
            fixed_edge_tour = egls.fixed_edge_tour(G, e, scale=1e6, max_trials=100, runs=10)
            fixed_edge_cost = egls.tour_cost(G, fixed_edge_tour)
            regret[e] = (fixed_edge_cost - optimal_cost)/optimal_cost

    return regret


def get_nearest_neighbours(G):
    neighbour_rank = {}

    for i in G.nodes:
        neighbours = [(j, G.edges[(i, j)]['weight']) for j in G.neighbors(i)]
        neighbours_sorted = sorted(neighbours, key=lambda e: e[1])
        neighbour_rank[i] = {j: k for k, (j, _) in enumerate(neighbours_sorted)}

    return neighbour_rank


def get_min_degree_graph(G, min_degree, weight='weight'):
    edges = sorted([(e, G.edges[e][weight]) for e in G.edges], key=lambda x: x[1], reverse=True)
    edges, _ = zip(*edges)
    edges = list(edges)

    H = G.edge_subgraph(edges)
    while min(dict(nx.degree(H)).values()) > min_degree:
        edges.pop(0)
        H = G.edge_subgraph(edges)

    return {e: e in edges for e in G.edges}


def get_depot_weight(G, depot, weight='weight'):
    depot_weight = {}
    for n in G.nodes:
        if n == depot:
            depot_weight[n] = 0
        else:
            depot_weight[n] = G.edges[(depot, n)][weight]
    return depot_weight


def set_features(G):
    optimal_cost = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            optimal_cost += G.edges[e]['weight']
    regret = get_regret(G, optimal_cost)

    greedy_solution = egls.greedy_tour(G, 0)
    in_greedy_solution = tour_to_edge_attribute(G, greedy_solution)

    knn = get_nearest_neighbours(G)

    min_degree_graph = get_min_degree_graph(G, 2)

    depot_weight = get_depot_weight(G, 0)

    sp_betweenness = nx.edge_betweenness_centrality(G)

    cf_betweenness = nx.edge_current_flow_betweenness_centrality(G)

    sp_closeness = nx.closeness_centrality(G)

    cf_closeness = nx.current_flow_closeness_centrality(G)

    clustering = nx.clustering(G)

    # hate this
    get_from_edge_dict = lambda d, k: d[k] if k in d else d[tuple(reversed(k))]

    for e in G.edges:
        i, j = e

        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
            get_from_edge_dict(in_greedy_solution, e),
            knn[i][j],
            knn[j][i],
            knn[i][j] == knn[j][i],
            knn[i][j] <= 0.1*len(G.nodes) or knn[j][i] <= 0.1*len(G.nodes),
            knn[i][j] <= 0.2*len(G.nodes) or knn[j][i] <= 0.2*len(G.nodes),
            knn[i][j] <= 0.3*len(G.nodes) or knn[j][i] <= 0.3*len(G.nodes),
            get_from_edge_dict(min_degree_graph, e),
            depot_weight[i],
            depot_weight[j],
            get_from_edge_dict(sp_betweenness, e),
            get_from_edge_dict(cf_betweenness, e),
            sp_closeness[i],
            sp_closeness[j],
            cf_closeness[i],
            cf_closeness[j],
            clustering[i],
            clustering[j],
            regret[e],
        ], dtype=np.float32)

    return G

def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.Graph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=p)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)

        opt_solution = egls.optimal_tour(G, scale=1e6)
        in_solution = tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

def to_dgl_graph(scaler, path):
    G = nx.read_gpickle(path)
    lG = nx.line_graph(G)

    features = np.vstack([G.edges[e]['features'] for e in lG.nodes])
    features_scaled = scaler.transform(features)

    x = {}
    y = {}
    for e, row in zip(lG.nodes, features_scaled):
        x[e] = row[:-1]
        y[e] = row[-1]

    nx.set_node_attributes(lG, x, 'x')
    nx.set_node_attributes(lG, y, 'regret')
    nx.set_node_attributes(lG, nx.get_edge_attributes(G, 'in_solution'), 'in_solution')

    H = dgl.from_networkx(lG, node_attrs=['x', 'in_solution', 'regret'])
    return H

if __name__ == '__main__':
    import tqdm.auto as tqdm
    import pathlib
    import uuid
    import argparse
    import pickle
    import multiprocessing as mp
    import functools

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    if args.dir.exists():
        raise Exception(f'Output directory {args.dir} exists.')
    else:
        args.dir.mkdir()

    pool = mp.Pool(processes=None)
    instance_gen = get_solved_instances(args.n_nodes, args.n_samples)
    for G in pool.imap_unordered(set_features, instance_gen):
        nx.write_gpickle(G, args.dir / f'{uuid.uuid4().hex}.pkl')

    # train test split
    data_set = list(args.dir.glob('*.pkl'))

    train_set, test_set = train_test_split(data_set, train_size=0.8, shuffle=True)
    train_set, val_set = train_test_split(train_set, train_size=0.8, shuffle=True)

    for data_set, file in zip([train_set, val_set, test_set], ['train.txt', 'val.txt', 'test.txt']):
        with open(args.dir / file, 'w') as data_file:
            for path in data_set:
                data_file.write(str(path.relative_to(args.dir)) + '\n')

    # normalise
    scaler = MinMaxScaler()

    for instance_path in tqdm.tqdm(train_set):
        G = nx.read_gpickle(instance_path)
        features = np.vstack([G.edges[e]['features'] for e in G.edges])
        scaler.partial_fit(np.vstack(features))

    pickle.dump(scaler, open(args.dir / 'scaler.pkl', 'wb'))

    f = functools.partial(to_dgl_graph, scaler)

    train_graphs = [G for G in pool.imap_unordered(f, train_set)]
    dgl.save_graphs(str(args.dir / 'train_graphs.bin'), train_graphs)

    val_graphs = [G for G in pool.imap_unordered(f, val_set)]
    dgl.save_graphs(str(args.dir / 'val_graphs.bin'), val_graphs)

    pool.close()
    pool.join()
