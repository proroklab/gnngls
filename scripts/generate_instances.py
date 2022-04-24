import gnngls
import networkx as nx
import numpy as np
import itertools

from gnngls import datasets

def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
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

        opt_solution = gnngls.optimal_tour(G, scale=1e6)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

if __name__ == '__main__':
    import pathlib
    import uuid
    import argparse
    import multiprocessing as mp

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
    for G in pool.imap_unordered(prepare_instance, instance_gen):
        nx.write_gpickle(G, args.dir / f'{uuid.uuid4().hex}.pkl')
    pool.close()
    pool.join()
