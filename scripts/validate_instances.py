import egls
from egls import algorithms
import networkx as nx
import numpy as np
import itertools


def get_node_width(G, depot):
    pos = []
    n2i = {}
    for i, n in enumerate(G.nodes):
        pos.append(G.nodes[n]['pos'])
        n2i[n] = i

    pos = np.vstack(pos)

    center = pos.mean(axis=0)
    center_line = center - pos[n2i[depot]]

    u = center_line/np.linalg.norm(center_line)
    n = np.array([-u[1], u[0]])

    v = pos - pos[n2i[depot]]

    width = np.apply_along_axis(np.dot, 1, v, n)

    return {n: width[n2i[n]] for n in G.nodes}


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


def get_mst(G, weight='weight'):
    mst = nx.minimum_spanning_edges(G, weight=weight, algorithm='kruskal', data=False)
    mst_attr = {e: False for e in G.edges}
    for e in mst:
        if e in mst_attr:
            mst_attr[e] = True
    assert sum(mst_attr.values()) == len(G.nodes) - 1
    return mst_attr


def check_features(G):
    # G = nx.read_gpickle(instance)

    # optimal_cost = 0
    # for e in G.edges:
    #     if G.edges[e]['in_solution']:
    #         optimal_cost += G.edges[e]['weight']

    # regret = get_regret(G, optimal_cost)

    nn_solution = algorithms.nearest_neighbor(G, 0, weight='weight')
    in_nn_solution = egls.tour_to_edge_attribute(G, nn_solution)

    fi_solution = algorithms.insertion(G, 0, mode='farthest', weight='weight')
    in_fi_solution = egls.tour_to_edge_attribute(G, fi_solution)

    # ni_solution = algorithms.insertion(G, 0, mode='nearest', weight='weight')
    # in_ni_solution = egls.tour_to_edge_attribute(G, ni_solution)

    width = get_node_width(G, 0)

    knn = get_nearest_neighbours(G)

    # min_degree_graph = get_min_degree_graph(G, 2)

    depot_weight = get_depot_weight(G, 0)

    mst = get_mst(G)

    # sp_betweenness = nx.edge_betweenness_centrality(G, weight='weight')

    # cf_betweenness = nx.edge_current_flow_betweenness_centrality(G, weight='weight')

    # sp_closeness = nx.closeness_centrality(G, distance='weight')

    # cf_closeness = nx.current_flow_closeness_centrality(G, weight='weight')

    # clustering = nx.clustering(G, weight='weight')

    # hate this
    get_from_edge_dict = lambda d, k: d[k] if k in d else d[tuple(reversed(k))]

    for e in G.edges:
        i, j = e

        expected_features = np.array([
            G.edges[e]['weight'],
            knn[i][j],
            knn[j][i],
            knn[i][j] == knn[j][i],
            knn[i][j] <= 0.1*len(G.nodes) or knn[j][i] <= 0.1*len(G.nodes),
            knn[i][j] <= 0.2*len(G.nodes) or knn[j][i] <= 0.2*len(G.nodes),
            knn[i][j] <= 0.3*len(G.nodes) or knn[j][i] <= 0.3*len(G.nodes),
            get_from_edge_dict(in_nn_solution, e),
            get_from_edge_dict(in_fi_solution, e),
            0, # get_from_edge_dict(in_ni_solution, e),
            0, # get_from_edge_dict(min_degree_graph, e),
            0, # get_from_edge_dict(sp_betweenness, e),
            0, # get_from_edge_dict(cf_betweenness, e),
            np.abs(width[i] - width[j]),
            get_from_edge_dict(mst, e),
        ], dtype=np.float32)

        match = np.isclose(expected_features, G.edges[e]['features'])
        match[[9, 10, 11, 12]] = True
        if not match.all():
            return False

        # G.edges[e]['regret'] = regret[e]

    for n in G.nodes:
        expected_features = np.array([
            depot_weight[n],
            0, # sp_closeness[n],
            0, # cf_closeness[n],
            0, # clustering[n],
            width[n],
        ], dtype=np.float32)

        match = np.isclose(expected_features, G.nodes[n]['features'])
        match[[1, 2, 3]] = True
        if not match.all():
            return False

    return True

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
        in_solution = egls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

if __name__ == '__main__':
    import pathlib
    import uuid
    import argparse
    import multiprocessing as mp
    import tqdm.auto as tqdm

    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    pool = mp.Pool(processes=None)
    instances = list(args.dir.glob('*.pkl'))
    for instance, is_good in tqdm.tqdm(pool.imap_unordered(check_features, instances), total=len(instances)):
        assert is_good, f'{instance} failed'
    pool.close()
    pool.join()
