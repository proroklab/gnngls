import os

backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import torch
import torch.utils.data
import dgl
import numpy as np
import networkx as nx
import pickle
import pathlib
import copy

from operator import itemgetter

from . import algorithms, tour_cost, tour_to_edge_attribute, fixed_edge_tour, optimal_cost as get_optimal_cost


def _get_from_edge_dict(d, k):
    return d[k] if k in d else d[tuple(reversed(k))]


def set_features(G, depot):
    for e in G.edges:
        i, j = e

        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)

def set_labels(G, depot):
    optimal_cost = get_optimal_cost(G)
    regret = get_regret(G, optimal_cost)

    for e in G.edges:
        G.edges[e]['regret'] = regret[e].astype(float32)


def get_regret(G, optimal_cost):
    regret = {}

    for e in G.edges:
        if G.edges[e]['in_solution']:
            regret[e] = 0.
        else:
            tour = fixed_edge_tour(G, e, scale=1e6, max_trials=100, runs=10)
            cost = tour_cost(G, tour)
            regret[e] = (cost - optimal_cost) / optimal_cost

    return regret


class TSPLIBDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, is_scaled=True):
        super().__init__()

        if is_scaled:
            pass
        else:
            raise Exception('NYI')

        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)

        self.instances = [line.strip() for line in open(instances_file)]
        self.root_dir = instances_file.parent

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        lG = nx.line_graph(G)
        for n in lG.nodes:
            lG.nodes[n]['e'] = n
            lG.nodes[n]['weight'] = G.edges[n]['weight']
        H = dgl.from_networkx(lG, node_attrs=['e', 'weight'])
        H.ndata['weight'] /= np.sqrt(2)  # scale lol
        return H


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, scalers_file=None, nfeat_drop_idx=[], efeat_drop_idx=[]):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)

        self.instances = [line.strip() for line in open(instances_file)]
        self.root_dir = instances_file.parent
        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        self.scalers = pickle.load(open(scalers_file, 'rb'))
        self.nfeat_drop_idx = nfeat_drop_idx
        self.efeat_drop_idx = efeat_drop_idx

        # cache the graph so we only need to create it once
        G = nx.read_gpickle(self.root_dir / self.instances[0])
        lG = nx.line_graph(G)
        for n in lG.nodes:
            lG.nodes[n]['e'] = n
        self.G = dgl.from_networkx(lG, node_attrs=['e'])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        H = self.get_scaled_features(G)
        return H

    def get_scaled_edge_weight(self, G):
        x = []
        for e_i in range(self.G.number_of_nodes()):
            e = tuple(self.G.ndata['e'][e_i].numpy())  # corresponding edge
            i, j = e  # nodes from edge

            w = G.edges[e]['weight']
            x.append([w])

        H = copy.deepcopy(self.G)
        H.ndata['features'] = torch.tensor(x, dtype=torch.float32) / np.sqrt(2)
        return H

    def get_scaled_features(self, G):
        e2i = {}
        regret = []
        efeats = []
        for i, e in enumerate(G.edges):
            e2i[e] = i
            regret.append(G.edges[e]['regret'])
            efeats.append(G.edges[e]['features'])
        regret = self.scalers['edges']['regret'].transform(np.vstack(regret))
        efeats = self.scalers['edges']['features'].transform(np.vstack(efeats))
        efeats = np.delete(efeats, self.efeat_drop_idx, axis=1)

        n2i = {}
        nfeats = []
        for i, n in enumerate(G.nodes):
            n2i[n] = i
            nfeats.append(G.nodes[n]['features'])
        nfeats = self.scalers['nodes']['features'].transform(np.vstack(nfeats))
        nfeats = np.delete(nfeats, self.nfeat_drop_idx, axis=1)

        x = []
        y = []
        for e_i in range(self.G.number_of_nodes()):
            e = tuple(self.G.ndata['e'][e_i].numpy())  # corresponding edge
            i, j = e  # nodes from edge

            x.append(np.hstack((
                nfeats[n2i[i]],
                efeats[e2i[e]],
                nfeats[n2i[j]]
            )))
            y.append(regret[e2i[e]])

        H = copy.deepcopy(self.G)
        H.ndata['features'] = torch.tensor(x, dtype=torch.float32)
        H.ndata['regret'] = torch.tensor(y, dtype=torch.float32)
        return H
