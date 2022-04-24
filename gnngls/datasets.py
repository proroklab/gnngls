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

from . import algorithms, tour_cost, fixed_edge_tour, optimal_cost as get_optimal_cost



def set_features(G):
    for e in G.edges:
        i, j = e

        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)

def set_labels(G):
    optimal_cost = get_optimal_cost(G)

    for e in G.edges:
        regret = 0.
        
        if not G.edges[e]['in_solution']:
            tour = fixed_edge_tour(G, e, scale=1e6, max_trials=100, runs=10)
            cost = tour_cost(G, tour)
            regret = (cost - optimal_cost) / optimal_cost
        
        G.edges[e]['regret'] = regret



class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, scalers_file=None, nfeat_drop_idx=[], efeat_drop_idx=[]):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent

        self.instances = [line.strip() for line in open(instances_file)]
        
        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        self.scalers = pickle.load(open(scalers_file, 'rb'))
        
        self.nfeat_drop_idx = nfeat_drop_idx
        self.efeat_drop_idx = efeat_drop_idx

        # only works for homogenous datasets
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

    def get_scaled_features(self, G):
        features = []
        regret = []
        in_solution = []
        for i in range(self.G.number_of_nodes()):
            e = tuple(self.G.ndata['e'][i].numpy())  # corresponding edge

            features.append(G.edges[e]['features'])
            regret.append(G.edges[e]['regret'])
            in_solution.append(G.edges[e]['in_solution'])

        features = np.vstack(features)
        features = np.delete(features, self.efeat_drop_idx, axis=1)
        features_transformed = self.scalers['features'].transform(features)
        regret = np.vstack(regret)
        regret_transformed = self.scalers['regret'].transform(regret)
        in_solution = np.vstack(in_solution)

        H = copy.deepcopy(self.G)
        H.ndata['features'] = torch.tensor(features_transformed, dtype=torch.float32)
        H.ndata['regret'] = torch.tensor(regret_transformed, dtype=torch.float32)
        H.ndata['in_solution'] = torch.tensor(regret, dtype=torch.float32)
        return H
