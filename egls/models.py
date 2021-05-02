import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import dgl
import dgl.nn
import numpy as np
import networkx as nx
import pickle
import pathlib

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, h):
        return self.embed(h)


class GATAndMLP(MLP):
    def __init__(self, embed_dim, hidden_dim, out_dim, n_heads):
        super().__init__(embed_dim, hidden_dim, out_dim)
        self.msg = dgl.nn.GATConv(embed_dim, embed_dim//n_heads, n_heads)

    def forward(self, G, h):
        h = self.msg(G, h).view(G.number_of_nodes(), -1)
        return super().forward(h)


class GGCNAndMLP(MLP):
    def __init__(self, embed_dim, hidden_dim, out_dim):
        super().__init__(embed_dim, hidden_dim, out_dim)
        self.msg = dgl.nn.GatedGraphConv(embed_dim, embed_dim, 1, 2)

    def forward(self, G, h, e):
        h = self.msg(G, h, e)
        return super().forward(h)


class EdgePropertyPredictionModel(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        out_dim,
        n_layers,
        layer_type,
        n_heads=1,
        activation=F.relu,
        dropout=0.,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.activation = activation

        self.embed_layer = nn.Linear(in_dim, embed_dim)

        # TODO: a better way to switch between models
        if layer_type == 'gcn':
            self.msg_layers = nn.ModuleList([
                dgl.nn.GraphConv(embed_dim, embed_dim) for _ in range(n_layers)
            ])
        elif layer_type == 'gat':
            self.msg_layers = nn.ModuleList([
                dgl.nn.GATConv(embed_dim, embed_dim//n_heads, n_heads) for _ in range(n_layers)
            ])
        elif layer_type == 'gated_gcn':
            self.msg_layers = nn.ModuleList([
                dgl.nn.GatedGraphConv(embed_dim, embed_dim, n_layers, 2)
            ])
        elif layer_type == 'gated_gcn_mlp':
            self.msg_layers = nn.ModuleList([
                GGCNAndMLP(embed_dim, 512, embed_dim) for _ in range(n_layers)
            ])
        elif layer_type == 'gat_mlp':
            self.msg_layers = nn.ModuleList([
                GATAndMLP(embed_dim, 512, embed_dim, n_heads) for _ in range(n_layers)
            ])
        else:
            raise Exception(f'Unsupported layer type: {layer_type}')

        self.decision_layer = nn.Linear(embed_dim, out_dim)

    def forward(self, G, x, e=None):
        h = self.embed_layer(x)
        for l in self.msg_layers:
            if isinstance(l, GGCNAndMLP) or isinstance(l, dgl.nn.GatedGraphConv):
                h = l(G, h, e)
            else:
                h = l(G, h).view(G.number_of_nodes(), -1)
        h = self.decision_layer(h)
        return h

class Dataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, scalers_file=None):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)

        self.instances = [line.strip() for line in open(instances_file)]
        self.root_dir = instances_file.parent
        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        self.scalers = pickle.load(open(scalers_file, 'rb'))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        H = self.nx_to_dgl(G)
        return H

    def nx_to_dgl(self, G):
        e2i = {}
        regret = []
        efeats = []
        for i, e in enumerate(G.edges):
            e2i[e] = i
            regret.append(G.edges[e]['regret'])
            efeats.append(G.edges[e]['features'])
        regret = self.scalers['edges']['regret'].transform(np.vstack(regret)).astype(np.float32)
        efeats = self.scalers['edges']['features'].transform(np.vstack(efeats))

        n2i = {}
        nfeats = []
        for i, n in enumerate(G.nodes):
            n2i[n] = i
            nfeats.append(G.nodes[n]['features'])
        nfeats = self.scalers['nodes']['features'].transform(np.vstack(nfeats))

        lG = nx.line_graph(G)
        for n in lG.nodes:
            i, j = n
            lG.nodes[n]['in_solution'] = np.array([G.edges[n]['in_solution']])
            lG.nodes[n]['regret'] = regret[e2i[n]]
            lG.nodes[n]['features'] = np.hstack((
                nfeats[n2i[i]],
                efeats[e2i[n]],
                nfeats[n2i[j]]
            ))
            # lG.nodes[n]['e'] = n # store edge id

        attrs = ['features', 'regret', 'in_solution'] # 'e']
        H = dgl.from_networkx(lG, node_attrs=attrs)
        return H
