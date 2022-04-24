import os

backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import torch.nn as nn
import torch.nn.functional as F
import dgl.nn


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, G=None):
        if G is not None:
            y = self.module(G, x).view(G.number_of_nodes(), -1)
        else:
            y = self.module(x)
        return x + y


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim):
        super().__init__()

        self.message_passing = SkipConnection(
            dgl.nn.GATConv(embed_dim, embed_dim // n_heads, n_heads)
        )

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim)
                ),
            ),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, G, x):
        h = self.message_passing(x, G=G).view(G.number_of_nodes(), -1)
        h = self.feed_forward(h)
        return h


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
        self.msg = dgl.nn.GATConv(embed_dim, embed_dim // n_heads, n_heads)

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
        # if layer_type == 'gcn':
        #     self.msg_layers = nn.ModuleList([
        #         dgl.nn.GraphConv(embed_dim, embed_dim) for _ in range(n_layers)
        #     ])
        # elif layer_type == 'gat':
        #     self.msg_layers = nn.ModuleList([
        #         dgl.nn.GATConv(embed_dim, embed_dim//n_heads, n_heads) for _ in range(n_layers)
        #     ])
        # elif layer_type == 'gated_gcn':
        #     self.msg_layers = nn.ModuleList([
        #         dgl.nn.GatedGraphConv(embed_dim, embed_dim, n_layers, 2)
        #     ])
        # elif layer_type == 'gated_gcn_mlp':
        #     self.msg_layers = nn.ModuleList([
        #         GGCNAndMLP(embed_dim, 512, embed_dim) for _ in range(n_layers)
        #     ])
        # elif layer_type == 'gat_mlp':
        #     self.msg_layers = nn.ModuleList([
        #         GATAndMLP(embed_dim, 512, embed_dim, n_heads) for _ in range(n_layers)
        #     ])
        # else:
        #     raise Exception(f'Unsupported layer type: {layer_type}')

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(embed_dim, n_heads, 512) for _ in range(n_heads))
        )

        self.decision_layer = nn.Linear(embed_dim, out_dim)

    def forward(self, G, x):
        h = self.embed_layer(x)
        for l in self.message_passing_layers:
            # if isinstance(l, GGCNAndMLP) or isinstance(l, dgl.nn.GatedGraphConv):
            #     h = l(G, h, e)
            # else:
            h = l(G, h)
        h = self.decision_layer(h)
        return h
