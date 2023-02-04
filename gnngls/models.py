import torch_geometric.nn as pyg
import torch.nn as nn


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
            pyg.GATv2Conv(embed_dim, embed_dim // n_heads, n_heads)
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


class EdgePropertyPredictionModel(nn.Module):
    def __init__(
            self,
            in_dim,
            embed_dim,
            out_dim,
            n_layers,
            n_heads=1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.embed_layer = nn.Linear(in_dim, embed_dim)

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(embed_dim, n_heads, 512) for _ in range(n_heads))
        )

        self.decision_layer = nn.Linear(embed_dim, out_dim)

    def forward(self, G, x):
        h = self.embed_layer(x)
        for l in self.message_passing_layers:
            h = l(G, h)
        h = self.decision_layer(h)
        return h
