#!/usr/bin/env python
# coding: utf-8

import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn
import numpy as np


class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_steps, activation=F.relu, dropout=0.0):
        super().__init__()

        self.activation = activation

        log2_in_size = max(np.floor(np.log2(in_size)).astype(int), 5) # min size = 32
        log2_hidden_size = np.floor(np.log2(hidden_size)).astype(int)
        log2_out_size = max(np.floor(np.log2(out_size)).astype(int), 5) # min size = 32

        embedding_layer_sizes = [2**x for x in range(log2_in_size, log2_hidden_size + 1)]
        embedding_layer_sizes.insert(0, in_size)

        decision_layer_sizes = [2**(-x) for x in range(-log2_hidden_size, -log2_out_size + 1)]
        decision_layer_sizes.append(out_size)

        self.embedding_layers = nn.ModuleList([nn.Linear(s1, s2) for s1, s2 in zip(embedding_layer_sizes[:-1], embedding_layer_sizes[1:])])

        self.msg_layer = dgl.nn.GatedGraphConv(hidden_size, hidden_size, n_steps=n_steps, n_etypes=1)

        self.decision_layers = nn.ModuleList([nn.Linear(s1, s2) for s1, s2 in zip(decision_layer_sizes[:-1], decision_layer_sizes[1:])])

    def forward(self, g, h):
        for l in self.embedding_layers:
            h = self.activation(l(h))

        etypes = torch.zeros(g.number_of_edges(), device=h.device)
        h = self.activation(self.msg_layer(g, h, etypes))

        for l in self.decision_layers[:-1]:
            h = self.activation(l(h))
        h = self.decision_layers[-1](h) # no activation on output layer

        return h


if __name__ == '__main__':

    import dgl
    import networkx as nx
    import tqdm.auto as tqdm
    import numpy as np
    import pathlib
    import argparse
    import datetime
    import json

    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score


    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('data_dir', type=pathlib.Path, help='Where to load dataset')
    parser.add_argument('model_dir', type=pathlib.Path, help='Where to save trained model')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Where to log Tensorboard data')
    parser.add_argument('--hidden_size', type=int, default=256, help='Maximum hidden feature dimension')
    parser.add_argument('--n_steps', type=int, default=8, help='Number of message passing steps')
    parser.add_argument('--activation', type=str, default='relu', choices=['elu', 'relu', 'leaky_relu'], help='Activation function')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()

    run_name = f'{args.hidden_size}_{args.n_steps}_{args.activation}_{args.lr_init}_{args.lr_decay}_{args.batch_size}_{args.n_epochs}'

    # Load dataset
    train_set, _ = dgl.load_graphs(str(args.data_dir / 'train_graphs.bin'))
    val_set, _ = dgl.load_graphs(str(args.data_dir / 'val_graphs.bin'))
    val_set = val_set[:1000] # trim so it fits on the GPU

    # use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    in_size = train_set[0].ndata['x'].shape[1]

    activation = getattr(F, args.activation)
    net = Net(in_size, args.hidden_size, 2, args.n_steps, activation)
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch)

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    tb_dir = args.tb_dir / f'{run_name}_{timestamp}'
    writer = SummaryWriter(tb_dir)

    # early stopping
    best_score = None
    min_delta = 1e-3
    counter = 0
    patience = 15

    pbar = tqdm.trange(args.n_epochs)
    for epoch in pbar:
        net.train()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.ndata['x']
            y = batch.ndata['y']

            pos_weight = len(y)/y.sum() - 1
            w = torch.FloatTensor([1, pos_weight]).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=w)

            optimizer.zero_grad()
            y_pred = net(batch, x)
            loss = criterion(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

        epoch_loss /= (batch_i + 1)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        with torch.no_grad():
            net.eval()
            batch = dgl.batch(val_set).to(device)

            x = batch.ndata['x']
            y = batch.ndata['y']

            pos_weight = len(y)/y.sum() - 1
            w = torch.FloatTensor([1, pos_weight]).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=w)

            y_pred = net(batch, x)
            val_loss = criterion(y_pred, y.squeeze())
            writer.add_scalar("Loss/validation", val_loss, epoch)

            y_prob = F.softmax(y_pred, dim=1).cpu()
            acc = accuracy_score(y.squeeze().cpu(), y_prob[:, 1] > 0.5)
            writer.add_scalar("Metrics/Accuracy/validation", acc, epoch)

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(val_loss),
        })

        if best_score is None:
            best_score = val_loss
        elif val_loss < best_score - min_delta:
            best_score = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            pbar.close()
            break

        lr_scheduler.step()
        # writer.flush()

    writer.close()
    '''json.dump({
        'config': vars(args),
        'results': {
            'validation': {
                'loss': float(val_loss),
                'accuracy': float(acc),
            },
            'training': {
                'loss': float(epoch_loss),
            }
        }
    }, open(args.model_dir / f'summary_{run_name}.json', 'w'))'''
    torch.save(net.state_dict(), args.model_dir / f'net_{run_name}.bin')
