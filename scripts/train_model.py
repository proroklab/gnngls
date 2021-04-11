#!/usr/bin/env python
# coding: utf-8

import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

from egls import models

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn
import numpy as np

import tqdm.auto as tqdm
import pathlib
import argparse
import datetime
import json

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('data_dir', type=str, help='Where to load dataset')
    parser.add_argument('model_dir', type=str, help='Where to save trained model')
    parser.add_argument('tb_dir', type=str, help='Where to log Tensorboard data')
    parser.add_argument('--embed_dim', type=int, default=128, help='Maximum hidden feature dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of message passing steps')
    parser.add_argument('--layer_type', type=str, default='gat', choices=['gcn', 'gated_gcn', 'gat', 'gat_mlp', 'gated_gcn_mlp'], help='GNN layer type')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads for GAT')
    # parser.add_argument('--activation', type=str, default='relu', choices=['elu', 'relu', 'leaky_relu'], help='Activation function')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--target', type=str, default='regret', choices=['regret', 'in_solution'])
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    args_list = list(vars(args).values())
    run_name = '_'.join(map(str, args_list[3:])) + '_' + timestamp
    print(run_name)

    data_dir = pathlib.Path(args.data_dir)
    model_dir = pathlib.Path(args.model_dir)
    tb_dir = pathlib.Path(args.tb_dir)

    # Load dataset
    train_set, _ = dgl.load_graphs(str(data_dir / 'train_graphs.bin'))
    val_set, _ = dgl.load_graphs(str(data_dir / 'val_graphs.bin'))

    # use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    # activation = getattr(F, args.activation)

    n_feats, feat_dim = train_set[0].ndata['features'].shape

    model = models.EdgePropertyPredictionModel(feat_dim,
        args.embed_dim,
        1,
        args.n_layers,
        args.layer_type,
        n_heads=args.n_heads,
        # activation=activation,
        dropout=0.
    )
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.target == 'regret':
        criterion = torch.nn.MSELoss()
    elif args.target == 'in_solution':
        # assuming all instances are the same size
        y = train_set[0].ndata['in_solution']
        pos_weight = len(y)/y.sum() - 1
        print('pos_weight =', pos_weight)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch)

    log_dir = tb_dir / run_name
    writer = SummaryWriter(log_dir)

    # early stopping
    best_score = None
    min_delta = 1e-3
    counter = 0
    patience = 20

    pbar = tqdm.trange(args.n_epochs)
    for epoch in pbar:
        model.train()

        epoch_loss = 0
        for batch_i, batch in enumerate(train_loader):
            batch = batch.to(device)
            x = batch.ndata['features']
            y = batch.ndata[args.target]
            e = batch.edata['is_depot'].view(-1)

            optimizer.zero_grad()
            y_pred = model(batch, x, e)
            loss = criterion(y_pred, y.type_as(y_pred))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

        epoch_loss /= (batch_i + 1)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        with torch.no_grad():
            model.eval()

            epoch_val_loss = 0
            for batch_i, batch in enumerate(val_loader):
                batch = batch.to(device)
                x = batch.ndata['features']
                y = batch.ndata[args.target]
                e = batch.edata['is_depot'].view(-1)

                y_pred = model(batch, x, e)
                loss = criterion(y_pred, y.type_as(y_pred))

                epoch_val_loss += loss.item()

            epoch_val_loss /= (batch_i + 1)
            writer.add_scalar("Loss/validation", epoch_val_loss, epoch)

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(epoch_val_loss),
        })

        if best_score is None:
            best_score = epoch_val_loss
        elif epoch_val_loss < best_score - min_delta:
            best_score = epoch_val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            pbar.close()
            break

        lr_scheduler.step()

    writer.close()

    summary = {
        'loss': float(epoch_loss),
        'val_loss': float(epoch_val_loss)
    }
    summary.update(vars(args)) # add all commandline arguments
    json.dump(summary, open(model_dir / f'summary_{run_name}.json', 'w'))
    torch.save(model.state_dict(), model_dir / f'model_{run_name}.bin')
