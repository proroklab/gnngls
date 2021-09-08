#!/usr/bin/env python
# coding: utf-8

import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

from egls import models, datasets

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
import uuid
import json

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train(model, data_loader, target, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x = batch.ndata['features']
        y = batch.ndata[target]

        optimizer.zero_grad()
        y_pred = model(batch, x)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss


def test(model, data_loader, target, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.ndata['features']
            y = batch.ndata[target]

            y_pred = model(batch, x)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        return epoch_loss

def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
    'val_loss': val_loss
    }, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('data_dir', type=pathlib.Path, help='Where to load dataset')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Where to log Tensorboard data')
    parser.add_argument('--embed_dim', type=int, default=128, help='Maximum hidden feature dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of message passing steps')
    parser.add_argument('--layer_type', type=str, default='gat', choices=['gcn', 'gated_gcn', 'gat', 'gat_mlp', 'gated_gcn_mlp'], help='GNN layer type')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads for GAT')
    # parser.add_argument('--activation', type=str, default='relu', choices=['elu', 'relu', 'leaky_relu'], help='Activation function')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint frequency')
    parser.add_argument('--efeat_drop_idx', type=int, nargs='+', default=[], help='Edge features to drop')
    parser.add_argument('--nfeat_drop_idx', type=int, nargs='+', default=[], help='Node features to drop')
    parser.add_argument('--target', type=str, default='regret', choices=['regret', 'in_solution'])
    args = parser.parse_args()


    # Load dataset
    train_set = datasets.TSPDataset(args.data_dir / 'train.txt', efeat_drop_idx=args.efeat_drop_idx, nfeat_drop_idx=args.nfeat_drop_idx)
    val_set  = datasets.TSPDataset(args.data_dir / 'val.txt', efeat_drop_idx=args.efeat_drop_idx, nfeat_drop_idx=args.nfeat_drop_idx)

    # use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    # activation = getattr(F, args.activation)

    n_nodes, feat_dim = train_set[0].ndata['features'].shape
    #print(n_nodes, feat_dim)

    model = models.EdgePropertyPredictionModel(
        feat_dim,
        args.embed_dim,
        1,
        args.n_layers,
        args.layer_type,
        n_heads=args.n_heads,
        # activation=activation,
        dropout=0.
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.target == 'regret':
        criterion = torch.nn.MSELoss()

    elif args.target == 'in_solution':
        # assuming all instances are the same size
        y = train_set[0].ndata['in_solution']
        pos_weight = len(y)/y.sum() - 1
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch, num_workers=os.cpu_count())
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch, num_workers=os.cpu_count())

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}'
    log_dir = args.tb_dir / run_name
    writer = SummaryWriter(log_dir)

    # early stopping
    best_score = None
    counter = 0

    pbar = tqdm.trange(args.n_epochs)
    for epoch in pbar:
        epoch_loss = train(model, train_loader, args.target, criterion, optimizer, device)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_val_loss = test(model, train_loader, args.target, criterion, device)
        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(epoch_val_loss),
        })

        if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_{epoch}.pt'
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / checkpoint_name)

        if best_score is None or epoch_val_loss < best_score - args.min_delta:
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_best_val.pt')

            best_score = epoch_val_loss
            counter = 0
        else:
            counter += 1

        if counter >= args.patience:
            pbar.close()
            break

        lr_scheduler.step()

    writer.close()

    params = dict(vars(args))
    params['data_dir'] = str(params['data_dir'])
    params['tb_dir'] = str(params['tb_dir'])
    json.dump(params, open(args.tb_dir / run_name / 'params.json', 'w'))

    save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_final.pt')
