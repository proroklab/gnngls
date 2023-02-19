#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import pathlib
import time
import uuid

import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

import gnngls
from gnngls import algorithms, models, datasets
from gnngls.aco_ga import aco_search_with_regret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('model_path', type=pathlib.Path)
    parser.add_argument('run_dir', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('--time_limit', type=float, default=10.)
    parser.add_argument('--perturbation_moves', type=int, default=20)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    params = json.load(open(args.model_path.parent / 'params.json'))
    if 'efeat_drop_idx' in params:
        test_set = datasets.TSPDataset(args.data_path, feat_drop_idx=params['efeat_drop_idx'])
    else:
        test_set = datasets.TSPDataset(args.data_path)

    if 'regret_pred' in args.guides:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        print('device =', device)

        _, feat_dim = test_set[0].ndata['features'].shape

        model = models.EdgePropertyPredictionModel(
            feat_dim,
            params['embed_dim'],
            1,
            params['n_layers'],
            n_heads=params['n_heads']
        ).to(device)

        checkpoint = torch.load(args.model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    pbar = tqdm.tqdm(test_set.instances)
    gaps = []
    search_progress = []
    for instance in pbar:
        G = nx.read_gpickle(test_set.root_dir / instance)

        opt_cost = gnngls.optimal_cost(G, weight='weight')

        t = time.time()
        search_progress.append({
            'instance': instance,
            'time': t,
            'opt_cost': opt_cost
        })

        if 'regret_pred' in args.guides:
            H = test_set.get_scaled_features(G).to(device)

            x = H.ndata['features']
            y = H.ndata['regret']
            with torch.no_grad():
                y_pred = model(H, x)

            regret_pred = test_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())

            es = H.ndata['e'].cpu().numpy()
            for e, regret_pred_i in zip(es, regret_pred):
                G.edges[e]['regret_pred'] = np.maximum(regret_pred_i.item(), 0)

            # Initialize the tour with nearest neighbor by the regrets
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')

        else:
            init_tour = algorithms.nearest_neighbor(G, 0, weight='weight')

        init_cost = gnngls.tour_cost(G, init_tour)

        # Use guided local search to find the best tour
        best_tour, best_cost, search_progress_i = aco_search_with_regret(G)

        for row in search_progress_i:
            row.update({
                'instance': instance,
                'opt_cost': opt_cost
            })
            search_progress.append(row)

        gap = (best_cost / opt_cost - 1) * 100
        gaps.append(gap)
        pbar.set_postfix({
            'Avg Gap': '{:.4f}'.format(np.mean(gaps)),
        })
        pbar.update(1)

    pbar.close()

    search_progress_df = pd.DataFrame.from_records(search_progress)
    search_progress_df['best_cost'] = search_progress_df.groupby('instance')['cost'].cummin()
    search_progress_df['gap'] = (search_progress_df['best_cost'] / search_progress_df['opt_cost'] - 1) * 100
    search_progress_df['dt'] = search_progress_df['time'] - search_progress_df.groupby('instance')['time'].transform(
        'min')

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}.pkl'
    if not args.run_dir.exists():
        args.run_dir.mkdir()
    search_progress_df.to_pickle(args.run_dir / run_name)
