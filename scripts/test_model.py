import egls
import torch
import numpy as np
import networkx as nx
import tqdm.auto as tqdm
import multiprocessing as mp
import pandas as pd
import time
import argparse
import pathlib
import datetime
import json

from egls import algorithms, operators, models, datasets
from validate_instances import check_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('profile_path', type=pathlib.Path)
    parser.add_argument('--model_path', type=pathlib.Path, default=None)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('--time_limit', type=float, default=10.)
    args = parser.parse_args()
    guides = args.guides

    if args.model_path is not None:
        params = json.load(open(args.model_path.parent / 'params.json'))
        ds = datasets.TSPDataset(args.data_path, efeat_drop_idx=params['efeat_drop_idx'], nfeat_drop_idx=params['nfeat_drop_idx'])
    else:
        ds = datasets.TSPDataset(args.data_path)

    if 'regret_pred' in guides:
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=device)

        model = models.EdgePropertyPredictionModel(
            25 - len(params['nfeat_drop_idx'])*2 - len(params['efeat_drop_idx']),
            params['embed_dim'],
            1,
            params['n_layers'],
            params['layer_type'],
            n_heads=params['n_heads']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        criterion = torch.nn.MSELoss()

    pbar = tqdm.trange(len(ds))
    gaps = []
    progress = []
    for instance in ds.instances:
        G = nx.read_gpickle(ds.root_dir / instance)

        opt_cost = egls.optimal_cost(G, weight='weight')

        # put this outside timing because it is part of the features, so it is recomputed
        init_tour = algorithms.insertion(G, 0, mode='farthest', weight='weight')
        init_cost = egls.tour_cost(G, init_tour)

        t = time.time()
        progress.append((instance, t, 0, opt_cost))

        if 'regret_pred' in guides:
            check_features(G)
            H = ds.get_scaled_features(G)

            x = H.ndata['features']
            y = H.ndata['regret']
            with torch.no_grad():
                 y_pred = model(H, x)
                 #print(criterion(y, y_pred))

            regret_pred = ds.scalers['edges']['regret'].inverse_transform(y_pred.numpy())

            for i in range(H.number_of_nodes()):
                e = H.ndata['e'][i].numpy()
                G.edges[e]['regret_pred'] = regret_pred[i]

        if 'weight' in guides:
            init_tour = algorithms.insertion(G, 0, mode='farthest', weight='weight')
            init_cost = egls.tour_cost(G, init_tour)
            
        if 'width' in guides:
            width = datasets.get_width(G, 0)
            for e in G.edges:
                i, j = e

                G.edges[e]['width'] = np.abs(width[i] - width[j])
                G.edges[e]['width_and_weight'] = G.edges[e]['width'] + G.edges[e]['weight']

        best_tour, best_cost, best_cost_progress = algorithms.guided_local_search(G, init_tour, init_cost, t + args.time_limit, weight='weight', guides=args.guides, first_improvement=False)
        gap = (best_cost/opt_cost - 1)*100
        gaps.append(gap)
        for p in best_cost_progress:
            progress.append((instance, *p, opt_cost))

        pbar.update(1)
        pbar.set_postfix({
            'Avg Gap': '{:.4f}'.format(np.mean(gaps)),
        })

    pbar.close()


    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    params = dict(vars(args))
    params['guides'] = '_'.join(params['guides'])
    params['timestamp'] = timestamp

    profile_name_parts = list(params.values())
    profile_path = args.profile_path / ('test_' + '_'.join(map(str, profile_name_parts[3:])) + '.pkl')
    profile = pd.DataFrame(progress, columns=['instance', 'time', 'cost', 'opt_cost'])
    profile.to_pickle(profile_path)
