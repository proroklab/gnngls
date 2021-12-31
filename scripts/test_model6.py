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
import uuid

from egls import algorithms, operators, models, datasets
#from validate_instances import check_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('profile_path', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('--model_path', type=pathlib.Path, default=None)
    parser.add_argument('--time_limit', type=float, default=10.)
    parser.add_argument('--perturbation_moves', type=int, default=30)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    guides = args.guides

    if args.model_path is not None:
        params = json.load(open(args.model_path.parent / 'params.json'))
        ds = datasets.TSPDataset(args.data_path, efeat_drop_idx=params['efeat_drop_idx'], nfeat_drop_idx=params['nfeat_drop_idx'])
    else:
        ds = datasets.TSPDataset(args.data_path)

    if 'regret_pred' in guides:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.model_path, map_location=device)

        model = models.EdgePropertyPredictionModel(
            25 - len(params['nfeat_drop_idx'])*2 - len(params['efeat_drop_idx']),
            params['embed_dim'],
            1,
            params['n_layers'],
            params['layer_type'],
            n_heads=params['n_heads']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    pbar = tqdm.trange(len(ds))
    gaps = []
    progress = []
    for instance in ds.instances:
        G = nx.read_gpickle(ds.root_dir / instance)

        opt_cost = egls.optimal_cost(G, weight='weight')

        t = time.time()
        progress.append((instance, t, np.nan, np.nan, opt_cost))

        # "calculate" other features
        width = datasets.get_width(G, 0)
        for e in G.edges:
            i, j = e

            G.edges[e]['width'] = np.abs(width[i] - width[j])
        datasets.get_nearest_neighbours(G, weight='weight')
        datasets.get_depot_weight(G, 0, weight='weight')

        if 'regret_pred' in guides:
            H = ds.get_scaled_features(G).to(device)

            x = H.ndata['features']
            y = H.ndata['regret']
            with torch.no_grad():
                 y_pred = model(H, x)

            regret_pred = ds.scalers['edges']['regret'].inverse_transform(y_pred.cpu().numpy())

            es = H.ndata['e'].cpu().numpy()
            for i in range(H.number_of_nodes()):
                e = es[i]
                G.edges[e]['regret_pred'] = np.maximum(regret_pred[i].item(), 0)
        
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')

        if 'width' in guides:
            width = datasets.get_width(G, 0)
            for e in G.edges:
                i, j = e

                G.edges[e]['width'] = np.abs(width[i] - width[j])
                G.edges[e]['width_and_weight'] = G.edges[e]['width'] + G.edges[e]['weight']

            #init_tour = algorithms.insertion(G, 0, mode='farthest', weight='weight')
            init_tour = algorithms.nearest_neighbor(G, 0, weight='weight')

        init_cost = egls.tour_cost(G, init_tour)

        best_tour, best_cost, best_cost_progress = algorithms.guided_local_search(G, init_tour, init_cost, t + args.time_limit, weight='weight', guides=args.guides, perturbation_moves=args.perturbation_moves, first_improvement=False)
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
    profile_name_parts = dict(vars(args))
    profile_name_parts['guides'] = '_'.join(profile_name_parts['guides'])
    profile_name_parts['timestamp'] = timestamp
    profile_name_parts['data_path'] = profile_name_parts['data_path'].parent.stem
    del profile_name_parts['profile_path']
    del profile_name_parts['model_path']
    profile_name_parts['uuid'] = uuid.uuid4().hex

    profile_path = args.profile_path / ('_'.join(map(str, profile_name_parts.values())) + '.pkl')
    profile = pd.DataFrame(progress, columns=['instance', 'time', 'cost', 'tour', 'opt_cost'])
    profile.to_pickle(profile_path)
