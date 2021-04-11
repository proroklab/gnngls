import os
backend = 'pytorch'
os.environ['DGLBACKEND'] = backend

import egls
import torch
import torch.nn.functional as F
import dgl
import pathlib
import pickle
import networkx as nx
import numpy as np
import argparse
import tqdm.auto as tqdm

# TODO: move prepare line graph to egls module
import preprocess_dataset

from matplotlib import pyplot as plt, colors

def local_search(G, init_tour, weight='weight', improvement_strategy='first'):
    assert improvement_strategy in ['best', 'first'], f'Unknown improvement strategy: {improvement_strategy}'

    init_cost = egls.tour_cost(G, init_tour, weight=weight)

    cur_tour, cur_cost = init_tour, init_cost
    best_tour, best_cost = init_tour, init_cost

    improved = True
    while improved:
        improved = False

        for new_tour in egls.two_opt_all_to_all(G, cur_tour):
            new_cost = egls.tour_cost(G, new_tour, weight=weight)

            if new_cost < best_cost:
                best_tour, best_cost = new_tour, new_cost

            if new_cost < cur_cost:
                improved = True
                cur_tour, cur_cost = new_tour, new_cost

                if improvement_strategy == 'first':
                    break
                elif improvement_strategy == 'best':
                    pass

    return best_tour

def guided_local_search(G, init_tour, n_iters, weight='weight', improvement_strategy='first', guide='regret', guide_importance=0.1):
    assert improvement_strategy in ['best', 'first'], f'Unknown improvement strategy: {improvement_strategy}'

    init_cost = egls.tour_cost(G, init_tour, weight=weight)

    cur_tour, cur_cost = init_tour, init_cost
    best_tour, best_cost = init_tour, init_cost

    k = guide_importance*init_cost/len(G.nodes)
    nx.set_edge_attributes(G, 0, '_penalty')

    for _ in range(n_iters):

        improved = True
        while improved:
            improved = False

            for new_tour in egls.two_opt_all_to_all(G, cur_tour):
                new_cost = egls.tour_cost(G, new_tour, weight=weight)
                new_guided_cost = new_cost + k*egls.tour_cost(G, new_tour, weight='_penalty')

                if new_cost < best_cost:
                    best_tour, best_cost = new_tour, new_cost

                if new_guided_cost < cur_cost:
                    cur_tour, cur_cost = new_tour, new_guided_cost
                    improved = True

                    if improvement_strategy == 'first':
                        break
                    elif improvement_strategy == 'best':
                        pass

        # penalize edge
        max_util = 0
        max_util_e = None
        for e in zip(cur_tour[:-1], cur_tour[1:]):
            util_e = G.edges[e][guide]/(1 + G.edges[e]['_penalty'])
            if util_e > max_util:
                max_util = util_e
                max_util_e = e

        if max_util_e is not None:
            G.edges[max_util_e]['_penalty'] += 1.
        else:
            break # all regret is 0

    return best_tour, nx.get_edge_attributes(G, '_penalty')

def optimal_tour_cost(G):
    c = 0.
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e]['weight']
    return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve a problem')
    parser.add_argument('data_path', type=str, help='Path to dataset')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('--n_instances', type=int, default=-1, help='Number of instances to evaluate')
    parser.add_argument('--init_strategy', type=str, choices=['greedy', 'beam_search'], default='greedy')
    parser.add_argument('--init_weight', type=str, default='weight')
    parser.add_argument('--init_guide', type=str, default='weight')
    parser.add_argument('--init_iters', type=int, default=1)
    parser.add_argument('--search_strategy', type=str, choices=['ls', 'gls', None], default=None)
    parser.add_argument('--search_weight', type=str, default='weight')
    parser.add_argument('--search_guide', type=str, default='weight')
    parser.add_argument('--search_guide_importance', type=float, default=0.1)
    parser.add_argument('--search_iters', type=int, default=-1)
    parser.add_argument('--display_last', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = egls.models.EdgePropertyPredictionModel(19, 128, 1, 6, 'gat_mlp', n_heads=8)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    data_path = pathlib.Path(args.data_path)
    test_set = [l.strip() for l in open(data_path / 'test.txt')]
    if args.n_instances > 0:
        test_set = test_set[:args.n_instances]
    scalers = pickle.load(open(data_path / 'scalers.pkl', 'rb'))

    gaps = []
    losses = []

    for instance in tqdm.tqdm(test_set):
        G = nx.read_gpickle(data_path / instance)
        H = preprocess_dataset.prepare_line_graph(scalers, data_path / instance)

        x = H.ndata['features']
        e = H.edata['is_depot']

        with torch.no_grad():
            y = model(H, x, e)
            loss = F.mse_loss(y, H.ndata['regret'])
            losses.append(loss.item())

        regret = scalers['edges']['regret'].inverse_transform(y.numpy()).squeeze()

        for i in range(H.number_of_nodes()):
            e = H.ndata['e'][i].numpy()
            G.edges[e]['regret_pred'] = regret[i]
            G.edges[e]['regret_pred_i'] = 1/regret[i]
            # G.edges[e]['regret_i'] = 1/np.float32(G.edges[e]['regret'])

        if args.init_strategy == 'greedy':
            init_tour = egls.greedy_tour(G, 0, weight=args.init_weight)
        elif args.init_strategy == 'beam_search':
            init_tour = egls.best_beam_search(G, 0, args.init_iters, prob=args.init_guide, weight=args.init_weight)
        else:
            raise Exception(f'Unknown initialisation strategy: {args.init_strategy}')

        if args.search_iters == -1:
            args.search_iters = 10000

        if args.search_strategy == 'ls':
            final_tour = local_search(G, init_tour, weight=args.search_weight)
        elif args.search_strategy == 'gls':
            final_tour, final_penalties = guided_local_search(G, init_tour, args.search_iters, weight=args.search_weight, guide=args.search_guide, guide_importance=args.search_guide_importance)
        else:
            final_tour = init_tour

        final_cost = egls.tour_cost(G, final_tour)

        opt_cost = optimal_tour_cost(G)
        gap = (final_cost - opt_cost)/opt_cost
        gaps.append(gap)

    print(f'Avg gap: {100*np.mean(gaps):.4f}%')
    print(f'Avg loss: {np.mean(losses):.4f}')

    if args.display_last:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

        cmap_colors = np.zeros((100, 4))
        cmap_colors[:, 0] = 1
        cmap_colors[:, 3] = np.linspace(0, 1, 100)
        cmap = colors.ListedColormap(cmap_colors)

        init_tour_edges = egls.tour_to_edge_attribute(G, init_tour)
        final_tour_edges = egls.tour_to_edge_attribute(G, final_tour)
        opt_tour_edges = nx.get_edge_attributes(G, 'in_solution')
        pos = nx.get_node_attributes(G, 'pos')
        if args.search_strategy == 'gls':
            penalties = nx.get_edge_attributes(G, '_penalty')
        else:
            penalties = {e: 0 for e in G.edges}

        nx.draw(G, pos, with_labels=True, edge_color=opt_tour_edges.values(), edge_cmap=cmap, ax=ax[0])
        nx.draw(G, pos, with_labels=True, edge_color=init_tour_edges.values(), edge_cmap=cmap, ax=ax[1])
        nx.draw(G, pos, with_labels=True, edge_color=final_tour_edges.values(), edge_cmap=cmap, ax=ax[2])
        nx.draw(G, pos, with_labels=True, edge_color=penalties.values(), edge_cmap=cmap, ax=ax[3])

        ax[0].set_title('Optimal tour')
        ax[1].set_title('Initial tour')
        ax[2].set_title(f'Final tour (gap={gap*100:.4f}%)')
        ax[3].set_title('Penalties')

        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
        ax[3].set_aspect('equal')

        plt.show()
