import concorde.tsp
import pathlib
import argparse
import networkx as nx
import numpy as np
from egls import datasets
import time
import egls
import pandas as pd
import lkh
import tqdm
import tsplib95

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=pathlib.Path)
parser.add_argument('--solver', type=str, default='concorde', choices=['concorde', 'lkh'])
args = parser.parse_args(['../data/tsp20_test/test.txt', '--solver', 'lkh'])

progress = []

ds = datasets.TSPDataset(args.data_path)
for instance_i, instance_path in tqdm.tqdm(enumerate(ds.instances), total=len(ds.instances)):
    instance = ds.root_dir / instance_path
    G = nx.read_gpickle(instance)
    opt_cost = egls.optimal_cost(G, weight='weight')
    t = time.time()
    progress.append((instance_i, instance, 0, t, np.nan, opt_cost))

    scale = 1e6
    pos = np.vstack([G.nodes[n]['pos'] for n in G.nodes])*scale
    if args.solver == 'concorde':
        problem = concorde.tsp.TSPSolver.from_data(pos[:, 0], pos[:, 1], 'EUC_2D')
        sol = problem.solve(verbose=False)
        tour = sol.tour.tolist()
        tour += [tour[0]]
        cost = egls.tour_cost(G, tour)
        t = time.time()
    else:
        pos_dict = {i + 1: n for i, n in enumerate(pos)}
        problem = tsplib95.models.StandardProblem(
            name=instance,
            type='TSP',
            node_coords=pos_dict,
            dimension=len(pos_dict),
            edge_weight_type='EUC_2D'
        )
        # problem = tsplib95.load_problem('../data/tsplib/rat99.tsp')
        solver_path = '../../LKH-3.0.6/LKH'
        tours = lkh.solve(solver_path, problem, max_trials=100, runs=10)
        tour = [n - 1 for n in tours[0]]
        cost = egls.tour_cost(G, tour)
        t = time.time()
    progress.append((instance_i, instance, 0, t, cost, opt_cost))

profile = pd.DataFrame(progress, columns=['instance', 'instance_name', 'run', 'time', 'cost', 'opt_cost'])
profile.to_pickle(f'{args.data_path.parent.stem}_{args.solver}.pkl')
