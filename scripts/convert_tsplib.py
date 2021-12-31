import pathlib
import tsplib95 as tsplib
import argparse
import networkx as nx
import numpy as np
from sklearn import preprocessing
import itertools
import egls

parser = argparse.ArgumentParser()
parser.add_argument('tsplib_dir', type=pathlib.Path)
parser.add_argument('out_dir', type=pathlib.Path)
args = parser.parse_args(['data/tsplib', 'data/tsplib'])

scaler = preprocessing.MinMaxScaler()

for instance_path in args.tsplib_dir.glob('*.tsp'):
    instance = tsplib.load(instance_path)
    if instance.dimension <= 200 and instance.edge_weight_type == 'EUC_2D':
        coords = np.array(list(instance.node_coords.values()))
        coords_scaled = scaler.fit_transform(coords)

        G = nx.Graph()
        for n, coord in enumerate(coords_scaled):
            G.add_node(n, pos=coord)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)

        opt_solution = egls.optimal_tour(G, scale=1e6)
        in_solution = egls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        nx.write_gpickle(G, args.out_dir / (instance_path.stem + '.pkl'))