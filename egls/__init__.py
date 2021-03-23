import numpy as np
import networkx as nx

def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c

def two_opt(tour, i, j):
    assert i > 0 and j > 0 and i < len(tour) and j < len(tour)
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j-1:i-1:-1] + tour[j:]

def two_opt_one_to_all(G, i, tour, cost_fun, cost_thresh=np.inf):
    for j in range(1, len(tour)):
        if i != j:
            new_tour = two_opt(tour, i, j)
            new_cost = cost_fun(G, new_tour)
            if new_cost < cost_thresh:
                yield new_tour, new_cost

def two_opt_all_to_all(G, tour, cost_fun, cost_thresh=np.inf):
    import itertools

    for i, j in itertools.combinations(range(1, len(tour)), 2):
        new_tour = two_opt(tour, i, j)
        new_cost = cost_fun(G, new_tour)
        if new_cost < cost_thresh:
            yield new_tour, new_cost

def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False

def greedy_tour(G, depot, weight='weight'):
    tour = [depot]
    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][weight]) for j in G.neighbors(i) if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    return tour

def beam_search(G, depot, p='p'):
    tour = [depot]

    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][p]) for j in G.neighbors(i) if j not in tour]

        nodes, p = zip(*neighbours)

        p = np.array(p)
        if np.sum(p) == 0:
            p[:] = 1.

        j = np.random.choice(nodes, p=p/np.sum(p))
        tour.append(j)

    tour.append(depot)
    return tour

def optimal_tour(G, scale=1e3):
    import concorde.tsp as concorde

    coords = scale*np.vstack([G.nodes[n]['pos'] for n in sorted(G.nodes)])
    solver = concorde.TSPSolver.from_data(coords[:,0], coords[:,1], norm="EUC_2D")
    solution = solver.solve()
    tour = solution.tour.tolist() + [0]
    return tour

def fixed_edge_tour(G, e, scale=1e3, lkh_path='LKH', **lkh_kwargs):
    import tsplib95
    import lkh

    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = len(G.nodes)
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {n + 1: scale*G.nodes[n]['pos'] for n in G.nodes}
    problem.fixed_edges = [[n + 1 for n in e]]

    solution = lkh.solve(lkh_path, problem=problem, **lkh_kwargs)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour
