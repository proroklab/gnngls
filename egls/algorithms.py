import numpy as np

from . import tour_cost, operators

def nearest_neighbor(G, depot, weight='weight'):
    tour = [depot]
    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][weight]) for j in G.neighbors(i) if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    return tour

def beam_search(G, depot, prob='prob'):
    tour = [depot]

    while len(tour) < len(G.nodes):
        i = tour[-1]

        neighbours = [(j, G.edges[(i, j)][prob]) for j in G.neighbors(i) if j not in tour]

        nodes, p = zip(*neighbours)

        p = np.array(p)

        # if there are any infinite values, make these 1 and others 0
        is_inf = np.isinf(p)
        if is_inf.any():
            p = is_inf

        # if there are all 0s, make everything 1
        if np.sum(p) == 0:
            p[:] = 1.

        j = np.random.choice(nodes, p=p/np.sum(p))
        tour.append(j)

    tour.append(depot)
    return tour

def cheapest_beam_search(G, depot, n_iters, prob='prob', weight='weight'):
    best_tour = None
    best_cost = 0

    for _ in range(n_iters):
        new_tour = beam_search(G, depot, prob)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour

def cheapest_insertion(G, sub_tour, n, weight='weight'):
    best_tour = None
    best_cost = 0

    for j in range(1, len(sub_tour)):
        new_tour = sub_tour.copy()
        new_tour.insert(j, n)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour

def insertion(G, depot, mode='farthest', weight='weight'):
    assert mode in ['random', 'nearest', 'farthest'], f'Unknown mode: {mode}'

    nodes = list(G.nodes)
    nodes.remove(depot)
    tour = [depot, depot]

    while len(nodes) > 0:
        if mode == 'random':
            next_node = np.random.choice(nodes)

        else:
            next_node = None
            next_cost = 0

            for i in tour:
                for j in nodes:
                    if (mode == 'nearest' and G.edges[i, j][weight] < next_cost) or \
                        (mode == 'farthest' and G.edges[i, j][weight] > next_cost) or \
                        (next_node is None):
                        next_node = j
                        next_cost = G.edges[i, j][weight]

        nodes.remove(next_node)
        tour = cheapest_insertion(G, tour, next_node, weight)

    return tour

def local_search(G, init_tour, init_cost, weight='weight'):
    cur_tour, cur_cost = init_tour, init_cost
    best_tour, best_cost = init_tour, init_cost

    improved = True
    while improved:
        improved = False

        for operator in [operators.two_opt_a2a, operators.relocate_a2a, operators.exchange_a2a]:

            for new_tour in operator(cur_tour):
                new_cost = egls.tour_cost(G, new_tour, weight=weight)

                if new_cost < best_cost:
                    best_tour, best_cost = new_tour, new_cost

                if new_cost < cur_cost:
                    improved = True
                    cur_tour, cur_cost = new_tour, new_cost

    return best_tour, best_cost

def guided_local_search(G, init_tour, init_cost, n_iters, weight='weight', guide='weight', perturbation_moves=30):
    k = 0.1*init_cost/len(G.nodes)
    nx.set_edge_attributes(G, 0, 'penalty')

    cur_tour, cur_cost = local_search(G, init_tour, init_cost, weight)
    best_tour, best_cost = cur_tour, cur_cost

    for _ in range(n_iters):
        # perturbation
        moves = 0
        while moves < perturbation_moves:
            # penalize edge
            max_util = 0
            max_util_e = None
            for e in zip(cur_tour[:-1], cur_tour[1:]):
                util = G.edges[e][guide]/(1 + G.edges[e]['penalty'])
                if util > max_util or max_util_e is None:
                    max_util = util
                    max_util_e = e

            G.edges[max_util_e]['penalty'] += 1.

            cur_guided_cost = cur_cost + k*egls.tour_cost(G, cur_tour, weight='penalty')

            # apply operator to edge
            for n in max_util_e:
                if n != 0: # not the depot
                    i = cur_tour.index(n)

                    for operator in [operators.two_opt_o2a, operators.relocate_o2a, operators.exchange_o2a]:
                        moved = False

                        for new_tour in operator(cur_tour, i):
                            new_cost = egls.tour_cost(G, new_tour, weight=weight)
                            new_guided_cost = new_cost + k*egls.tour_cost(G, new_tour, weight='penalty')

                            if new_cost < best_cost:
                                best_tour, best_cost = new_tour, new_cost

                            if new_guided_cost < cur_guided_cost:
                                cur_tour, cur_cost, cur_guided_cost = new_tour, new_cost, new_guided_cost
                                moved = True

                        moves += moved

        # optimisation
        cur_tour, cur_cost = local_search(G, cur_tour, cur_cost, weight)
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour, cur_cost

    return best_tour, best_cost
