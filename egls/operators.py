import itertools

def two_opt(tour, i, j):
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j-1:i-1:-1] + tour[j:]

def two_opt_a2a(tour):
    idxs = range(1, len(tour) - 1)
    for i, j in itertools.combinations(idxs, 2):
        if abs(i - j) < 2:
            continue

        new_tour = two_opt(tour, i, j)
        yield new_tour

def two_opt_o2a(tour, i):
    assert i > 0 and i < len(tour) - 1

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if abs(i - j) < 2:
            continue

        new_tour = two_opt(tour, i, j)
        yield new_tour

def relocate(tour, i, j):
    new_tour = tour.copy()
    n = new_tour.pop(i)
    new_tour.insert(j, n)
    return new_tour

def relocate_o2a(tour, i):
    assert i > 0 and  i < len(tour) - 1

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if i == j:
            continue

        new_tour = relocate(tour, i, j)
        yield new_tour

def relocate_a2a(tour):
    idxs = range(1, len(tour) - 1)
    for i, j in itertools.permutations(idxs, 2):
        if i - j == 1: # e.g. relocate 2 -> 3 == relocate 3 -> 2
            continue

        new_tour = relocate(tour, i, j)
        yield new_tour

def exchange(tour, i, j):
    new_tour = tour.copy()
    n, m = new_tour[i], new_tour[j]
    new_tour[j], new_tour[i] = n, m
    return new_tour

def exchange_o2a(tour, i):
    assert i > 0 and  i < len(tour) - 1

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if i == j:
            continue

        new_tour = exchange(tour, i, j)
        yield new_tour

def exchange_a2a(tour):
    idxs = range(1, len(tour) - 1)
    for i, j in itertools.combinations(idxs, 2):
        new_tour = exchange(tour, i, j)
        yield new_tour
