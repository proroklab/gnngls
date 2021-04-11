import itertools

from . import is_valid_tour

def two_opt(tour, i, j):
    assert i > 0 and j > 0 and i < len(tour) and j < len(tour)
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j-1:i-1:-1] + tour[j:]

def two_opt_one_to_all(tour, i):
    for j in range(1, len(tour)):
        if i != j:
            new_tour = two_opt(tour, i, j)
            yield new_tour

# TODO: there are some moves that give the inital tour back
def two_opt_all_to_all(tour):
    for i, j in itertools.combinations(range(1, len(tour)), 2):
        new_tour = two_opt(tour, i, j)
        yield new_tour

def relocate_one_to_all(tour, i):
    assert i > 0 and  i < len(tour) - 1
    sub_tour = tour.copy()
    n = sub_tour.pop(i)

    for j in range(1, len(sub_tour)):
        if i != j and i != j - 1:
            new_tour = sub_tour.copy()
            new_tour.insert(j, n)
            yield new_tour

def relocate_all_to_all(tour):
    for i in range(1, len(tour) - 1):
        for new_tour in relocate_one_to_all(tour, i):
            yield new_tour

def exchange(tour, i, j):
    assert i > 0 and j > 0 and i < len(tour) and j < len(tour)
    new_tour = tour.copy()
    n, m = new_tour[i], new_tour[j]
    new_tour[j], new_tour[i] = n, m
    return new_tour
