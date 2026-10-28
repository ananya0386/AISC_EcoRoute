import math
import random

import pytest

from ecoroute.algorithms import ACOParams, ant_colony_optimization, exact_tour, nearest_neighbor_tour
from ecoroute.graph_utils import path_distance
from ecoroute.route_utils import build_distance_matrix, expand_tour
from ecoroute.tours import is_valid_tour, tour_distance

# 5 locations on a line: depot 0 at x=0, stops at x = 1, 2, 3, 10 (symmetric |xi - xj|).
LINE = [[abs(a - b) for b in (0, 1, 2, 3, 10)] for a in (0, 1, 2, 3, 10)]

# Classic trap for greedy: NN goes to the nearby stop first and pays heavily later.
TRAP = [
    [0, 1, 2, 3],
    [1, 0, 1, 10],
    [2, 1, 0, 1],
    [3, 10, 1, 0],
]


def test_tour_distance_formula():
    # d(0,1) + d(1,2) + d(2,3) + d(3,0) = 1 + 1 + 1 + 3
    assert tour_distance(TRAP, [0, 1, 2, 3, 0]) == 6


def test_is_valid_tour():
    assert is_valid_tour([0, 2, 1, 3, 0], 4)
    assert not is_valid_tour([0, 2, 2, 3, 0], 4)  # repeats a stop
    assert not is_valid_tour([0, 2, 1, 0], 4)  # misses a stop
    assert not is_valid_tour([1, 2, 3, 0, 1], 4)  # does not start at the depot


def test_nearest_neighbor_produces_valid_greedy_tour():
    result = nearest_neighbor_tour(LINE)
    assert is_valid_tour(result.tour, 5)
    assert result.tour == [0, 1, 2, 3, 4, 0]
    assert result.distance == tour_distance(LINE, result.tour) == 20


def test_nearest_neighbor_tie_break_is_deterministic():
    m = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert nearest_neighbor_tour(m).tour == [0, 1, 2, 0]


def test_aco_produces_valid_tour_and_never_beats_the_optimum():
    rng = random.Random(7)
    pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(8)]
    m = [[math.dist(p, q) for q in pts] for p in pts]
    aco = ant_colony_optimization(m, ACOParams(n_ants=10, n_iterations=30), seed=1)
    optimum = exact_tour(m).distance
    assert is_valid_tour(aco.best_tour, 8)
    assert math.isclose(aco.best_distance, tour_distance(m, aco.best_tour))
    assert aco.best_distance >= optimum - 1e-9  # no test assumes ACO reaches the optimum


def test_aco_history_is_monotone_and_complete():
    aco = ant_colony_optimization(LINE, ACOParams(n_ants=5, n_iterations=25), seed=3)
    h = aco.best_so_far_history
    assert len(h) == 25 == len(aco.iteration_mean_history)
    assert all(b <= a + 1e-12 for a, b in zip(h, h[1:]))
    assert h[-1] == aco.best_distance
    assert 1 <= aco.best_found_iteration <= 25


def test_aco_is_reproducible_with_fixed_seed():
    rng = random.Random(3)
    pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(12)]
    m = [[math.dist(p, q) for q in pts] for p in pts]
    p = ACOParams(n_ants=8, n_iterations=20)
    a, b = ant_colony_optimization(m, p, seed=99), ant_colony_optimization(m, p, seed=99)
    assert a.best_tour == b.best_tour
    assert a.best_so_far_history == b.best_so_far_history


@pytest.mark.parametrize("bad", [
    ACOParams(n_ants=0), ACOParams(n_iterations=0), ACOParams(alpha=-1), ACOParams(evaporation_rate=0),
    ACOParams(evaporation_rate=1.5), ACOParams(q=0), ACOParams(initial_pheromone=0), ACOParams(alpha=0, beta=0),
])
def test_aco_rejects_invalid_parameters(bad):
    with pytest.raises(ValueError):
        ant_colony_optimization(LINE, bad, seed=0)


def test_exact_solver_finds_optimum_of_trap():
    # NN: 0->1->2->3->0 = 1+1+1+3 = 6 ; every tour here is 6 or worse.
    assert exact_tour(TRAP).distance == 6


def test_distance_matrix_is_consistent_with_expanded_routes(sample_dataset):
    g, depot = sample_dataset.graph, sample_dataset.depot
    stops = sample_dataset.deliveries[:6]
    dm = build_distance_matrix(g, depot, stops)
    assert dm.locations[0] == depot
    assert len(dm.searches) == 7 * 6 // 2  # one A* per unordered pair
    for i in range(dm.size):
        assert dm.matrix[i][i] == 0
        for j in range(dm.size):
            assert math.isclose(dm.matrix[i][j], dm.matrix[j][i])
            assert math.isclose(path_distance(g, dm.paths[(i, j)]), dm.matrix[i][j])

    # Both methods are measured with the same matrix and their expanded road routes agree with it.
    for tour in (nearest_neighbor_tour(dm.matrix).tour, ant_colony_optimization(dm.matrix, ACOParams(n_iterations=10), seed=0).best_tour):
        road = expand_tour(dm, tour)
        assert road[0] == road[-1] == depot
        assert set(stops) <= set(road)
        assert math.isclose(path_distance(g, road), tour_distance(dm.matrix, tour))
