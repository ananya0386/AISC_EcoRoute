import itertools
import math

from ecoroute.algorithms.astar import astar, euclidean_heuristic, zero_heuristic
from ecoroute.graph_utils import path_distance

from .conftest import make_graph


def test_astar_finds_known_shortest_path(small_graph):
    result = astar(small_graph, "A", "E")
    assert result.found
    assert result.path == ["A", "B", "C", "E"]
    assert result.distance == 3


def test_astar_handles_unreachable_goal(small_graph):
    result = astar(small_graph, "A", "Z")
    assert not result.found
    assert result.path == []
    assert math.isinf(result.distance)


def test_astar_start_equals_goal(small_graph):
    result = astar(small_graph, "C", "C")
    assert result.found and result.path == ["C"] and result.distance == 0


def test_path_distance_equals_sum_of_edge_weights(small_graph):
    result = astar(small_graph, "D", "C")
    assert math.isclose(path_distance(small_graph, result.path), result.distance)


def test_astar_prefers_longer_hop_count_when_shorter_in_km():
    # Direct road S-G is 10 km; the 3-road detour is only 3 km.
    g = make_graph({"S": (0, 0), "M1": (0.5, 0.5), "M2": (1, 0.5), "G": (1.5, 0)},
                   [("S", "G", 10), ("S", "M1", 1), ("M1", "M2", 1), ("M2", "G", 1)])
    assert astar(g, "S", "G").path == ["S", "M1", "M2", "G"]


def test_heuristic_scale_keeps_heuristic_admissible():
    # This road (1 km) is SHORTER than its straight line (5 km): plain Euclidean
    # would overestimate. heuristic_scale() must shrink h so A* stays optimal.
    g = make_graph({"A": (0, 0), "B": (5, 0), "C": (5, 5)}, [("A", "B", 1), ("B", "C", 9), ("A", "C", 20)])
    assert math.isclose(g.heuristic_scale(), 0.2)
    assert astar(g, "A", "C").distance == 10  # A-B-C, the true shortest


def test_astar_matches_dijkstra_on_every_pair_of_sample_city(sample_dataset):
    g = sample_dataset.graph
    h = euclidean_heuristic(g)
    total_astar = total_dijkstra = 0
    for a, b in itertools.combinations(sorted(g.nodes), 2):
        with_h = astar(g, a, b, h)
        without_h = astar(g, a, b, zero_heuristic)
        assert math.isclose(with_h.distance, without_h.distance), (a, b)
        assert math.isclose(path_distance(g, with_h.path), with_h.distance)
        total_astar += with_h.nodes_expanded
        total_dijkstra += without_h.nodes_expanded
    assert total_astar < total_dijkstra  # the heuristic actually prunes the search
