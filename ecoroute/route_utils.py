"""Distance matrix (built with A*) and route expansion.

This module is the bridge between the two algorithms:

    A*  answers  "what is the shortest ROAD path from location a to location b?"
    ACO answers  "in what ORDER should the delivery stops be visited?"

We run A* once for every pair of selected locations (depot + stops), store the
distance in a matrix and keep the road path so the final route can be drawn.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .algorithms.astar import AStarResult, astar, euclidean_heuristic
from .graph_utils import InputError, RoadGraph, path_distance
from .tours import tour_distance, validate_tour

__all__ = ["DistanceMatrix", "build_distance_matrix", "expand_tour", "tour_distance", "validate_tour"]


@dataclass
class DistanceMatrix:
    locations: list[str]  # node ids; index 0 is ALWAYS the depot
    matrix: list[list[float]]  # matrix[i][j] = A* shortest road distance (km)
    paths: dict[tuple[int, int], list[str]]  # (i, j) -> road path as node ids
    searches: list[AStarResult] = field(default_factory=list)  # one A* run per unordered pair
    runtime_ms: float = 0.0

    @property
    def size(self) -> int:
        return len(self.locations)

    @property
    def total_expanded(self) -> int:
        return sum(s.nodes_expanded for s in self.searches)


def build_distance_matrix(graph: RoadGraph, depot: str, deliveries: list[str]) -> DistanceMatrix:
    """Run A* between every pair of {depot} U deliveries.

    Roads are undirected, so d(a, b) = d(b, a): we search each unordered pair once
    and reuse the reversed path. That is n(n-1)/2 A* searches for n locations.
    """
    t0 = time.perf_counter()
    locations = [depot, *deliveries]
    n = len(locations)
    h = euclidean_heuristic(graph)
    matrix = [[0.0] * n for _ in range(n)]
    paths: dict[tuple[int, int], list[str]] = {(i, i): [locations[i]] for i in range(n)}
    searches: list[AStarResult] = []

    for i in range(n):
        for j in range(i + 1, n):
            result = astar(graph, locations[i], locations[j], h)
            searches.append(result)
            if not result.found:
                a, b = graph.nodes[locations[i]], graph.nodes[locations[j]]
                raise InputError(f"A* found no road path between {a.name} ({a.id}) and {b.name} ({b.id}). "
                                 "These locations are in disconnected parts of the network.")
            matrix[i][j] = matrix[j][i] = result.distance
            paths[(i, j)] = result.path
            paths[(j, i)] = list(reversed(result.path))

    return DistanceMatrix(locations, matrix, paths, searches, (time.perf_counter() - t0) * 1000)


def expand_tour(dm: DistanceMatrix, tour: list[int]) -> list[str]:
    """Replace each stop-to-stop leg of the tour with its full A* road path."""
    validate_tour(tour, dm.size)
    route: list[str] = [dm.locations[tour[0]]]
    for a, b in zip(tour, tour[1:]):
        route.extend(dm.paths[(a, b)][1:])  # skip first node: it is the end of the previous leg
    return route


def check_expanded_distance(graph: RoadGraph, dm: DistanceMatrix, tour: list[int]) -> float:
    """Distance of the expanded road route; must equal the matrix tour distance (consistency check)."""
    road_km = path_distance(graph, expand_tour(dm, tour))
    matrix_km = tour_distance(dm.matrix, tour)
    if not math.isclose(road_km, matrix_km, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"Expanded route {road_km} km != matrix tour {matrix_km} km")
    return road_km
