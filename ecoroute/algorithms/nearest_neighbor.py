"""Nearest-neighbour baseline for ordering delivery stops.

Greedy rule: from the current location, always drive to the closest stop not yet
visited; after the last stop, return to the depot. It is fast (O(n^2)) and often
reasonable, but it never reconsiders an earlier choice, so an early "cheap" step
can force a very expensive step later (typically the final return to the depot).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..tours import DistanceMatrixLike, tour_distance


@dataclass
class NearestNeighborResult:
    tour: list[int]
    distance: float
    runtime_ms: float


def nearest_neighbor_tour(matrix: DistanceMatrixLike, start: int = 0) -> NearestNeighborResult:
    t0 = time.perf_counter()
    n = len(matrix)
    unvisited = set(range(n)) - {start}
    tour = [start]
    current = start
    while unvisited:
        # Key (distance, index): ties are broken by the lowest index, so the result is deterministic.
        nearest = min(unvisited, key=lambda j: (matrix[current][j], j))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    tour.append(start)  # return to the depot
    return NearestNeighborResult(tour, tour_distance(matrix, tour), (time.perf_counter() - t0) * 1000)
