"""Exhaustive (brute-force) reference solver -- EVALUATION ONLY, not one of the main algorithms.

Tries every possible order of the m delivery stops (m! orders) and keeps the
shortest. This gives the true optimum, which lets us measure how close NN and
ACO get. Because m! explodes (10! = 3,628,800), it is limited to small instances.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from ..tours import DistanceMatrixLike, tour_distance

MAX_EXACT_STOPS = 9  # 9! = 362,880 orders: a couple of seconds in pure Python


@dataclass
class ExactResult:
    tour: list[int]
    distance: float
    runtime_ms: float
    orders_checked: int


def exact_tour(matrix: DistanceMatrixLike, max_stops: int = MAX_EXACT_STOPS) -> ExactResult:
    stops = list(range(1, len(matrix)))
    if len(stops) > max_stops:
        raise ValueError(f"Exhaustive search is limited to {max_stops} stops (got {len(stops)}).")
    t0 = time.perf_counter()
    best_tour: list[int] = []
    best_distance = float("inf")
    checked = 0
    for order in itertools.permutations(stops):
        checked += 1
        tour = [0, *order, 0]
        d = tour_distance(matrix, tour)
        if d < best_distance:
            best_tour, best_distance = tour, d
    return ExactResult(best_tour, best_distance, (time.perf_counter() - t0) * 1000, checked)
