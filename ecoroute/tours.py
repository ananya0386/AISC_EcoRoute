"""Tour representation shared by every ordering algorithm.

A tour is a list of matrix indices that starts AND ends at the depot (index 0),
for example [0, 3, 1, 2, 0]. Every method (nearest neighbour, ACO, exact search)
is scored with the same tour_distance() function, which keeps the comparison fair.

    Total Distance = d(0, p1) + sum_i d(p_i, p_{i+1}) + d(p_m, 0)
"""

from __future__ import annotations

from typing import Sequence

DistanceMatrixLike = Sequence[Sequence[float]]


def tour_distance(matrix: DistanceMatrixLike, tour: Sequence[int]) -> float:
    return float(sum(matrix[a][b] for a, b in zip(tour, tour[1:])))


def is_valid_tour(tour: Sequence[int], n_locations: int) -> bool:
    """True if the tour starts/ends at the depot and visits every other location exactly once."""
    if len(tour) != n_locations + 1 or tour[0] != 0 or tour[-1] != 0:
        return False
    return sorted(tour[1:-1]) == list(range(1, n_locations))


def validate_tour(tour: Sequence[int], n_locations: int) -> None:
    if not is_valid_tour(tour, n_locations):
        raise ValueError(f"Invalid tour {list(tour)} for {n_locations} locations (depot + {n_locations - 1} stops).")
