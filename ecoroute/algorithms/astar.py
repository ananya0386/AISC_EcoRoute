"""A* Search: shortest road path between two locations.

    f(n) = g(n) + h(n)

    g(n) : road distance already travelled from the start to n   (known exactly)
    h(n) : estimated remaining distance from n to the goal       (heuristic)
    f(n) : estimated length of the best complete path through n

A* always expands the frontier node with the smallest f. If h never overestimates
the true remaining distance (admissible), the first time the goal is *popped*
from the priority queue its g value is the shortest possible distance.

Heuristic used here: h(n) = k * straight_line_distance(n, goal), where k is
computed by RoadGraph.heuristic_scale() (k = 1 when every road is at least as
long as the straight line between its ends, which is true for the sample city).
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from ..graph_utils import RoadGraph, reconstruct_path

Heuristic = Callable[[str, str], float]


@dataclass
class AStarResult:
    start: str
    goal: str
    found: bool
    path: list[str]  # node ids from start to goal (empty if unreachable)
    distance: float  # km; math.inf if unreachable
    nodes_expanded: int  # nodes popped from the queue and whose neighbours were examined
    nodes_generated: int  # entries pushed onto the priority queue
    max_frontier: int  # largest size the priority queue reached
    runtime_ms: float
    expanded_order: list[str] = field(default_factory=list)  # for visualising the search


def euclidean_heuristic(graph: RoadGraph) -> Heuristic:
    """Admissible straight-line heuristic, scaled by k so it never overestimates on this graph."""
    k = graph.heuristic_scale()

    def h(node: str, goal: str) -> float:
        return k * graph.euclidean(node, goal)

    return h


def zero_heuristic(node: str, goal: str) -> float:
    """h(n) = 0 turns A* into Dijkstra's algorithm (useful as a comparison)."""
    return 0.0


def astar(graph: RoadGraph, start: str, goal: str, heuristic: Heuristic | None = None) -> AStarResult:
    if start not in graph.nodes or goal not in graph.nodes:
        raise ValueError(f"Unknown start or goal: {start!r}, {goal!r}")
    h = heuristic if heuristic is not None else euclidean_heuristic(graph)
    t0 = time.perf_counter()

    g_score: dict[str, float] = {start: 0.0}  # best known distance from start
    came_from: dict[str, str] = {}  # predecessor on the best known path
    tie_breaker = itertools.count()  # keeps heap ordering stable when f values are equal

    # Priority queue (binary min-heap) of (f, tie, g, node).
    frontier: list[tuple[float, int, float, str]] = [(h(start, goal), next(tie_breaker), 0.0, start)]
    nodes_generated, max_frontier = 1, 1
    expanded_order: list[str] = []

    while frontier:
        f, _, g, current = heapq.heappop(frontier)

        # Lazy deletion: the heap may hold an outdated (longer) entry for a node we
        # have since reached more cheaply. Skip it instead of searching the heap.
        if g > g_score[current]:
            continue

        # Goal test happens when the node is POPPED (not when it is generated);
        # this is what guarantees optimality with an admissible heuristic.
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return AStarResult(start, goal, True, path, g, len(expanded_order), nodes_generated, max_frontier,
                               (time.perf_counter() - t0) * 1000, expanded_order)

        expanded_order.append(current)
        for neighbor, road_km in graph.neighbors(current):
            tentative_g = g + road_km
            if tentative_g < g_score.get(neighbor, math.inf):
                # Found a shorter way to reach `neighbor`: record it and (re)queue it.
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(frontier, (tentative_g + h(neighbor, goal), next(tie_breaker), tentative_g, neighbor))
                nodes_generated += 1
        max_frontier = max(max_frontier, len(frontier))

    # Queue exhausted without reaching the goal: it is unreachable.
    return AStarResult(start, goal, False, [], math.inf, len(expanded_order), nodes_generated, max_frontier,
                       (time.perf_counter() - t0) * 1000, expanded_order)
