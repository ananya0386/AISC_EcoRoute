"""Plain-language explanation of a NN vs ACO result, generated from the actual numbers."""

from __future__ import annotations

from .graph_utils import RoadGraph
from .pipeline import OptimizationResult, RouteSummary


def leg_table(graph: RoadGraph, result: OptimizationResult, route: RouteSummary) -> list[dict]:
    """One row per stop-to-stop leg: where from, where to, A* road distance."""
    dm = result.distance_matrix
    rows = []
    for k, (a, b) in enumerate(zip(route.tour, route.tour[1:]), start=1):
        path = dm.paths[(a, b)]
        rows.append({
            "Leg": k,
            "From": graph.nodes[dm.locations[a]].name,
            "To": graph.nodes[dm.locations[b]].name,
            "Road distance (km)": round(dm.matrix[a][b], 2),
            "Intersections passed": max(len(path) - 2, 0),
        })
    return rows


def _longest_leg(graph: RoadGraph, result: OptimizationResult, route: RouteSummary) -> tuple[str, str, float]:
    dm = result.distance_matrix
    a, b = max(zip(route.tour, route.tour[1:]), key=lambda leg: dm.matrix[leg[0]][leg[1]])
    return graph.nodes[dm.locations[a]].name, graph.nodes[dm.locations[b]].name, dm.matrix[a][b]


def explain_result(graph: RoadGraph, result: OptimizationResult) -> list[str]:
    nn, aco = result.nearest_neighbor, result.aco
    dm = result.distance_matrix
    diff = nn.distance_km - aco.distance_km
    lines: list[str] = []

    if diff > 1e-6:
        lines.append(f"ACO's delivery order is **{diff:.2f} km shorter** ({result.reduction_pct:.1f}%) than the "
                     "nearest-neighbour order, using the *same* A* distance matrix and the same start/return rule.")
    elif diff < -1e-6:
        lines.append(f"In this run ACO's route is **{-diff:.2f} km longer** than nearest neighbour. ACO is stochastic and "
                     "is not guaranteed to win on every instance. Try more iterations or ants, a higher beta, or another seed.")
    else:
        lines.append("Both methods found routes of **identical length**. On small or simple instances the greedy order is "
                     "often already optimal or near-optimal, so there is nothing left for ACO to improve.")

    nn_from, nn_to, nn_len = _longest_leg(graph, result, nn)
    aco_from, aco_to, aco_len = _longest_leg(graph, result, aco)
    lines.append(f"Longest single leg: NN **{nn_from} → {nn_to} ({nn_len:.2f} km)** vs "
                 f"ACO **{aco_from} → {aco_to} ({aco_len:.2f} km)**.")

    nn_return = dm.matrix[nn.tour[-2]][0]
    aco_return = dm.matrix[aco.tour[-2]][0]
    if nn_return > aco_return + 1e-6:
        lines.append(f"Nearest neighbour's final drive back to the depot is {nn_return:.2f} km versus {aco_return:.2f} km for ACO. "
                     "This is the typical greedy failure: always taking the closest next stop leaves a distant stop for last, "
                     "and the vehicle then has to drive all the way back.")

    if result.exact is not None:
        opt = result.exact.distance_km
        nn_gap = (nn.distance_km - opt) / opt * 100 if opt > 0 else 0.0
        aco_gap = (aco.distance_km - opt) / opt * 100 if opt > 0 else 0.0
        lines.append(f"Exhaustive search (reference only) proves the best possible route is **{opt:.2f} km**: "
                     f"NN is {nn_gap:.2f}% above it and ACO is {aco_gap:.2f}% above it.")

    lines.append("Fuel and cost figures are *estimates* derived from distance with your chosen km/L and price. "
                 "The algorithms minimise distance only; real fuel use also depends on traffic, load, speed and gradients.")
    return lines
