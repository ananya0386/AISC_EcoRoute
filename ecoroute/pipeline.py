"""End-to-end optimisation workflow used by the app, the experiment script and the tests.

    1. validate depot + deliveries on the road graph
    2. A* for every pair of locations          -> distance matrix + road paths
    3. Nearest neighbour on the matrix          -> baseline order
    4. ACO on the same matrix                   -> optimised order
    5. (optional) exhaustive search             -> true optimum, small instances only
    6. expand each order into its A* road path  -> drawable routes
    7. distance, estimated fuel, estimated cost, runtime, % reduction
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .algorithms.aco import ACOParams, ACOResult, ant_colony_optimization
from .algorithms.exact import MAX_EXACT_STOPS, exact_tour
from .algorithms.nearest_neighbor import nearest_neighbor_tour
from .evaluation import VehicleConfig, estimate_fuel, percentage_reduction
from .graph_utils import RoadGraph, validate_selection
from .route_utils import DistanceMatrix, build_distance_matrix, check_expanded_distance, expand_tour


@dataclass
class RouteSummary:
    method: str
    tour: list[int]  # matrix indices, e.g. [0, 3, 1, 2, 0]
    stop_order: list[str]  # delivery node ids in visiting order (depot excluded)
    road_path: list[str]  # full A*-expanded node path, depot -> ... -> depot
    distance_km: float
    fuel_litres: float
    fuel_cost: float
    runtime_ms: float  # ordering time only; the shared A* matrix time is reported separately


@dataclass
class OptimizationResult:
    depot: str
    deliveries: list[str]
    vehicle: VehicleConfig
    seed: int | None
    distance_matrix: DistanceMatrix
    nearest_neighbor: RouteSummary
    aco: RouteSummary
    aco_details: ACOResult
    exact: RouteSummary | None = None
    reduction_pct: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def routes(self) -> list[RouteSummary]:
        return [r for r in (self.nearest_neighbor, self.aco, self.exact) if r is not None]


def _summarize(method: str, tour: list[int], runtime_ms: float, graph: RoadGraph, dm: DistanceMatrix, vehicle: VehicleConfig) -> RouteSummary:
    distance = check_expanded_distance(graph, dm, tour)  # also asserts road route == matrix tour
    litres, cost = estimate_fuel(distance, vehicle)
    return RouteSummary(
        method=method,
        tour=list(tour),
        stop_order=[dm.locations[i] for i in tour[1:-1]],
        road_path=expand_tour(dm, tour),
        distance_km=distance,
        fuel_litres=litres,
        fuel_cost=cost,
        runtime_ms=runtime_ms,
    )


def optimize_routes(
    graph: RoadGraph,
    depot: str,
    deliveries: list[str],
    aco_params: ACOParams | None = None,
    seed: int | None = 42,
    vehicle: VehicleConfig | None = None,
    compute_exact: bool = False,
) -> OptimizationResult:
    vehicle = vehicle or VehicleConfig()
    vehicle.validate()
    aco_params = aco_params or ACOParams()
    aco_params.validate()
    validate_selection(graph, depot, deliveries)

    dm = build_distance_matrix(graph, depot, deliveries)

    nn_raw = nearest_neighbor_tour(dm.matrix)
    nn = _summarize("Nearest Neighbour", nn_raw.tour, nn_raw.runtime_ms, graph, dm, vehicle)

    aco_raw = ant_colony_optimization(dm.matrix, aco_params, seed=seed)
    aco = _summarize("Ant Colony Optimization", aco_raw.best_tour, aco_raw.runtime_ms, graph, dm, vehicle)

    exact = None
    notes: list[str] = []
    if compute_exact:
        if len(deliveries) <= MAX_EXACT_STOPS:
            ex = exact_tour(dm.matrix)
            exact = _summarize("Exhaustive optimum (reference)", ex.tour, ex.runtime_ms, graph, dm, vehicle)
        else:
            notes.append(f"Exhaustive reference skipped: {len(deliveries)} stops > limit of {MAX_EXACT_STOPS}.")

    return OptimizationResult(
        depot=depot,
        deliveries=list(deliveries),
        vehicle=vehicle,
        seed=seed,
        distance_matrix=dm,
        nearest_neighbor=nn,
        aco=aco,
        aco_details=aco_raw,
        exact=exact,
        reduction_pct=percentage_reduction(nn.distance_km, aco.distance_km),
        notes=notes,
    )
