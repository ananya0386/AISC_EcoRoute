"""Metrics and experiment execution.

Fuel is NOT optimised separately. The algorithms minimise distance; fuel and fuel
cost are *estimates* derived from distance with a fixed, user-chosen efficiency:

    Estimated Fuel (L)    = Total Distance (km) / Vehicle Efficiency (km/L)
    Estimated Fuel Cost   = Estimated Fuel (L) x Fuel Price per Litre
    Distance Reduction %  = (Baseline - ACO) / Baseline x 100
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .algorithms.aco import ACOParams, ant_colony_optimization
from .algorithms.exact import MAX_EXACT_STOPS, exact_tour
from .algorithms.nearest_neighbor import nearest_neighbor_tour
from .graph_utils import InputError, RoadGraph, validate_selection
from .route_utils import build_distance_matrix


@dataclass
class VehicleConfig:
    km_per_litre: float = 12.0  # ASSUMED average efficiency of a small delivery van
    fuel_price_per_litre: float = 100.0  # ASSUMED price (Rs/L); change it in the app

    def validate(self) -> None:
        if not self.km_per_litre > 0:
            raise InputError("Vehicle fuel efficiency must be greater than 0 km/L.")
        if self.fuel_price_per_litre < 0:
            raise InputError("Fuel price cannot be negative.")


def percentage_reduction(baseline: float, optimized: float) -> float | None:
    """Positive = optimised route is shorter. None when the baseline is 0 (undefined)."""
    if baseline <= 0:
        return None
    return (baseline - optimized) / baseline * 100.0


def gap_to_optimum(optimal: float, value: float) -> float | None:
    """How much longer than the true optimum a route is, in %. 0 = optimal."""
    if optimal <= 0:
        return None
    return (value - optimal) / optimal * 100.0


def estimate_fuel(distance_km: float, vehicle: VehicleConfig) -> tuple[float, float]:
    vehicle.validate()
    litres = distance_km / vehicle.km_per_litre
    return litres, litres * vehicle.fuel_price_per_litre


def compare_label(nn_km: float, aco_km: float, tol: float = 1e-6) -> str:
    if aco_km < nn_km - tol:
        return "ACO shorter"
    if aco_km > nn_km + tol:
        return "ACO longer"
    return "Tie"


# ----------------------------------------------------------------- experiments
ProgressCallback = Callable[[float, str], None]


def run_experiments(
    graph: RoadGraph,
    depot: str,
    candidate_pool: list[str],
    sizes: list[int],
    instances_per_size: int = 3,
    trials_per_instance: int = 5,
    aco_params: ACOParams | None = None,
    base_seed: int = 2026,
    include_exact: bool = True,
    exact_limit: int = 8,
    progress: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    """Benchmark NN vs ACO over several delivery-set sizes.

    For every size we draw `instances_per_size` random delivery sets (reproducible
    seeds). For each set, A* builds the matrix once, NN runs once (it is
    deterministic) and ACO runs `trials_per_instance` times with different seeds.
    Returns three tables: every run, per-instance statistics and per-size statistics.
    """
    aco_params = aco_params or ACOParams()
    aco_params.validate()
    pool = sorted(set(candidate_pool) - {depot})
    if not sizes:
        raise InputError("Choose at least one delivery-set size.")
    if max(sizes) > len(pool):
        raise InputError(f"Largest size {max(sizes)} exceeds the {len(pool)} available locations.")
    if min(sizes) < 2:
        raise InputError("Each delivery-set size must be at least 2.")
    exact_limit = min(exact_limit, MAX_EXACT_STOPS)

    rows: list[dict] = []
    total_steps = len(sizes) * instances_per_size
    step = 0
    for size in sizes:
        for instance in range(instances_per_size):
            instance_seed = base_seed + 1000 * size + instance
            stops = random.Random(instance_seed).sample(pool, size)
            validate_selection(graph, depot, stops)
            dm = build_distance_matrix(graph, depot, stops)
            nn = nearest_neighbor_tour(dm.matrix)
            exact = exact_tour(dm.matrix) if include_exact and size <= exact_limit else None

            for trial in range(trials_per_instance):
                aco_seed = base_seed + trial
                aco = ant_colony_optimization(dm.matrix, aco_params, seed=aco_seed)
                rows.append({
                    "stops": size,
                    "instance": instance + 1,
                    "instance_seed": instance_seed,
                    "aco_seed": aco_seed,
                    "nn_km": nn.distance,
                    "aco_km": aco.best_distance,
                    "optimal_km": exact.distance if exact else None,
                    "reduction_pct": percentage_reduction(nn.distance, aco.best_distance),
                    "result": compare_label(nn.distance, aco.best_distance),
                    "nn_gap_to_opt_pct": gap_to_optimum(exact.distance, nn.distance) if exact else None,
                    "aco_gap_to_opt_pct": gap_to_optimum(exact.distance, aco.best_distance) if exact else None,
                    "aco_best_iteration": aco.best_found_iteration,
                    "astar_matrix_ms": dm.runtime_ms,
                    "astar_searches": len(dm.searches),
                    "nn_ms": nn.runtime_ms,
                    "aco_ms": aco.runtime_ms,
                    "exact_ms": exact.runtime_ms if exact else None,
                })
            step += 1
            if progress:
                progress(step / total_steps, f"{size} stops, instance {instance + 1}/{instances_per_size}")

    runs = pd.DataFrame(rows)
    return {"runs": runs, "instances": summarize_instances(runs), "sizes": summarize_sizes(runs)}


def summarize_instances(runs: pd.DataFrame) -> pd.DataFrame:
    """Per delivery set: NN distance vs ACO mean / std / best / worst over the trials."""
    out = []
    for (size, instance), grp in runs.groupby(["stops", "instance"], sort=True):
        aco = grp["aco_km"].tolist()
        nn_km = float(grp["nn_km"].iloc[0])
        opt = grp["optimal_km"].iloc[0]
        out.append({
            "stops": size,
            "instance": instance,
            "trials": len(aco),
            "nn_km": nn_km,
            "aco_mean_km": statistics.fmean(aco),
            "aco_std_km": statistics.stdev(aco) if len(aco) > 1 else 0.0,
            "aco_best_km": min(aco),
            "aco_worst_km": max(aco),
            "optimal_km": opt if pd.notna(opt) else None,
            "mean_reduction_pct": percentage_reduction(nn_km, statistics.fmean(aco)),
            "best_reduction_pct": percentage_reduction(nn_km, min(aco)),
            "aco_found_optimum_runs": int((abs(grp["aco_km"] - opt) < 1e-6).sum()) if pd.notna(opt) else None,
        })
    return pd.DataFrame(out)


def summarize_sizes(runs: pd.DataFrame) -> pd.DataFrame:
    """Per delivery-set size, aggregated over all instances and trials."""
    out = []
    for size, grp in runs.groupby("stops", sort=True):
        red = grp["reduction_pct"].astype(float)
        row = {
            "stops": size,
            "instances": grp["instance"].nunique(),
            "aco_runs": len(grp),
            "mean_nn_km": grp.drop_duplicates("instance")["nn_km"].mean(),
            "mean_aco_km": grp["aco_km"].mean(),
            "mean_reduction_pct": red.mean(),
            "std_reduction_pct": red.std(ddof=1) if len(red) > 1 else 0.0,
            "best_reduction_pct": red.max(),
            "worst_reduction_pct": red.min(),
            "aco_shorter_runs": int((grp["result"] == "ACO shorter").sum()),
            "tie_runs": int((grp["result"] == "Tie").sum()),
            "aco_longer_runs": int((grp["result"] == "ACO longer").sum()),
            "mean_astar_matrix_ms": grp.drop_duplicates("instance")["astar_matrix_ms"].mean(),
            "mean_nn_ms": grp.drop_duplicates("instance")["nn_ms"].mean(),
            "mean_aco_ms": grp["aco_ms"].mean(),
        }
        if grp["optimal_km"].notna().all():
            row["mean_nn_gap_to_opt_pct"] = grp.drop_duplicates("instance")["nn_gap_to_opt_pct"].mean()
            row["mean_aco_gap_to_opt_pct"] = grp["aco_gap_to_opt_pct"].mean()
            row["mean_exact_ms"] = grp.drop_duplicates("instance")["exact_ms"].mean()
        out.append(row)
    return pd.DataFrame(out)
