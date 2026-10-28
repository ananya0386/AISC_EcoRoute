"""ACO parameter sensitivity sweep on the sample city (used to choose the default parameters).

For several (tau_0, Q, rho, beta) settings, runs ACO on fixed random delivery sets
(8, 15 and 25 stops; 4 sets each; 3 seeds each) and reports the mean % distance
reduction vs nearest neighbour and, for 8 stops, the mean gap to the exhaustive optimum.

Run:  python scripts/parameter_sweep.py      (takes several minutes)
Writes: results/parameter_sweep.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ecoroute.algorithms import ACOParams, ant_colony_optimization, exact_tour, nearest_neighbor_tour  # noqa: E402
from ecoroute.evaluation import gap_to_optimum, percentage_reduction  # noqa: E402
from ecoroute.graph_utils import load_dataset_json  # noqa: E402
from ecoroute.route_utils import build_distance_matrix  # noqa: E402

SIZES = (8, 15, 25)
INSTANCES = 4
SEEDS = 3
SETTINGS = [  # (tau_0, Q, rho)
    (1.0, 100.0, 0.3),
    (1.0, 1.0, 0.3),
    (0.1, 1.0, 0.1),
    (0.1, 1.0, 0.3),
    (0.1, 1.0, 0.5),
]
BETAS = (2.0, 3.0, 5.0)


def main() -> None:
    dataset = load_dataset_json(ROOT / "data" / "sample_city.json")
    graph, depot = dataset.graph, dataset.depot
    pool = sorted(set(graph.nodes) - {depot})

    instances = []
    for size in SIZES:
        for k in range(INSTANCES):
            stops = random.Random(size * 100 + k).sample(pool, size)
            dm = build_distance_matrix(graph, depot, stops)
            nn = nearest_neighbor_tour(dm.matrix).distance
            opt = exact_tour(dm.matrix).distance if size <= 8 else None
            instances.append((size, k, dm.matrix, nn, opt))

    rows = []
    for tau0, q, rho in SETTINGS:
        for beta in BETAS:
            params = ACOParams(n_ants=20, n_iterations=100, alpha=1.0, beta=beta, evaporation_rate=rho, q=q, initial_pheromone=tau0)
            for size, k, matrix, nn, opt in instances:
                for seed in range(SEEDS):
                    aco = ant_colony_optimization(matrix, params, seed=seed)
                    rows.append({"tau0": tau0, "Q": q, "rho": rho, "beta": beta, "stops": size, "instance": k, "seed": seed,
                                 "nn_km": nn, "aco_km": aco.best_distance, "reduction_pct": percentage_reduction(nn, aco.best_distance),
                                 "gap_to_opt_pct": gap_to_optimum(opt, aco.best_distance) if opt else None})
            print(f"done tau0={tau0} Q={q} rho={rho} beta={beta}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "parameter_sweep.csv", index=False)
    summary = (df.groupby(["tau0", "Q", "rho", "beta", "stops"])
                 .agg(mean_reduction_pct=("reduction_pct", "mean"), worst_reduction_pct=("reduction_pct", "min"),
                      mean_gap_to_opt_pct=("gap_to_opt_pct", "mean"))
                 .round(2).unstack("stops"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
