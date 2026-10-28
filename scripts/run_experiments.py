"""Command-line experiment runner (same code path as the app's Experiments page).

Run:  python scripts/run_experiments.py
      python scripts/run_experiments.py --sizes 5 10 15 20 --instances 3 --trials 10 --seed 2026

Writes CSVs to results/: experiment_runs.csv, experiment_instances.csv, experiment_sizes.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ecoroute.algorithms import ACOParams  # noqa: E402
from ecoroute.evaluation import run_experiments  # noqa: E402
from ecoroute.graph_utils import load_dataset_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "sample_city.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 7, 10, 15, 20, 25])
    parser.add_argument("--instances", type=int, default=3, help="random delivery sets per size")
    parser.add_argument("--trials", type=int, default=5, help="ACO seeds per delivery set")
    parser.add_argument("--seed", type=int, default=2026, help="base random seed")
    parser.add_argument("--exact-limit", type=int, default=8, help="run exhaustive reference up to this many stops")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    dataset = load_dataset_json(args.dataset)
    params = ACOParams()
    print(f"Dataset: {dataset.name} | depot {dataset.depot} | ACO params {params.as_dict()}")
    t0 = time.perf_counter()
    tables = run_experiments(
        dataset.graph, dataset.depot, list(dataset.graph.nodes), args.sizes,
        instances_per_size=args.instances, trials_per_instance=args.trials, aco_params=params,
        base_seed=args.seed, exact_limit=args.exact_limit,
        progress=lambda frac, msg: print(f"  [{frac:5.0%}] {msg}", flush=True),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"experiment_{name}.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\nFinished in {time.perf_counter() - t0:.1f} s. CSVs written to {out}\n")
    print("Per-size summary:")
    print(tables["sizes"].round(2).to_string(index=False))
    print("\nPer-instance summary:")
    print(tables["instances"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
