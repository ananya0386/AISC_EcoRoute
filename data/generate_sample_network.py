"""Generate the SYNTHETIC "EcoRoute Sample City" road network.

This is NOT real city data. It is a reproducible, made-up network designed to
look like a small city so the algorithms can be demonstrated offline:

* a 9 x 7 grid of intersections (~1 km apart, jittered so it is not a perfect grid)
* a river between rows 3 and 4 that can only be crossed at 3 bridges
  (this makes straight-line distance misleading, so A* and ACO have real work to do)
* some grid roads removed (dead ends / missing links) and some diagonal arterial roads
* road length = straight-line length x a random "curviness" factor in [1.03, 1.30],
  rounded UP to 0.01 km, so every road is at least as long as its straight line
  (this keeps the Euclidean A* heuristic admissible)

Run:  python data/generate_sample_network.py
Outputs (next to this script): sample_city.json, sample_nodes.csv, sample_edges.csv
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import deque
from pathlib import Path

SEED = 42
COLS, ROWS = 9, 7
SPACING_KM = 1.0
RIVER_BELOW_ROW = 3  # river lies between row 3 and row 4
RIVER_GAP_KM = 0.6  # extra width added by the river
BRIDGE_COLS = (1, 5, 7)
DEPOT_RC = (1, 4)

DELIVERY_NAMES = [
    "Lotus Pharmacy", "Banyan Bookstore", "Saffron Bakery", "Neem Clinic",
    "Peacock Electronics", "Marigold Florist", "Indigo Tailors", "Jasmine Cafe",
    "Teak Furniture", "Monsoon Sports", "Coral Opticals", "Amber Grocers",
    "Cedar School", "Pebble Hardware", "Orchid Salon", "Willow Library",
    "Mango Mart", "Silver Oak Hotel", "Comet Courier Point", "Riverfront Office Park",
]
DEFAULT_SELECTION_SIZE = 10


def node_id(r: int, c: int) -> str:
    return f"N{r}{c}"


def main() -> None:
    rng = random.Random(SEED)
    out_dir = Path(__file__).resolve().parent

    # 1. Intersections with jittered coordinates.
    coords: dict[str, tuple[float, float]] = {}
    for r in range(ROWS):
        for c in range(COLS):
            x = c * SPACING_KM + rng.uniform(-0.22, 0.22)
            y = r * SPACING_KM + rng.uniform(-0.22, 0.22) + (RIVER_GAP_KM if r > RIVER_BELOW_ROW else 0.0)
            coords[node_id(r, c)] = (round(x, 3), round(y, 3))

    # 2. Candidate roads: grid neighbours (river crossings only at bridges) + diagonals.
    roads: set[tuple[str, str]] = set()
    for r in range(ROWS):
        for c in range(COLS):
            if c + 1 < COLS:
                roads.add((node_id(r, c), node_id(r, c + 1)))
            if r + 1 < ROWS:
                crosses_river = r == RIVER_BELOW_ROW
                if not crosses_river or c in BRIDGE_COLS:
                    roads.add((node_id(r, c), node_id(r + 1, c)))
    for _ in range(10):  # diagonal arterial roads, never across the river
        r, c = rng.randrange(ROWS - 1), rng.randrange(COLS - 1)
        if r == RIVER_BELOW_ROW:
            continue
        if rng.random() < 0.5:
            roads.add((node_id(r, c), node_id(r + 1, c + 1)))
        else:
            roads.add((node_id(r, c + 1), node_id(r + 1, c)))

    def connected(edge_set: set[tuple[str, str]]) -> bool:
        adj: dict[str, list[str]] = {n: [] for n in coords}
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        start = next(iter(coords))
        seen, queue = {start}, deque([start])
        while queue:
            for v in adj[queue.popleft()]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return len(seen) == len(coords)

    # 3. Remove ~12% of non-bridge roads, but never disconnect the city.
    bridge_roads = {(node_id(RIVER_BELOW_ROW, c), node_id(RIVER_BELOW_ROW + 1, c)) for c in BRIDGE_COLS}
    removable = sorted(roads - bridge_roads)
    rng.shuffle(removable)
    target_removals = int(0.12 * len(roads))
    removed = 0
    for road in removable:
        if removed >= target_removals:
            break
        trial = roads - {road}
        if connected(trial):
            roads = trial
            removed += 1

    # 4. Road lengths: straight line x curviness, rounded UP (keeps heuristic admissible).
    edges = []
    for u, v in sorted(roads):
        (x1, y1), (x2, y2) = coords[u], coords[v]
        straight = math.hypot(x1 - x2, y1 - y2)
        length = math.ceil(straight * rng.uniform(1.03, 1.30) * 100) / 100
        edges.append({"source": u, "target": v, "distance_km": length})

    # 5. Depot and named delivery locations (spread across both river banks).
    depot = node_id(*DEPOT_RC)
    candidates = [n for n in coords if n != depot]
    delivery_ids = rng.sample(candidates, len(DELIVERY_NAMES))
    names = {depot: "EcoRoute Central Depot"}
    names.update(dict(zip(delivery_ids, DELIVERY_NAMES)))

    nodes = []
    for nid, (x, y) in coords.items():
        kind = "depot" if nid == depot else "delivery" if nid in delivery_ids else "intersection"
        name = names.get(nid, f"Junction {nid[1]}-{nid[2]}")
        nodes.append({"id": nid, "name": name, "x": x, "y": y, "kind": kind})

    dataset = {
        "metadata": {
            "name": "EcoRoute Sample City (synthetic)",
            "description": (
                "Synthetic, reproducible road network generated by data/generate_sample_network.py "
                f"(seed {SEED}). Coordinates and road lengths are in kilometres. All place names are "
                "fictional. This is NOT collected real-world city data."
            ),
            "synthetic": True,
            "units": "km",
            "seed": SEED,
            # Purely decorative: the app shades this horizontal band as the river.
            "river_y_km": [RIVER_BELOW_ROW * SPACING_KM + 0.3, (RIVER_BELOW_ROW + 1) * SPACING_KM + RIVER_GAP_KM - 0.3],
        },
        "nodes": nodes,
        "edges": edges,
        "depot": depot,
        "deliveries": delivery_ids[:DEFAULT_SELECTION_SIZE],
    }

    (out_dir / "sample_city.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    with open(out_dir / "sample_nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "x", "y", "kind"])
        writer.writeheader()
        writer.writerows(nodes)
    with open(out_dir / "sample_edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "distance_km"])
        writer.writeheader()
        writer.writerows(edges)
    print(f"Wrote {len(nodes)} nodes, {len(edges)} roads, depot {depot}, {len(delivery_ids)} delivery locations to {out_dir}")


if __name__ == "__main__":
    main()
