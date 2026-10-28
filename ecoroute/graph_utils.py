"""Road-graph representation, dataset loading/validation and path helpers.

The road network is an undirected weighted graph:
    * nodes  = intersections / addresses, each with planar (x, y) coordinates in km
    * edges  = road segments, each with a road length in km (the edge weight)

We deliberately use a plain adjacency-list dictionary instead of a graph library
so that every step of A* (in algorithms/astar.py) is visible in our own code.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

VALID_KINDS = {"depot", "delivery", "intersection"}

# Minimum number of delivery stops for a meaningful ordering problem.
# With 1 stop there is only one possible route; with 2 stops on an undirected
# graph both orders have identical length. We require at least 2 so the
# pipeline runs, and the UI recommends 4+ for an interesting comparison.
MIN_DELIVERIES = 2


class DatasetError(ValueError):
    """Raised when a road-network file is malformed."""


class InputError(ValueError):
    """Raised when the user's depot/delivery selection or parameters are invalid."""


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    x: float  # km, planar coordinate
    y: float  # km, planar coordinate
    kind: str = "intersection"  # "depot", "delivery" or "intersection"


class RoadGraph:
    """Undirected weighted graph stored as an adjacency list.

    adjacency[u][v] = road length (km) of the segment u <-> v.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.adjacency: dict[str, dict[str, float]] = {}

    # ----------------------------------------------------------------- building
    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise DatasetError(f"Duplicate node id '{node.id}'.")
        self.nodes[node.id] = node
        self.adjacency[node.id] = {}

    def add_edge(self, u: str, v: str, distance_km: float) -> None:
        if u not in self.nodes or v not in self.nodes:
            missing = u if u not in self.nodes else v
            raise DatasetError(f"Road {u} - {v} refers to unknown node '{missing}'.")
        if u == v:
            raise DatasetError(f"Road {u} - {v} is a self-loop; remove it.")
        if not math.isfinite(distance_km) or distance_km <= 0:
            raise DatasetError(f"Road {u} - {v} has invalid distance {distance_km!r}; it must be > 0.")
        # If the same road appears twice we keep the shorter one.
        current = self.adjacency[u].get(v)
        if current is None or distance_km < current:
            self.adjacency[u][v] = distance_km
            self.adjacency[v][u] = distance_km

    # ------------------------------------------------------------------ queries
    def neighbors(self, u: str) -> Iterable[tuple[str, float]]:
        return self.adjacency[u].items()

    def edge_weight(self, u: str, v: str) -> float:
        return self.adjacency[u][v]

    def has_edge(self, u: str, v: str) -> bool:
        return v in self.adjacency.get(u, {})

    def edges(self) -> list[tuple[str, str, float]]:
        """Each undirected edge once, as (u, v, weight) with u < v."""
        return [(u, v, w) for u, nbrs in self.adjacency.items() for v, w in nbrs.items() if u < v]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self.adjacency.values()) // 2

    def euclidean(self, u: str, v: str) -> float:
        a, b = self.nodes[u], self.nodes[v]
        return math.hypot(a.x - b.x, a.y - b.y)

    def reachable_from(self, start: str) -> set[str]:
        """Breadth-first search: every node reachable from `start` (used for validation)."""
        seen = {start}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self.adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return seen

    def heuristic_scale(self) -> float:
        """Largest k <= 1 such that  k * straight_line(u, v) <= road_length(u, v)  for every edge.

        Why this matters: A* is only guaranteed optimal if h(n) never overestimates
        the true remaining cost. If every road is at least k times its straight-line
        length, then (by the triangle inequality) every *path* from n to the goal is
        at least k * straight_line(n, goal). So h(n) = k * straight_line(n, goal) is
        admissible (and also consistent). For realistic road data k = 1.
        """
        k = 1.0
        for u, v, w in self.edges():
            straight = self.euclidean(u, v)
            if straight > 0:
                k = min(k, w / straight)
        return k


@dataclass
class Dataset:
    graph: RoadGraph
    depot: str | None
    deliveries: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", "Custom road network"))


# --------------------------------------------------------------------- loading
def _to_float(value: Any, what: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise DatasetError(f"{what} must be a number, got {value!r}.") from None
    if not math.isfinite(result):
        raise DatasetError(f"{what} must be a finite number, got {value!r}.")
    return result


def parse_dataset(data: dict[str, Any]) -> Dataset:
    """Build a Dataset from the JSON-style dictionary format (see data/sample_city.json)."""
    if not isinstance(data, dict):
        raise DatasetError("Dataset must be a JSON object with 'nodes' and 'edges'.")
    raw_nodes = data.get("nodes")
    raw_edges = data.get("edges")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise DatasetError("Dataset needs a non-empty 'nodes' list.")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise DatasetError("Dataset needs a non-empty 'edges' list.")

    graph = RoadGraph()
    for i, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict) or "id" not in raw:
            raise DatasetError(f"Node #{i + 1} is missing an 'id'.")
        node_id = str(raw["id"]).strip()
        if not node_id:
            raise DatasetError(f"Node #{i + 1} has an empty 'id'.")
        if "x" not in raw or "y" not in raw:
            raise DatasetError(f"Node '{node_id}' needs 'x' and 'y' coordinates (km).")
        kind = str(raw.get("kind") or "intersection").strip().lower()
        if kind not in VALID_KINDS:
            raise DatasetError(f"Node '{node_id}' has unknown kind '{kind}'. Use one of {sorted(VALID_KINDS)}.")
        graph.add_node(
            Node(
                id=node_id,
                name=str(raw.get("name") or node_id).strip(),
                x=_to_float(raw["x"], f"x of node '{node_id}'"),
                y=_to_float(raw["y"], f"y of node '{node_id}'"),
                kind=kind,
            )
        )

    for i, raw in enumerate(raw_edges):
        if not isinstance(raw, dict) or "source" not in raw or "target" not in raw:
            raise DatasetError(f"Road #{i + 1} needs 'source' and 'target'.")
        if "distance_km" not in raw:
            raise DatasetError(f"Road #{i + 1} ({raw['source']} - {raw['target']}) needs 'distance_km'.")
        u, v = str(raw["source"]).strip(), str(raw["target"]).strip()
        graph.add_edge(u, v, _to_float(raw["distance_km"], f"distance_km of road {u} - {v}"))

    depot = data.get("depot")
    if depot is None:
        depots = [n.id for n in graph.nodes.values() if n.kind == "depot"]
        depot = depots[0] if depots else None
    elif str(depot) not in graph.nodes:
        raise DatasetError(f"Default depot '{depot}' is not a node in the dataset.")

    deliveries = data.get("deliveries")
    if deliveries is None:
        deliveries = [n.id for n in graph.nodes.values() if n.kind == "delivery"]
    unknown = [d for d in deliveries if str(d) not in graph.nodes]
    if unknown:
        raise DatasetError(f"Default deliveries refer to unknown nodes: {unknown}.")

    return Dataset(
        graph=graph,
        depot=str(depot) if depot is not None else None,
        deliveries=[str(d) for d in deliveries],
        metadata=dict(data.get("metadata") or {}),
    )


def load_dataset_json(source: str | Path | bytes) -> Dataset:
    """Load a dataset from a JSON file path or raw JSON bytes (e.g. a Streamlit upload)."""
    try:
        if isinstance(source, (bytes, bytearray)):
            data = json.loads(source.decode("utf-8-sig"))
        else:
            data = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON: {exc}") from None
    return parse_dataset(data)


def _read_csv_rows(source: str | Path | bytes, label: str) -> list[dict[str, str]]:
    text = source.decode("utf-8-sig") if isinstance(source, (bytes, bytearray)) else Path(source).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise DatasetError(f"{label} CSV is empty.")
    reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]
    return [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]


def load_dataset_csv(nodes_source: str | Path | bytes, edges_source: str | Path | bytes, name: str = "Uploaded CSV network") -> Dataset:
    """Load from two CSVs.

    nodes.csv : id,name,x,y,kind      (kind = depot | delivery | intersection)
    edges.csv : source,target,distance_km
    """
    nodes = _read_csv_rows(nodes_source, "Nodes")
    edges = _read_csv_rows(edges_source, "Edges")
    for col in ("id", "x", "y"):
        if nodes and col not in nodes[0]:
            raise DatasetError(f"Nodes CSV is missing column '{col}'. Expected: id,name,x,y,kind")
    for col in ("source", "target", "distance_km"):
        if edges and col not in edges[0]:
            raise DatasetError(f"Edges CSV is missing column '{col}'. Expected: source,target,distance_km")
    return parse_dataset({"metadata": {"name": name}, "nodes": nodes, "edges": edges})


def dataset_to_dict(dataset: Dataset) -> dict[str, Any]:
    g = dataset.graph
    return {
        "metadata": dataset.metadata,
        "nodes": [{"id": n.id, "name": n.name, "x": n.x, "y": n.y, "kind": n.kind} for n in g.nodes.values()],
        "edges": [{"source": u, "target": v, "distance_km": w} for u, v, w in g.edges()],
        "depot": dataset.depot,
        "deliveries": dataset.deliveries,
    }


# ------------------------------------------------------------------ validation
def validate_selection(graph: RoadGraph, depot: str, deliveries: list[str]) -> None:
    """Check the user's depot/delivery choice before running any algorithm."""
    if not depot:
        raise InputError("Please choose a depot.")
    if depot not in graph.nodes:
        raise InputError(f"Depot '{depot}' is not a location in this road network.")
    unknown = [d for d in deliveries if d not in graph.nodes]
    if unknown:
        raise InputError(f"These delivery locations are not in the road network: {unknown}.")
    if depot in deliveries:
        raise InputError(f"The depot ({graph.nodes[depot].name}) cannot also be a delivery stop.")
    if len(set(deliveries)) != len(deliveries):
        raise InputError("Each delivery location can be selected only once.")
    if len(deliveries) < MIN_DELIVERIES:
        raise InputError(f"Select at least {MIN_DELIVERIES} delivery locations (you selected {len(deliveries)}).")

    # Cheap connectivity check (BFS) so we can give a clear message before running A*.
    reachable = graph.reachable_from(depot)
    stranded = [d for d in deliveries if d not in reachable]
    if stranded:
        names = ", ".join(f"{graph.nodes[d].name} ({d})" for d in stranded)
        raise InputError(
            f"No road connects the depot to: {names}. The road network is disconnected, "
            "so no complete delivery route exists. Remove these stops or add roads."
        )


# ------------------------------------------------------------- path utilities
def reconstruct_path(came_from: dict[str, str], start: str, goal: str) -> list[str]:
    """Walk predecessor links backwards from goal to start, then reverse."""
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def path_distance(graph: RoadGraph, path: list[str]) -> float:
    """Sum of edge weights along a node path. Raises if two consecutive nodes are not joined by a road."""
    total = 0.0
    for u, v in zip(path, path[1:]):
        if not graph.has_edge(u, v):
            raise ValueError(f"Path uses non-existent road {u} - {v}.")
        total += graph.edge_weight(u, v)
    return total
