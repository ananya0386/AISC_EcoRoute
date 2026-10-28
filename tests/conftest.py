from pathlib import Path

import pytest

from ecoroute.graph_utils import Node, RoadGraph, load_dataset_json

DATA = Path(__file__).resolve().parents[1] / "data"


def make_graph(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str, float]]) -> RoadGraph:
    g = RoadGraph()
    for node_id, (x, y) in nodes.items():
        g.add_node(Node(node_id, node_id, x, y))
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


@pytest.fixture
def small_graph() -> RoadGraph:
    """
        A --1-- B --1-- C
        |               |
        4               1
        |               |
        D ------6------ E        plus an isolated node Z (unreachable)

    Shortest A -> E is A-B-C-E = 3 (not A-D-E = 10).
    """
    return make_graph(
        {"A": (0, 1), "B": (1, 1), "C": (2, 1), "D": (0, 0), "E": (2, 0), "Z": (9, 9)},
        [("A", "B", 1), ("B", "C", 1), ("C", "E", 1), ("A", "D", 4), ("D", "E", 6)],
    )


@pytest.fixture(scope="session")
def sample_dataset():
    return load_dataset_json(DATA / "sample_city.json")
