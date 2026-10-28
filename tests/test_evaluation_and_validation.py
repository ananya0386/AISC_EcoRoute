import json
import math

import pytest

from ecoroute.algorithms import ACOParams
from ecoroute.evaluation import VehicleConfig, estimate_fuel, gap_to_optimum, percentage_reduction, run_experiments
from ecoroute.graph_utils import DatasetError, InputError, load_dataset_csv, load_dataset_json, parse_dataset, validate_selection
from ecoroute.pipeline import optimize_routes
from ecoroute.tours import is_valid_tour

from .conftest import DATA

FAST = ACOParams(n_ants=8, n_iterations=15)


# ----------------------------------------------------------------- metrics
def test_percentage_reduction():
    assert percentage_reduction(50, 40) == pytest.approx(20.0)
    assert percentage_reduction(40, 50) == pytest.approx(-25.0)  # ACO worse -> negative, reported honestly
    assert percentage_reduction(30, 30) == 0
    assert percentage_reduction(0, 0) is None  # zero baseline handled safely


def test_gap_to_optimum():
    assert gap_to_optimum(40, 44) == pytest.approx(10.0)
    assert gap_to_optimum(0, 1) is None


def test_fuel_estimate():
    litres, cost = estimate_fuel(60, VehicleConfig(km_per_litre=12, fuel_price_per_litre=100))
    assert litres == pytest.approx(5.0)
    assert cost == pytest.approx(500.0)


@pytest.mark.parametrize("vehicle", [VehicleConfig(km_per_litre=0), VehicleConfig(km_per_litre=-3), VehicleConfig(fuel_price_per_litre=-1)])
def test_invalid_vehicle_rejected(vehicle):
    with pytest.raises(InputError):
        estimate_fuel(10, vehicle)


# -------------------------------------------------------------- validation
def test_selection_validation(small_graph):
    with pytest.raises(InputError, match="not a location"):
        validate_selection(small_graph, "Q", ["B", "C"])
    with pytest.raises(InputError, match="not in the road network"):
        validate_selection(small_graph, "A", ["B", "Q"])
    with pytest.raises(InputError, match="cannot also be a delivery"):
        validate_selection(small_graph, "A", ["A", "B"])
    with pytest.raises(InputError, match="only once"):
        validate_selection(small_graph, "A", ["B", "B"])
    with pytest.raises(InputError, match="at least"):
        validate_selection(small_graph, "A", ["B"])
    with pytest.raises(InputError, match="disconnected"):
        validate_selection(small_graph, "A", ["B", "Z"])
    validate_selection(small_graph, "A", ["B", "E"])  # valid -> no exception


def test_pipeline_reports_disconnected_graph(small_graph):
    with pytest.raises(InputError):
        optimize_routes(small_graph, "A", ["C", "Z"], FAST)


@pytest.mark.parametrize("data, message", [
    ({"nodes": [], "edges": []}, "non-empty 'nodes'"),
    ({"nodes": [{"id": "A", "x": 0, "y": 0}], "edges": [{"source": "A", "target": "B", "distance_km": 1}]}, "unknown node"),
    ({"nodes": [{"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 1, "y": 0}], "edges": [{"source": "A", "target": "B", "distance_km": -2}]}, "must be > 0"),
    ({"nodes": [{"id": "A", "x": 0, "y": 0}, {"id": "A", "x": 1, "y": 0}], "edges": [{"source": "A", "target": "A", "distance_km": 1}]}, "Duplicate"),
    ({"nodes": [{"id": "A", "x": "abc", "y": 0}], "edges": [{"source": "A", "target": "A", "distance_km": 1}]}, "must be a number"),
    ({"nodes": [{"id": "A", "y": 0}], "edges": [{"source": "A", "target": "A", "distance_km": 1}]}, "coordinates"),
])
def test_malformed_datasets_rejected(data, message):
    with pytest.raises(DatasetError, match=message):
        parse_dataset(data)


def test_invalid_json_bytes_rejected():
    with pytest.raises(DatasetError, match="Invalid JSON"):
        load_dataset_json(b"{not json")


def test_csv_and_json_sample_are_identical():
    from_json = load_dataset_json(DATA / "sample_city.json")
    from_csv = load_dataset_csv(DATA / "sample_nodes.csv", DATA / "sample_edges.csv")
    assert from_json.graph.num_nodes == from_csv.graph.num_nodes
    assert sorted(from_json.graph.edges()) == sorted(from_csv.graph.edges())
    assert from_json.depot == from_csv.depot


def test_sample_dataset_is_labelled_synthetic():
    meta = json.loads((DATA / "sample_city.json").read_text(encoding="utf-8"))["metadata"]
    assert meta["synthetic"] is True and "NOT" in meta["description"]


# ------------------------------------------------------------- end to end
def test_pipeline_end_to_end_is_consistent_and_reproducible(sample_dataset):
    g, depot, stops = sample_dataset.graph, sample_dataset.depot, sample_dataset.deliveries[:7]
    vehicle = VehicleConfig(km_per_litre=10, fuel_price_per_litre=90)
    r1 = optimize_routes(g, depot, stops, FAST, seed=5, vehicle=vehicle, compute_exact=True)
    r2 = optimize_routes(g, depot, stops, FAST, seed=5, vehicle=vehicle, compute_exact=True)

    n = len(stops) + 1
    for route in r1.routes:
        assert is_valid_tour(route.tour, n)
        assert sorted(route.stop_order) == sorted(stops)
        assert route.fuel_litres == pytest.approx(route.distance_km / 10)
        assert route.fuel_cost == pytest.approx(route.fuel_litres * 90)
    assert r1.exact is not None
    assert r1.exact.distance_km <= r1.aco.distance_km + 1e-9
    assert r1.exact.distance_km <= r1.nearest_neighbor.distance_km + 1e-9
    assert r1.reduction_pct == pytest.approx(percentage_reduction(r1.nearest_neighbor.distance_km, r1.aco.distance_km))
    assert r1.aco.tour == r2.aco.tour  # same seed -> same answer


def test_experiments_produce_tables(sample_dataset):
    g, depot = sample_dataset.graph, sample_dataset.depot
    pool = [n for n in g.nodes if n != depot]
    out = run_experiments(g, depot, pool, sizes=[4, 6], instances_per_size=2, trials_per_instance=3, aco_params=FAST)
    assert len(out["runs"]) == 2 * 2 * 3
    assert list(out["sizes"]["stops"]) == [4, 6]
    assert (out["runs"]["aco_km"] >= out["runs"]["optimal_km"] - 1e-9).all()
    assert out["instances"]["trials"].eq(3).all()
    assert not math.isnan(out["sizes"]["mean_reduction_pct"].iloc[0])


def test_experiments_reject_bad_sizes(sample_dataset):
    g, depot = sample_dataset.graph, sample_dataset.depot
    with pytest.raises(InputError):
        run_experiments(g, depot, list(g.nodes), sizes=[1])
    with pytest.raises(InputError):
        run_experiments(g, depot, list(g.nodes), sizes=[500])
