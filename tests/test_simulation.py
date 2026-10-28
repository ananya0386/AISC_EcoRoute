import pytest

from ecoroute.algorithms import ACOParams
from ecoroute.pipeline import optimize_routes
from ecoroute.simulation import aco_decisions, leg_points, nn_decisions


@pytest.fixture(scope="module")
def result(sample_dataset):
    return optimize_routes(sample_dataset.graph, sample_dataset.depot, sample_dataset.deliveries[:6],
                           ACOParams(n_ants=8, n_iterations=15), seed=3)


def test_nn_always_picks_the_closest_unvisited_stop(sample_dataset, result):
    g, dm, tour = sample_dataset.graph, result.distance_matrix, result.nearest_neighbor.tour
    decisions = nn_decisions(g, dm, tour)
    assert len(decisions) == len(tour) - 1
    for d in decisions[:-1]:
        assert d.to_idx == min(d.options, key=lambda o: (o.distance_km, o.idx)).idx
    assert decisions[-1].options == [] and decisions[-1].to_idx == 0
    assert sum(d.leg_km for d in decisions) == pytest.approx(result.nearest_neighbor.distance_km)


def test_aco_probabilities_sum_to_one_and_include_chosen_stop(sample_dataset, result):
    g, dm, tour = sample_dataset.graph, result.distance_matrix, result.aco.tour
    decisions = aco_decisions(g, dm, tour, result.aco_details)
    for d in decisions[:-1]:
        assert sum(o.probability for o in d.options) == pytest.approx(1.0)
        assert d.to_idx in {o.idx for o in d.options}
    assert sum(d.leg_km for d in decisions) == pytest.approx(result.aco.distance_km)


def test_leg_points_start_and_end_at_the_leg_ends(sample_dataset, result):
    g, dm = sample_dataset.graph, result.distance_matrix
    pts = leg_points(g, dm, 0, 3)
    assert pts[0] == (g.nodes[dm.locations[0]].x, g.nodes[dm.locations[0]].y)
    assert pts[-1] == (g.nodes[dm.locations[3]].x, g.nodes[dm.locations[3]].y)


def test_timeline_and_final_state_match_the_routes(sample_dataset, result):
    from ecoroute.simulation import side_state, timeline

    g, dm = sample_dataset.graph, result.distance_matrix
    for route, decisions in ((result.nearest_neighbor, nn_decisions(g, dm, result.nearest_neighbor.tour)),
                             (result.aco, aco_decisions(g, dm, result.aco.tour, result.aco_details))):
        legs = [leg_points(g, dm, a, b) for a, b in zip(route.tour, route.tour[1:])]
        frames = [len(leg) for leg in legs]
        steps = timeline(frames)
        assert steps[0][0] == -1  # starts "not started"
        start = side_state(dm, route.tour, decisions, legs, *steps[0])
        end = side_state(dm, route.tour, decisions, legs, *steps[-1])
        assert start.driven == 0 and start.visited == {}
        assert end.driven == pytest.approx(route.distance_km)
        assert len(end.visited) == 6 and sorted(end.visited.values()) == [1, 2, 3, 4, 5, 6]


def test_animation_figure_has_one_frame_per_timeline_step(sample_dataset, result):
    from ecoroute.simulation import timeline
    from ecoroute.visualization import simulation_animation

    g, dm = sample_dataset.graph, result.distance_matrix
    sides = {}
    for key, route, decisions in (("nn", result.nearest_neighbor, nn_decisions(g, dm, result.nearest_neighbor.tour)),
                                  ("aco", result.aco, aco_decisions(g, dm, result.aco.tour, result.aco_details))):
        sides[key] = {"route": route, "decisions": decisions,
                      "legs": [leg_points(g, dm, a, b) for a, b in zip(route.tour, route.tour[1:])]}
    fig = simulation_animation(g, result, sides, 200, sample_dataset.metadata.get("river_y_km"))
    n_legs = len(sides["nn"]["legs"])
    expected = len(timeline([max(len(sides["nn"]["legs"][k]), len(sides["aco"]["legs"][k])) for k in range(n_legs)]))
    assert len(fig.frames) == expected
    assert len(fig.data) == 14
