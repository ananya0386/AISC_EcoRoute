"""Step-by-step decision data for the live simulation (no UI code here).

For every leg of a finished tour we record *what the method was choosing between*:

* Nearest neighbour: the unvisited stops and their road distances; it picks the closest.
* ACO: the same stops with the probability P(i -> j) = tau^alpha * (1/d)^beta / sum(...),
  computed from the FINAL pheromone matrix. It shows how strongly the colony prefers each
  stop after learning (an ant picks randomly with these odds, so the chosen stop is
  usually, but not always, the most likely one).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np

from .algorithms.aco import ACOResult
from .graph_utils import RoadGraph
from .route_utils import DistanceMatrix

MAX_FRAMES_PER_LEG = 8


@dataclass
class Option:
    idx: int
    name: str
    distance_km: float
    probability: float | None = None  # ACO only


@dataclass
class Decision:
    step: int
    from_idx: int
    to_idx: int
    from_name: str
    to_name: str
    leg_km: float
    options: list[Option]  # candidates that were still unvisited (empty on the final return leg)
    reason: str


def _name(graph: RoadGraph, dm: DistanceMatrix, idx: int) -> str:
    return graph.nodes[dm.locations[idx]].name


def nn_decisions(graph: RoadGraph, dm: DistanceMatrix, tour: list[int]) -> list[Decision]:
    out = []
    for step, (a, b) in enumerate(zip(tour, tour[1:])):
        unvisited = [j for j in tour[step + 1:-1]]  # stops not yet visited (excludes final depot)
        options = [Option(j, _name(graph, dm, j), dm.matrix[a][j]) for j in unvisited]
        if options:
            reason = f"Closest unvisited stop is {_name(graph, dm, b)} ({dm.matrix[a][b]:.2f} km), so go there."
        else:
            reason = "Every stop is done, so drive back to the depot."
        out.append(Decision(step, a, b, _name(graph, dm, a), _name(graph, dm, b), dm.matrix[a][b], options, reason))
    return out


def aco_decisions(graph: RoadGraph, dm: DistanceMatrix, tour: list[int], aco: ACOResult) -> list[Decision]:
    tau = np.asarray(aco.pheromone, dtype=float)
    p = aco.params
    out = []
    for step, (a, b) in enumerate(zip(tour, tour[1:])):
        unvisited = [j for j in tour[step + 1:-1]]
        weights = {j: (tau[a, j] ** p.alpha) * ((1.0 / dm.matrix[a][j]) ** p.beta) for j in unvisited}
        total = sum(weights.values())
        options = [Option(j, _name(graph, dm, j), dm.matrix[a][j], weights[j] / total if total > 0 else None) for j in unvisited]
        if options:
            chosen = next(o for o in options if o.idx == b)
            reason = (f"The colony prefers {chosen.name} here ({chosen.probability:.0%} chance): "
                      "it is close and its pheromone is strong.")
        else:
            reason = "Every stop is done, so drive back to the depot."
        out.append(Decision(step, a, b, _name(graph, dm, a), _name(graph, dm, b), dm.matrix[a][b], options, reason))
    return out


def leg_points(graph: RoadGraph, dm: DistanceMatrix, a: int, b: int) -> list[tuple[float, float]]:
    """Coordinates the truck passes on the A* road path a -> b, thinned to a few animation frames."""
    pts = [(graph.nodes[n].x, graph.nodes[n].y) for n in dm.paths[(a, b)]]
    stride = max(1, len(pts) // MAX_FRAMES_PER_LEG)
    thinned = pts[::stride]
    if thinned[-1] != pts[-1]:
        thinned.append(pts[-1])
    return thinned


@dataclass
class SideState:
    """What one truck's panel shows at one animation frame."""
    trail: list[tuple[float, float]]
    truck: tuple[float, float]
    visited: dict[str, int]  # node id -> visiting order number
    target: str | None  # node id the truck is heading to
    driven: float  # km driven so far
    decision: Decision | None


def side_state(dm: DistanceMatrix, tour: list[int], decisions: list[Decision], legs: list[list[tuple[float, float]]],
               step: int, frame: int, thinking: bool) -> SideState:
    """State of one truck. step = -1 means 'not started'; thinking = paused at the start of a leg, deciding."""
    if step < 0:
        return SideState([], legs[0][0], {}, None, 0.0, None)
    leg = legs[step]
    fi = 0 if thinking else min(frame, len(leg) - 1)
    arrived = (not thinking) and fi >= len(leg) - 1
    trail = [pt for earlier in legs[:step] for pt in earlier] + leg[: fi + 1]
    if arrived:
        part = decisions[step].leg_km
    elif thinking:
        part = 0.0
    else:
        part = decisions[step].leg_km * (fi + 1) / len(leg)
    driven = sum(d.leg_km for d in decisions[:step]) + part
    upto = step + (1 if arrived else 0)
    visited = {dm.locations[idx]: pos for pos, idx in enumerate(tour[: upto + 1]) if idx != 0}
    return SideState(trail, leg[fi], visited, dm.locations[tour[step + 1]], driven, decisions[step])


def timeline(frames_per_leg: list[int]) -> list[tuple[int, int, bool]]:
    """(step, frame, thinking) for every animation frame; 'thinking' frames repeat so the pause is visible."""
    out: list[tuple[int, int, bool]] = [(-1, 0, False)] * 2
    for step, n in enumerate(frames_per_leg):
        out += [(step, 0, True)] * 3
        out += [(step, f, False) for f in range(n)]
    out += [(len(frames_per_leg) - 1, frames_per_leg[-1] - 1, False)] * 3
    return out


def decision_lines(key: str, state: SideState) -> list[str]:
    """Text for the 'what is being calculated' panel (plotly mini-HTML; padded with no-break spaces)."""
    d = state.decision
    if d is None:
        return ["Truck is at the depot.", "Press Play to start."]
    if not d.options:
        return ["<b>All stops done.</b>", "Driving back to the depot."]
    rows = sorted(d.options, key=(lambda o: -(o.probability or 0)) if key == "aco" else (lambda o: o.distance_km))
    top = rows[:5]
    if not any(o.idx == d.to_idx for o in top):
        top.append(next(o for o in rows if o.idx == d.to_idx))
    lines = [f"<b>At {d.from_name[:18]}: next stop?</b>"]
    for o in top:
        row = f"{'>' if o.idx == d.to_idx else ' '} {o.name[:18]:<18} {o.distance_km:5.2f} km"
        if key == "aco":
            row += f"  {o.probability:>4.0%}"
        lines.append(row.replace(" ", " "))
    lines += textwrap.wrap(d.reason, 38)
    return lines
