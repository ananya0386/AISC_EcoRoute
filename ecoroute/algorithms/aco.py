"""Ant Colony Optimization (Ant System, Dorigo et al.) for ordering delivery stops.

Idea: many artificial "ants" build delivery routes at random, but the randomness
is biased by two things:

    tau[i][j]  pheromone   - "how good was going i -> j in previous good routes"  (learned)
    eta[i][j]  desirability = 1 / d(i, j)  - "j is close to i"                   (greedy hint)

Probability that an ant at stop i picks unvisited stop j:

                    tau[i][j]^alpha * eta[i][j]^beta
    P(i -> j) = -------------------------------------------
                 sum over unvisited k of (tau[i][k]^alpha * eta[i][k]^beta)

After all ants have built their routes in an iteration:

    1. Evaporation:  tau <- (1 - rho) * tau            (forget old information, avoid stagnation)
    2. Deposit:      tau[i][j] += Q / L_ant            for every leg i -> j in each ant's route
                     (shorter route L  =>  more pheromone  =>  more ants follow it next time)

Over the iterations pheromone accumulates on legs that belong to short routes.
The best route ever seen is tracked and returned. ACO is stochastic: it usually
finds very good routes but is NOT guaranteed to find the optimum.

d(i, j) here is the A*-computed shortest road distance (the distance matrix).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..tours import DistanceMatrixLike, tour_distance

MIN_PHEROMONE = 1e-12  # floor to prevent pheromone underflowing to exactly zero


@dataclass
class ACOParams:
    # Defaults chosen with scripts/parameter_sweep.py on the sample city. Q and tau_0
    # only matter relative to each other and to route length (deposit = Q / L km).
    n_ants: int = 20  # m: routes constructed per iteration
    n_iterations: int = 100  # how many times the colony builds routes and updates pheromone
    alpha: float = 1.0  # pheromone influence (0 = ignore what the colony learned)
    beta: float = 2.0  # heuristic influence (0 = ignore distances, pure pheromone)
    evaporation_rate: float = 0.1  # rho in (0, 1]: fraction of pheromone lost per iteration
    q: float = 1.0  # pheromone deposit constant Q
    initial_pheromone: float = 0.1  # tau_0, the same on every leg at the start

    def validate(self) -> None:
        if int(self.n_ants) < 1:
            raise ValueError("Number of ants must be at least 1.")
        if int(self.n_iterations) < 1:
            raise ValueError("Number of iterations must be at least 1.")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("Alpha and beta must be non-negative.")
        if self.alpha == 0 and self.beta == 0:
            raise ValueError("Alpha and beta cannot both be 0 (selection would be purely random).")
        if not 0 < self.evaporation_rate <= 1:
            raise ValueError("Evaporation rate must be in (0, 1].")
        if self.q <= 0:
            raise ValueError("Deposit constant Q must be positive.")
        if self.initial_pheromone <= 0:
            raise ValueError("Initial pheromone must be positive.")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ACOResult:
    best_tour: list[int]
    best_distance: float
    best_found_iteration: int  # 1-based iteration where the final best tour first appeared
    best_so_far_history: list[float]  # best distance found up to and including each iteration
    iteration_best_history: list[float]  # best distance among the ants of each iteration
    iteration_mean_history: list[float]  # mean distance of the ants of each iteration
    pheromone: np.ndarray  # final pheromone matrix (for visualisation)
    runtime_ms: float
    seed: int | None
    params: ACOParams = field(default_factory=ACOParams)


def _construct_tour(choice_info: np.ndarray, rng: np.random.Generator) -> list[int]:
    """One ant builds a complete route: depot -> every stop once -> depot.

    choice_info[i][j] = tau[i][j]^alpha * eta[i][j]^beta  (the numerator of P(i -> j)).
    """
    n = choice_info.shape[0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    tour = [0]
    current = 0
    for _ in range(n - 1):
        candidates = np.flatnonzero(unvisited)
        weights = choice_info[current, candidates]
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            chosen = int(rng.choice(candidates))  # numerical fallback: uniform choice
        else:
            # Roulette-wheel selection: stop j gets a slice of [0, total) proportional to its weight.
            spin = rng.random() * total
            index = int(np.searchsorted(np.cumsum(weights), spin, side="right"))
            chosen = int(candidates[min(index, len(candidates) - 1)])
        tour.append(chosen)
        unvisited[chosen] = False
        current = chosen
    tour.append(0)
    return tour


def ant_colony_optimization(matrix: DistanceMatrixLike, params: ACOParams | None = None, seed: int | None = None) -> ACOResult:
    params = params or ACOParams()
    params.validate()
    t0 = time.perf_counter()

    dist = np.asarray(matrix, dtype=float)
    n = dist.shape[0]
    if dist.shape != (n, n) or n < 2:
        raise ValueError("Distance matrix must be square with at least 2 locations.")
    symmetric = bool(np.allclose(dist, dist.T))

    # Heuristic desirability eta = 1/d (closer = more attractive). Diagonal is never used.
    with np.errstate(divide="ignore"):
        eta = np.where(dist > 0, 1.0 / np.maximum(dist, 1e-12), 0.0)
    eta_beta = eta ** params.beta  # constant for the whole run, so compute it once
    tau = np.full((n, n), params.initial_pheromone, dtype=float)  # pheromone initialisation
    rng = np.random.default_rng(seed)  # fixed seed -> reproducible results

    best_tour: list[int] = []
    best_distance = float("inf")
    best_found_iteration = 0
    best_so_far_history: list[float] = []
    iteration_best_history: list[float] = []
    iteration_mean_history: list[float] = []

    for iteration in range(1, int(params.n_iterations) + 1):
        # 1. Every ant constructs a route probabilistically. Pheromone only changes
        #    between iterations, so the selection weights are computed once per iteration.
        choice_info = (tau ** params.alpha) * eta_beta
        tours = [_construct_tour(choice_info, rng) for _ in range(int(params.n_ants))]
        lengths = [tour_distance(dist, t) for t in tours]

        # 2. Track the best solution.
        best_index = int(np.argmin(lengths))
        if lengths[best_index] < best_distance - 1e-9:
            best_distance = lengths[best_index]
            best_tour = tours[best_index]
            best_found_iteration = iteration

        # 3. Evaporation on every leg.
        tau *= 1.0 - params.evaporation_rate

        # 4. Deposit: each ant adds Q / L on the legs it used (quality-proportional).
        for tour, length in zip(tours, lengths):
            deposit = params.q / length if length > 0 else params.q
            for a, b in zip(tour, tour[1:]):
                tau[a, b] += deposit
                if symmetric:  # undirected roads: i -> j and j -> i are the same leg
                    tau[b, a] += deposit
        np.maximum(tau, MIN_PHEROMONE, out=tau)

        best_so_far_history.append(best_distance)
        iteration_best_history.append(lengths[best_index])
        iteration_mean_history.append(float(np.mean(lengths)))

    return ACOResult(
        best_tour=best_tour,
        best_distance=best_distance,
        best_found_iteration=best_found_iteration,
        best_so_far_history=best_so_far_history,
        iteration_best_history=iteration_best_history,
        iteration_mean_history=iteration_mean_history,
        pheromone=tau,
        runtime_ms=(time.perf_counter() - t0) * 1000,
        seed=seed,
        params=params,
    )
