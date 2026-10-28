# EcoRoute: Intelligent Multi-Stop Delivery Route Optimization Using A* Search and Ant Colony Optimization

**Course:** Artificial Intelligence and Soft Computing (AISC). B.Tech Computer Science and Engineering, Manipal University Jaipur
**Project type:** Course project (report draft)
**Author:** _[your name, registration number]_
**Supervisor:** _[faculty name]_

> All experimental numbers in this report were produced by the code in this repository (`scripts/run_experiments.py`, `scripts/parameter_sweep.py`). The raw outputs are in `results/`. The road network is **synthetic** (see §5.1).

---

## Abstract

Delivery vehicles must visit several customers and return to the depot. Minimising total distance requires two decisions: the road path between any two locations, and the order in which to visit them. EcoRoute separates these decisions. **A\* Search** with an admissible straight-line heuristic computes exact shortest road paths between every pair of selected locations, producing a distance matrix. **Ant Colony Optimization (Ant System)** then searches for a short visiting order on that matrix. A **nearest-neighbour** heuristic serves as the baseline, and an **exhaustive search** gives the true optimum for small instances as an evaluation reference. Both orders are expanded back into road paths and reported with total distance, estimated fuel use, estimated fuel cost and runtime. All algorithms are implemented from scratch in Python, with a Streamlit interface for interactive use.

On a reproducible synthetic 63-intersection network, across 35 random delivery sets of 5–25 stops and 10 random seeds each (350 ACO runs), ACO produced a shorter route than nearest neighbour in 330 runs and an equal route in 20 runs. It was never longer. The mean distance reduction was 11.71%. On the 15 delivery sets small enough to solve exactly (5–8 stops), ACO found the proven optimum in 145 of 150 runs (worst gap 3.78%), while nearest neighbour was optimal on only 2 of the 15 sets (mean gaps 7.2–13.1%).

---

## 1. Introduction

Last-mile delivery is a major part of logistics cost, and distance driven is directly linked to fuel use and emissions. A dispatcher who has a list of addresses must decide a route. A common manual strategy is "always go to the nearest remaining address". It is simple but can be far from optimal.

The underlying combinatorial problem is the **Travelling Salesman Problem (TSP)**: visit *m* locations once each and return to the start with minimum total cost. TSP is NP-hard, and the number of possible orders, *m!*, grows explosively. On a real road network there is an additional layer: the "distance" between two addresses is itself the result of a shortest-path search over many intersections.

This project builds a working system that addresses both layers with two classic AI techniques from the AISC syllabus: **informed search (A\*)** for paths and **swarm intelligence (ACO)** for ordering. It then measures how much the optimisation actually helps.

## 2. Problem statement

Given:
- a weighted, undirected road graph G = (V, E), where each edge has a road length in km and each node has planar coordinates;
- a depot *s ∈ V*;
- a set of *m ≥ 2* distinct delivery locations D ⊂ V \ {s};
- vehicle assumptions: efficiency η_v (km/L) and fuel price *p* (₹/L);

find a closed route that starts at *s*, visits every location in D exactly once, returns to *s*, and has **minimum total road distance**. Report the route, its distance, estimated fuel, estimated fuel cost and runtime, and compare it with a nearest-neighbour baseline.

**Scope (version 1):** one vehicle; no capacity, time windows, live traffic or dynamic rerouting; two-way roads.

## 3. Objectives

1. Represent a road network as a weighted graph and load it from a built-in sample or from user CSV/JSON files, with validation.
2. Implement A\* search with an admissible heuristic, returning the path, its distance and search statistics, and handling unreachable goals.
3. Build a distance matrix of A\* shortest-path distances between the depot and all delivery locations.
4. Implement a nearest-neighbour baseline and Ant Colony Optimization on the same matrix.
5. Expand each visiting order into its full road route and compute distance, estimated fuel, estimated cost and runtime.
6. Evaluate ACO against the baseline (and against the exact optimum on small instances) over many random instances and seeds.
7. Provide an interactive web application that visualises routes, convergence and experiment results and explains them.

## 4. Mathematical formulation

Let the selected locations be indexed 0 … m, with 0 the depot and 1 … m the delivery stops. Let *d(i, j)* be the **shortest road distance** from location *i* to location *j* in G, computed by A\*. Because roads are two-way, d(i, j) = d(j, i).

A route is a permutation π = (π₁, …, πₘ) of {1, …, m}. Its length is

  **L(π) = d(0, π₁) + Σᵢ₌₁ᵐ⁻¹ d(πᵢ, πᵢ₊₁) + d(πₘ, 0)**

and the optimisation problem is **min over π of L(π)**.

Derived quantities (estimates, not optimised separately):

| Quantity | Formula |
|---|---|
| Estimated fuel (L) | L(π) / η_v |
| Estimated fuel cost (₹) | (L(π) / η_v) × p |
| Distance reduction (%) | (L_NN − L_ACO) / L_NN × 100, undefined if L_NN = 0 |
| Gap to optimum (%) | (L_method − L*) / L* × 100 |

With a constant km/L, minimising distance is equivalent to minimising this estimated fuel. Real fuel consumption depends on factors the model excludes (speed, traffic, load, gradient, idling), so fuel savings are **inferred from the simplified model**, not measured.

## 5. Methodology

### 5.1 Dataset

The built-in network, "EcoRoute Sample City (synthetic)", is generated by `data/generate_sample_network.py` with a fixed seed (42). It is **not real city data**, and all place names are fictional.

- 63 intersections on a 9 × 7 grid with about 1 km spacing, each jittered by up to ±0.22 km.
- A river between rows 3 and 4 (0.6 km extra gap) that can be crossed only at **3 bridges**.
- 12% of non-bridge grid roads removed (never disconnecting the network), and 10 diagonal roads added.
- Road length = straight-line length × a random factor in [1.03, 1.30], **rounded up** to 0.01 km. This guarantees every road is at least as long as its straight line.
- Result: 97 roads, total length 119.8 km, one depot ("EcoRoute Central Depot", node N14) and 20 named delivery points. Any intersection can also be chosen as a delivery address.

The same network is exported as `sample_city.json`, `sample_nodes.csv` and `sample_edges.csv`, which also serve as templates for user data. The river makes straight-line proximity misleading, which is precisely the situation in which road-distance computation (A\*) matters.

### 5.2 Workflow

1. Load or construct the road graph and validate it (unique ids, numeric coordinates, known endpoints, positive distances).
2. Select the depot and delivery stops; validate them (existence, no duplicates, depot not a stop, ≥ 2 stops, reachability by BFS).
3. Run A\* for each unordered pair of the n = m + 1 locations (n(n−1)/2 searches), storing distances and paths.
4. Build the n × n distance matrix.
5. Run nearest neighbour on the matrix (baseline order).
6. Run ACO on the same matrix (optimised order).
7. (Optional, m ≤ 9) Run exhaustive search for the exact optimum.
8. Expand each order into its road path by concatenating the stored A\* paths, and assert that the expanded road length equals the matrix tour length.
9. Compute distance, estimated fuel, estimated cost, runtime and % reduction.
10. Display and explain the results.

**A\* optimises the path between a pair of locations; ACO optimises the order of the stops.** ACO never sees the road graph.

### 5.3 Fairness of the comparison

Nearest neighbour, ACO and the exhaustive reference all use the same distance matrix, depot, stops and return-to-depot rule, and the same `tour_distance()` function. Runtimes are reported for the ordering step only; the shared A\* matrix time is reported separately.

## 6. Algorithms

### 6.1 A\* Search (`ecoroute/algorithms/astar.py`)

**Evaluation function:** f(n) = g(n) + h(n), where g(n) is the road distance from the start to n, h(n) is the estimated remaining distance, and f(n) is the estimated length of the cheapest path through n.

**Heuristic:** h(n) = k · ‖n − goal‖₂, with k = min(1, min over roads (u,v) of length(u,v) / ‖u − v‖₂).

*Admissibility:* every road satisfies length ≥ k·‖u − v‖. For any path from n to the goal, summing over its roads and applying the triangle inequality gives path length ≥ k·‖n − goal‖ = h(n). *Consistency:* for every road, h(u) − h(v) ≤ k‖u − v‖ ≤ length(u, v). For the sample city k = 1 (plain Euclidean distance). The scaling keeps A\* optimal even on user data where some recorded road lengths are shorter than their straight lines.

```
function A_STAR(G, start, goal, h):
    g[start] ← 0 ;  came_from ← {}
    OPEN ← min-heap containing (h(start, goal), 0, start)
    while OPEN not empty:
        (f, g_n, n) ← pop minimum from OPEN
        if g_n > g[n]: continue                      // stale entry (lazy deletion)
        if n = goal:
            return RECONSTRUCT(came_from, start, goal), g_n
        for each road (n, v, w) in G:
            t ← g_n + w
            if t < g[v] (default ∞):
                g[v] ← t ; came_from[v] ← n
                push (t + h(v, goal), t, v) onto OPEN
    return FAILURE (goal unreachable, distance ∞)

function RECONSTRUCT(came_from, start, goal):
    path ← [goal]
    while last(path) ≠ start: append came_from[last(path)] to path
    return reverse(path)
```

Implementation notes: Python `heapq` (binary heap) with an insertion counter as a tie-breaker; the goal test is applied when a node is **popped**; statistics returned include nodes expanded, nodes generated, maximum frontier size, runtime and the expansion order (used to visualise the search).

### 6.2 Nearest-neighbour baseline (`nearest_neighbor.py`)

```
function NEAREST_NEIGHBOUR(D):
    tour ← [0] ; current ← 0 ; U ← {1, …, m}
    while U not empty:
        j ← argmin over j ∈ U of (D[current][j], j)    // ties → lowest index
        append j to tour ; remove j from U ; current ← j
    append 0 to tour
    return tour, TOUR_LENGTH(D, tour)
```

### 6.3 Ant Colony Optimization: Ant System (`aco.py`)

**Transition rule** for an ant at location i choosing among the unvisited set U:

  P(i → j) = τᵢⱼ^α · ηᵢⱼ^β / Σₖ∈U τᵢₖ^α · ηᵢₖ^β,  with ηᵢⱼ = 1 / d(i, j).

**Pheromone update** after all ants finish an iteration:

  τᵢⱼ ← (1 − ρ) · τᵢⱼ + Σ over ants a whose route uses (i,j) of Q / Lₐ

(deposited on both (i,j) and (j,i) because the matrix is symmetric).

```
function ACO(D, m_ants, I, α, β, ρ, Q, τ0, seed):
    rng ← random generator(seed)
    η[i][j] ← 1 / D[i][j] for i ≠ j ;  τ[i][j] ← τ0
    best ← none ; L_best ← ∞
    for it = 1 … I:
        C ← τ^α ⊙ η^β                                   // "choice info", once per iteration
        for a = 1 … m_ants:
            tour ← [0] ; U ← {1, …, m}
            while U not empty:
                i ← last(tour)
                choose j ∈ U by roulette wheel with weights C[i][j]
                append j ; remove j from U
            append 0 ; L_a ← TOUR_LENGTH(D, tour)
        if min_a L_a < L_best: best, L_best ← that ant's tour, L_a
        τ ← (1 − ρ) · τ                                  // evaporation
        for each ant a, for each leg (i, j) of its tour:
            τ[i][j] += Q / L_a ; τ[j][i] += Q / L_a       // quality-based deposit
        record L_best, best-of-iteration, mean-of-iteration
    return best, L_best, histories, τ
```

**Parameters (defaults):** m = 20 ants, I = 100 iterations, α = 1, β = 2, ρ = 0.1, Q = 1, τ₀ = 0.1. All are configurable in the UI. Results are reproducible for a fixed seed (`numpy.random.default_rng`). A floor of 10⁻¹² prevents pheromone from underflowing to zero. If all selection weights are zero or non-finite, the ant chooses uniformly (a numerical safeguard).

### 6.4 Exhaustive reference (`exact.py`) — evaluation only

Enumerates all m! permutations with the depot fixed as the start, and returns the minimum-length tour. It is limited to m ≤ 9 (9! = 362,880). It is **not** one of the proposed methods; it exists only to measure optimality gaps.

## 7. Complexity analysis

Notation: V intersections, E roads, m delivery stops, n = m + 1 locations, I iterations, a ants.

| Component | Time | Space | Assumptions |
|---|---|---|---|
| A\* (one search) | O((V + E) log V) worst case | O(V) | Binary heap, consistent heuristic; a good heuristic reduces work in practice, not the worst-case bound |
| Distance matrix | O(n² (V + E) log V) | O(n² + n²·P) | n(n−1)/2 searches (symmetric roads); P = average stored path length |
| BFS reachability check | O(V + E) | O(V) | |
| Nearest neighbour | O(n²) | O(n) | Linear scan of unvisited stops at each step |
| ACO | O(I · a · n²) | O(n² + a·n) | Each ant: n steps × O(n) candidate scan; evaporation and choice-info O(n²) per iteration |
| Route expansion | O(n · P) | O(n · P) | |
| Exhaustive reference | O(m! · m) | O(m) | Feasible only for small m |

## 8. System architecture

```
app.py  (Streamlit UI: pages, widgets, session state; no algorithm logic)
   │
   ▼
ecoroute/pipeline.py ── optimize_routes(): validate → matrix → NN → ACO → (exact) → expand → metrics
   │            │                │
   │            ▼                ▼
   │   route_utils.py      algorithms/ (astar, nearest_neighbor, aco, exact)
   │   (A* matrix,              │
   │    expansion,              ▼
   │    consistency)       tours.py (tour representation, tour_distance)
   ▼
graph_utils.py (RoadGraph, JSON/CSV loading, validation, BFS, path reconstruction)
evaluation.py  (fuel/cost/% metrics, experiment runner)   explain.py (plain-language result explanation)
simulation.py  (per-leg decisions, animation frame states)   visualization.py (Plotly figures, incl. animated simulation)
```

The same `optimize_routes()` function is used by the UI, the experiment scripts and the integration tests, so what is tested is what is demonstrated. Libraries: `numpy` (arrays and random numbers inside ACO), `pandas` (tables and CSV), `plotly` (interactive charts), `streamlit` (UI) and `pytest` (tests). No graph or optimisation library is used for A\*, NN or ACO.

## 9. Testing

`tests/` contains **49 automated tests** (all passing), including:

- A\* returns the known shortest path on a hand-built graph (A→B→C→E, 3 km, not the direct-looking A→D→E at 10 km); handles unreachable goals (found = False, ∞) and start = goal; and its path distance equals the sum of edge weights.
- The heuristic scale shrinks h when a road is shorter than its straight line, keeping A\* optimal.
- **A\* equals Dijkstra on all 1,953 node pairs** of the sample city, with fewer total expansions.
- Tour validity checks; nearest-neighbour correctness and deterministic tie-breaking.
- ACO returns a valid tour whose length equals `tour_distance()` and is ≥ the exact optimum. **No test assumes ACO finds the optimum.**
- ACO is reproducible with a fixed seed, its best-so-far history is monotone, and invalid parameters are rejected.
- Distance matrix: symmetric, zero diagonal, stored paths match matrix entries, and expanded routes match tour lengths for both NN and ACO.
- Metrics: % reduction (including a negative value and the zero-baseline case), gap to optimum, fuel and cost; invalid vehicle settings are rejected.
- Input validation: unknown depot or stop, depot used as a stop, duplicates, too few stops, disconnected stops, and malformed datasets (empty, unknown node, negative distance, duplicate id, non-numeric coordinate, missing coordinates, invalid JSON).
- The CSV and JSON versions of the sample are identical, and the sample is labelled synthetic.
- Live simulation: nearest-neighbour decisions always pick the closest unvisited stop; ACO option probabilities sum to 1 and include the chosen stop; the animation timeline starts un-started and ends with the driven distance equal to each route's distance; the animated figure has one frame per timeline step.
- End to end: the pipeline produces valid, consistent and reproducible results, and the experiment runner produces correct table shapes.

## 10. Experimental setup

- **Network:** sample city (§5.1); depot N14.
- **Instances:** for each size m ∈ {5, 7, 8, 10, 15, 20, 25}, 5 random delivery sets drawn from all non-depot intersections (instance seed = 2026 + 1000·m + instance index).
- **ACO:** default parameters (§6.3), 10 seeds per delivery set (seeds 2026 … 2035), giving **350 ACO runs** in total.
- **Baseline:** nearest neighbour (deterministic, one run per set).
- **Reference:** exhaustive optimum for m ≤ 8.
- **Hardware/software:** a Windows 11 laptop with an Intel 64-bit CPU, Python 3.12.10, single-threaded. Runtimes are machine-dependent and indicative only.
- **Reproduce:** `python scripts/run_experiments.py --sizes 5 7 8 10 15 20 25 --instances 5 --trials 10 --seed 2026`

## 11. Results

### 11.1 Distance: ACO vs nearest neighbour

Each row aggregates 5 delivery sets × 10 seeds = 50 ACO runs (`results/experiment_sizes.csv`).

| Stops | Mean NN (km) | Mean ACO (km) | Mean reduction (%) | Std. (%) | Best (%) | Worst (%) | ACO shorter / tie / longer |
|---|---|---|---|---|---|---|---|
| 5  | 30.88 | 28.97 | 6.12  | 7.19 | 17.99 | 0.00  | 30 / 20 / 0 |
| 7  | 38.22 | 34.21 | 9.87  | 6.76 | 20.97 | 0.88  | 50 / 0 / 0 |
| 8  | 39.91 | 35.21 | 10.62 | 7.99 | 26.00 | 3.95  | 50 / 0 / 0 |
| 10 | 41.69 | 34.61 | 15.64 | 9.93 | 31.39 | 5.74  | 50 / 0 / 0 |
| 15 | 48.78 | 43.69 | 9.87  | 5.05 | 19.05 | 4.97  | 50 / 0 / 0 |
| 20 | 62.89 | 55.08 | 11.40 | 9.24 | 24.89 | 1.81  | 50 / 0 / 0 |
| 25 | 67.96 | 55.44 | 18.43 | 3.59 | 22.61 | 10.33 | 50 / 0 / 0 |

**Overall:** 330 shorter, 20 ties, 0 longer. Mean reduction 11.71% (median 9.36%). All 20 ties occurred at 5 stops, on the two delivery sets where nearest neighbour was already optimal. The standard deviation here mostly reflects differences **between delivery sets** (some sets are easy for the greedy rule), not randomness in ACO (see §11.3).

### 11.2 Distance to the true optimum (m ≤ 8)

| Stops | NN mean gap to optimum (%) | ACO mean gap to optimum (%) | ACO runs reaching the optimum |
|---|---|---|---|
| 5 | 7.16  | 0.00 | 50 / 50 |
| 7 | 11.60 | 0.00 | 50 / 50 |
| 8 | 13.05 | 0.17 | 45 / 50 |

ACO reached the proven optimum in **145 of 150** runs. The 5 misses were all on one 8-stop set, where the worst ACO run was 3.78% above the optimum (that set's 10-seed mean was still 6.68% shorter than NN). Nearest neighbour was optimal on 2 of 15 sets, and its largest gap was 35.13%.

### 11.3 Variation across seeds

In 23 of the 35 delivery sets, all 10 seeds produced the same route length (to 0.01 km). The largest per-set standard deviation over seeds was 0.78 km, on a 25-stop set with a mean ACO route of about 56 km. Variation appears mainly at 15+ stops. The mean iteration at which the final best route was first found grew with problem size: 1.3 (5 stops), 4.4 (10), 14.7 (15), 36.9 (20) and 46.6 (25). In other words, the 100-iteration budget is well used for larger instances.

### 11.4 Runtime

| Stops | A\* matrix (ms) | NN (ms) | ACO (ms) | Exhaustive (ms) |
|---|---|---|---|---|
| 5  | 0.69  | 0.02 | 152.5 | 0.20 |
| 7  | 1.11  | 0.03 | 225.5 | 9.27 |
| 8  | 1.50  | 0.03 | 238.0 | 81.59 |
| 10 | 2.94  | 0.05 | 322.7 | n/a |
| 15 | 4.20  | 0.05 | 473.5 | n/a |
| 20 | 10.00 | 0.09 | 625.8 | n/a |
| 25 | 11.12 | 0.12 | 770.7 | n/a |

Nearest neighbour is three to four orders of magnitude faster than ACO. Although ACO's asymptotic cost is O(I·a·n²), its measured time grew only about 5× from 5 to 25 stops (n² grew about 19×). At these sizes the fixed per-step Python/NumPy overhead dominates the O(n) candidate scan. The exhaustive search grows factorially: about ×9 from 7 to 8 stops, and it would take hours at 12+ stops. Building the A\* matrix is cheap at these sizes.

### 11.5 Effect of the A\* heuristic

Over all 1,953 node pairs of the sample city, A\* expanded **26,605** nodes in total versus **62,095** for Dijkstra (h = 0), a **57.2% reduction**, with identical distances on every pair.

### 11.6 ACO parameter sensitivity (`results/parameter_sweep.csv`)

Fixed instances: 4 sets each of 8, 15 and 25 stops, 3 seeds each, with 20 ants and 100 iterations. The table shows the mean % reduction vs NN.

| τ₀ | Q | ρ | β | 8 stops | 15 stops | 25 stops | Worst run, 25 stops |
|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.1 | 2 | 4.75 | 19.38 | **16.18** | **6.53** |
| 0.1 | 1 | 0.1 | 3 | 4.75 | 19.60 | 15.72 | 5.59 |
| 0.1 | 1 | 0.1 | 5 | 4.75 | 19.20 | 14.84 | 6.33 |
| 0.1 | 1 | 0.5 | 2 | 4.75 | 19.66 | 15.49 | 5.59 |
| 0.1 | 1 | 0.5 | 5 | 4.66 | 18.76 | 13.18 | 5.11 |
| 1.0 | 1 | 0.3 | 2 | 4.75 | 19.69 | 15.96 | 5.59 |
| 1.0 | 100 | 0.3 | 5 | 4.75 | 18.10 | 13.50 | 5.59 |

(See the CSV for all 15 combinations.) All settings performed within about 3 percentage points of each other. Large β (5) was consistently slightly worse at 25 stops, consistent with over-greedy behaviour. The first row was chosen as the default. These conclusions apply to this network only.

### 11.7 Discussion

- **Why ACO beats nearest neighbour.** NN commits greedily and often leaves a distant stop for last, which forces long cross-river legs and a long return to the depot. In the default 10-stop demonstration, NN's final return leg is 14.03 km versus 3.86 km for ACO (`explain.py` generates this observation automatically). ACO's probabilistic construction, guided by accumulated pheromone, explores orders that avoid such traps.
- **Small instances.** With 5 stops the search space is tiny (120 orders), NN is sometimes already optimal (ties), and ACO finds the optimum almost immediately.
- **Larger instances.** Gains remain substantial (≈ 10–18% on average). ACO's run-to-run variation appears and the best solution is found later, which suggests that larger instances would benefit from more iterations or local search.
- **Not universal.** ACO never lost to NN in these 350 runs, but this is not guaranteed in general. The application reports any loss explicitly.

## 12. Comparison of approaches

| Sub-problem | Chosen | Why it is suitable | Alternatives not selected and why |
|---|---|---|---|
| Pairwise road paths | **A\*** with admissible Euclidean heuristic | Exact shortest paths; the heuristic uses node coordinates to cut expansions (57% here); classic informed search | **Dijkstra**: also exact, but uninformed (more expansions); used only as a test oracle. **BFS**: ignores edge weights. **Floyd–Warshall**: O(V³) all-pairs over the whole graph, which is wasteful when only n ≪ V locations matter. **Greedy best-first**: fast but not optimal |
| Visiting order | **ACO (Ant System)** | Constructive metaheuristic designed for TSP-like problems; uses problem knowledge (η = 1/d); learns through pheromone; tunable; anytime best-so-far | **Exhaustive search**: exact but O(m!), so only a reference. **Held–Karp DP**: exact, O(n²2ⁿ), still exponential. **Genetic Algorithm**: excluded by the project brief, and needs permutation-preserving operators. **Simulated annealing / tabu search**: single-solution methods needing cooling or tabu design; viable future work. **2-opt local search**: an improvement step rather than a constructive AI method; future work |
| Baseline | **Nearest neighbour** | What a simple dispatcher or system would do; deterministic; O(n²) | Random order: too weak to be meaningful. Christofides: requires metric matching machinery beyond scope |
| Fuel model | **Distance ÷ km/L** | Transparent and configurable; clearly labelled as an estimate | Speed/load/gradient fuel models need data we do not have |

## 13. Limitations

1. **Synthetic data.** The network is generated, not surveyed. Results describe this network, not any real city.
2. **Single vehicle, distance only.** No capacities, time windows, driver hours or multiple vehicles.
3. **Static, symmetric costs.** Two-way roads, no traffic, no turn penalties or one-way streets.
4. **Fuel is an estimate** from a constant km/L. Real consumption varies with speed, load, gradient and congestion.
5. **ACO is stochastic with no optimality guarantee.** Its parameters were tuned on this network only, and performance on other networks may differ.
6. **Scalability.** All-pairs A\* requires n(n−1)/2 searches, and pure-Python ACO takes about 0.8 s at 25 stops (20 ants, 100 iterations). Neither is designed for hundreds of stops.
7. **The exact reference** is limited to ≤ 9 stops, so optimality gaps are only known for small instances.

## 14. Future enhancements (not implemented)

- Capacitated and multi-vehicle routing (CVRP), and time windows (VRPTW).
- One-way streets and turn restrictions (an asymmetric distance matrix; the ACO code already detects asymmetry and deposits one-way).
- Time-dependent travel costs from historical traffic data; a multi-objective cost (distance, time, emissions).
- A physics-based fuel/emission model (speed, load, gradient).
- Optional import of real road data (for example an OpenStreetMap extract, cached for offline use).
- Max–Min Ant System, elitist reinforcement, or ACO + 2-opt local search for larger instances.
- Vectorised or compiled ACO (NumPy batching, Numba) for speed.
- Held–Karp dynamic programming to extend exact references to about 15 stops.

## 15. Conclusion

EcoRoute demonstrates a clean division of labour between two AI techniques. **A\*** solves the polynomial sub-problem (exact shortest road paths) efficiently with an admissible heuristic. **ACO** tackles the NP-hard sub-problem (stop ordering) on the resulting distance matrix. On a reproducible synthetic network, ACO reduced route distance by 11.71% on average compared with a nearest-neighbour baseline over 350 runs, never produced a longer route in those runs, and matched the proven optimum in 145 of 150 runs where the optimum could be computed. The system is implemented from scratch, tested with 49 automated tests, and packaged as an interactive application that explains its own results. It also states its assumptions and limitations, especially that fuel figures are estimates derived from distance.

## References

1. P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 1968.
2. M. Dorigo, V. Maniezzo, A. Colorni, "Ant System: Optimization by a Colony of Cooperating Agents," *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, 26(1), 1996.
3. M. Dorigo, T. Stützle, *Ant Colony Optimization*, MIT Press, 2004.
4. S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2020 (Chapter 3: informed search).
5. D. J. Rosenkrantz, R. E. Stearns, P. M. Lewis, "An Analysis of Several Heuristics for the Traveling Salesman Problem," *SIAM Journal on Computing*, 6(3), 1977.

---

### Appendix A: Implemented vs proposed

| Feature | Status |
|---|---|
| A\* with admissible scaled-Euclidean heuristic, statistics, unreachable handling | Implemented |
| ACO (Ant System) with all configurable parameters, seeds, convergence history | Implemented |
| Nearest-neighbour baseline; exhaustive reference (≤ 9 stops) | Implemented |
| JSON/CSV upload with validation; BFS reachability check | Implemented |
| Streamlit UI: overview, setup, results, convergence, experiments, CSV export | Implemented |
| Live Simulation page: two trucks driving side by side with per-leg decision panels (Plotly frames) | Implemented |
| A\* vs Dijkstra search visualisation; pheromone heatmap; multi-seed robustness check | Implemented |
| 49 automated tests; experiment and parameter-sweep scripts | Implemented |
| Everything in §14 | **Proposed (not implemented)** |
