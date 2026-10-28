# EcoRoute: Intelligent Multi-Stop Delivery Route Optimization Using A* Search and Ant Colony Optimization

EcoRoute is an AISC course project. It plans a single delivery vehicle's route: start at a depot, visit every selected delivery stop exactly once, and return to the depot, with the **shortest total road distance**.

It combines two algorithms that are implemented from scratch in this repository:

| Question | Algorithm | File |
|---|---|---|
| What is the shortest road path between two locations? | **A\* Search** | `ecoroute/algorithms/astar.py` |
| In what order should the stops be visited? | **Ant Colony Optimization (Ant System)** | `ecoroute/algorithms/aco.py` |
| What would a simple greedy driver do? (baseline) | **Nearest Neighbour** | `ecoroute/algorithms/nearest_neighbor.py` |
| What is the true best order? (evaluation reference, small instances only) | Exhaustive search | `ecoroute/algorithms/exact.py` |

No routing API, optimisation library or internet connection is used. NetworkX is **not** used: the graph is a plain Python adjacency list, so every step of A* is visible in the code.

> **Data notice:** the built-in "EcoRoute Sample City" is **synthetic**. It is generated reproducibly by `data/generate_sample_network.py` (seed 42), all place names are fictional, and it is not collected real-world data.

---

## 1. Quick start on Windows

You need **Python 3.10 or newer** (tested with 3.12). Check it in Command Prompt or PowerShell:

```bash
python --version
```

### Option A: double-click

1. Double-click **`setup_windows.bat`** (one time only; it creates `.venv` and installs the packages).
2. Double-click **`run_app.bat`**. Your browser opens at http://localhost:8501.

### Option B: commands

Open a terminal in the project folder:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

If Streamlit asks for an e-mail address on its first run, just press **Enter**.

### Run the tests

```bash
.venv\Scripts\python.exe -m pytest -v
```

(or double-click `run_tests.bat`). There are 49 tests covering A* correctness, unreachable goals, tour validity, ACO reproducibility, metrics, input validation and the end-to-end pipeline.

### Reproduce the experiments in the report

```bash
.venv\Scripts\python.exe scripts\run_experiments.py --sizes 5 7 8 10 15 20 25 --instances 5 --trials 10 --seed 2026
```

This writes `results/experiment_runs.csv`, `experiment_instances.csv` and `experiment_sizes.csv`. The ACO parameter sweep used to choose the defaults is `scripts\parameter_sweep.py`. It takes several minutes and writes `results/parameter_sweep.csv`.

---

## 2. Using the app

| Page | What you do there |
|---|---|
| **Live Simulation** (opens first) | Pick 3–12 stops and press ▶ Play on the chart: two trucks drive side by side while each panel shows the decision being made (nearest neighbour: closest stop; ACO: the colony's percentage chance per stop) |
| **Overview** | Problem statement, A* vs ACO, workflow diagram, instructions |
| **Network & Setup** | Load the sample city or upload your own CSV/JSON network, choose the depot and delivery stops (manual, random or defaults), and see live validation |
| **Optimization Results** | NN vs ACO metrics, an auto-generated explanation, both routes drawn on the map, leg-by-leg tables, A* statistics, and an "A* vs Dijkstra" search visualisation |
| **ACO Convergence** | Best-so-far and mean-ant distance per iteration, a multi-seed robustness check, and the pheromone heatmap |
| **Experiments** | Benchmark NN vs ACO over many random instances and sizes, with charts and CSV export |

The **sidebar** holds the vehicle assumptions (km/L, fuel price), the ACO parameters (ants, iterations, α, β, ρ, Q, τ₀, seed), the exact-optimum option and the **Optimise routes** button. You can rerun from any page.

---

## 3. How it works (short version)

```
road graph ──► A* for every pair of {depot + stops} ──► distance matrix d(i,j) + road paths
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                       ▼
                    Nearest Neighbour (baseline order)                      ACO (optimised order)
                              │                                                       │
                              └──────── expand each leg into its A* road path ────────┘
                                                          ▼
                       distance · estimated fuel · estimated cost · runtime · % reduction
```

**Objective (minimised).** For a visiting order π that starts and ends at the depot (index 0):

    Total Distance = d(0, π1) + Σ d(πi, πi+1) + d(πm, 0)

**Fuel is estimated, not optimised separately:**

    Estimated Fuel (L)  = Total Distance (km) / Vehicle Efficiency (km/L)
    Estimated Fuel Cost = Estimated Fuel × Fuel Price per Litre
    Distance Reduction (%) = (NN distance − ACO distance) / NN distance × 100   (undefined if NN distance = 0)

**A\*** uses f(n) = g(n) + h(n), where h(n) = k × straight-line distance to the goal. `RoadGraph.heuristic_scale()` computes k = min(1, min over roads of road length ÷ straight-line length). This guarantees that h never overestimates, so A\* is optimal on **any** dataset you load. For the sample city k = 1.

**ACO** ants choose the next stop with probability ∝ τ^α · (1/d)^β. After each iteration pheromone evaporates (τ ← (1−ρ)τ) and every ant deposits Q/L on the legs it used. The best route ever found is returned.

**Fairness:** NN and ACO use the same A\* distance matrix, the same depot, the same stops, the same return-to-depot rule and the same `tour_distance()` function.

Full details are in [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

---

## 4. Project structure

```
AISC/
├── app.py                      Streamlit UI (no algorithm logic)
├── ecoroute/
│   ├── algorithms/
│   │   ├── astar.py            A* search (heapq priority queue, predecessor map, statistics)
│   │   ├── aco.py              Ant Colony Optimization (Ant System)
│   │   ├── nearest_neighbor.py Greedy baseline
│   │   └── exact.py            Brute-force reference (evaluation only, ≤ 9 stops)
│   ├── graph_utils.py          RoadGraph, dataset loading (JSON/CSV), validation, path helpers
│   ├── route_utils.py          A* distance matrix, route expansion, consistency check
│   ├── tours.py                Tour representation and the shared tour_distance()
│   ├── simulation.py           Step-by-step decision data for the live simulation
│   ├── pipeline.py             End-to-end workflow used by app, scripts and tests
│   ├── evaluation.py           Fuel/cost/% metrics and the experiment runner
│   ├── explain.py              Plain-language explanation of each result
│   └── visualization.py        Plotly figures
├── data/
│   ├── generate_sample_network.py   Reproducible synthetic city generator
│   ├── sample_city.json             Generated network (JSON format)
│   ├── sample_nodes.csv / sample_edges.csv   Same network in CSV format (editable template)
├── scripts/
│   ├── run_experiments.py      CLI benchmark → results/*.csv
│   └── parameter_sweep.py      ACO parameter sensitivity study
├── results/                    Experiment outputs quoted in the report
├── tests/                      pytest unit and integration tests
├── docs/
│   ├── PROJECT_REPORT.md       Report draft (methodology, pseudocode, complexity, results)
│   ├── VIVA_GUIDE.md           Likely viva questions with answers
│   └── DEMO_SCRIPT.md          Step-by-step demonstration script
├── requirements.txt
└── setup_windows.bat / run_app.bat / run_tests.bat
```

---

## 5. Using your own road network

**JSON** (one file; see `data/sample_city.json`):

```json
{
  "metadata": {"name": "My network"},
  "nodes": [{"id": "A", "name": "Warehouse", "x": 0.0, "y": 0.0, "kind": "depot"},
            {"id": "B", "name": "Shop 1",    "x": 1.2, "y": 0.4, "kind": "delivery"}],
  "edges": [{"source": "A", "target": "B", "distance_km": 1.5}],
  "depot": "A",
  "deliveries": ["B"]
}
```

**CSV** (two files; see `data/sample_nodes.csv` and `data/sample_edges.csv`):

- `nodes.csv`: `id,name,x,y,kind` (kind = `depot`, `delivery` or `intersection`)
- `edges.csv`: `source,target,distance_km`

Rules: coordinates and distances are in **km**, roads are **two-way**, distances must be > 0, and node ids must be unique. Invalid files are rejected with a message that names the problem (unknown node, negative distance, missing column, and so on). If stops lie in disconnected parts of the network, the app says which ones.

The coordinates are only used for drawing and for the A\* heuristic. The road `distance_km` values are what gets optimised.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python from python.org and tick **"Add python.exe to PATH"**, or use `py -3.12` instead of `python`. |
| Port 8501 is already in use | Close the other Streamlit window, or run `.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502`. |
| The browser does not open | Open http://localhost:8501 manually. |
| Changes to the code don't show | Press **R** in the app or restart `run_app.bat` (the sample dataset is cached). |
