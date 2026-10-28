"""EcoRoute -- Streamlit user interface.

Run:  streamlit run app.py

This file only handles the UI. All algorithm logic lives in the `ecoroute` package:
A* (ecoroute/algorithms/astar.py), ACO (aco.py), nearest neighbour (nearest_neighbor.py).
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import pandas as pd
import streamlit as st

from ecoroute.algorithms import MAX_EXACT_STOPS, ACOParams, ant_colony_optimization, astar, euclidean_heuristic, zero_heuristic
from ecoroute.evaluation import VehicleConfig, estimate_fuel, gap_to_optimum, run_experiments
from ecoroute.explain import explain_result, leg_table
from ecoroute.graph_utils import Dataset, DatasetError, InputError, dataset_to_dict, load_dataset_csv, load_dataset_json, validate_selection
from ecoroute.simulation import aco_decisions, leg_points, nn_decisions
from ecoroute.pipeline import OptimizationResult, optimize_routes
from ecoroute.visualization import (
    COLOR_ACO,
    COLOR_NN,
    astar_search_figure,
    convergence_figure,
    experiment_distance_figure,
    experiment_reduction_figure,
    experiment_runtime_figure,
    matrix_table,
    network_figure,
    overlay_figure,
    learning_figure,
    pheromone_heatmap,
    simulation_animation,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_city.json"
CURRENCY = "₹"
ACO_DEFAULTS = ACOParams()

st.set_page_config(page_title="EcoRoute", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      .eco-hero {font-size: 2.2rem; font-weight: 700; line-height: 1.15; margin-bottom: .2rem;}
      .eco-sub {font-size: 1.05rem; opacity: .75; margin-bottom: 1.2rem;}
      .eco-pill {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem;
                 border:1px solid rgba(128,128,128,.35); margin-right:.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================= state
@st.cache_resource
def load_sample_dataset() -> Dataset:
    return load_dataset_json(SAMPLE_PATH)


def set_dataset(dataset: Dataset, source: str) -> None:
    ss = st.session_state
    ss.dataset = dataset
    ss.dataset_source = source
    ss.depot = dataset.depot if dataset.depot in dataset.graph.nodes else next(iter(dataset.graph.nodes))
    ss.deliveries = [d for d in dataset.deliveries if d != ss.depot]
    ss.result = None
    ss.result_signature = None
    ss.run_error = None
    ss.experiments = None
    ss.seed_study = None
    for key in ("w_depot", "w_deliveries"):
        ss.pop(key, None)


# Default value of every sidebar setting. Settings are always read through setting(), so a
# missing widget state (e.g. a click while the page is still loading) falls back to the
# correct default instead of a slider minimum.
SETTING_DEFAULTS = {
    "km_per_l": 12.0, "fuel_price": 100.0, "compute_exact": True, "seed": 42,
    "n_ants": ACO_DEFAULTS.n_ants, "n_iterations": ACO_DEFAULTS.n_iterations, "alpha": ACO_DEFAULTS.alpha,
    "beta": ACO_DEFAULTS.beta, "rho": ACO_DEFAULTS.evaporation_rate, "q": ACO_DEFAULTS.q,
    "tau0": ACO_DEFAULTS.initial_pheromone,
}
ACO_KEYS = ("n_ants", "n_iterations", "alpha", "beta", "rho", "q", "tau0", "seed")


def setting(key: str):
    return st.session_state.get(key, SETTING_DEFAULTS[key])


def init_state() -> None:
    ss = st.session_state
    if "dataset" not in ss:
        set_dataset(load_sample_dataset(), "Built-in sample city")
    # Seed every settings key on every run (no-op if already set). Widgets then read their
    # value from session state, and callbacks may change it without Streamlit warnings.
    for key, value in SETTING_DEFAULTS.items():
        ss.setdefault(key, value)


def reset_aco_defaults() -> None:
    # Runs as a callback, so assigning the widget keys updates both the stored values and the widgets on screen.
    for key in ACO_KEYS:
        st.session_state[key] = SETTING_DEFAULTS[key]


def next_seed_and_rerun() -> None:
    # Runs as a callback (before widgets are created), the only time a widget's value may be changed.
    st.session_state.seed = int(setting("seed")) + 1
    st.session_state.pending_run = "stay"  # rerun the optimisation but stay on the current page


def current_params() -> ACOParams:
    return ACOParams(n_ants=int(setting("n_ants")), n_iterations=int(setting("n_iterations")), alpha=float(setting("alpha")),
                     beta=float(setting("beta")), evaporation_rate=float(setting("rho")), q=float(setting("q")),
                     initial_pheromone=float(setting("tau0")))


def current_vehicle() -> VehicleConfig:
    return VehicleConfig(km_per_litre=float(setting("km_per_l")), fuel_price_per_litre=float(setting("fuel_price")))


def run_signature() -> tuple:
    ss = st.session_state
    return (id(ss.dataset), ss.depot, tuple(ss.deliveries), tuple(sorted(current_params().as_dict().items())),
            int(setting("seed")), bool(setting("compute_exact")))


def label(node_id: str) -> str:
    node = st.session_state.dataset.graph.nodes[node_id]
    return node_id if node.name == node_id else f"{node.name} ({node_id})"


def run_optimization() -> bool:
    ss = st.session_state
    ss.run_error = None
    try:
        with st.spinner("Running A* for every pair of locations, then nearest neighbour and ACO..."):
            ss.result = optimize_routes(ss.dataset.graph, ss.depot, list(ss.deliveries), current_params(), seed=int(setting("seed")),
                                        vehicle=current_vehicle(), compute_exact=bool(setting("compute_exact")))
        ss.result_signature = run_signature()
        ss.seed_study = None
        return True
    except (InputError, ValueError) as exc:
        ss.result = None
        ss.run_error = str(exc)
        return False


def river() -> list[float] | None:
    """Decorative river band of the sample city (None for other datasets)."""
    return st.session_state.dataset.metadata.get("river_y_km")


def fmt_km(x: float) -> str:
    return f"{x:,.2f} km"


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}%"


# ===================================================================== sidebar
def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.caption(f"**Dataset:** {ss.dataset.name}  \n**Depot:** {label(ss.depot)}  \n**Delivery stops:** {len(ss.deliveries)}")
        # The click only sets a flag; the run happens below, after every settings widget exists.
        st.button("Optimise routes", type="primary", icon=":material/play_arrow:", width="stretch",
                  on_click=lambda: ss.update(pending_run="results"))
        error_slot = st.empty()

        with st.expander("Vehicle assumptions", expanded=False, icon=":material/local_gas_station:"):
            st.number_input("Fuel efficiency (km per litre)", min_value=0.5, max_value=60.0, step=0.5, key="km_per_l",
                            help="Assumed average efficiency. Used ONLY to convert distance into an estimated fuel figure.")
            st.number_input(f"Fuel price ({CURRENCY} per litre)", min_value=0.0, max_value=1000.0, step=1.0,
                            key="fuel_price")
            st.caption("Fuel = distance ÷ km/L; cost = fuel × price. These are estimates, not measurements.")

        with st.expander("Advanced: ACO parameters", expanded=False, icon=":material/tune:"):
            st.slider("Number of ants (m)", 1, 100, key="n_ants", help="Routes built per iteration.")
            st.slider("Iterations", 1, 500, key="n_iterations", help="Pheromone update cycles.")
            st.slider("Alpha α: pheromone influence", 0.0, 5.0, step=0.1, key="alpha")
            st.slider("Beta β: distance (heuristic) influence", 0.0, 10.0, step=0.5, key="beta")
            st.slider("Evaporation rate ρ", 0.01, 1.0, step=0.01, key="rho",
                      help="Fraction of pheromone that evaporates each iteration.")
            st.number_input("Deposit constant Q", min_value=0.001, max_value=10000.0, step=0.5, key="q",
                            help="Each ant deposits Q / (its route length) on every leg it used.")
            st.number_input("Initial pheromone τ₀", min_value=0.0001, max_value=1000.0, step=0.05, format="%.4f",
                            key="tau0")
            st.number_input("Random seed", min_value=0, max_value=1_000_000, step=1, key="seed",
                            help="Same seed + same inputs = identical ACO result.")
            st.button("Reset to defaults", on_click=reset_aco_defaults, width="stretch")

        st.checkbox(f"Also compute exact optimum (≤ {MAX_EXACT_STOPS} stops)", key="compute_exact",
                    help="Brute-force reference used only to evaluate NN and ACO. Too slow for large instances.")

        target = ss.pop("pending_run", None)
        if target and run_optimization() and target == "results":
            st.switch_page(PAGES["results"])
        if ss.get("run_error"):
            error_slot.error(ss.run_error)


# ====================================================================== pages
def page_overview() -> None:
    st.markdown('<div class="eco-hero">EcoRoute</div>', unsafe_allow_html=True)
    st.markdown('<div class="eco-sub">Intelligent multi-stop delivery route optimisation using '
                '<b>A* Search</b> and <b>Ant Colony Optimization</b></div>', unsafe_allow_html=True)

    st.subheader("The problem")
    st.markdown(
        "A delivery van leaves a **depot**, must visit **every delivery location exactly once**, and return to the depot. "
        "We want the **shortest total driving distance** (and hence, approximately, the least fuel). Two separate decisions are hidden in this:"
    )
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("#### 1 · How do I drive from A to B?  →  **A\\* Search**")
        st.markdown(
            "Between any two locations there are many possible road paths. A* finds the **shortest road path** "
            "by always expanding the intersection with the smallest  \n"
            "**f(n) = g(n) + h(n)**  \n"
            "- *g(n)*: km already driven from the start\n"
            "- *h(n)*: straight-line km from *n* to the goal (never more than the real road distance → **admissible**)\n"
            "- *f(n)*: estimated length of the best complete path through *n*"
        )
    with c2, st.container(border=True):
        st.markdown("#### 2 · In what order do I visit the stops?  →  **ACO**")
        st.markdown(
            "With *m* stops there are *m!* possible orders (10 stops → 3,628,800). Ant Colony Optimization lets many "
            "artificial ants build routes, choosing the next stop with probability ∝ **τ^α · η^β**, where "
            "τ = pheromone (learned from good routes) and η = 1/distance (closer is better). Pheromone **evaporates** and "
            "is **re-deposited** in proportion to route quality (Q / L), so the colony gradually concentrates on short routes."
        )
    st.info("**Key distinction:** A* optimises the *path between a pair of locations*; ACO optimises the *order of the stops*. "
            "ACO never looks at the road graph directly. It works on the distance matrix that A* produced.", icon=":material/lightbulb:")

    st.subheader("Optimisation workflow")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box style="rounded,filled" fillcolor="#eef4fc" color="#2a78d6" fontname="Helvetica" fontsize=11];
          edge [color="#8a8983"];
          G [label="Road graph\\n(nodes, roads, km)"]; S [label="Depot +\\ndelivery stops"];
          A [label="A* for every pair\\n→ shortest paths"]; M [label="Distance matrix\\nd(i, j)"];
          N [label="Nearest neighbour\\n(baseline order)" fillcolor="#fdeee7" color="#eb6834"];
          C [label="ACO\\n(optimised order)"];
          E [label="Expand legs into\\nA* road paths"]; R [label="Distance, est. fuel,\\ncost, runtime, % saved"];
          G -> A; S -> A; A -> M; M -> N; M -> C; N -> E; C -> E; E -> R;
        }
        """,
        width="stretch",
    )
    st.markdown(
        "**Objective (minimised):**  Total Distance = d(0, π₁) + Σ d(πᵢ, πᵢ₊₁) + d(πₘ, 0), where *d* is the A* shortest-path distance "
        "and π is the visiting order. Both methods are scored with the same formula on the same matrix, which keeps the comparison fair."
    )

    st.subheader("How to use this app")
    st.markdown(
        "1. **Network & Setup**: choose the depot and delivery stops (or load your own CSV/JSON network).\n"
        "2. Adjust **vehicle assumptions** and **ACO parameters** in the sidebar, then press **Optimise routes**.\n"
        "3. **Optimization Results**: compare the nearest-neighbour and ACO routes, metrics and A* statistics.\n"
        "4. **ACO Convergence**: see how the best route improved over iterations; rerun with different seeds.\n"
        "5. **Experiments**: benchmark both methods on many random instances and export CSV results."
    )
    meta = st.session_state.dataset.metadata
    if meta.get("synthetic"):
        st.warning(f"**Data notice:** {meta.get('description', 'This dataset is synthetic.')}", icon=":material/info:")


def page_setup() -> None:
    ss = st.session_state
    ds: Dataset = ss.dataset
    g = ds.graph
    st.title("Network & delivery setup")

    with st.expander("Data source", expanded=False, icon=":material/database:"):
        st.markdown(f"Currently loaded: **{ds.name}** ({ss.dataset_source}): {g.num_nodes} locations, {g.num_edges} roads.")
        source = st.radio("Load a network", ["Built-in sample city", "Upload JSON", "Upload CSV (nodes + edges)"], horizontal=True)
        if source == "Built-in sample city":
            if st.button("Load / reset sample city", icon=":material/restart_alt:"):
                set_dataset(load_sample_dataset(), "Built-in sample city")
                st.rerun()
        elif source == "Upload JSON":
            st.caption("Format: `{\"nodes\": [{\"id\",\"name\",\"x\",\"y\",\"kind\"}], \"edges\": [{\"source\",\"target\",\"distance_km\"}], "
                       "\"depot\": id, \"deliveries\": [ids]}`. Coordinates in km.")
            up = st.file_uploader("Network JSON", type=["json"])
            if up is not None and st.button("Load JSON network"):
                try:
                    set_dataset(load_dataset_json(up.getvalue()), f"Uploaded: {up.name}")
                    st.rerun()
                except DatasetError as exc:
                    st.error(f"Could not load this file: {exc}")
        else:
            st.caption("nodes.csv: `id,name,x,y,kind` (kind = depot | delivery | intersection) · edges.csv: `source,target,distance_km`")
            c1, c2 = st.columns(2)
            nodes_up = c1.file_uploader("nodes.csv", type=["csv"])
            edges_up = c2.file_uploader("edges.csv", type=["csv"])
            if nodes_up is not None and edges_up is not None and st.button("Load CSV network"):
                try:
                    set_dataset(load_dataset_csv(nodes_up.getvalue(), edges_up.getvalue(), name=f"Uploaded CSV ({nodes_up.name})"),
                                f"Uploaded: {nodes_up.name} + {edges_up.name}")
                    st.rerun()
                except DatasetError as exc:
                    st.error(f"Could not load these files: {exc}")
        st.download_button("Download current network as JSON (editable template)",
                           json.dumps(dataset_to_dict(ds), indent=2), file_name="ecoroute_network.json", mime="application/json")

    if ds.metadata.get("synthetic"):
        st.caption(f":material/info: {ds.metadata.get('description')}")

    all_nodes = list(g.nodes)
    named_first = sorted(all_nodes, key=lambda n: (g.nodes[n].kind != "delivery", g.nodes[n].name))

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("Depot")
        ss.w_depot = ss.depot

        def on_depot_change() -> None:
            ss.depot = ss.w_depot
            ss.deliveries = [d for d in ss.deliveries if d != ss.depot]

        st.selectbox("Depot (start and end of the route)", all_nodes, key="w_depot", format_func=label, on_change=on_depot_change)

        st.subheader("Delivery stops")
        candidates = [n for n in g.nodes.values() if n.kind == "delivery" and n.id != ss.depot]
        max_stops = len(all_nodes) - 1
        n_pick = st.slider("Number of stops for a random selection", 2, max(2, max_stops), min(10, max_stops), key="n_pick")
        pick_seed = st.number_input("Selection seed", 0, 100_000, 7, key="pick_seed")

        def pick_random() -> None:
            pool = [n.id for n in candidates] if n_pick <= len(candidates) else [n for n in all_nodes if n != ss.depot]
            ss.deliveries = random.Random(int(pick_seed)).sample(sorted(pool), n_pick)

        def use_defaults() -> None:
            ss.deliveries = [d for d in ds.deliveries if d != ss.depot]

        with st.container(horizontal=True, wrap=True):
            st.button("Random pick", on_click=pick_random, icon=":material/casino:")
            st.button("Defaults", on_click=use_defaults, icon=":material/restart_alt:")
            st.button("Clear", on_click=lambda: ss.update(deliveries=[]), icon=":material/clear_all:")
        st.caption(f"Random pick uses the {len(candidates)} named delivery points when possible, otherwise any intersection.")

        ss.w_deliveries = [d for d in ss.deliveries if d in g.nodes]
        st.multiselect("Selected stops (edit freely: any intersection can be a delivery address)",
                       [n for n in named_first if n != ss.depot], key="w_deliveries", format_func=label,
                       on_change=lambda: ss.update(deliveries=list(ss.w_deliveries)))

        st.metric("Selected delivery stops", len(ss.deliveries))
        try:
            validate_selection(g, ss.depot, list(ss.deliveries))
            orders = 1
            for k in range(2, len(ss.deliveries) + 1):
                orders *= k
            st.success(f"Selection is valid. There are {orders:,} possible visiting orders.", icon=":material/check_circle:")
            if len(ss.deliveries) < 4:
                st.caption("Tip: use 4 or more stops for a meaningful NN vs ACO comparison.")
        except InputError as exc:
            st.error(str(exc), icon=":material/error:")

        if st.button("Optimise routes", type="primary", icon=":material/play_arrow:", width="stretch", key="setup_run"):
            if run_optimization():
                st.switch_page(PAGES["results"])
            else:
                st.error(ss.run_error)

    with right:
        st.plotly_chart(network_figure(g, ss.depot, list(ss.deliveries), title="Depot ★ and delivery stops ●", height=560, river=river()))
        k = g.heuristic_scale()
        st.caption(f"{g.num_nodes} locations · {g.num_edges} roads · total road length {sum(w for *_, w in g.edges()):.1f} km · "
                   f"A* heuristic scale k = {k:.3f} ({'plain straight-line distance is admissible' if k == 1 else 'straight-line distance scaled down to stay admissible'}). "
                   "Hover a road or intersection for details.")


def stale_warning() -> None:
    ss = st.session_state
    if ss.result is not None and ss.result_signature != run_signature():
        st.warning("Your selection or ACO settings have changed since this result was computed. Press **Optimise routes** to update.",
                   icon=":material/sync_problem:")


def page_results() -> None:
    ss = st.session_state
    st.title("Optimization results")
    result: OptimizationResult | None = ss.result
    if result is None:
        st.info("No results yet. Choose stops on **Network & Setup** and press **Optimise routes**.", icon=":material/info:")
        if ss.get("run_error"):
            st.error(ss.run_error)
        return
    stale_warning()
    g = ss.dataset.graph
    vehicle = current_vehicle()
    nn, aco = result.nearest_neighbor, result.aco
    nn_l, nn_cost = estimate_fuel(nn.distance_km, vehicle)
    aco_l, aco_cost = estimate_fuel(aco.distance_km, vehicle)

    st.markdown(f"<span class='eco-pill'>{len(result.deliveries)} stops</span>"
                f"<span class='eco-pill'>depot: {g.nodes[result.depot].name}</span>"
                f"<span class='eco-pill'>ACO seed {result.seed}</span>"
                f"<span class='eco-pill'>{vehicle.km_per_litre:g} km/L · {CURRENCY}{vehicle.fuel_price_per_litre:g}/L (assumed)</span>",
                unsafe_allow_html=True)
    st.write("")

    m = st.columns(5)
    m[0].metric("Nearest neighbour distance", fmt_km(nn.distance_km), border=True)
    m[1].metric("ACO distance", fmt_km(aco.distance_km),
                delta=f"{aco.distance_km - nn.distance_km:+.2f} km vs NN", delta_color="inverse", border=True)
    m[2].metric("Distance reduction", fmt_pct(result.reduction_pct).replace("+", ""),
                help="(NN − ACO) / NN × 100. Negative means ACO was longer in this run.", border=True)
    m[3].metric("Est. fuel saved", f"{nn_l - aco_l:.2f} L", help="Estimate = distance ÷ assumed km/L.", border=True)
    m[4].metric("Est. fuel cost saved", f"{CURRENCY}{nn_cost - aco_cost:,.2f}", border=True)

    rows = []
    opt = result.exact.distance_km if result.exact else None
    for r in result.routes:
        litres, cost = estimate_fuel(r.distance_km, vehicle)
        rows.append({
            "Method": r.method,
            "Distance (km)": round(r.distance_km, 2),
            "Est. fuel (L)": round(litres, 2),
            f"Est. fuel cost ({CURRENCY})": round(cost, 2),
            "Ordering runtime (ms)": round(r.runtime_ms, 2),
            "Gap to optimum (%)": round(gap_to_optimum(opt, r.distance_km), 2) if opt else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Shared preprocessing: {len(result.distance_matrix.searches)} A* searches built the distance matrix in "
               f"{result.distance_matrix.runtime_ms:.1f} ms (used by every method). " + " ".join(result.notes))

    st.subheader("Why the routes differ")
    for line in explain_result(g, result):
        st.markdown(f"- {line}")

    st.subheader("Routes on the road network")
    tabs = st.tabs(["Side by side", "Overlay", *(["Exact optimum"] if result.exact else [])])
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(network_figure(g, result.depot, result.deliveries, nn, title=f"Nearest neighbour: {fmt_km(nn.distance_km)}", height=480, river=river()))
        c2.plotly_chart(network_figure(g, result.depot, result.deliveries, aco, title=f"ACO: {fmt_km(aco.distance_km)}", height=480, river=river()))
        st.caption("Numbers on the stops are the visiting order. Arrows show the direction of travel along the A* road paths. "
                   "NN is drawn orange/dashed, ACO blue/solid.")
    with tabs[1]:
        st.plotly_chart(overlay_figure(g, result.depot, result.deliveries, [nn, aco], river=river()))
    if result.exact:
        with tabs[2]:
            st.plotly_chart(network_figure(g, result.depot, result.deliveries, result.exact,
                                           title=f"Exhaustive optimum (reference): {fmt_km(result.exact.distance_km)}", height=560, river=river()))

    st.subheader("Visiting order, leg by leg")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**<span style='color:{COLOR_NN}'>■</span> Nearest neighbour**", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(leg_table(g, result, nn)), hide_index=True)
    with c2:
        st.markdown(f"**<span style='color:{COLOR_ACO}'>■</span> Ant Colony Optimization**", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(leg_table(g, result, aco)), hide_index=True)

    st.subheader("A* search statistics")
    dm = result.distance_matrix
    s = st.columns(4)
    s[0].metric("A* searches", len(dm.searches), help="One per unordered pair of locations: n(n−1)/2.", border=True)
    s[1].metric("Nodes expanded (total)", f"{dm.total_expanded:,}", border=True)
    s[2].metric("Avg. expanded per search", f"{dm.total_expanded / max(len(dm.searches), 1):.1f}", border=True)
    s[3].metric("Matrix build time", f"{dm.runtime_ms:.1f} ms", border=True)
    with st.expander("Distance matrix (A* shortest road distances, km)"):
        st.dataframe(matrix_table(g, dm.locations, dm.matrix))
    with st.expander("Per-pair A* statistics"):
        st.dataframe(pd.DataFrame([{
            "From": g.nodes[r.start].name, "To": g.nodes[r.goal].name, "Distance (km)": round(r.distance, 2),
            "Road segments": len(r.path) - 1, "Nodes expanded": r.nodes_expanded, "Nodes generated": r.nodes_generated,
            "Max frontier": r.max_frontier, "Time (ms)": round(r.runtime_ms, 3)} for r in dm.searches]), hide_index=True)

    with st.container(border=True):
        st.markdown("#### Inspect one A* search: does the heuristic help?")
        st.caption("Compare A* (h = straight-line distance) with Dijkstra (h = 0) on the same leg. Both must find the same distance; "
                   "A* should expand fewer intersections because the heuristic steers it towards the goal.")
        c1, c2 = st.columns(2)
        src = c1.selectbox("From", dm.locations, format_func=label, key=f"leg_from_{id(result)}")
        dst = c2.selectbox("To", dm.locations, index=min(1, len(dm.locations) - 1), format_func=label, key=f"leg_to_{id(result)}")
        a_res = astar(g, src, dst, euclidean_heuristic(g))
        d_res = astar(g, src, dst, zero_heuristic)
        c1.plotly_chart(astar_search_figure(g, a_res, f"A*: {a_res.nodes_expanded} nodes expanded", COLOR_ACO, river=river()))
        c2.plotly_chart(astar_search_figure(g, d_res, f"Dijkstra (h=0): {d_res.nodes_expanded} nodes expanded", COLOR_NN, river=river()))
        if d_res.nodes_expanded:
            st.markdown(f"Both found **{a_res.distance:.2f} km**. A* expanded **{a_res.nodes_expanded}** vs **{d_res.nodes_expanded}** "
                        f"nodes ({100 * (1 - a_res.nodes_expanded / d_res.nodes_expanded):.0f}% fewer).")


def page_convergence() -> None:
    ss = st.session_state
    st.title("ACO convergence")
    result: OptimizationResult | None = ss.result
    if result is None:
        st.info("Run the optimisation first (sidebar → **Optimise routes**).", icon=":material/info:")
        return
    stale_warning()
    aco = result.aco_details
    g = ss.dataset.graph

    m = st.columns(4)
    m[0].metric("Best ACO distance", fmt_km(aco.best_distance), border=True)
    m[1].metric("First found at iteration", f"{aco.best_found_iteration} / {len(aco.best_so_far_history)}", border=True)
    m[2].metric("Iteration-1 best", fmt_km(aco.best_so_far_history[0]), border=True)
    m[3].metric("ACO runtime", f"{aco.runtime_ms:.0f} ms", border=True)

    st.plotly_chart(convergence_figure(aco, result.nearest_neighbor.distance_km, result.exact.distance_km if result.exact else None))
    p = aco.params
    st.caption(f"Parameters: m={p.n_ants} ants, {p.n_iterations} iterations, α={p.alpha}, β={p.beta}, ρ={p.evaporation_rate}, "
               f"Q={p.q}, τ₀={p.initial_pheromone}, seed={aco.seed}.")
    st.info(
        "**How to read this:** the dark-blue step line is the best route found *so far*; it can only go down. The light line is the "
        "average ant in each iteration: it typically falls in the early iterations as pheromone builds up on good legs (the colony *learning*), "
        "then levels off while ants keep exploring alternatives. "
        "ACO is **stochastic**: different seeds can give different answers, and there is no guarantee of reaching the global optimum. "
        "If the mean line collapses onto the best line very early, the colony has *stagnated* (try a higher ρ or lower α).",
        icon=":material/school:",
    )

    c1, c2 = st.columns([1, 1])
    c1.button("Rerun with next seed", icon=":material/refresh:", width="stretch", on_click=next_seed_and_rerun)
    n_seeds = c2.number_input("Seeds for a robustness check", 3, 50, 10, key="n_seed_study")
    if st.button(f"Run ACO with {int(n_seeds)} different seeds on this instance", icon=":material/science:"):
        dists = []
        prog = st.progress(0.0)
        for i in range(int(n_seeds)):
            dists.append(ant_colony_optimization(result.distance_matrix.matrix, current_params(), seed=i).best_distance)
            prog.progress((i + 1) / int(n_seeds))
        prog.empty()
        ss.seed_study = dists
    if ss.get("seed_study"):
        d = ss.seed_study
        nn_km = result.nearest_neighbor.distance_km
        s = st.columns(5)
        s[0].metric("Runs", len(d))
        s[1].metric("Mean", fmt_km(statistics.fmean(d)))
        s[2].metric("Std. dev.", f"{statistics.stdev(d):.3f} km" if len(d) > 1 else "n/a")
        s[3].metric("Best", fmt_km(min(d)))
        s[4].metric("Runs beating NN", f"{sum(x < nn_km - 1e-6 for x in d)} / {len(d)}")
        st.caption(f"Seeds 0 to {len(d) - 1} with the current sidebar parameters on the same distance matrix. NN = {nn_km:.2f} km"
                   + (f", optimum = {result.exact.distance_km:.2f} km." if result.exact else "."))

    with st.expander("Final pheromone matrix", icon=":material/grid_on:"):
        labels = [("Depot" if i == 0 else f"{i}: {g.nodes[n].name}") for i, n in enumerate(result.distance_matrix.locations)]
        st.plotly_chart(pheromone_heatmap(aco, labels))
        st.caption("Each cell is τ(i, j) after the last iteration. Legs on the best tour carry the most pheromone.")


SPEEDS = {"Slow": 600, "Normal": 300, "Fast": 100}  # milliseconds per animation frame


def sim_stops() -> list[str]:
    ss = st.session_state
    g = ss.dataset.graph
    pool = [n.id for n in g.nodes.values() if n.kind == "delivery" and n.id != ss.depot]
    if len(pool) < int(ss.sim_n):
        pool = [n for n in g.nodes if n != ss.depot]
    return random.Random(int(ss.sim_seed)).sample(sorted(pool), int(ss.sim_n))


def sim_side_data(g, result: OptimizationResult) -> dict:
    dm = result.distance_matrix
    sides = {}
    for key, route, decisions in (
        ("nn", result.nearest_neighbor, nn_decisions(g, dm, result.nearest_neighbor.tour)),
        ("aco", result.aco, aco_decisions(g, dm, result.aco.tour, result.aco_details)),
    ):
        legs = [leg_points(g, dm, a, b) for a, b in zip(route.tour, route.tour[1:])]
        sides[key] = {"route": route, "decisions": decisions, "legs": legs}
    return sides


def page_simulation() -> None:
    ss = st.session_state
    g = ss.dataset.graph
    st.title("Live simulation")
    st.caption("Two trucks, same stops. Watch each one decide where to go next. "
               "**Left:** nearest neighbour always picks the closest stop. "
               "**Right:** the ant colony picks using what it learned (pheromone) plus distance.")

    c1, c2, c3 = st.columns([1.3, 1, 1.3])
    c1.slider("Delivery stops", 3, 12, 6, key="sim_n")
    c2.number_input("Stop layout #", 0, 9999, 1, key="sim_seed", help="Change this to get a different random set of stops.")
    c3.select_slider("Speed", list(SPEEDS), value="Normal", key="sim_speed")

    stops = sim_stops()
    signature = (id(ss.dataset), ss.depot, tuple(stops), tuple(sorted(current_params().as_dict().items())), int(setting("seed")))
    if ss.get("sim_signature") != signature:
        try:
            with st.spinner("Preparing: A* finds every road distance, then both methods plan their routes..."):
                ss.sim_result = optimize_routes(g, ss.depot, stops, current_params(), seed=int(setting("seed")),
                                                vehicle=current_vehicle(), compute_exact=False)
            ss.sim_signature = signature
        except (InputError, ValueError) as exc:
            st.error(str(exc))
            return
    result: OptimizationResult = ss.sim_result

    nn, aco = result.nearest_neighbor, result.aco
    saved = nn.distance_km - aco.distance_km
    verdict = (f"ACO's order is **{saved:.2f} km shorter** ({result.reduction_pct:.1f}%)." if saved > 1e-6
               else f"ACO's order is **{-saved:.2f} km longer** this time (ACO is random; try another layout)." if saved < -1e-6
               else "Both orders are the same length.")
    st.info(f"Press **▶ Play** under the maps (or drag the slider). Final result: nearest neighbour {nn.distance_km:.2f} km, "
            f"ACO {aco.distance_km:.2f} km. {verdict}", icon=":material/play_circle:")

    st.plotly_chart(simulation_animation(g, result, sim_side_data(g, result), SPEEDS[ss.sim_speed], river()))

    with st.expander("How the ants learned this route (convergence)"):
        hist = result.aco_details.best_so_far_history
        st.plotly_chart(learning_figure(hist, len(hist), "Best route found so far, per iteration"))
        st.caption("Each iteration every ant builds a full route; shorter routes leave more pheromone for the next ants.")


def page_experiments() -> None:
    ss = st.session_state
    st.title("Experiments")
    st.markdown("Benchmark nearest neighbour vs ACO over **random delivery sets of increasing size**. For each set, A* builds the matrix once, "
                "NN runs once (deterministic) and ACO runs once per seed. All numbers below come from live algorithm runs.")
    g, depot = ss.dataset.graph, ss.depot
    max_size = g.num_nodes - 1
    with st.form("experiment_form"):
        c1, c2, c3, c4 = st.columns(4)
        size_options = [s for s in (3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40) if s <= max_size]
        sizes = c1.multiselect("Delivery-set sizes", size_options, default=[s for s in (5, 7, 10, 15, 20) if s <= max_size])
        instances = c2.number_input("Random delivery sets per size", 1, 20, 3)
        trials = c3.number_input("ACO seeds per set", 1, 30, 5)
        base_seed = c4.number_input("Base seed", 0, 1_000_000, 2026)
        exact = st.checkbox("Compute exact optimum for sizes ≤ 8 (reference)", value=True)
        est_runs = len(sizes) * int(instances) * int(trials)
        st.caption(f"Depot: {label(depot)} · ACO parameters from the sidebar · this will run ACO {est_runs} times.")
        submitted = st.form_submit_button("Run experiments", type="primary", icon=":material/science:")
    if submitted:
        prog = st.progress(0.0, text="Starting...")
        try:
            ss.experiments = run_experiments(
                g, depot, list(g.nodes), sorted(sizes), int(instances), int(trials), current_params(), int(base_seed),
                include_exact=exact, exact_limit=8, progress=lambda f, msg: prog.progress(f, text=msg))
            ss.experiments_params = current_params().as_dict()
        except (InputError, ValueError) as exc:
            st.error(str(exc))
        prog.empty()

    tables = ss.get("experiments")
    if not tables:
        return
    runs, inst, sizes_df = tables["runs"], tables["instances"], tables["sizes"]
    total = len(runs)
    shorter = int((runs["result"] == "ACO shorter").sum())
    ties = int((runs["result"] == "Tie").sum())
    longer = int((runs["result"] == "ACO longer").sum())

    k = st.columns(4)
    k[0].metric("ACO runs", total, border=True)
    k[1].metric("Mean reduction vs NN", f"{runs['reduction_pct'].mean():.2f}%", border=True)
    k[2].metric("ACO shorter / tie / longer", f"{shorter} / {ties} / {longer}", border=True)
    if runs["optimal_km"].notna().any():
        sub = runs[runs["optimal_km"].notna()]
        k[3].metric("ACO hit the optimum", f"{int((sub['aco_gap_to_opt_pct'].abs() < 1e-6).sum())} / {len(sub)} runs",
                    help="Only for sizes where the exhaustive reference was computed.", border=True)

    st.plotly_chart(experiment_reduction_figure(runs))
    st.plotly_chart(experiment_distance_figure(inst))
    st.plotly_chart(experiment_runtime_figure(sizes_df))
    st.caption("Runtime note: A* matrix and NN are fast polynomial steps; ACO costs more because it builds "
               "ants × iterations complete routes; the exhaustive reference grows factorially and is only run for small sizes.")

    st.subheader("Summary by number of stops")
    st.dataframe(sizes_df.round(3), hide_index=True)
    st.subheader("Per delivery set (ACO mean, std. dev., best over seeds)")
    st.dataframe(inst.round(3), hide_index=True)
    with st.expander("All individual runs"):
        st.dataframe(runs.round(3), hide_index=True)

    notes = [f"Each size used {int(inst.groupby('stops').size().iloc[0])} random delivery sets × "
             f"{int(inst['trials'].iloc[0])} ACO seeds. Parameters: {ss.get('experiments_params')}."]
    if longer:
        notes.append(f"ACO was longer than NN in {longer} run(s). This is reported, not hidden: with a fixed iteration budget "
                     "ACO can miss a good order that the greedy rule happens to find, especially as the number of stops grows.")
    if ties:
        notes.append(f"{ties} run(s) tied: typically small instances where NN is already optimal.")
    for n in notes:
        st.markdown(f"- {n}")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download runs CSV", runs.to_csv(index=False), "ecoroute_experiment_runs.csv", "text/csv", icon=":material/download:")
    d2.download_button("Download per-set CSV", inst.to_csv(index=False), "ecoroute_experiment_instances.csv", "text/csv", icon=":material/download:")
    d3.download_button("Download per-size CSV", sizes_df.to_csv(index=False), "ecoroute_experiment_sizes.csv", "text/csv", icon=":material/download:")


# ======================================================================= main
init_state()
PAGES = {
    "simulation": st.Page(page_simulation, title="Live Simulation", icon=":material/local_shipping:", default=True, url_path="simulation"),
    "overview": st.Page(page_overview, title="Overview", icon=":material/home:", url_path="overview"),
    "setup": st.Page(page_setup, title="Network & Setup", icon=":material/map:", url_path="setup"),
    "results": st.Page(page_results, title="Optimization Results", icon=":material/route:", url_path="results"),
    "convergence": st.Page(page_convergence, title="ACO Convergence", icon=":material/trending_down:", url_path="convergence"),
    "experiments": st.Page(page_experiments, title="Experiments", icon=":material/science:", url_path="experiments"),
}
navigation = st.navigation(list(PAGES.values()))
sidebar()
navigation.run()
