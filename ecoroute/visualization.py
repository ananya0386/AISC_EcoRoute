"""Plotly figures for the Streamlit app (presentation only -- no algorithm logic here)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .algorithms.aco import ACOResult
from .algorithms.astar import AStarResult
from .graph_utils import RoadGraph
from .pipeline import RouteSummary
from .simulation import decision_lines, side_state, timeline

# Fixed role colours (validated categorical palette). Identity never relies on colour
# alone: each method also has its own line dash, and every chart has a legend.
COLOR_ACO = "#2a78d6"  # blue, solid
COLOR_NN = "#eb6834"  # orange, dashed
COLOR_OPT = "#1baf7a"  # aqua, dotted
COLOR_ROAD = "#b5b3ad"
COLOR_JUNCTION = "#8a8983"
COLOR_DEPOT = "#0b0b0b"
COLOR_STOP = "#52514e"
METHOD_STYLE = {
    "Nearest Neighbour": (COLOR_NN, "dash"),
    "Ant Colony Optimization": (COLOR_ACO, "solid"),
    "Exhaustive optimum (reference)": (COLOR_OPT, "dot"),
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _base_layout(fig: go.Figure, title: str, height: int = 520, has_x_axis: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=90 if has_x_axis else 10),
        # Legend sits below the plot (and below the x-axis title, if any) so it can wrap
        # on narrow screens without covering the title or the axis.
        legend=dict(orientation="h", yanchor="top", y=-0.22 if has_x_axis else -0.02, xanchor="left", x=0),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def _short(graph: RoadGraph, node_id: str) -> str:
    return graph.nodes[node_id].name


def _road_traces(graph: RoadGraph) -> list[go.Scatter]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for u, v, _ in graph.edges():
        a, b = graph.nodes[u], graph.nodes[v]
        xs += [a.x, b.x, None]
        ys += [a.y, b.y, None]
    roads = go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=COLOR_ROAD, width=1.5),
                       hoverinfo="skip", name="Roads", showlegend=False)
    # Invisible midpoint markers so hovering a road shows its length.
    mids = graph.edges()
    road_hover = go.Scatter(
        x=[(graph.nodes[u].x + graph.nodes[v].x) / 2 for u, v, _ in mids],
        y=[(graph.nodes[u].y + graph.nodes[v].y) / 2 for u, v, _ in mids],
        mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"), showlegend=False,
        text=[f"Road {u} - {v}: {w:.2f} km" for u, v, w in mids], hoverinfo="text",
    )
    junctions = [n for n in graph.nodes.values()]
    junction_trace = go.Scatter(
        x=[n.x for n in junctions], y=[n.y for n in junctions], mode="markers",
        marker=dict(size=5, color=COLOR_JUNCTION), showlegend=False,
        text=[f"{n.name} ({n.id})" for n in junctions], hoverinfo="text",
    )
    return [roads, road_hover, junction_trace]


def _add_river(fig: go.Figure, graph: RoadGraph, river: list[float] | None) -> None:
    """Optional decoration: shade the river band given in the dataset metadata (sample city only)."""
    if not river:
        return
    xs = [n.x for n in graph.nodes.values()]
    fig.add_shape(type="rect", x0=min(xs) - 0.3, x1=max(xs) + 0.3, y0=river[0], y1=river[1], layer="below",
                  line=dict(width=0), fillcolor="rgba(57,135,229,0.13)")
    fig.add_annotation(x=min(xs) - 0.25, y=(river[0] + river[1]) / 2, text="River", showarrow=False, xanchor="left",
                       font=dict(size=11, color="#256abf"))


def _equal_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title=None)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title=None, scaleanchor="x", scaleratio=1)


def _route_trace(graph: RoadGraph, route: RouteSummary, name: str | None = None) -> go.Scatter:
    color, dash = METHOD_STYLE.get(route.method, (COLOR_ACO, "solid"))
    xs = [graph.nodes[n].x for n in route.road_path]
    ys = [graph.nodes[n].y for n in route.road_path]
    return go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=name or f"{route.method} ({route.distance_km:.2f} km)",
        line=dict(color=color, width=4, dash=dash),
        # Arrow markers point in the direction of travel.
        marker=dict(symbol="arrow", angleref="previous", size=11, color=color),
        hoverinfo="skip",
    )


def network_figure(graph: RoadGraph, depot: str | None, deliveries: list[str], route: RouteSummary | None = None,
                   title: str = "Road network", height: int = 560, river: list[float] | None = None) -> go.Figure:
    """Road graph with depot + delivery stops; optionally one route with numbered visiting order."""
    fig = go.Figure(_road_traces(graph))
    _add_river(fig, graph, river)
    if route is not None:
        fig.add_trace(_route_trace(graph, route))

    if deliveries:
        order = {stop: i + 1 for i, stop in enumerate(route.stop_order)} if route else {}
        color = METHOD_STYLE.get(route.method, (COLOR_STOP, ""))[0] if route else COLOR_STOP
        fig.add_trace(go.Scatter(
            x=[graph.nodes[d].x for d in deliveries], y=[graph.nodes[d].y for d in deliveries],
            mode="markers+text", name="Delivery stop",
            marker=dict(size=22 if route else 13, color=color, line=dict(color="white", width=2)),
            text=[str(order[d]) if d in order else "" for d in deliveries],
            textfont=dict(color="white", size=11), textposition="middle center",
            hovertext=[f"{_short(graph, d)} ({d})" + (f"<br>Visit #{order[d]}" if d in order else "") for d in deliveries],
            hoverinfo="text",
        ))
    if depot:
        n = graph.nodes[depot]
        fig.add_trace(go.Scatter(
            x=[n.x], y=[n.y], mode="markers", name="Depot",
            marker=dict(size=24, symbol="star", color=COLOR_DEPOT, line=dict(color="white", width=2)),
            hovertext=[f"DEPOT: {n.name} ({n.id})"], hoverinfo="text",
        ))
    _equal_axes(fig)
    return _base_layout(fig, title, height)


def overlay_figure(graph: RoadGraph, depot: str, deliveries: list[str], routes: list[RouteSummary], height: int = 600,
                   river: list[float] | None = None) -> go.Figure:
    fig = go.Figure(_road_traces(graph))
    _add_river(fig, graph, river)
    for route in routes:
        fig.add_trace(_route_trace(graph, route))
    fig.add_trace(go.Scatter(
        x=[graph.nodes[d].x for d in deliveries], y=[graph.nodes[d].y for d in deliveries], mode="markers",
        name="Delivery stop", marker=dict(size=13, color=COLOR_STOP, line=dict(color="white", width=2)),
        hovertext=[f"{_short(graph, d)} ({d})" for d in deliveries], hoverinfo="text",
    ))
    n = graph.nodes[depot]
    fig.add_trace(go.Scatter(x=[n.x], y=[n.y], mode="markers", name="Depot",
                             marker=dict(size=24, symbol="star", color=COLOR_DEPOT, line=dict(color="white", width=2)),
                             hovertext=[f"DEPOT: {n.name}"], hoverinfo="text"))
    _equal_axes(fig)
    return _base_layout(fig, "Routes overlaid on the road network", height)


def simulation_animation(graph: RoadGraph, result, sides: dict, frame_ms: int, river: list[float] | None = None) -> go.Figure:
    """Both trucks in ONE figure with Plotly frames, so playback runs smoothly in the browser (no server reruns).

    Every frame redraws, for each side: driven trail, waiting vs visited stops, the next target ring, the truck,
    a title with km driven, and a text panel with the decision being made.
    """
    dm = result.distance_matrix
    cfg = {"nn": (COLOR_NN, "dash", "Nearest neighbour", ""), "aco": (COLOR_ACO, "solid", "Ant Colony Optimization", "2")}
    roads = _road_traces(graph)[0]
    depot = graph.nodes[result.depot]
    stops = result.deliveries
    xs = [n.x for n in graph.nodes.values()]
    river_label_x = min(xs) - 0.2

    def dyn_traces(key: str, st) -> list[go.Scatter]:
        color, dash, _, ax = cfg[key]
        axes = dict(xaxis=f"x{ax}", yaxis=f"y{ax}", hoverinfo="skip", showlegend=False)
        waiting = [s for s in stops if s not in st.visited]
        done = [s for s in stops if s in st.visited]
        target = graph.nodes[st.target] if st.target else None
        return [
            go.Scatter(x=[p[0] for p in st.trail], y=[p[1] for p in st.trail], mode="lines", line=dict(color=color, width=5, dash=dash), **axes),
            go.Scatter(x=[graph.nodes[s].x for s in waiting], y=[graph.nodes[s].y for s in waiting], mode="markers",
                       marker=dict(size=14, color="#b5b3ad", line=dict(color="white", width=2)), **axes),
            go.Scatter(x=[graph.nodes[s].x for s in done], y=[graph.nodes[s].y for s in done], mode="markers+text",
                       marker=dict(size=22, color=color, line=dict(color="white", width=2)),
                       text=[str(st.visited[s]) for s in done], textfont=dict(color="white", size=11), **axes),
            go.Scatter(x=[target.x] if target else [], y=[target.y] if target else [], mode="markers",
                       marker=dict(size=32, symbol="circle-open", color=color, line=dict(width=3)), **axes),
            go.Scatter(x=[st.truck[0]], y=[st.truck[1]], mode="text", text=["\U0001F69A"], textfont=dict(size=26),
                       textposition="middle center", **axes),
        ]

    def annotations(states: dict) -> list[dict]:
        out = []
        for key, st in states.items():
            color, _, name, ax = cfg[key]
            x0 = 0.0 if key == "nn" else 0.52
            waiting = len(stops) - len(st.visited)
            out.append(dict(xref="paper", yref="paper", x=x0 + 0.24, y=1.0, yanchor="bottom", showarrow=False,
                            text=f"<b>{name}</b><br>{st.driven:.2f} km driven<br>{waiting} stops waiting", font=dict(size=12)))
            out.append(dict(xref="paper", yref="paper", x=x0, y=-0.03, xanchor="left", yanchor="top", align="left", showarrow=False,
                            text="<br>".join(decision_lines(key, st)), font=dict(family="Courier New, monospace", size=11)))
            if river:
                out.append(dict(xref=f"x{ax}", yref=f"y{ax}", x=river_label_x, y=sum(river) / 2, text="River", showarrow=False,
                                xanchor="left", font=dict(size=11, color="#256abf")))
        return out

    frames_per_leg = [max(len(sides["nn"]["legs"][k]), len(sides["aco"]["legs"][k])) for k in range(len(sides["nn"]["legs"]))]
    steps = timeline(frames_per_leg)

    def states_at(step: int, frame: int, thinking: bool) -> dict:
        return {k: side_state(dm, sides[k]["route"].tour, sides[k]["decisions"], sides[k]["legs"], step, frame, thinking)
                for k in ("nn", "aco")}

    # Static traces: [roads, depot] per side; dynamic traces (5 per side) follow and are replaced by every frame.
    static = []
    for key in ("nn", "aco"):
        ax = cfg[key][3]
        static += [go.Scatter(x=roads.x, y=roads.y, mode="lines", line=dict(color=COLOR_ROAD, width=1.5), hoverinfo="skip",
                              showlegend=False, xaxis=f"x{ax}", yaxis=f"y{ax}"),
                   go.Scatter(x=[depot.x], y=[depot.y], mode="markers", marker=dict(size=22, symbol="star", color=COLOR_DEPOT,
                              line=dict(color="white", width=2)), hovertext=["Depot"], hoverinfo="text", showlegend=False,
                              xaxis=f"x{ax}", yaxis=f"y{ax}")]
    n_static = len(static)
    dyn_idx = list(range(n_static, n_static + 10))

    def frame_data(states: dict) -> list[go.Scatter]:
        return dyn_traces("nn", states["nn"]) + dyn_traces("aco", states["aco"])

    first = states_at(*steps[0])
    fig = go.Figure(data=static + frame_data(first))
    fig.frames = [go.Frame(data=frame_data(s), traces=dyn_idx, name=str(t), layout=dict(annotations=annotations(s)))
                  for t, s in ((t, states_at(*steps[t])) for t in range(len(steps)))]

    axis = dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain")
    fig.update_layout(
        xaxis=dict(domain=[0, 0.48], **axis), yaxis=dict(domain=[0, 1], scaleanchor="x", scaleratio=1, **axis),
        xaxis2=dict(domain=[0.52, 1], anchor="y2", **axis), yaxis2=dict(domain=[0, 1], anchor="x2", scaleanchor="x2", scaleratio=1, **axis),
        annotations=annotations(first), height=800, margin=dict(l=10, r=10, t=75, b=290), showlegend=False,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.56, xanchor="left", yanchor="top", showactive=False, buttons=[
            dict(label="\u25b6 Play", method="animate", args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                 transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="\u23f8 Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                 transition=dict(duration=0), mode="immediate")])])],
        sliders=[dict(x=0.17, y=-0.56, len=0.83, xanchor="left", yanchor="top", currentvalue=dict(visible=False), ticklen=0,
                      steps=[dict(method="animate", label="", args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))]) for t in range(len(steps))])],
    )
    if river:
        for ax in ("", "2"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax}", x0=min(xs) - 0.3, x1=max(xs) + 0.3, y0=river[0], y1=river[1],
                          layer="below", line=dict(width=0), fillcolor="rgba(57,135,229,0.13)")
    return fig


def learning_figure(history: list[float], upto: int, title: str, height: int = 440) -> go.Figure:
    """ACO 'learning' replay: best route found so far, revealed up to iteration `upto`."""
    lo, hi = min(history), max(history)
    pad = (hi - lo) * 0.15 + 0.5
    fig = go.Figure(go.Scatter(x=list(range(1, upto + 1)), y=history[:upto], mode="lines",
                               line=dict(color=COLOR_ACO, width=4, shape="hv")))
    fig.update_xaxes(title="Iteration", range=[1, len(history)], showgrid=False)
    fig.update_yaxes(title="Best route so far (km)", range=[lo - pad, hi + pad], gridcolor="rgba(128,128,128,0.15)")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, title, height, has_x_axis=True)


def astar_search_figure(graph: RoadGraph, result: AStarResult, title: str, color: str = COLOR_ACO,
                        river: list[float] | None = None) -> go.Figure:
    """Highlight which intersections A* (or Dijkstra) expanded while searching one leg."""
    fig = go.Figure(_road_traces(graph))
    _add_river(fig, graph, river)
    expanded = result.expanded_order
    fig.add_trace(go.Scatter(
        x=[graph.nodes[n].x for n in expanded], y=[graph.nodes[n].y for n in expanded], mode="markers",
        name=f"Expanded nodes ({len(expanded)})", marker=dict(size=11, color=color, opacity=0.35),
        text=[f"{graph.nodes[n].name}: expanded #{i + 1}" for i, n in enumerate(expanded)], hoverinfo="text",
    ))
    if result.found:
        fig.add_trace(go.Scatter(
            x=[graph.nodes[n].x for n in result.path], y=[graph.nodes[n].y for n in result.path],
            mode="lines", name=f"Shortest path ({result.distance:.2f} km)", line=dict(color=COLOR_DEPOT, width=4),
            hoverinfo="skip",
        ))
    for node_id, label, symbol in ((result.start, "Start", "star"), (result.goal, "Goal", "diamond")):
        n = graph.nodes[node_id]
        fig.add_trace(go.Scatter(x=[n.x], y=[n.y], mode="markers", name=label,
                                 marker=dict(size=18, symbol=symbol, color=COLOR_DEPOT, line=dict(color="white", width=2)),
                                 hovertext=[f"{label}: {n.name}"], hoverinfo="text"))
    _equal_axes(fig)
    return _base_layout(fig, title, 430)


def convergence_figure(aco: ACOResult, nn_distance: float, optimal_distance: float | None = None) -> go.Figure:
    iterations = list(range(1, len(aco.best_so_far_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=aco.iteration_mean_history, mode="lines", name="Mean of ants (iteration)",
                             line=dict(color="#86b6ef", width=2), hovertemplate="Iteration %{x}<br>Mean ant route: %{y:.2f} km<extra></extra>"))
    fig.add_trace(go.Scatter(x=iterations, y=aco.iteration_best_history, mode="lines", name="Best ant (iteration)",
                             line=dict(color="#6da7ec", width=1.5, dash="dot"), hovertemplate="Iteration %{x}<br>Best ant this iteration: %{y:.2f} km<extra></extra>"))
    fig.add_trace(go.Scatter(x=iterations, y=aco.best_so_far_history, mode="lines", name="ACO best so far",
                             line=dict(color=COLOR_ACO, width=3, shape="hv"), hovertemplate="Iteration %{x}<br>Best so far: %{y:.2f} km<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1, iterations[-1]], y=[nn_distance] * 2, mode="lines", name=f"Nearest Neighbour ({nn_distance:.2f} km)",
                             line=dict(color=COLOR_NN, width=2, dash="dash"), hoverinfo="skip"))
    if optimal_distance is not None:
        fig.add_trace(go.Scatter(x=[1, iterations[-1]], y=[optimal_distance] * 2, mode="lines",
                                 name=f"True optimum ({optimal_distance:.2f} km)", line=dict(color=COLOR_OPT, width=2, dash="dot"),
                                 hoverinfo="skip"))
    fig.update_xaxes(title="Iteration", showgrid=False)
    fig.update_yaxes(title="Route distance (km)", gridcolor="rgba(128,128,128,0.15)")
    fig.update_layout(hovermode="x unified")
    return _base_layout(fig, "ACO convergence: route distance across iterations", 480, has_x_axis=True)


def pheromone_heatmap(aco: ACOResult, labels: list[str]) -> go.Figure:
    tau = np.array(aco.pheromone, dtype=float)
    np.fill_diagonal(tau, np.nan)
    fig = go.Figure(go.Heatmap(
        z=tau, x=labels, y=labels, colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        hovertemplate="From %{y}<br>To %{x}<br>Pheromone %{z:.3g}<extra></extra>", colorbar=dict(title="tau"),
    ))
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, "Final pheromone matrix (darker = stronger learned preference)", 560)


def experiment_distance_figure(instances: pd.DataFrame) -> go.Figure:
    """Per delivery set: NN distance vs ACO mean (error bar = std over seeds), optimum when known."""
    df = instances.copy()
    df["label"] = df["stops"].astype(str) + " stops · set " + df["instance"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["label"], y=df["nn_km"], name="Nearest Neighbour", marker_color=COLOR_NN,
                         hovertemplate="%{x}<br>NN: %{y:.2f} km<extra></extra>"))
    fig.add_trace(go.Bar(x=df["label"], y=df["aco_mean_km"], name="ACO (mean of seeds ± std)", marker_color=COLOR_ACO,
                         error_y=dict(type="data", array=df["aco_std_km"], color="#52514e", thickness=1.5),
                         hovertemplate="%{x}<br>ACO mean: %{y:.2f} km<extra></extra>"))
    if df["optimal_km"].notna().any():
        fig.add_trace(go.Scatter(x=df["label"], y=df["optimal_km"], mode="markers", name="True optimum",
                                 marker=dict(symbol="line-ew-open", size=14, color=COLOR_OPT, line=dict(width=3)),
                                 hovertemplate="%{x}<br>Optimum: %{y:.2f} km<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="Route distance (km)", gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(tickangle=-35)
    return _base_layout(fig, "Route distance per delivery set", 500, has_x_axis=True)


def experiment_reduction_figure(runs: pd.DataFrame) -> go.Figure:
    """Every ACO run's % reduction vs NN, grouped by size (negative = ACO was worse)."""
    fig = go.Figure()
    fig.add_trace(go.Box(x=runs["stops"].astype(str), y=runs["reduction_pct"], name="ACO runs", boxpoints="all",
                         jitter=0.4, pointpos=0, marker=dict(color=COLOR_ACO, size=8), line=dict(color=COLOR_ACO, width=1.5),
                         fillcolor="rgba(42,120,214,0.12)", hovertemplate="%{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#52514e", width=1, dash="dash"), annotation_text="No improvement",
                  annotation_position="bottom right")
    fig.update_xaxes(title="Number of delivery stops", type="category")
    fig.update_yaxes(title="Distance reduction vs NN (%)", gridcolor="rgba(128,128,128,0.15)")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, "Distance reduction of every ACO run (each dot = one seed on one delivery set)", 440, has_x_axis=True)


def experiment_runtime_figure(sizes: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [("mean_astar_matrix_ms", "A* distance matrix", "#6250d6", "solid"),
              ("mean_nn_ms", "Nearest Neighbour", COLOR_NN, "dash"),
              ("mean_aco_ms", "ACO", COLOR_ACO, "solid")]
    if "mean_exact_ms" in sizes:
        series.append(("mean_exact_ms", "Exhaustive (reference)", COLOR_OPT, "dot"))
    for col, name, color, dash in series:
        if col in sizes and sizes[col].notna().any():
            fig.add_trace(go.Scatter(x=sizes["stops"], y=sizes[col], mode="lines+markers", name=name,
                                     line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
                                     hovertemplate=name + ": %{y:.2f} ms<extra></extra>"))
    fig.update_xaxes(title="Number of delivery stops", tickmode="array", tickvals=list(sizes["stops"]))
    fig.update_yaxes(title="Mean runtime (ms, log scale)", type="log", dtick=1, gridcolor="rgba(128,128,128,0.15)")
    fig.update_layout(hovermode="x unified")
    return _base_layout(fig, "Runtime as the number of stops grows", 480, has_x_axis=True)


def matrix_table(graph: RoadGraph, locations: list[str], matrix: list[list[float]]) -> pd.DataFrame:
    labels = [("DEPOT · " if i == 0 else f"{i} · ") + graph.nodes[n].name for i, n in enumerate(locations)]
    return pd.DataFrame([[round(v, 2) for v in row] for row in matrix], index=labels, columns=labels)
