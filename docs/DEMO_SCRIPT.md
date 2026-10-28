# EcoRoute: Demonstration Script (about 12 minutes)

**Before the demo**
- Run `run_app.bat` (or `.venv\Scripts\python.exe -m streamlit run app.py`) and open http://localhost:8501.
- Make the browser full-screen. Leave the sidebar ACO parameters at their defaults (press **Reset to defaults** if unsure).
- Optional: run `run_tests.bat` beforehand and keep the "49 passed" window open as evidence.

The numbers quoted below come from the default settings (sample city, default 10 stops, seed 42). If you change anything, read the numbers off the screen instead. Every figure is computed live.

---

## 0. Opening hook: the live simulation (Live Simulation page, ~2 min)

The app opens on this page. Do this first, because it shows the whole idea before any theory.

- Leave **6 stops** and set **Speed** to Normal. Press **▶ Play** under the two maps (you can also drag the slider, or press Pause).
- Both 🚚 trucks drive the same stops at the same time. Before each leg they pause and the panel under each map shows the decision:
  - Left, nearest neighbour: "It lists the stops left with their road distances and takes the closest. It never looks ahead."
  - Right, ACO: "It shows the colony's percentage chance for each stop, from pheromone and distance. The row marked with > is the one it picked."
- Point to the titles: km driven so far and stops still waiting. Numbers on the stops are the visiting order.
- When they finish, read the blue box above the maps (final distances and the saving). Then say: "Same stops, same road distances; only the *order* differs. Let me explain how each part works."
- Open **How the ants learned this route** to show the convergence curve.
- If ACO happens to tie or lose on this layout, say so honestly and change **Stop layout #** (ACO is random).
- Tip: use **Fast** if you are short on time. Numbers are computed live, so they may differ from other runs.

## 1. The problem (Overview page, ~1.5 min)

> "A delivery van leaves a depot, visits every stop once and comes back. We want the shortest total distance. There are really **two** problems hidden here."

- Point to the two cards. **A\*** answers *how to drive between two places*; **ACO** answers *in what order to visit the stops*.
- Point to the blue info box: A\* optimises paths, ACO optimises the order, and ACO works on the matrix that A\* produces.
- Point to the workflow diagram, left to right.
- Point to the data notice: "The city is synthetic and generated reproducibly. I don't claim it's real data."

## 2. The network (Network & Setup page, ~1.5 min)

- Show the map: "63 intersections, 97 roads, and a **river with only 3 bridges**, so straight-line distance is often misleading."
- Hover over a road to show its length, then over a stop to show its name.
- Point to the caption "A\* heuristic scale k = 1.000": "Every road is at least as long as its straight line, so the straight-line heuristic never overestimates. It's **admissible**."
- Show the validation: press **Clear**. A red error appears ("Select at least 2…"). Press **Defaults** to restore 10 stops, and point out "3,628,800 possible visiting orders".
- (Optional) Mention that CSV/JSON upload is under *Data source*.

## 3. Run it (sidebar, ~0.5 min)

- Point to the sidebar: vehicle assumptions (12 km/L, ₹100/L) and the ACO parameters (20 ants, 100 iterations, α = 1, β = 2, ρ = 0.1).
- Click **Optimise routes**. The app jumps to the results.

## 4. Results (Optimization Results page, ~3 min)

- **Metric cards:** NN ≈ 50.67 km, ACO ≈ 40.75 km, **≈ 19.6% shorter**, plus estimated fuel and cost saved.
  > "Fuel and cost are *estimates*: distance divided by an assumed km/L. We optimise distance only."
- **"Why the routes differ":** read the third bullet aloud. NN's final drive back to the depot is about 14 km versus about 4 km for ACO.
  > "This is the classic greedy failure: always taking the nearest stop leaves a far stop for last."
- **Side-by-side maps:** follow the numbers 1→10 on each map. NN zig-zags across the river; ACO sweeps around in one loop. Then switch to the **Overlay** tab.
- **Leg-by-leg tables:** "Every leg is an A\* shortest road path; the numbers add up to the totals."
- **A\* statistics:** 55 A\* searches (one per pair of the 11 locations) built the matrix in a few milliseconds.
- **"Inspect one A\* search":** pick any two locations.
  > "Both find the same distance, but A\* expands far fewer intersections than Dijkstra because the heuristic pulls it towards the goal."

## 5. Convergence (ACO Convergence page, ~1.5 min)

- The dark-blue step line is the best route found so far. It drops quickly in the first iterations, then stays flat.
- The light line is the average ant. It drifts down as pheromone accumulates on good legs: the colony learns.
- The orange dashed line is NN, for reference.
- Click **Run ACO with 10 different seeds**. Point to the mean, std. dev., best, and "Runs beating NN".
  > "ACO is random, so I report several runs, not one lucky one."
- Open **Final pheromone matrix**: the darker cells are the legs of the best route.

## 6. Proof against the true optimum (~1 min)

- Go back to **Network & Setup**, move the slider to **8**, and press **Random pick**, then **Optimise routes**.
- With ≤ 9 stops the app also runs the exhaustive search (the "Exact optimum" tab and the "Gap to optimum" column).
  > "Brute force checks all 40,320 orders and proves the best route. ACO's gap is usually 0%; NN is noticeably above it."
- Stress that the exhaustive solver is **only a reference**; it becomes impossible above about 10 stops.

## 7. Experiments (Experiments page, ~1 min, or show the pre-computed CSVs)

- Press **Run experiments** with the defaults (5–20 stops, 3 sets × 5 seeds = 75 ACO runs; takes about 30–40 s).
- Box plot: every dot is one ACO run. All dots are at or above 0%, meaning ACO was never worse in this experiment. Point to the "shorter / tie / longer" counter.
  > "If ACO had lost somewhere, the counter and this text would say so; nothing is hidden."
- Runtime chart (log scale): NN takes about 0.1 ms, ACO a few hundred ms, and the exhaustive search explodes, which is why heuristics are needed.
- Click **Download runs CSV**.

## 8. Close (~0.5 min)

> "A\* gives exact shortest paths between stops; ACO uses those distances to find a much better visiting order than the greedy baseline, often the true optimum on small instances. Everything is implemented from scratch, tested with 49 unit tests, and every number is generated live."

---

### If something goes wrong
- **Error in the sidebar:** read it aloud. Validation errors are a feature (for example, a disconnected stop or too few stops).
- **The app looks frozen:** a large experiment may still be running; wait for the progress bar.
- **"Settings changed" warning:** press **Optimise routes** again.
- **Port in use:** `python -m streamlit run app.py --server.port 8502`.
