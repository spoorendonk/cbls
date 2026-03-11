# CBLS Engine — Context Dump for Claude Code
*Flowty — March 2026. This document captures the full research conversation and project spec for handoff to Claude Code.*

---

## What This Is

A research + implementation project to build an open-source **Constraint-Based Local Search (CBLS) engine** with a high-level, problem-agnostic modeling API modeled on Hexaly (formerly LocalSolver). The engine targets problems that fall between LP-based MIP solvers and dedicated specialist tools — specifically problems with nonlinear objectives, coupled discrete+continuous decisions, or nonconvex feasible regions where LP relaxation is weak or meaningless.

**Repository owner:** Simon Friis Vindum, Flowty  
**Context:** Side project / research artifact at PhD-level OR. Goals are both a publishable artifact and a foundation for commercial tool positioning.

---

## Key Design Decisions (from conversation)

### Why not existing tools

| Tool | Why it doesn't cover the target niche |
|---|---|
| Gurobi / HiGHS | Require piecewise linearization of nonlinear terms; accuracy loss is the point |
| CP-SAT | Integer domains only; no continuous variables; no nonlinear objectives |
| Timefold / OptaPlanner | Discrete planning variables only; no continuous variables; no LP subproblem hook |
| BARON / SCIP | Spatial branch-and-bound with convex relaxations; weak on target problems |
| Hexaly | Commercial monopoly; no open-source equivalent with modeling API exists |

### Meta-solver routing (where this engine sits)

1. Pure VRP → PyVRP / HGS
2. Pure scheduling → Timefold / CP-SAT
3. VRP or scheduling + mild extensions → extend the specialist
4. **Deeply hybrid, nonlinear, or black-box evaluation → this CBLS engine** ←
5. Linear network/flow → dedicated network simplex (Flowty MCF core)

---

## Architecture

### The full engine stack

```
Variables + expression DAG          ← CBLS modeling layer
         ↓
Neighborhood generation             ← derived automatically from variable types
         ↓
Delta evaluation of F = f + λ·V    ← ViolationLS constraint handling
         ↓
Move selection / search strategy    ← SA, tabu, LNS
         ↓
Inner solver hook                   ← exact continuous subproblem on fixed combinatorics
```

### Variable types

| Type | Domain | Auto-generated moves |
|---|---|---|
| `ListVar` | ordered subset of {0..n-1} | insert, remove, swap, reverse, relocate, 2-opt |
| `SetVar` | unordered subset of {0..n-1} | add, remove, swap |
| `IntVar` | [lo, hi] ⊆ ℤ | increment, decrement, random jump |
| `FloatVar` | [lo, hi] ⊆ ℝ | δ-perturbation, or replaced by inner exact solver |
| `BoolVar` | {0, 1} | flip |

### Expression graph

The objective and constraints are DAGs over variables. Key operators:

```
sum(e1..en), prod(e1, e2), pow(e, k)
max(e1, e2), min(e1, e2)
at(list, i), count(list)
lambda(i, f(i))          -- nonlinear term per element of a list
if(cond, e_then, e_else) -- indicator logic
```

The DAG enables **incremental (delta) evaluation**: when a move changes k variables, only the affected subgraph is recomputed. This is what makes move evaluation cheap on nonlinear problems.

### Constraint handling: ViolationLS

Constraints are not enforced by propagation. The engine maintains a violation score V(x) ≥ 0 and searches on the augmented objective:

```
F(x) = f(x) + λ · V(x)
```

where V(x) = Σ max(0, constraint_violation_i(x)) and λ is an adaptive penalty weight. This is exact at λ → ∞; in practice λ is tuned adaptively:
- Increase λ when solutions are consistently infeasible
- Decrease λ when search is stuck in feasible region

ViolationLS is **fully general on nonlinear problems** — it requires only delta evaluation of F, which the DAG provides regardless of the shape of f or V.

### Search loop

```
Phase 1 — FJ-NL (nonlinear greedy violation descent):
    repeat:
        for each variable x_j:
            evaluate delta_V for each candidate value (via DAG delta eval)
        execute move with highest violation reduction
    until V(x) = 0 or stagnation detected
    on stagnation: random perturbation + restart, or hand to Phase 2 with high λ

Phase 2 — ViolationLS:
    minimize F = f + λ·V with adaptive λ
    allow brief infeasibility excursions to escape local optima

Escape — LNS:
    when stuck in local minimum of F:
    destroy subset of variables (random, or guided by violation/cost contribution)
    repair via greedy or inner exact solver
    iterate
```

### Why FJ-NL not FJ

Feasibility Jump (Lethé, Vayatis & Vidal, 2023) is an LP-free MIP feasibility heuristic. Its O(1) per-variable jump score computation relies on **linear constraint structure** — violation of a linear constraint is affine in any single variable, so the optimal value for each variable has a closed form.

For nonlinear constraints this breaks. FJ-NL is the generalization:
- Same greedy violation-reduction selection criterion
- Jump score computed via DAG delta evaluation instead of closed form
- No O(1) guarantee, but O(subgraph depth) per move — still cheap
- For booleans: 2 evaluations per variable, trivially cheap
- For small-domain integers: enumerate domain
- For floats: grid search or gradient step on V

FJ-NL requires nothing beyond the DAG infrastructure ViolationLS already needs. The same delta evaluation mechanism serves both. It is the natural initialization strategy for a general nonlinear CBLS engine.

### Variable decomposition for continuous decisions

When the model contains float variables coupled to discrete decisions, pure LS on floats is inefficient. The right pattern:

1. Fix all discrete/list/set variables to current values
2. Solve float variables exactly given the fixed combinatorial skeleton — LP, QP, or closed-form (e.g., optimal speed on each ship leg from cubic fuel equation)
3. Return (combinatorial + optimal-continuous) as a single point to the LS engine

This is exposed as an **inner solver hook** — a callable that the engine invokes whenever the combinatorial state changes.

---

## Scope

### In scope (v1)
- Expression DAG with typed variables and auto-derived neighborhoods
- Delta evaluation infrastructure
- FJ-NL initialization phase
- ViolationLS with adaptive λ
- LNS with configurable destroy operators
- Inner solver hook interface (caller provides LP or closed-form solver)
- Multi-threaded parallel restarts with shared best solution
- Python API, NumPy-compatible data interface

### Out of scope (v1)
- Full constraint propagation engine (arc consistency, global constraints)
- Competing on pure VRP benchmarks vs. HGS/PyVRP
- GPU-accelerated move evaluation
- JVM interface or cloud API

---

## Benchmark Suite

All four benchmarks are **nonlinear by design**. This is the selection criterion. LP-based methods and CP-SAT fail on these problems precisely because of the nonlinearity — that's why CBLS is the right tool. FJ (the linear version) is therefore a poor fit for all four; FJ-NL and ViolationLS are the right approaches.

### Summary

| Domain | Problem | Core nonlinearity | Benchmark source | Scale |
|---|---|---|---|---|
| Energy | CHPED — Heat/Power Plant Scheduling | Valve-point sinusoidal cost, nonconvex FOR polygons, POZs | Standard literature instances | 4–96 units |
| Energy | ROADEF 2010 — Nuclear Outage Scheduling | Stochastic continuous production coupled to binary outage dates | challenge.roadef.org/2010 | 18–100 plants |
| Maritime | Fleet Bunker + ECA Optimization | Cubic fuel (v³), switching discontinuities, nonlinear tank dynamics | LINER-LIB + synthetic extension | 10–50 vessels |
| Manufacturing | Pharma GLSP + Shelf-Life | Coupled lot-sizing/sequencing, shelf-life cross-stage nonlinearity | Haase-Kimms + arXiv 2602.13668 | 5–20 products, 2–8 lines |

---

### Benchmark 1: CHPED (Combined Heat and Power Economic Dispatch)

**Problem.** Day-ahead scheduling of CHP plants, power-only, and heat-only units over 24–48 hour horizon. Decisions: unit commitment (boolean per unit per period) + dispatch levels (float, power and heat output). Objective: minimize fuel cost.

**Nonlinearity.**
- Feasible operating regions (FORs): nonconvex polygons in heat-power space. LP approximates as convex hull → feasibility distortion
- Valve-point loading: cost = a + b·P + c·P² + |d·sin(e·(P_min − P))| — sinusoidal, nonconvex in P
- Prohibited operating zones (POZs): disconnected feasible sets per unit

**CBLS model sketch.**
- BoolVar per unit per period: on/off commitment
- FloatVar per unit per period: dispatch level (P, H)
- FOR polygon constraints as ViolationLS penalty terms
- Valve-point cost as nonlinear lambda expression over dispatch float vars
- Inner solver hook: given fixed commitment, solve optimal dispatch as nonlinear 1D problem per unit per period

**Why specialist tools fail.**
- CP-SAT: integer domains only; no float dispatch variables
- SCIP: fails on 120+/182 realistic test days at standard time limits
- IPOPT: finds poor local optima on nonconvex instances
- Timefold: no continuous variables

**Instances.** 4, 7, 11, 24, 48, 84, 96 unit systems. All parameters in paper appendices — fully reconstructable. 24-unit/24-hour is canonical medium-scale with 100+ published results. 96-unit is current large-scale frontier.

**Open-source baseline.** SCIP on ≤11 units (exact); DEAP/pymoo GA/PSO (representative metaheuristic). pglib-uc for instance data.

---

### Benchmark 2: ROADEF/EURO 2010 — Nuclear Outage Scheduling

**Problem.** Schedule maintenance and refueling outages for EDF's French nuclear fleet over a multi-year horizon. Two-stage stochastic: first stage decides outage dates (binary), second stage optimizes continuous production levels across 500 demand scenarios to minimize expected cost.

**Nonlinearity.** The coupling between binary outage decisions and continuous production optimization across 500 scenarios is what resists LP/MIP decomposition. Best competition results came from pure local search directly modifying both.

**CBLS model sketch.**
- BoolVar or IntVar per plant: outage start period
- FloatVar per plant per period per scenario: production level
- Stochastic objective: mean cost across scenarios (evaluable as lambda over scenario set)
- Inner solver hook: given fixed outage dates, solve optimal production as LP per scenario

**Instances.** Fully public at challenge.roadef.org/2010. Sets A, B, X. B8/B9 are hardest. 44 teams from 25 countries; full solution comparison baseline available.

**Open-source baseline.** OR-Tools CP-SAT on simplified (deterministic, small) version; Jost & Savourey ILP+heuristic (first prize).

---

### Benchmark 3: Maritime Fleet Bunker + ECA Optimization

**Problem.** Given fixed port rotation (liner) or contracted cargoes (tramp), optimize: sailing speed on each leg (float), bunkering quantity at each port call (float), fuel type at ECA zone boundaries (bool). Minimize total fuel cost subject to time windows and ECA compliance.

**Nonlinearity.**
- Cubic fuel: consumption ∝ v³ × distance/v = v² × distance
- ECA switching: binary fuel-type decisions create cost discontinuities
- Tank level dynamics: fuel(t) = fuel(t−1) − f(v, dist, load) — nonlinear state evolution
- Bunkering arbitrage: optimal buy at port A depends on prices at all downstream ports

**CBLS model sketch.**
- FloatVar per leg: sailing speed
- FloatVar per port call: bunker quantity
- BoolVar per ECA crossing: fuel type (MGO vs VLSFO)
- Cubic cost as pow(speed, 2) × distance expression per leg
- Tank level as running sum constraint (ViolationLS penalty if negative)
- Inner solver hook: given fixed fuel-type switching decisions, solve speed + bunkering jointly as a nonlinear program (closed-form optimal speed per leg known from calculus)

**Tramp shipping note.** The Norstad et al. (2011) benchmark on TSRSP-SO already uses local search + exact speed subproblem. This is not a novel architecture for tramp — it's validation that the CBLS pattern is known to work. What's novel is implementing it in a general modeling framework rather than problem-specific code.

**Instances.** LINER-LIB (Brouer et al., 2014) for liner base structure. Hemmati et al. (2014) for tramp variants. ECA layer + nonlinear fuel parameters added synthetically from IMO zone definitions + Ship & Bunker price data. Vilhelmsen et al. (2024) ALNS matheuristic is primary recent baseline (3.35% gap vs. relaxed bound).

---

### Benchmark 4: Pharma Campaign Scheduling (GLSP + Shelf-Life)

**Problem.** Multi-product, multi-line pharmaceutical production planning. Decisions: production sequence on each line (list of products per line) + lot sizes per campaign (float). Objective: minimize changeover + inventory + tardiness cost subject to demand, capacity, and shelf-life constraints.

**Nonlinearity.**
- Shelf-life coupling: batch expires unless stage-2 starts within W days of stage-1 completion; the gap depends on lot sizes of all intervening campaigns on other lines — nonlinear feasibility condition coupling all float variables across all lines
- Sequence-dependent changeovers interact with lot sizes nonlinearly through capacity constraints

**Why Timefold cannot model this.**
- Timefold has no continuous planning variables — no mechanism for lot sizes
- The standard GLSP decomposition (fix sequence → solve LP for lot sizes) requires calling an LP solver as part of move evaluation; no hook for this in Timefold's architecture
- Shelf-life constraints create nonlinear feasibility conditions that Timefold's discrete constraint model cannot express

**CBLS model sketch.**
- ListVar per manufacturing line: production sequence (list of product indices)
- FloatVar per campaign: lot size
- Changeover cost as lambda over consecutive pairs in each list
- Shelf-life as ViolationLS penalty: max(0, stage2_start − stage1_end − lot_size_weighted_gap)
- Inner solver hook: given fixed sequence, solve optimal lot sizes as LP

**Danish industry connection.** Novo Nordisk (Kalundborg, Hillerød), Leo Pharma (Ballerup), ALK (Hørsholm). All operate multi-product, multi-stage facilities with exactly this structure.

**Instances.** Haase & Kimms (2000): canonical GLSP dataset, 50+ papers. Almada-Lobo et al. parallel-machine extensions. arXiv 2602.13668 (Feb 2026): real pharma plant data, three sizes, directly citable as baseline.

---

## Open-Source Baselines

### General

| Tool | Role | Limitation |
|---|---|---|
| OR-Tools / CP-SAT | Discrete scheduling baseline | Integer domains only; no nonlinear objectives |
| Timefold | Discrete sequencing baseline | No continuous variables |
| SCIP | Exact baseline on small instances | Fails at realistic scale on all four benchmarks |
| DEAP / pymoo | Metaheuristic baseline on CHPED | GA/PSO/DE; matches literature comparisons |
| PyVRP | Routing baseline (maritime) | Fixed speed; no nonlinear fuel |
| CVXPY / Pyomo + IPOPT | Inner solver for continuous subproblems | Not competitive as full solver |

### Per benchmark comparison structure

| Benchmark | Baseline 1 | Baseline 2 | Primary comparison |
|---|---|---|---|
| CHPED | Best GA/PSO/HS from literature | SCIP on ≤11 units | CBLS engine |
| ROADEF 2010 | Jost & Savourey (1st prize) | CP-SAT on simplified version | CBLS engine |
| Maritime | Vilhelmsen et al. ALNS (2024) | PyVRP fixed-speed | CBLS engine |
| Pharma GLSP | Almada-Lobo VNS/TS | Gurobi on small Haase-Kimms + Timefold sequence-only | CBLS engine |

---

## References

### CBLS and Modeling Frameworks
1. **Hexaly / LocalSolver.** hexaly.com. Commercial CBLS; primary reference for modeling API design.
2. **Pisinger, D. & Ropke, S. (2010).** Large neighborhood search. *Handbook of Metaheuristics*, 399–419. Foundational LNS.
3. **Shaw, P. (1998).** Using CP and local search to solve VRP. *CP 1998.* Original LNS.
4. **Laborie, P. et al. (2018).** IBM ILOG CP Optimizer for scheduling. *Constraints* 23(2), 210–250.
5. **Timefold Solver.** timefold.ai / github.com/TimefoldAI/timefold-solver. Discrete CBLS; direct comparison point.

### LP-Free Heuristics
6. **Lethé, G., Vayatis, N. & Vidal, T. (2023/2024).** Feasibility Jump: an LP-free Lagrangian MIP feasibility heuristic. *Mathematical Programming Computation.* Core FJ paper — linear constraints only.
7. **Porumbel, D. (2022).** A ViolationLS approach for weighted constraint satisfaction. *IJCAI 2022.*

### CHPED
8. **Vasebi, A., Fesanghary, M. & Bathaee, S. (2007).** CHPED by harmony search. *Int. J. Electrical Power & Energy Systems* 29(10). Canonical 5-unit instance.
9. **pglib-uc.** github.com/power-grid-lib/pglib-uc. UC instance data.
10. **UnitCommitment.jl.** github.com/ANL-CEEESA/UnitCommitment.jl. Security-constrained UC.

### ROADEF 2010
11. **Porcheron, M. et al. (2010).** ROADEF/EURO 2010 problem specification. EDF R&D.
12. **Jost, V. & Savourey, D. (2013).** ILP approach for nuclear outage scheduling. *Journal of Scheduling* 16(6). First-prize solution.
13. **Instances:** challenge.roadef.org/2010 (sets A, B, X).
14. **rte-france/challenge-roadef-2020.** GitHub. 2020 variant reference implementation.

### Maritime
15. **Brouer, B.D. et al. (2014).** LINER-LIB. *Transportation Science* 48(2), 281–312.
16. **Hemmati, A. et al. (2014).** Benchmark suite for industrial and tramp ship routing. *INFOR* 52(1), 28–38.
17. **Norstad, I., Fagerholt, K. & Laporte, G. (2011).** Tramp ship routing with speed optimization. *Transportation Research Part C* 19(5), 853–865. Establishes LS + inner speed subproblem pattern.
18. **Vilhelmsen, C. et al. (2024).** Fleet repositioning with bunker optimization. *European Journal of Operational Research.* Primary recent baseline.
19. **Stålhane, M. et al. (2025).** Rich TSRSP model: speed, chartering, bunkering, hull cleaning. *Transportation Research Part E.* Most complete current model.
20. **Ship & Bunker.** shipandbunker.com. Historical bunker price data.

### Pharma GLSP
21. **Haase, K. & Kimms, A. (2000).** GLSP with sequence-dependent setups. *European Journal of Operational Research* 125(2). Canonical benchmark.
22. **Fleischmann, B. & Meyr, H. (1997).** The general lotsizing and scheduling problem. *OR Spektrum* 19.
23. **Almada-Lobo, B. et al. (2007).** Multi-product capacitated lot sizing with sequence-dependent setups. *International Journal of Production Economics.*
24. **Shaik, M.A. & Floudas, C.A. (2007).** STN-based biopharma scheduling with shelf-life. *Ind. & Eng. Chemistry Research* 46(5), 1764.
25. **arXiv 2602.13668 (Feb 2026).** Data-driven pharma production scheduling. Real plant data, three snapshot sizes. Primary current baseline.

---

## Implementation Notes for Claude Code

### Where to start

The critical path is the **expression DAG + delta evaluation infrastructure**. Everything else (FJ-NL, ViolationLS, LNS) is built on top of it. Without efficient delta evaluation, move evaluation is O(n) per move and the engine is too slow to be competitive.

Suggested order:
1. Variable types with domain representation
2. Expression node base class + operator nodes (sum, prod, pow, at, lambda, if)
3. DAG construction and topological ordering
4. Full evaluation (forward pass)
5. Delta evaluation (touched-nodes-only forward pass from a change set)
6. Violation expression nodes: max(0, g(x)) terms
7. Augmented objective F = f + λ·V
8. Single-variable move loop (FJ-NL phase)
9. Adaptive λ controller
10. LNS destroy/repair scaffold
11. Inner solver hook interface
12. Python API wrapper

### Language recommendation

C++ core with Python bindings (pybind11). The delta evaluation inner loop is the hot path — needs to be fast. Python API for model construction and result inspection is fine.

Alternatively: pure Python with NumPy for a prototype, then port hot paths.

### Key performance targets

- Delta evaluation per single-variable flip: O(subgraph depth), targeting < 1μs for typical models
- Moves per second on a 96-unit CHPED instance: > 100k/s as a sanity check
- Time to first feasible solution on CHPED 24-unit: < 1 second

### Testing strategy

CHPED is the right first benchmark to implement against:
- Simple binary + float variable structure
- Well-understood nonlinear objective (valve-point formula is explicit)
- 100+ published results for comparison at every scale
- Clear failure mode for LP-based solvers to motivate the CBLS approach

Get CHPED working well first, then the other three follow the same architecture with different expression graphs.
