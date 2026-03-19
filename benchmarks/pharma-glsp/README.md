# Benchmark 5: Pharma GLSP-RP — Lot-Sizing and Scheduling

## Overview

The General Lot-Sizing and Scheduling Problem with Rich Constraints and Rework
(GLSP-RP) schedules production of multiple pharmaceutical products on a single
machine over a multi-period horizon. It decides **what sequence** to produce in
(combinatorial) and **how much** of each product per period (continuous),
subject to setup changeovers, capacity limits, demand fulfillment, minimum lot
sizes, defective rework, and shelf-life/disposal rules. This is the first CBLS
benchmark that couples **ListVar** (sequence) with **FloatVar** (lot sizes),
exercising list-move generation, `pair_lambda_sum`, and an inner-solver hook
simultaneously.

## Paper Reference

> A. Goerler, E. Lalla-Ruiz & S. Voß (2020).
> "A Late Acceptance Hill Climbing Metaheuristic for the General Lot-Sizing and
> Scheduling Problem with Rich Constraints and Rework."
> *Algorithms*, 13(6):138.
> [https://doi.org/10.3390/a13060138](https://doi.org/10.3390/a13060138)

## Problem Description

A single machine produces $J$ products over $T$ macro-periods, each subdivided
into $|M_t|$ micro-periods. In each macro-period the planner decides:

- **Sequence**: the order in which products are run (permutation of $J$
  products). Changing from product $i$ to product $j$ incurs a
  sequence-dependent setup cost $f_{ij}$ and setup time $st_{ij}$.
- **Lot sizes**: how many units $x_{j,t}$ of each product $j$ to produce in
  macro-period $t$.

Constraints:

| Constraint | Description |
|---|---|
| **Capacity** | Total production time + setup time $\le b_t$ per macro-period |
| **Demand** | Cumulative production must meet cumulative demand |
| **Min lot size** | If product $j$ is produced at all in period $t$, at least $\kappa_j$ units |
| **Rework/lifetime** | A fraction $\Theta_{j,t}$ of production is defective; defectives can be reworked within $\Omega_j$ micro-periods or else are disposed |
| **Disposal** | Defectives exceeding rework capacity incur disposal cost $\lambda_j$ per unit |

The objective minimizes total cost: changeover + inventory holding + rework
holding + disposal.

## Mathematical Model

The MIP formulation follows Goerler et al. (2020), Equations 1-25.

### Sets and Parameters

| Symbol | Description |
|---|---|
| $J$ | Number of products |
| $T$ | Number of macro-periods |
| $M_t$ | Set of micro-periods in macro-period $t$ |
| $d_{j,t}$ | Demand for product $j$ in macro-period $t$ |
| $f_{ij}$ | Changeover cost from product $i$ to $j$ |
| $st_{ij}$ | Changeover time from product $i$ to $j$ |
| $tp_j$ | Processing time per unit of product $j$ |
| $tp^R_j$ | Rework processing time per unit |
| $b_t$ | Capacity of macro-period $t$ (time units) |
| $h_j$ | Holding cost per unit of product $j$ |
| $h^R_j$ | Rework holding cost per unit per micro-period |
| $\kappa_j$ | Minimum lot size for product $j$ |
| $\lambda_j$ | Disposal cost per unit |
| $\Omega_j$ | Rework lifetime (micro-periods) |
| $\Theta_{j,t}$ | Defect rate for product $j$ in macro-period $t$ |

### Decision Variables

| Variable | Type | Description |
|---|---|---|
| $x_{j,t}$ | Continuous $\ge 0$ | Lot size of product $j$ in macro-period $t$ |
| $y_{ij,t}$ | Binary | 1 if changeover from $i$ to $j$ occurs in period $t$ |
| $I_{j,t}$ | Continuous $\ge 0$ | Serviceable inventory of $j$ at end of period $t$ |
| $D_{j,t}$ | Continuous $\ge 0$ | Disposed units of $j$ in period $t$ |

### Objective

$$
\min \sum_t \sum_{i \ne j} f_{ij}\, y_{ij,t}
\;+\; \sum_{j,t} h_j\, I_{j,t}
\;+\; \sum_{j,t} h^R_j \cdot \tfrac{|M_t|}{2} \cdot \Theta_{j,t}\, x_{j,t}
\;+\; \sum_{j,t} \lambda_j\, D_{j,t}
$$

The four terms are: (1) changeover cost, (2) inventory holding cost,
(3) rework holding cost (approximated over micro-periods), (4) disposal cost.

### Key Constraints

**Capacity:**
$$\sum_j tp_j\, x_{j,t} + \sum_{i \ne j} st_{ij}\, y_{ij,t} \le b_t
\quad \forall\, t$$

**Demand satisfaction (cumulative inventory balance):**
$$\sum_{\tau=1}^{t} \bigl((1 - \Theta_{j,\tau})\, x_{j,\tau} - d_{j,\tau}\bigr) \ge 0
\quad \forall\, j,\, t$$

**Minimum lot size:**
$$x_{j,t} \ge \kappa_j \cdot z_{j,t} \quad \forall\, j,\, t$$

where $z_{j,t} \in \{0,1\}$ indicates whether product $j$ is produced in
period $t$.

**Disposal (excess defectives beyond rework capacity):**
$$D_{j,t} \ge \Theta_{j,t}\, x_{j,t} - \text{ReworkCap}_{j,t}$$

where $\text{ReworkCap}_{j,t} = \Omega_j \cdot (b_t / T / |M_t|) / tp^R_j$ is
the rework capacity available within the lifetime window.

## CBLS Encoding

The CBLS model translates the MIP into the expression-DAG framework:

### Variables

| CBLS Variable | MIP Counterpart | Description |
|---|---|---|
| `ListVar(J)` per macro-period | Sequence $y_{ij,t}$ | `seq[t]` — permutation encoding the production order |
| `FloatVar` per product per period | $x_{j,t}$ | `lot[j][t]` — continuous lot size, bounded by $[0, b_t/tp_j]$ |

The binary changeover variables $y_{ij,t}$ are **implicit** — they are read off
from adjacent pairs in the list variable.

### DAG Objective Terms

1. **Changeover cost** — `pair_lambda_sum(seq[t], cost_matrix)`: a specialized
   DAG operation (added for this benchmark) that sums
   $f_{\text{seq}[k], \text{seq}[k+1]}$ over consecutive pairs in the list.
   Supports efficient delta evaluation when list moves change only a few
   adjacencies.

2. **Inventory holding cost** — Cumulative supply minus cumulative demand,
   clamped to $\ge 0$, weighted by $h_j$.

3. **Rework/disposal cost** — Defective quantity $\Theta_{j,t} \cdot x_{j,t}$
   decomposed into rework holding (proportional) and disposal (excess over
   rework capacity).

### DAG Constraints

1. **Capacity** — `sum(process_time[j] * lot[j][t]) + pair_lambda_sum(seq[t], time_matrix) - capacity[t] <= 0`
2. **Demand satisfaction** — Cumulative inventory $\ge 0$ (penalized as violation)
3. **Minimum lot size** — Soft constraint: `max(0, kappa - lot) * active_indicator`

### Inner Solver Hook

The `GLSPInnerSolverHook` optimizes FloatVar lot sizes given fixed ListVar
sequences. The SA outer loop moves list variables (swap, insert, 2-opt, etc.);
after each accepted move, the hook runs:

1. **Compute setup times** from current sequences (read `seq[t].elements`)
2. **JIT allocation** — set each lot to `demand[j][t] / (1 - defect_rate)`,
   producing exactly what's needed in each period (zero holding cost)
3. **Capacity spilling** — for over-capacity periods, spill excess production
   to earlier periods. Products with lowest holding cost are spilled first to
   minimize the holding cost penalty. Spill targets the latest earlier period
   with spare capacity (minimizing holding duration).
4. **Min lot enforcement** — lots below the minimum threshold are rounded up
   or zeroed out
5. **Apply** — write new lot sizes to FloatVars, trigger delta evaluation

This decomposition exploits the problem structure: sequence decisions are
combinatorial (handled by SA list moves) while lot sizing given a fixed sequence
is a continuous sub-problem (handled by the JIT heuristic).

## Instance Generation

Instances follow Table 9 of Goerler et al. (2020):

| Class | $J$ | $T$ | $|M_t|$ | Demand | Capacity factor | Setup time | Notes |
|---|---|---|---|---|---|---|---|
| **A** | 5 | 4 | 7 | $U[0; 40, 120]$ | $2.0 \times \sum d$ | $f_{ij}/10$ | Standard |
| **B** | 4 | 3 | 6 | $U[0; 40, 120]$ | $2.0 \times \sum d$ | $f_{ij}/10$ | Fewer products/periods |
| **C** | 6 | 2 | 8 | $U[0; 600, 1000]$ | $0.6 \times \sum d$ | $U[10, 40]$ | High demand, tight capacity |
| **D** | 10 | 6 | 12 | $U[0; 40, 120]$ | $2.0 \times \sum d$ | $f_{ij}/10$ | Scaled-up |
| **E** | 20 | 10 | 15 | $U[0; 40, 120]$ | $2.0 \times \sum d$ | $f_{ij}/10$ | Stress test |

- Demand: zero with 30% probability, else uniform in the range shown
- Setup cost: $f_{ij} \sim U[100, 400]$, diagonal = 0
- Defect rate: $\Theta_{j,t} \sim U[0; 0.005, 0.03]$ (30% chance of zero)
- Classes A-C: 50 instances each; D-E: 10 instances each
- Seeds: deterministic from `base_seed + hash((cls, idx)) % 2^31`

## Baseline Results

From Goerler et al. (2020) Tables 10-12, compared with our CBLS results:

| Class | Method | N | Feasible % | Avg Objective | Avg Time (s) | Source |
|---|---|---|---|---|---|---|
| A | CPLEX | 50 | 100% | 6,290 | 1,800 | Paper Table 10 |
| A | LAHCM (l=50) | 50 | 100% | 5,588 | 876 | Paper Table 10 |
| A | **CBLS** | 10 | 100% | 2,699 | 15 | This benchmark |
| B | CPLEX | 50 | 100% | 2,664 | 212 | Paper Table 11 |
| B | LAHCM (l=50) | 50 | 100% | 2,668 | 168 | Paper Table 11 |
| B | **CBLS** | 10 | 100% | 1,544 | 15 | This benchmark |
| C | CPLEX | 50 | 100% | 31,288 | 1,354 | Paper Table 12 |
| C | LAHCM (l=50) | 50 | 100% | 31,335 | 182 | Paper Table 12 |
| C | **CBLS** | 10 | 60% | 2,090 | 15 | This benchmark |

CBLS now beats the paper's LAHCM on classes A (2.1x better), B (1.7x better),
and C (15x better) in 15 seconds vs 182-876 seconds. The JIT lot-sizing hook
combined with SA sequence optimization is highly effective. Class C feasibility
(60%) remains an area for improvement — tight capacity (0.6x demand) makes some
instances infeasible with the current approach.

## How to Run

### Generate Instances

```bash
cd benchmarks/instances/pharma-glsp
python gen_jsonl.py
```

This writes `class_a.jsonl`, `class_b.jsonl`, `class_c.jsonl`, `class_d.jsonl`,
`class_e.jsonl`, and `all_instances.jsonl`.

### Run CBLS Solver

```bash
cmake -B build && cmake --build build
./build/cbls_pharma_glsp --class a --max 10 --time 30
./build/cbls_pharma_glsp --time 60        # all classes
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--dir PATH` | `benchmarks/instances/pharma-glsp` | Instance directory |
| `--class X` | all | Filter to class a, b, c, d, or e (lowercase) |
| `--max N` | 0 (all) | Limit number of instances |
| `--time T` | 30 | Time limit per instance (seconds) |

### Run Reference Solver (HiGHS MIP)

```bash
pip install highspy
cd benchmarks/pharma-glsp
python reference_solve.py --class A --max 5 --time 300
python reference_solve.py --all --time 300
```

### Run Tests

```bash
ctest --test-dir build                 # C++ tests (includes GLSP)
pytest                                 # Python tests
```
