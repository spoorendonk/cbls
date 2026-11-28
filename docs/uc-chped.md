# Application: unit commitment with valve-point costs

Unit commitment decides which thermal generators run in each hour of a horizon
and how much each one produces, so that demand and a spinning reserve are met at
the least fuel and start-up cost. With the valve-point effect, each unit's fuel
cost carries a term `|d·sin(e·(Pmin − P))|`. The cost is then non-convex and
non-smooth in the dispatch, a term CP-SAT cannot state. This page solves the
published formulation and instances of Pedroso, Kubo and Viana,
[*Unit commitment with valve-point loading effect*](https://arxiv.org/abs/1404.4944)
(2014), and compares cbls with SCIP on them.

## The problem

Units $u \in \mathcal{U}$, periods $t \in \{1, \ldots, T\}$. The commitment
$y_{u,t} \in \{0, 1\}$ and the dispatch $p_{u,t} \geq 0$ are the decisions.

$$\min \sum_{u, t} \left[ y_{u,t} F_u(p_{u,t}) + SC_{u,t} \right], \qquad
F_u(P) = a_u + b_u P + c_u P^2 + \left| d_u \sin\big(e_u (P_u^{\min} - P)\big) \right|$$

subject to, for every $u$ and $t$:

- **demand:** $\sum_u p_{u,t} = D_t$;
- **spinning reserve:** $\sum_u P_u^{\max} y_{u,t} \geq D_t + R_t$, on committed
  capacity, not on dispatch;
- **dispatch limits:** $P_u^{\min} y_{u,t} \leq p_{u,t} \leq P_u^{\max} y_{u,t}$;
- **minimum up and down times**, including what each unit's state before the
  horizon still owes;
- **start-up cost** $SC_{u,t}$: hot if the unit ran within its last
  $\tau_u^{\text{cold}}$ periods, cold otherwise. On these instances every
  start-up is cold.

There are no ramp rates, in the source or here.

| Instance | Units | Periods | Source |
|---|---:|---:|---|
| `ucp13` | 13 | 1-24 | Pedroso et al.'s data, checked field by field |
| `ucp40` | 40 | 1-24 | Pedroso et al.'s data, checked field by field |
| `ucp100` | 100 | 1-24, 48, 168 | this repository's construction from `ucp40` |
| `ucp200` | 200 | 1-24, 48, 168 | this repository's construction from `ucp40` |

Pedroso et al. publish bounds for `ucp13` and `ucp40` only. The construction of
the larger instances, the field-by-field check and the instance files are in
[docs/benchmarks/uc-chped/README.md](https://github.com/spoorendonk/cbls/blob/main/docs/benchmarks/uc-chped/README.md).

### Where cbls's model differs

[benchmarks/uc-chped/FIDELITY.md](https://github.com/spoorendonk/cbls/blob/main/benchmarks/uc-chped/FIDELITY.md)
audits the model equation by equation against the source. The objective, the
start-up costs, the reserve and the minimum up and down times match. One row
differs:

- **Demand is `>=` by default.** The runner solves
  $\sum_u p_{u,t} \geq D_t$, where the source has `=`. That is a relaxation: a
  valve-point cost is not monotone, so producing more than demand can cost less.
  It leaves the proven 1- and 3-period optima unchanged (FIDELITY.md, section
  1.4).
- **`--demand-equality` solves the source's `=`.** It adds
  $\sum_u p_{u,t} - D_t \leq 0$ to every period, the verifier then rejects
  supply above demand too, and each row records `demand=eq`, so the scorer will
  not mix the two models (FIDELITY.md, section 2.4).

The [equal-time results](#results) below use `--demand-equality`; #257's and
#441's cbls columns are on `>=`.

In cbls, each $y_{u,t}$ is a `Bool` and each $p_{u,t}$ a `Float`. The fuel cost
is an expression over `sin` and `abs` nodes, and the hot or cold start-up price
an `if_then_else` over a `max` of the past commitments. The model is
[benchmarks/uc-chped/uc_model.h](https://github.com/spoorendonk/cbls/blob/main/benchmarks/uc-chped/uc_model.h).

## How to run it

The runner is the `cbls_uc_chped` target. Build it from the repository root,
with the build directory configured as in the README:

```bash
cmake --build build --target cbls_uc_chped
```

It reads the instance files in `benchmarks/instances/uc-chped/`. This solves
`ucp13` over 3 periods for 2 seconds with the source's equality demand, checks
the result with the runner's verifier (`--verify`), writes the row to
`ucp13.csv` and the solution to `ucp13.jsonl`:

```bash
./build/cbls_uc_chped benchmarks/instances/uc-chped --instance ucp13 --periods 3 \
    --time-limit 2 --seed 1 --demand-equality --verify \
    --out ucp13.csv --solution-out ucp13.jsonl
```

`check_solutions.py` then checks each solution again, independently of the
engine: it recomputes feasibility and cost from the instance file alone, against
Pedroso et al.'s `=` model at a tolerance of 1e-6, and prints `OK` per
solution:

```bash
.venv/bin/python benchmarks/uc-chped/check_solutions.py benchmarks/instances/uc-chped ucp13.jsonl
```

Leave out `--instance` and `--periods` for the whole roster at its per-horizon
budgets (10 s at 1 period up to 600 s at 168). `--threads N` runs the parallel
portfolio, as the results below do at 16. `cbls_uc_chped --help` lists every
flag; docs/benchmarks/uc-chped/README.md says what each one records, and how
`score.py` scores a multi-seed campaign against the SCIP references. The SCIP
references are in
[benchmarks/uc-chped/pedroso/](https://github.com/spoorendonk/cbls/blob/main/benchmarks/uc-chped/pedroso/README.md),
a GPL-3.0-or-later directory: Pedroso et al.'s Algorithm 1 ported from Gurobi to
SCIP (`pwl`), and SCIP's MINLP with the true valve-point term (`exact`).

## Results

The tables are on the [benchmark results](results.md#uc-chped) page, each
copied from the issue comment that scored it, with its run record. This page
states each one's conditions and what it shows; every number below is from
those tables.

| Table | cbls | Compared against | Conditions |
|---|---|---|---|
| [Equal time and threads](results.md#cbls-against-scip-at-equal-time-and-threads) (#442) | seeds 1, 2 and 42; seed 42 alone on the 4 rows of 48 and 168 periods | `pwl` and `exact` on SCIP 11 preview | demand `=`; 16 threads each; the same wall budget per horizon, 10-600 s; scored by primal integral and final gap |
| [Best known bounds](results.md#best-known-bounds) (#443) | - | - | the best upper and lower bound known per row, from methods an agent wrote around stock SCIP 10 for the purpose; not a timed comparison |
| [cbls against the best known bounds](results.md#cbls-against-the-best-known-bounds) | #442's medians | #443's bounds | computed for that page from the posted numbers |
| [SCIP at one hour](results.md#scip-references-at-one-hour-cbls-on) (#441) | #257's three seeds on `>=`, 10-600 s on 16 threads | `pwl` and `exact` on SCIP 10 | SCIP at 3600 s on one thread; not equal time |
| [Pedroso's published bounds](results.md#cbls-against-pedroso-s-published-bounds-cbls-on) (#257) | seeds 1, 2 and 3 on `>=`, 16 threads, 10-600 s | Pedroso et al.'s Table 2 | `ucp13` and `ucp40` only; superseded for the `=` model by #442 |

**At equal time and threads (#442),** cbls has the lowest primal integral on 23
of the 24 rows: it reaches good solutions first on every instance size. On the
other row, `ucp40` at 3 periods, `pwl`'s 0.0210 is below cbls's median 0.0214
and inside its seed range. By final gap, `pwl` ends best or tied on 21 rows and
cbls alone on the 3 largest (`ucp100` at 168 periods, `ucp200` at 48 and 168).
On the two `ucp200` ones, neither SCIP method has a solution within 600 s.

**Against the best known bounds (#443),** cbls's median ends 0.00 % above the
best known upper bound on the three rows Pedroso et al. proved optimal, and
0.36-0.69 % above on the other `ucp13` rows, 1.05-2.60 % on the other `ucp40`
rows, 1.28-6.45 % on `ucp100` and `ucp200` up to 24 periods, and 6.38-7.76 % on
the four rows of 48 and 168 periods. #443's methods also prove two more rows
optimal: `ucp13` at 6 periods and `ucp40` at 3.

**SCIP at one hour (#441)** shows the references' own limits: `exact` crashed
inside SCIP on six rows (`ucp100` at 12 and 24 periods, `ucp200` at 6, 12, 24
and 48), each reproduced, and `pwl`'s lower bound is not a bound on any `ucp40`,
`ucp100` or `ucp200` row, as some unit breaks the method's condition
`c <= e*f^2/2`.

**Caveats.**
- The long rows of the equal-time table are one cbls seed.
- On the 3 largest rows the equal-time reference is cbls's own final result, so
  its 0.00 gap there is measured against itself; the best known bounds table
  places it 7.3-7.8 % above.
- Read each table at its engine commit: a search-trajectory change invalidates
  every table ([docs/measuring.md](https://github.com/spoorendonk/cbls/blob/main/docs/measuring.md)).
  #442's cbls runs are at `2eb27278`, #257's at `02c78ad`, with the same `src/`
  and `include/`. Six commits have touched those directories since, as of
  `318bc409`, among them an opt-in hook call in the search.
