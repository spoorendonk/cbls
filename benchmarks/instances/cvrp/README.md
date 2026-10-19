# CVRPLIB X set (#196)

The instance data and pinned best-known solutions for the CVRP benchmark of epic
#182, the worked example for `List` variables. This directory holds data only;
the runner is `benchmarks/cvrp/` (#197).

## Source

- **Instances:** the X set of Uchoa, Pecin, Pessoa, Poggi, Subramanian and Vidal,
  "New benchmark instances for the Capacitated Vehicle Routing Problem", EJOR
  257(3), 2017, <https://doi.org/10.1016/j.ejor.2016.08.012>. 100 instances,
  100 to 1000 customers.
- **Served by:** CVRPLIB, <https://galgos.inf.puc-rio.br/cvrplib/>. Per instance,
  `index.php/en/download/instance/<id>` (TSPLIB `.vrp`) and
  `index.php/en/download/bks/<id>` (`.sol`); the ids, the listed BKS and the
  optimality marking come from the listing page `index.php/en/`. The older
  `vrp.galgos.inf.puc-rio.br/media/com_vrp/...` paths return 404.
- **Snapshot date: 2026-10-14.** CVRPLIB runs an open BKS challenge, so its BKS
  move. The values here are that day's, pinned, and they are the yardstick
  until someone re-snapshots on purpose (below).
- **Terms:** CVRPLIB states no licence for the instance or solution files; the
  site links a terms-of-use page about the site itself. They are published for
  research use. Cite Uchoa et al. (2017) for the instances and CVRPLIB for the
  BKS.

## Files

| File | What it is |
|---|---|
| `X/<name>.vrp` | The 100 instance files, committed (with the `.sol` files, 1.1 MB). |
| `X/<name>.sol` | The BKS solution of each, as CVRPLIB served it on the snapshot date. |
| `manifest.csv` | `instance,n,capacity,k_min,sha256,bytes` per `.vrp`. `n` is the customer count (the TSPLIB `DIMENSION` minus the depot). |
| `roster.csv` | `instance,bks,bks_kind,bks_routes`, all 100. `bks_kind` is `opt` where CVRPLIB marks the BKS proven optimal (61 of 100), `ub` otherwise. `bks_routes` counts the non-empty routes of the pinned `.sol`. |
| `smoke.csv` | Five instances across the size range, for checking that a runner works. Not for results. |
| `dev.csv` | The 20 instances arms and parameters are chosen on (next section). |
| `report.csv` | The other 80: the roster results are reported on. |
| `references.csv` | sha256 and byte count of every `.sol`, the four roster tables and `manifest.csv` (whose `k_min` comes from CVRPLIB's listing, not from a file). |
| `download.py` | How all of the above was produced, and the offline check of it. |

The files are committed rather than fetched on demand, the choice `minlplib`
makes, for the reason `minlplib/HELDOUT.md` gives: a result must be about exactly
these bytes, and CVRPLIB revises what it serves.

```sh
.venv/bin/python benchmarks/instances/cvrp/download.py --verify   # offline
.venv/bin/python benchmarks/instances/cvrp/download.py            # re-fetch, compare to the pins
```

`--verify` checks every pin and also re-derives each roster row from its `.sol`
and each subset table from the draw, so a hand-edited roster fails as surely as a
substituted file. A re-fetch whose instance bytes moved is refused without
`--update-manifest`; one whose BKS, route counts or `.sol` bytes moved is refused
without `--update-references`, which prints the per-instance changes first.
Nothing is written on a refusal. `tests/python/test_cvrp_download.py` pins the
parsers, the committed snapshot and the refusals.

## Conventions

**Distance.** `EDGE_WEIGHT_TYPE: EUC_2D` with TSPLIB rounding:
`d(i, j) = nint(sqrt((x_i - x_j)^2 + (y_i - y_j)^2))`, rounded per edge. Every
BKS cost is in that metric. Every X instance uses it, and `download.py` refuses
any other `TYPE` or `EDGE_WEIGHT_TYPE` rather than guess.

**Fleet.** The `k` in an X name is a **minimum** vehicle count, not a fleet
limit, and the BKS may use more routes: X-n101-k25's uses 26, X-n936-k151's 157.
A model or verifier must not cap the fleet at `k`. `bks_routes` records what
each BKS actually uses.

`k_min` is taken from CVRPLIB's listing, which is not always the name's `k`:
X-n247-k50, X-n524-k153 and X-n670-k130 are listed with minimum fleets of 47,
137 and 126. Their BKS use 51, 155 and 133 routes.

**Fleet size `R` in the model.** `cbls_cvrp` (`benchmarks/cvrp/`) gives the
model `R = k + max(3, ceil(0.1 k))` Lists, `k` being the name's `k`
(`default_fleet` in `benchmarks/cvrp/cvrp_model.h`), unless `--fleet R` says
otherwise. That covers `bks_routes` on all 100 instances (the largest excess is
X-n936-k151's 6 routes, under its 16) and leaves slack for the search to move
through. Unused Lists cost nothing in the objective, an empty route being 0,
but each is a structure the batch samples and moves; what that costs in search
is not measured yet and is #204's to measure on `dev.csv`. Every result row
records its `fleet`.

**Search arm for the smoke check.** At 10 s and seed 1 on the five `smoke.csv`
instances (engine `edc3500`, one run each, a runner check and not a result),
the engine's default structural arm -- `first_improving`, no neighbour list --
was feasible on the three smaller instances but not on X-n801-k40 or
X-n1001-k43. With `--structural-selection violation_guided --neighbours 20`,
the arm #197's smoke evidence used, both were feasible and verified. So check
the runner with that arm. Since #203 the engine's default selection with a
neighbour list is granular best improvement, which beat `violation_guided`
15/15 paired on the smoke roster at 10 s (#203); the driver's `default` arm is
therefore `--neighbours 20` and #204 runs it, with `guided` as the comparison
arm.

## Running and scoring

`benchmarks/cvrp/run_benchmark.py` runs `cbls_cvrp` over a roster x arms x
budgets x seeds x threads, one process per job, resumably;
`benchmarks/cvrp/verify_solution.py` re-checks every `.sol` against its `.vrp`
with no code shared with the C++ runner; and `benchmarks/cvrp/score.py` turns a
results directory into gap-to-BKS tables, per-band summaries and paired A/B
intervals (#199). Each module's docstring states its rules and refusals.

The reference baseline is PyVRP (#200, `benchmarks/cvrp/reference_solve.py`,
pinned in the `benchmarks` extra), run by the driver as the `pyvrp` arm on the
same machine, budget and seeds, over the distance matrix the verifier scores
with. It is single-threaded, so it runs at `--threads 1` and is paired only with
`cbls_cvrp --threads 1`. Hexaly is quoted, not re-run: its published X-set
figures (60 s, AMD Ryzen 7 7700, 8 cores; 0.7% average, 0.1 / 0.5 / 0.6 / 0.8 /
0.9% per size band) are different hardware and threads, and `score.py` prints
them labelled so beside its per-band table.

```sh
.venv/bin/python benchmarks/cvrp/run_benchmark.py --roster smoke --budget 10 \
    --arms guided pyvrp --results /tmp/cvrp-smoke
.venv/bin/python benchmarks/cvrp/score.py /tmp/cvrp-smoke --roster smoke \
    --ab cbls/guided pyvrp
```

## Dev and report rosters

Arms, the fleet rule `R` and the neighbour count `k` are chosen on `dev.csv`, so
a table over all 100 would partly be reported on the instances it was tuned on.
Following `benchmarks/instances/minlplib/HELDOUT.md`, the dev set is drawn with a
seed fixed in advance, and **results are reported on `report.csv`**, the 80
instances outside it. A table that includes dev instances must flag them.

The draw, as `download.py` implements it:

1. Bin the 100 instances into Hexaly's five size bands by customer count:
   `[100, 200)`, `[200, 400)`, `[400, 600)`, `[600, 800)`, `[800, 1000]`.
2. Within each band, order instances by
   `sha256("cbls-cvrp-dev:196:<name>")`. A hash rather than `random.shuffle`,
   whose output Python does not promise across versions.
3. Take the first four of each band: 20 dev instances.

`HELDOUT_SEED = 196` was fixed before any membership was computed, and when the
draw was made no CVRP result of this engine existed on any X instance other than
the single-seed smoke runs on X-n101-k25, X-n251-k28, X-n502-k39 and
X-n1001-k43 quoted in #197 and #201. Of those, X-n1001-k43 fell into `dev.csv`.

`smoke.csv` is independent of the split: it checks that a runner works and
carries no result, so it may overlap either roster (X-n1001-k43 is in both
`smoke.csv` and `dev.csv`).
