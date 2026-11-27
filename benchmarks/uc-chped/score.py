"""Score a UC-CHPED campaign: gaps to Pedroso's bounds over seeds (#363, campaign #257).

`cbls_uc_chped` writes one `comparison.csv` per seed: its `CBLS ViolationLS` rows
beside the `Pedroso MIP (1hr)` rows restated from each instance's bounds. This
module reads one table per seed and writes the campaign's Markdown table. It
solves nothing.

Usage, from any directory:

    .venv/bin/python benchmarks/uc-chped/score.py RESULTS/seed*/comparison.csv \\
        [--out table.md] [--roster NAME:T,T ...] [--scip pwl.jsonl --scip exact.jsonl] \\
        [--trace SEED:anytime_trace.csv ...]

THE MEASURE, one line per (instance, horizon):

* Gap to LB is `100 * (objective - lb) / lb`, the runner's own `gap_pct` formula,
  recomputed from `objective` because that column is rounded to two decimals.
  Gap to UB is the same against Pedroso's one-hour incumbent; below 0 is better
  than it. Each is the median over the seeds that solved it, with [min, max].
* The baseline is Pedroso's own gap on the row, a cited result and not a
  measurement.
* `--scip FILE.jsonl` (once per file) adds a second table from the SCIP references'
  JSON lines (`benchmarks/uc-chped/pedroso/ucp_valve_scip.py`, read as data): each
  method's LB, UB and gap, and cbls's gap to the best SCIP LB and UB, median
  [min, max] over the seeds that solved it. A `pwl` LB from a run where some unit
  breaks `c <= e*f^2/2` is no bound: it is starred and left out of the best LB. A
  pair with no record for a method shows `-` and is named under the table.
* `--trace SEED:CSV` (once per seed: `uc_chped --trace`) adds the equal-time table
  (#442): per row, the primal integral over [0, budget] and the final primal gap of
  cbls (median [min, max] over seeds) and of each SCIP method's driver trajectory,
  against the row's reference, the lowest of Pedroso's UB, every final objective
  and every trajectory point. The gap and integral are
  `benchmarks/common/primal_integral.py`'s. A row with no incumbent anywhere has no
  reference and is named under the table.
* The instances Pedroso did not publish (ucp100, ucp200 and their extensions) have
  objectives and no gaps: `-`, never an invented bound.
* A seed whose row is infeasible, or feasible with no finite objective, has no gap.
  The `solved` column counts it and the seed is named with the row's note. It is a
  result, so it does not refuse.
* A verified objective below Pedroso's LB is a negative gap, and its row is named
  under the table. It is not refused: that LB comes from a piecewise-linear MIP and
  is printed rounded, so a slip below it need not be an error, but it is never a gap
  to read past.

REFUSALS (exit 2, naming what is wrong; nothing is written):

* a roster pair missing from any table, a pair outside the roster, a row twice;
* a measured row that failed verification, or a feasible row never verified;
* a row the runner skipped (`skipped: T > n_periods`), which measured nothing;
* a row that lost portfolio workers, which measured fewer threads than it claims;
* a cell that should be a number and is not, or a measured row with no budget;
* more than one engine commit, search configuration or feasibility tolerance, or a
  commit the runner was not told (empty or `unknown`);
* one horizon at two budgets. The default map gives horizons different budgets, so
  the check is per horizon, across instances and seeds;
* two tables at one seed, or one table at two;
* Pedroso's cited numbers differing between tables, or from a measured row's bounds;
* a table whose columns are not `COLUMNS`: an older published table, or a torn file;
* an `--out` that is one of the tables, or the runner's published `comparison.csv`;
* with `--scip`: a line that is no JSON object of the port's fields, an unknown
  method, a pair outside the roster, a (pair, method) twice, a record whose
  solution failed the port's re-check (`check`), two SCIP versions, backends or
  thread counts, two budgets for one method (driver: for one method and horizon), a
  driver record whose trajectory is not time-ordered finite pairs. A pair whose
  best SCIP LB is above its best SCIP UB is named under the table, not refused. An
  LB that is not positive gives no gap;
* with `--trace`: a file not in the trace's columns, a seed no table has or a seed
  twice, a row no table of that seed has, a commit or budget other than the tables',
  a table row with no trace; with `--scip` too, records that are not the driver's or
  whose threads or per-horizon budget differ from cbls's.

`--roster` replaces the published `ROSTER` for a smoke, and every output says so on
its first line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.primal_integral import primal_gap, primal_integral  # noqa: E402
from benchmarks.common.provenance import REPO_ROOT  # noqa: E402
from benchmarks.common.records import atomic_write, csv_number  # noqa: E402

#: The runner's published table; `--out` never replaces it.
PUBLISHED_COMPARISON = REPO_ROOT / "benchmarks" / "instances" / "uc-chped" / "comparison.csv"

#: `kCsvHeader` in `uc_chped.cpp`; a test pins the two together.
COLUMNS = [
    "instance",
    "periods",
    "method",
    "objective",
    "lb",
    "gap_pct",
    "time_s",
    "source",
    "note",
    "ub",
    "time_limit_s",
    "seed",
    "feasible",
    "verified",
    "max_violation",
    "feas_tol",
    "commit_sha",
    "search_config",
]

#: The runner's default roster (`uc_chped.cpp`, `run_benchmark`); a test pins the two.
ROSTER: dict[str, tuple[int, ...]] = {
    "ucp13": (1, 3, 6, 12, 24),
    "ucp40": (1, 3, 6, 12, 24),
    "ucp100": (1, 3, 6, 12, 24),
    "ucp100-48p": (48,),
    "ucp100-168p": (168,),
    "ucp200": (1, 3, 6, 12, 24),
    "ucp200-48p": (48,),
    "ucp200-168p": (168,),
}

MEASURED_METHOD = "CBLS ViolationLS"
CITED_METHOD = "Pedroso MIP (1hr)"
SKIPPED_NOTE = "skipped:"
LOST_WORKERS_NOTE = "lost-workers"
#: The runner's `--commit` default: no provenance, as an empty cell is none.
NO_COMMIT = ("", "unknown")
EXIT_REFUSED = 2

Pair = tuple[str, int]


class RefusedError(ValueError):
    """The tables cannot honestly give the table; the message says why."""


@dataclass(frozen=True)
class Measured:
    instance: str
    periods: int
    seed: str
    objective: float
    lb: float
    ub: float
    budget: float
    feasible: bool
    note: str

    @property
    def solved(self) -> bool:
        return self.feasible and math.isfinite(self.objective)


@dataclass(frozen=True)
class Cited:
    lb: float
    ub: float
    gap_pct: float


@dataclass(frozen=True)
class Campaign:
    measured: dict[Pair, dict[str, Measured]]
    cited: dict[Pair, Cited]
    seeds: list[str]
    commit: str
    search_config: str
    feas_tol: str
    budgets: dict[int, float]


#: `ucp_valve_scip.py --method`, in column order.
SCIP_METHODS = ("pwl", "exact")


#: `ucp_valve_scip.py --backend`: the wheel in process (#441) or `cbls_scip_traj` (#442).
SCIP_BACKENDS = ("pyscipopt", "driver")

Trajectory = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ScipResult:
    """One `ucp_valve_scip.py` JSON line. `lb`/`ub` are None where it has none."""

    lb: float | None
    ub: float | None
    status: str
    backend: str
    threads: int
    #: pyscipopt: process CPU s (`cpu_limit_s`); driver: wall s (`time_limit_s`).
    budget_s: float
    #: `pwl` only: a unit breaks `c <= e*f^2/2`, so the LB is not a bound.
    lb_not_a_bound: bool
    #: driver only: the running best as (wall s from the method's start, objective).
    trajectory: Trajectory | None = None

    @property
    def bound_lb(self) -> float | None:
        """The LB as a bound to take a gap to: none when it is not one or is not positive."""
        if self.lb is None or self.lb <= 0 or self.lb_not_a_bound:
            return None
        return self.lb


#: `kTraceHeader` in `trace_recorder.h`; a test pins the two together.
TRACE_COLUMNS = [
    "instance",
    "periods",
    "time_limit_s",
    "time_seconds",
    "batches",
    "objective",
    "new_best",
    "commit_sha",
]

#: cbls's incumbents per (pair, seed): (seconds from the solve's start, objective).
Traces = dict[Pair, dict[str, list[tuple[float, float]]]]


@dataclass(frozen=True)
class Scip:
    results: dict[Pair, dict[str, ScipResult]]
    version: str
    backend: str
    threads: int
    #: pyscipopt: one CPU budget per method; driver: one wall budget per (method, horizon).
    budgets: dict[tuple[str, int | None], float]


def read_rows(path: Path) -> list[dict[str, str]]:
    """The table's rows, its `#` provenance block aside, or a refusal if it is not this layout."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        rows = list(reader)
        if reader.fieldnames != COLUMNS:
            raise RefusedError(f"{path}: columns are not the runner's: {reader.fieldnames}")
    for row in rows:
        if None in row or None in row.values():
            raise RefusedError(f"{path}: torn row for {row.get('instance')}")
    return rows


def number_cell(name: str, row: dict[str, str], column: str) -> float:
    """A cell's number, NaN when the runner left it empty or wrote NaN; garbage is refused,
    since `csv_number` alone would read it as "no value" and print it as an unsolved seed."""
    cell = row[column]
    value = csv_number(cell)
    if math.isnan(value) and cell.strip().lower() not in ("", "nan"):
        raise RefusedError(f"{name}: {column} is {cell!r}, not a number")
    return value


def periods_cell(name: str, row: dict[str, str]) -> int:
    try:
        return int(row["periods"])
    except ValueError as err:
        raise RefusedError(f"{name}: periods is {row['periods']!r}") from err


def parse_measured(path: Path, row: dict[str, str]) -> Measured:
    """One `CBLS ViolationLS` row, refused if it is not a verified measurement."""
    name = f"{path}: {row['instance']} T={row['periods']}"
    if row["note"].startswith(SKIPPED_NOTE):
        raise RefusedError(f"{name} measured nothing: {row['note']}")
    if row["verified"] == "false" or row["note"].startswith("verify-fail"):
        raise RefusedError(f"{name} failed verification: {row['note']}")
    if row["note"].startswith(LOST_WORKERS_NOTE):
        raise RefusedError(f"{name} lost portfolio workers: re-run it (#170)")
    if row["feasible"] not in ("true", "false"):
        raise RefusedError(f"{name}: feasible is {row['feasible']!r}")
    feasible = row["feasible"] == "true"
    if feasible and row["verified"] != "true":
        raise RefusedError(f"{name} is feasible and was never verified: run with --verify")
    budget = number_cell(name, row, "time_limit_s")
    # 0 is how the runner writes --no-time-limit.
    if not (math.isfinite(budget) and budget > 0):
        raise RefusedError(f"{name}: no budget in time_limit_s; a campaign row has one")
    return Measured(
        instance=row["instance"],
        periods=periods_cell(name, row),
        seed=row["seed"],
        objective=number_cell(name, row, "objective"),
        lb=number_cell(name, row, "lb"),
        ub=number_cell(name, row, "ub"),
        budget=budget,
        feasible=feasible,
        note=row["note"],
    )


def same_number(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def add_cited(path: Path, row: dict[str, str], cited: dict[Pair, Cited]) -> None:
    name = f"{path}: {row['instance']} T={row['periods']}"
    pair = (row["instance"], periods_cell(name, row))
    new = Cited(
        number_cell(name, row, "lb"),
        number_cell(name, row, "ub"),
        number_cell(name, row, "gap_pct"),
    )
    if cited.setdefault(pair, new) != new:
        raise RefusedError(f"{path}: Pedroso's numbers for {pair} differ from another table's")


def one_value(values: set[str], what: str) -> str:
    if len(values) != 1:
        raise RefusedError(f"mixed {what}: {sorted(values)}")
    return next(iter(values))


def record_table_seed(path: Path, rows: Sequence[Measured], seen: dict[str, Path]) -> None:
    """The one seed a table holds; two tables at one seed would count it twice."""
    seed = one_value({row.seed for row in rows}, f"seeds in {path}")
    if not seed.isdecimal():
        raise RefusedError(f"{path}: the seed cell is {seed!r}")
    if seed in seen:
        raise RefusedError(f"seed {seed} is in both {seen[seed]} and {path}")
    seen[seed] = path


def horizon_budgets(rows: Sequence[Measured]) -> dict[int, float]:
    budgets: dict[int, float] = {}
    for row in rows:
        if budgets.setdefault(row.periods, row.budget) != row.budget:
            raise RefusedError(
                f"horizon T={row.periods} at two budgets: {budgets[row.periods]:g} s "
                f"and {row.budget:g} s ({row.instance}, seed {row.seed})"
            )
    return budgets


def check_roster(
    measured: dict[Pair, dict[str, Measured]], roster: dict[str, tuple[int, ...]], seeds: list[str]
) -> None:
    wanted = {(name, t) for name, horizons in roster.items() for t in horizons}
    outside = sorted(set(measured) - wanted)
    if outside:
        raise RefusedError(f"rows outside the roster: {outside}")
    missing = [
        f"{name} T={t} seed {seed}"
        for name, t in sorted(wanted)
        for seed in seeds
        if seed not in measured.get((name, t), {})
    ]
    if missing:
        raise RefusedError(f"incomplete roster, {len(missing)} missing: " + "; ".join(missing))


def check_bounds(rows: Sequence[Measured], cited: dict[Pair, Cited]) -> None:
    for row in rows:
        reference = cited.get((row.instance, row.periods))
        lb, ub = (reference.lb, reference.ub) if reference else (math.nan, math.nan)
        if not (same_number(row.lb, lb) and same_number(row.ub, ub)):
            raise RefusedError(
                f"{row.instance} T={row.periods} seed {row.seed}: bounds {row.lb:g}/{row.ub:g} "
                f"are not Pedroso's {lb:g}/{ub:g}"
            )


def load_campaign(tables: Sequence[Path], roster: dict[str, tuple[int, ...]]) -> Campaign:
    measured: dict[Pair, dict[str, Measured]] = {}
    cited: dict[Pair, Cited] = {}
    seen: dict[str, Path] = {}
    every: list[Measured] = []
    commits: set[str] = set()
    configs: set[str] = set()
    tolerances: set[str] = set()
    for path in tables:
        rows = read_rows(path)
        own: list[Measured] = []
        for row in rows:
            if row["method"] == CITED_METHOD:
                add_cited(path, row, cited)
                continue
            if row["method"] != MEASURED_METHOD:
                raise RefusedError(f"{path}: unknown method {row['method']!r}")
            parsed = parse_measured(path, row)
            own.append(parsed)
            commits.add(row["commit_sha"])
            configs.add(row["search_config"])
            tolerances.add(row["feas_tol"])
            by_seed = measured.setdefault((parsed.instance, parsed.periods), {})
            if parsed.seed in by_seed:
                raise RefusedError(
                    f"{path}: {parsed.instance} T={parsed.periods} seed {parsed.seed} is twice"
                )
            by_seed[parsed.seed] = parsed
        if not own:
            raise RefusedError(f"{path}: no {MEASURED_METHOD} rows")
        record_table_seed(path, own, seen)
        every += own
    seeds = sorted(seen, key=int)
    check_roster(measured, roster, seeds)
    check_bounds(every, cited)
    commit = one_value(commits, "engine commits")
    if commit in NO_COMMIT:
        raise RefusedError(f"the commit_sha cell is {commit!r}: run with --commit")
    return Campaign(
        measured=measured,
        cited=cited,
        seeds=seeds,
        commit=commit,
        search_config=one_value(configs, "search configurations"),
        feas_tol=one_value(tolerances, "feasibility tolerances"),
        budgets=horizon_budgets(every),
    )


def roster_pair(instance: str, periods: int, roster: dict[str, tuple[int, ...]]) -> Pair:
    """The port names a base instance and a horizon; the roster names `ucp100-48p` for T=48."""
    extended = f"{instance}-{periods}p"
    return (extended if periods in roster.get(extended, ()) else instance, periods)


def scip_bound(where: str, record: dict[str, object], key: str) -> float | None:
    return json_number(where, key, record.get(key))


def json_number(where: str, key: str, value: object) -> float | None:
    """A JSON number as a finite float, None for null; anything else is refused."""
    if value is None:
        return None
    number = math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:  # an integer literal past float's range
            number = math.inf
    if not math.isfinite(number):
        raise RefusedError(f"{where}: {key} {value!r} is not a finite number or null")
    return number


def scip_trajectory(where: str, record: dict[str, object]) -> Trajectory:
    """The driver's running best: finite (seconds, objective) pairs, time not falling."""
    raw = record.get("trajectory")
    if not isinstance(raw, list):
        raise RefusedError(f"{where}: a driver record has no trajectory list")
    points: list[tuple[float, float]] = []
    for item in raw:
        if not (isinstance(item, list) and len(item) == 2):
            raise RefusedError(f"{where}: trajectory point {item!r} is not [seconds, objective]")
        t = json_number(where, "trajectory time", item[0])
        z = json_number(where, "trajectory objective", item[1])
        if t is None or z is None or (points and t < points[-1][0]):
            raise RefusedError(f"{where}: trajectory point {item!r} is null or out of time order")
        points.append((t, z))
    return tuple(points)


def scip_backend(
    where: str, record: dict[str, object]
) -> tuple[str, int, float, Trajectory | None]:
    """(backend, threads, budget, trajectory): records from before #442 are pyscipopt's."""
    backend = record.get("backend", "pyscipopt")
    if backend not in SCIP_BACKENDS:
        raise RefusedError(f"{where}: unknown backend {backend!r}")
    threads = record.get("threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise RefusedError(f"{where}: threads {threads!r} is not a positive integer")
    key = "cpu_limit_s" if backend == "pyscipopt" else "time_limit_s"
    budget = scip_bound(where, record, key)
    if budget is None or budget <= 0:
        raise RefusedError(f"{where}: {key} {record.get(key)!r} is not a positive number")
    trajectory = scip_trajectory(where, record) if backend == "driver" else None
    return str(backend), threads, budget, trajectory


def parse_scip_record(where: str, record: object) -> tuple[str, int, str, str, ScipResult]:
    """(instance, periods, method, SCIP version, result) from one JSON line."""
    if not isinstance(record, dict):
        raise RefusedError(f"{where}: not a JSON object")
    instance, periods, method = record.get("instance"), record.get("periods"), record.get("method")
    status, version = record.get("status"), record.get("scip")
    check = record.get("check", [])
    violated = record.get("concavity_violated_units")
    if not (
        isinstance(instance, str)
        and isinstance(periods, int)
        and not isinstance(periods, bool)
        and isinstance(method, str)
        and isinstance(status, str)
        and isinstance(version, str)
        and isinstance(check, list)
        and isinstance(violated, list)
    ):
        raise RefusedError(f"{where}: not a ucp_valve_scip.py record")
    if method not in SCIP_METHODS:
        raise RefusedError(f"{where}: unknown method {method!r}")
    if record.get("scip_clock") != "wall":
        # The table text says wall-limited; older records ran on SCIP's CPU clock (#441).
        raise RefusedError(f"{where}: scip_clock is not 'wall': a record from before #441's fix")
    if check:
        raise RefusedError(f"{where}: its solution failed the port's re-check: {check[:3]}")
    backend, threads, budget, trajectory = scip_backend(where, record)
    result = ScipResult(
        lb=scip_bound(where, record, "lb"),
        ub=scip_bound(where, record, "ub"),
        status=status,
        backend=backend,
        threads=threads,
        budget_s=budget,
        lb_not_a_bound=method == "pwl" and bool(violated),
        trajectory=trajectory,
    )
    return instance, periods, method, version, result


def load_scip(paths: Sequence[Path], roster: dict[str, tuple[int, ...]]) -> Scip:
    results: dict[Pair, dict[str, ScipResult]] = {}
    versions: set[str] = set()
    backends: set[str] = set()
    threads: set[str] = set()
    limits: dict[tuple[str, int | None], set[float]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except ValueError as err:  # JSONDecodeError, or an integer past Python's digit limit
                raise RefusedError(f"{where}: not JSON ({err})") from err
            instance, periods, method, version, result = parse_scip_record(where, record)
            pair = roster_pair(instance, periods, roster)
            if pair[1] not in roster.get(pair[0], ()):
                raise RefusedError(f"{where}: {instance} T={periods} is not in the roster")
            by_method = results.setdefault(pair, {})
            if method in by_method:
                raise RefusedError(f"{where}: {pair[0]} T={periods} {method} is twice")
            by_method[method] = result
            versions.add(version)
            backends.add(result.backend)
            threads.add(str(result.threads))
            # The driver runs #442's per-horizon wall budgets; the wheel one CPU budget.
            horizon = periods if result.backend == "driver" else None
            limits.setdefault((method, horizon), set()).add(result.budget_s)
    if not results:
        raise RefusedError(f"--scip {' '.join(map(str, paths))}: no records")
    mixed = {
        m if t is None else f"{m} T={t}": sorted(v) for (m, t), v in limits.items() if len(v) != 1
    }
    if mixed:
        raise RefusedError(f"mixed SCIP budgets: {mixed}")
    return Scip(
        results=results,
        version=one_value(versions, "SCIP versions"),
        backend=one_value(backends, "SCIP backends"),
        threads=int(one_value(threads, "SCIP thread counts")),
        budgets={k: next(iter(v)) for k, v in limits.items()},
    )


def parse_trace_arg(item: str) -> tuple[str, Path]:
    seed, _, path = item.partition(":")
    if not seed.isdecimal() or not path:
        raise RefusedError(f"--trace wants SEED:CSV, got {item!r}")
    return seed, Path(path)


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if reader.fieldnames != TRACE_COLUMNS:
            raise RefusedError(f"{path}: columns are not the trace's: {reader.fieldnames}")
    for row in rows:
        if None in row or None in row.values():
            raise RefusedError(f"{path}: torn row for {row.get('instance')}")
    return rows


def add_trace_rows(path: Path, seed: str, campaign: Campaign, traces: Traces) -> None:
    """One seed's trace rows into `traces`, each held to the seed's table."""
    for row in read_trace(path):
        name = f"{path}: {row['instance']} T={row['periods']}"
        pair = (row["instance"], periods_cell(name, row))
        if seed not in campaign.measured.get(pair, {}):
            raise RefusedError(f"{name}: no seed {seed} row in the tables")
        if row["commit_sha"] != campaign.commit:
            raise RefusedError(f"{name}: commit {row['commit_sha']!r}, the tables' is not")
        budget = number_cell(name, row, "time_limit_s")
        if budget != campaign.budgets[pair[1]]:
            raise RefusedError(f"{name}: budget {budget:g} s, the tables' is not")
        t = number_cell(name, row, "time_seconds")
        objective = number_cell(name, row, "objective")
        # An empty objective is the closing row of a solve with no incumbent.
        if not math.isfinite(t) or math.isinf(objective):
            raise RefusedError(f"{name}: time {t} or objective {objective} is not finite")
        points = traces.setdefault(pair, {}).setdefault(seed, [])
        if not math.isnan(objective):
            points.append((t, objective))


def load_traces(items: Sequence[str], campaign: Campaign) -> Traces:
    """cbls's anytime traces (`uc_chped --trace`), one per seed, covering every table row."""
    traces: Traces = {}
    seen: set[str] = set()
    for item in items:
        seed, path = parse_trace_arg(item)
        if seed not in campaign.seeds or seed in seen:
            raise RefusedError(f"--trace {item}: seed {seed} is not a table's, or is twice")
        seen.add(seed)
        add_trace_rows(path, seed, campaign, traces)
    missing = [
        f"{pair[0]} T={pair[1]} seed {seed}"
        for pair, by_seed in campaign.measured.items()
        for seed in by_seed
        if seed not in traces.get(pair, {})
    ]
    if missing:
        raise RefusedError(f"no trace for {len(missing)} rows: " + "; ".join(sorted(missing)))
    return traces


def check_equal_time(campaign: Campaign, scip: Scip) -> None:
    """The primal integrals compare only at one wall budget and thread count (#442)."""
    if scip.backend != "driver":
        raise RefusedError(
            f"--trace scores at equal time: the SCIP records are {scip.backend}'s, "
            "with no trajectory; use ucp_valve_scip.py --backend driver"
        )
    threads = config_threads(campaign.search_config)
    if str(scip.threads) != threads:
        raise RefusedError(f"SCIP ran {scip.threads} threads, cbls {threads}")
    for (method, t), budget in scip.budgets.items():
        # Every driver budget names its horizon, and the roster gives cbls each one.
        cbls = campaign.budgets[t] if t is not None else math.nan
        if budget != cbls:
            raise RefusedError(f"SCIP {method} T={t} ran {budget:g} s, cbls {cbls:g} s")


def config_threads(search_config: str) -> str:
    """The portfolio's size: the runner records `threads=N` only off its default of 1."""
    for item in search_config.split(";"):
        key, _, value = item.partition("=")
        if key == "threads":
            return value
    return "1"


def gap_pct(objective: float, bound: float) -> float:
    return 100.0 * (objective - bound) / bound if bound else math.nan


def spread(values: Sequence[float], fmt: str = ".2f") -> str:
    """Median [min, max], or `-` when no seed has a value."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "-"
    return f"{statistics.median(finite):{fmt}} [{min(finite):{fmt}}, {max(finite):{fmt}}]"


def table_line(pair: Pair, campaign: Campaign) -> str:
    by_seed = campaign.measured[pair]
    solved = [row for row in by_seed.values() if row.solved]
    reference = campaign.cited.get(pair)
    objective = f"{statistics.median(r.objective for r in solved):.1f}" if solved else "-"
    gap_lb = spread([gap_pct(r.objective, r.lb) for r in solved])
    gap_ub = spread([gap_pct(r.objective, r.ub) for r in solved])
    baseline = f"{reference.gap_pct:.2f}" if reference else "-"
    return (
        f"| {pair[0]} | {pair[1]} | {campaign.budgets[pair[1]]:g} | {len(solved)}/{len(by_seed)} "
        f"| {objective} | {gap_lb} | {gap_ub} | {baseline} |"
    )


def number(value: float | None, star: bool = False) -> str:
    return "-" if value is None else f"{value:.2f}{'*' if star else ''}"


def scip_cells(result: ScipResult | None) -> list[str]:
    """LB, UB and gap % of one method's record, `-` where it has none."""
    if result is None:
        return ["-", "-", "-"]
    lb = result.bound_lb
    gap = None if lb is None or result.ub is None else gap_pct(result.ub, lb)
    return [number(result.lb, result.lb_not_a_bound), number(result.ub), number(gap)]


def best_scip_bounds(by_method: dict[str, ScipResult]) -> tuple[float | None, float | None]:
    """The highest LB that is a bound, and the lowest UB, over the methods."""
    lbs = [r.bound_lb for r in by_method.values() if r.bound_lb is not None]
    ubs = [r.ub for r in by_method.values() if r.ub is not None]
    return (max(lbs) if lbs else None, min(ubs) if ubs else None)


def scip_line(pair: Pair, campaign: Campaign, scip: Scip) -> tuple[str, float | None]:
    """The pair's line, and its best SCIP LB for the below-LB check."""
    solved = [row for row in campaign.measured[pair].values() if row.solved]
    by_method = scip.results.get(pair, {})
    best_lb, best_ub = best_scip_bounds(by_method)
    objective = f"{statistics.median(r.objective for r in solved):.1f}" if solved else "-"
    cells = [cell for method in SCIP_METHODS for cell in scip_cells(by_method.get(method))]
    to_lb = "-" if best_lb is None else spread([gap_pct(r.objective, best_lb) for r in solved])
    to_ub = "-" if best_ub is None else spread([gap_pct(r.objective, best_ub) for r in solved])
    line = f"| {pair[0]} | {pair[1]} | {objective} | {' | '.join(cells)} | {to_lb} | {to_ub} |"
    return line, best_lb


def scip_budget_line(scip: Scip) -> str:
    """How the SCIP records were timed, which differs by backend."""
    limits = "; ".join(
        f"{m}{'' if t is None else f' T={t}'} {budget:g}"
        for (m, t), budget in sorted(
            scip.budgets.items(), key=lambda kv: (SCIP_METHODS.index(kv[0][0]), kv[0][1] or 0)
        )
    )
    if scip.backend == "driver":
        return (
            f"- SCIP {scip.version} through cbls_scip_traj at {scip.threads} threads; budget "
            f"(wall s from the method's start, writing and reading included): {limits} "
            "(ucp_valve_scip.py --backend driver records, #442)."
        )
    return (
        f"- SCIP {scip.version}; budget (process CPU s): {limits} (ucp_valve_scip.py records). "
        "Each SCIP solve is limited in wall time to what is left, as ucp_valve.py limited "
        "Gurobi: SCIP's CPU clock is a times() syscall costly enough to overrun the budget "
        "by a third (#441)."
    )


def render_scip(campaign: Campaign, scip: Scip, pairs: Sequence[Pair]) -> list[str]:
    heads = [f"{m} {what}" for m in SCIP_METHODS for what in ("LB", "UB", "gap %")]
    lines = [
        "",
        "## UC-CHPED: SCIP references",
        "",
        f"| instance | T | cbls objective (median) | {' | '.join(heads)} "
        "| cbls gap to best SCIP LB %: median [min, max] "
        "| cbls gap to best SCIP UB %: median [min, max] |",
        "|---|---|---|" + "---|" * len(heads) + "---|---|",
    ]
    below: list[str] = []
    for pair in pairs:
        line, best_lb = scip_line(pair, campaign, scip)
        lines.append(line)
        below += [
            f"{pair[0]} T={pair[1]} seed {seed}: {row.objective:.2f} against {best_lb:.2f}"
            for seed, row in sorted(campaign.measured[pair].items(), key=lambda kv: int(kv[0]))
            if best_lb is not None and row.solved and row.objective < best_lb
        ]
    missing = [
        f"{pair[0]} T={pair[1]} {method}"
        for pair in pairs
        for method in SCIP_METHODS
        if method not in scip.results.get(pair, {})
    ]
    unproven = [
        f"{pair[0]} T={pair[1]} {method}: {result.status}"
        for pair in pairs
        for method in SCIP_METHODS
        if (result := scip.results.get(pair, {}).get(method)) is not None and result.ub is None
    ]
    # A best LB above the best UB is named, not refused: the port's re-check lets a
    # dispatch fall 1e-6 * demand short, which can price a UB just under the optimum.
    crossed = [
        f"{pair[0]} T={pair[1]}: LB {lb:.2f} above UB {ub:.2f}"
        for pair in pairs
        for lb, ub in [best_scip_bounds(scip.results.get(pair, {}))]
        if lb is not None and ub is not None and lb > ub
    ]
    no_bound = [
        f"{pair[0]} T={pair[1]}"
        for pair in pairs
        if (r := scip.results.get(pair, {}).get("pwl")) is not None and r.lb_not_a_bound
    ]
    lines += [
        "",
        scip_budget_line(scip),
        "- rule: SCIP gap = 100 * (UB - LB) / LB per method; cbls's gaps as above, against "
        "the highest LB that is a bound and the lowest UB over the methods. `pwl` is Pedroso "
        "et al.'s Algorithm 1 ported to SCIP, `exact` SCIP's MINLP with the true valve-point "
        "term. Both solve the source's model (demand =); "
        + (
            "so does cbls (--demand-equality, search config demand=eq)."
            if "demand=eq" in campaign.search_config.split(";")
            else "cbls's has demand >= (FIDELITY.md section 1.4)."
        ),
        f"- pwl LB not a bound (some unit breaks c <= e*f^2/2), starred: {len(no_bound)}",
        *[f"  - {line}" for line in no_bound],
        f"- no SCIP record: {len(missing)}",
        *[f"  - {line}" for line in missing],
        f"- no SCIP solution: {len(unproven)}",
        *[f"  - {line}" for line in unproven],
        f"- best SCIP LB above the best SCIP UB, to check: {len(crossed)}",
        *[f"  - {line}" for line in crossed],
        f"- cbls below the best SCIP LB, to check before reading the gap: {len(below)}",
        *[f"  - {line}" for line in below],
    ]
    return lines


def reference_value(
    pair: Pair, campaign: Campaign, scip: Scip | None, traces: Traces
) -> float | None:
    """The row's z*: the lowest of Pedroso's UB, every final objective and trajectory point."""
    cited = campaign.cited.get(pair)
    values = [cited.ub] if cited is not None and math.isfinite(cited.ub) else []
    values += [row.objective for row in campaign.measured[pair].values() if row.solved]
    values += [z for points in traces.get(pair, {}).values() for _, z in points]
    for result in (scip.results.get(pair, {}) if scip else {}).values():
        values += [] if result.ub is None else [result.ub]
        values += [z for _, z in result.trajectory or ()]
    return min(values) if values else None


def integral_cells(
    trajectories: Sequence[Sequence[tuple[float, float]]],
    finals: Sequence[float | None],
    reference: float,
    budget: float,
) -> list[str]:
    """Primal integral over every run, and final primal gap % over the runs with a solution."""
    integrals = [primal_integral(list(points), reference, budget) for points in trajectories]
    gaps = [100.0 * primal_gap(z, reference) for z in finals if z is not None]
    return [spread(integrals, ".4f"), spread(gaps)]


def integral_line(
    pair: Pair, campaign: Campaign, scip: Scip | None, traces: Traces
) -> tuple[str, bool]:
    """The pair's line, and whether it has a reference at all."""
    budget = campaign.budgets[pair[1]]
    head = f"| {pair[0]} | {pair[1]} | {budget:g} |"
    reference = reference_value(pair, campaign, scip, traces)
    if reference is None:
        return head + " - |" + " - |" * 2 * (1 + len(SCIP_METHODS)), False
    by_seed = campaign.measured[pair]
    cells = integral_cells(
        [traces[pair][seed] for seed in by_seed],
        [row.objective if row.solved else None for row in by_seed.values()],
        reference,
        budget,
    )
    for method in SCIP_METHODS:
        result = scip.results.get(pair, {}).get(method) if scip else None
        if result is None or result.trajectory is None:
            cells += ["-", "-"]
            continue
        cells += integral_cells([result.trajectory], [result.ub], reference, budget)
    return f"{head} {reference:.2f} | {' | '.join(cells)} |", True


def render_integrals(
    campaign: Campaign, scip: Scip | None, traces: Traces, pairs: Sequence[Pair]
) -> list[str]:
    heads = [f"{who} {what}" for who in ("cbls", *SCIP_METHODS) for what in ("PI", "final gap %")]
    lines = [
        "",
        "## UC-CHPED: equal time, primal integral",
        "",
        f"| instance | T | budget (s) | reference | {' | '.join(heads)} |",
        "|---|---|---|---|" + "---|" * len(heads),
    ]
    unreferenced: list[str] = []
    for pair in pairs:
        line, referenced = integral_line(pair, campaign, scip, traces)
        lines.append(line)
        if not referenced:
            unreferenced.append(f"{pair[0]} T={pair[1]}")
    lines += [
        "",
        "- rule: benchmarks/common/primal_integral.py's MIPfeas primal gap "
        "|z - ref| / max(|z|, |ref|), 2 before the first incumbent, and PI its time average "
        "over [0, budget]; median [min, max]. cbls's PI is over every seed, its final gap over "
        "the seeds that solved it. The reference is the lowest of Pedroso's UB, every final "
        "objective and every trajectory point, over all seeds and methods.",
        "- cbls: its --trace, seconds from the solve's start, "
        f"{config_threads(campaign.search_config)} threads. SCIP: the driver's running best, "
        "wall seconds from the method's start "
        "(model build and file I/O included). exact's points are SCIP's objective, an upper "
        "bound on that incumbent's true cost; its final gap uses the true cost.",
        f"- no incumbent from any method, so no reference: {len(unreferenced)}",
        *[f"  - {line}" for line in unreferenced],
    ]
    return lines


def render_markdown(
    campaign: Campaign,
    roster: dict[str, tuple[int, ...]],
    published_roster: bool,
    scip: Scip | None = None,
    traces: Traces | None = None,
) -> str:
    lines = [] if published_roster else ["NOT THE PUBLISHED ROSTER (--roster): no result.", ""]
    lines += [
        "## UC-CHPED: gap to Pedroso's bounds",
        "",
        "| instance | T | budget (s) | solved | objective (median) | gap to LB %: median [min, max]"
        " | gap to UB %: median [min, max] | Pedroso gap % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    pairs = [(name, t) for name, horizons in roster.items() for t in horizons]
    lines += [table_line(pair, campaign) for pair in pairs]
    # Roster and seed order, so one campaign gives one table whatever order the tables came in.
    rows = [campaign.measured[pair][seed] for pair in pairs for seed in campaign.seeds]
    unsolved = [
        f"{row.instance} T={row.periods} seed {row.seed}: {row.note or 'no note recorded'}"
        for row in rows
        if not row.solved
    ]
    below_lb = [
        f"{row.instance} T={row.periods} seed {row.seed}: {row.objective:.2f} against {row.lb:.2f}"
        for row in rows
        if row.solved and row.objective < row.lb
    ]
    lines += [
        "",
        f"- engine commit: {campaign.commit} (commit_sha column)",
        f"- seeds: {', '.join(campaign.seeds)} (seed column)",
        f"- cbls threads: {config_threads(campaign.search_config)} (search_config column)",
        f"- search config: {campaign.search_config} (search_config column)",
        f"- feasibility tolerance: {campaign.feas_tol} (feas_tol column)",
        "- rule: gap = 100 * (objective - bound) / bound over the seeds that solved it, and a "
        "seed that did not is counted in `solved` and named below. UB below 0 beats Pedroso's "
        "one-hour incumbent. Pedroso's gap is a cited result (Pedroso et al. 2014, Table 2), "
        "not a measurement. `-`: no published bound, or no seed solved it.",
        f"- no solution: {len(unsolved)}",
        *[f"  - {line}" for line in unsolved],
        f"- below Pedroso's LB, to check before reading the gap: {len(below_lb)}",
        *[f"  - {line}" for line in below_lb],
    ]
    if scip is not None:
        lines += render_scip(campaign, scip, pairs)
    if traces is not None:
        lines += render_integrals(campaign, scip, traces, pairs)
    return "\n".join(lines) + "\n"


def parse_roster(items: Sequence[str]) -> dict[str, tuple[int, ...]]:
    roster: dict[str, tuple[int, ...]] = {}
    for item in items:
        name, _, horizons = item.partition(":")
        if not name or not horizons or name in roster:
            raise RefusedError(f"--roster wants NAME:T,T once per name, got {item!r}")
        try:
            roster[name] = tuple(int(t) for t in horizons.split(","))
        except ValueError as err:
            raise RefusedError(f"--roster wants NAME:T,T, got {item!r}") from err
        if len(set(roster[name])) != len(roster[name]):
            raise RefusedError(f"--roster repeats a horizon in {item!r}")
    return roster


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", type=Path, nargs="+", help="one comparison.csv per seed")
    parser.add_argument("--out", type=Path, help="write the table here as well as printing it")
    parser.add_argument(
        "--roster", nargs="+", default=[], metavar="NAME:T,T", help="a smoke's roster"
    )
    parser.add_argument(
        "--scip",
        type=Path,
        action="append",
        default=[],
        metavar="JSONL",
        help="a ucp_valve_scip.py --out file, once per flag: adds the SCIP references' table",
    )
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        metavar="SEED:CSV",
        help="the seed's uc_chped --trace file, once per seed: adds the equal-time primal "
        "integral table (#442)",
    )
    return parser.parse_args(argv)


def check_out(out: Path | None, tables: Sequence[Path]) -> None:
    """Markdown over a results table would lose the measurements it was scored from."""
    if out is None:
        return
    target = out.resolve()
    if target == PUBLISHED_COMPARISON.resolve() or target in {t.resolve() for t in tables}:
        raise RefusedError(f"--out {out} would overwrite a results table")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        trace_paths = [parse_trace_arg(item)[1] for item in args.trace]
        check_out(args.out, [*args.tables, *args.scip, *trace_paths])
        roster = parse_roster(args.roster) if args.roster else ROSTER
        campaign = load_campaign(args.tables, roster)
        scip = load_scip(args.scip, roster) if args.scip else None
        traces = load_traces(args.trace, campaign) if args.trace else None
        if traces is not None and scip is not None:
            check_equal_time(campaign, scip)
    except (RefusedError, OSError, UnicodeDecodeError, csv.Error) as err:
        print(f"refused: {err}", file=sys.stderr)
        return EXIT_REFUSED
    text = render_markdown(
        campaign, roster, published_roster=not args.roster, scip=scip, traces=traces
    )
    # Printed first: an --out that cannot be written must not lose the table.
    print(text, end="")
    if args.out is not None:
        try:
            atomic_write(args.out, text)
        except OSError as err:
            print(f"cannot write {args.out}: {err}", file=sys.stderr)
            return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
