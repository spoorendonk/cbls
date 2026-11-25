"""Score a UC-CHPED campaign: gaps to Pedroso's bounds over seeds (#363, campaign #257).

`cbls_uc_chped` writes one `comparison.csv` per seed: its `CBLS ViolationLS` rows
beside the `Pedroso MIP (1hr)` rows restated from each instance's bounds. This
module reads one table per seed and writes the campaign's Markdown table. It
solves nothing.

Usage, from any directory:

    .venv/bin/python benchmarks/uc-chped/score.py RESULTS/seed*/comparison.csv \\
        [--out table.md] [--roster NAME:T,T ...] [--scip pwl.jsonl exact.jsonl]

THE MEASURE, one line per (instance, horizon):

* Gap to LB is `100 * (objective - lb) / lb`, the runner's own `gap_pct` formula,
  recomputed from `objective` because that column is rounded to two decimals.
  Gap to UB is the same against Pedroso's one-hour incumbent; below 0 is better
  than it. Each is the median over the seeds that solved it, with [min, max].
* The baseline is Pedroso's own gap on the row, a cited result and not a
  measurement.
* `--scip FILE.jsonl ...` adds a second table from the SCIP references'
  JSON lines (`benchmarks/uc-chped/pedroso/ucp_valve_scip.py`, read as data): each
  method's LB, UB and gap, and cbls's gap to the best SCIP LB and UB, median
  [min, max] over the seeds that solved it. A `pwl` LB from a run where some unit
  breaks `c <= e*f^2/2` is no bound: it is starred and left out of the best LB. A
  pair with no record for a method shows `-` and is named under the table.
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
  solution failed the port's re-check (`check`), or two SCIP versions.

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


@dataclass(frozen=True)
class ScipResult:
    """One `ucp_valve_scip.py` JSON line. `lb`/`ub` are None where it has none."""

    lb: float | None
    ub: float | None
    status: str
    cpu_limit_s: float
    #: `pwl` only: a unit breaks `c <= e*f^2/2`, so the LB is not a bound.
    lb_not_a_bound: bool


@dataclass(frozen=True)
class Scip:
    results: dict[Pair, dict[str, ScipResult]]
    version: str
    cpu_limits: dict[str, set[float]]


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
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RefusedError(f"{where}: {key} {value!r} is not a finite number or null")
    return float(value)


def parse_scip_record(where: str, record: object) -> tuple[str, int, str, str, ScipResult]:
    """(instance, periods, method, SCIP version, result) from one JSON line."""
    if not isinstance(record, dict):
        raise RefusedError(f"{where}: not a JSON object")
    instance, periods, method = record.get("instance"), record.get("periods"), record.get("method")
    status, version = record.get("status"), record.get("scip")
    limit, check = record.get("cpu_limit_s"), record.get("check", [])
    violated = record.get("concavity_violated_units")
    if not (
        isinstance(instance, str)
        and isinstance(periods, int)
        and not isinstance(periods, bool)
        and isinstance(method, str)
        and isinstance(status, str)
        and isinstance(version, str)
        and isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and isinstance(check, list)
        and isinstance(violated, list)
    ):
        raise RefusedError(f"{where}: not a ucp_valve_scip.py record")
    if method not in SCIP_METHODS:
        raise RefusedError(f"{where}: unknown method {method!r}")
    if check:
        raise RefusedError(f"{where}: its solution failed the port's re-check: {check[:3]}")
    result = ScipResult(
        lb=scip_bound(where, record, "lb"),
        ub=scip_bound(where, record, "ub"),
        status=status,
        cpu_limit_s=float(limit),
        lb_not_a_bound=method == "pwl" and bool(violated),
    )
    return instance, periods, method, version, result


def load_scip(paths: Sequence[Path], roster: dict[str, tuple[int, ...]]) -> Scip:
    results: dict[Pair, dict[str, ScipResult]] = {}
    versions: set[str] = set()
    cpu_limits: dict[str, set[float]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise RefusedError(f"{where}: not JSON ({err.msg})") from err
            instance, periods, method, version, result = parse_scip_record(where, record)
            pair = roster_pair(instance, periods, roster)
            if pair[1] not in roster.get(pair[0], ()):
                raise RefusedError(f"{where}: {instance} T={periods} is not in the roster")
            by_method = results.setdefault(pair, {})
            if method in by_method:
                raise RefusedError(f"{where}: {pair[0]} T={periods} {method} is twice")
            by_method[method] = result
            versions.add(version)
            cpu_limits.setdefault(method, set()).add(result.cpu_limit_s)
    if not results:
        raise RefusedError(f"--scip {' '.join(map(str, paths))}: no records")
    return Scip(
        results=results, version=one_value(versions, "SCIP versions"), cpu_limits=cpu_limits
    )


def config_threads(search_config: str) -> str:
    """The portfolio's size: the runner records `threads=N` only off its default of 1."""
    for item in search_config.split(";"):
        key, _, value = item.partition("=")
        if key == "threads":
            return value
    return "1"


def gap_pct(objective: float, bound: float) -> float:
    return 100.0 * (objective - bound) / bound if bound else math.nan


def spread(values: Sequence[float]) -> str:
    """Median [min, max], or `-` when no seed has a value."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "-"
    return f"{statistics.median(finite):.2f} [{min(finite):.2f}, {max(finite):.2f}]"


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
    gap = (
        None
        if result.lb is None or result.ub is None or result.lb_not_a_bound
        else gap_pct(result.ub, result.lb)
    )
    return [number(result.lb, result.lb_not_a_bound), number(result.ub), number(gap)]


def best_scip_bounds(by_method: dict[str, ScipResult]) -> tuple[float | None, float | None]:
    """The highest LB that is a bound, and the lowest UB, over the methods."""
    lbs = [r.lb for r in by_method.values() if r.lb is not None and not r.lb_not_a_bound]
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
        for method, result in sorted(scip.results.get(pair, {}).items())
        if result.ub is None
    ]
    no_bound = [
        f"{pair[0]} T={pair[1]}"
        for pair in pairs
        if (r := scip.results.get(pair, {}).get("pwl")) is not None and r.lb_not_a_bound
    ]
    limits = "; ".join(
        f"{m} {', '.join(f'{s:g}' for s in sorted(scip.cpu_limits[m]))}"
        for m in SCIP_METHODS
        if m in scip.cpu_limits
    )
    lines += [
        "",
        f"- SCIP {scip.version}; CPU limit (s): {limits} (ucp_valve_scip.py records)",
        "- rule: SCIP gap = 100 * (UB - LB) / LB per method; cbls's gaps as above, against "
        "the highest LB that is a bound and the lowest UB over the methods. `pwl` is Pedroso "
        "et al.'s Algorithm 1 ported to SCIP, `exact` SCIP's MINLP with the true valve-point "
        "term. Both solve the source's model (demand =); cbls's has demand >= "
        "(FIDELITY.md section 1.4).",
        f"- pwl LB not a bound (some unit breaks c <= e*f^2/2), starred: {len(no_bound)}",
        *[f"  - {line}" for line in no_bound],
        f"- no SCIP record: {len(missing)}",
        *[f"  - {line}" for line in missing],
        f"- no SCIP solution: {len(unproven)}",
        *[f"  - {line}" for line in unproven],
        f"- cbls below the best SCIP LB, to check before reading the gap: {len(below)}",
        *[f"  - {line}" for line in below],
    ]
    return lines


def render_markdown(
    campaign: Campaign,
    roster: dict[str, tuple[int, ...]],
    published_roster: bool,
    scip: Scip | None = None,
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
        nargs="+",
        default=[],
        metavar="JSONL",
        help="ucp_valve_scip.py --out files: adds the SCIP references' table",
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
        check_out(args.out, [*args.tables, *args.scip])
        roster = parse_roster(args.roster) if args.roster else ROSTER
        campaign = load_campaign(args.tables, roster)
        scip = load_scip(args.scip, roster) if args.scip else None
    except (RefusedError, OSError, UnicodeDecodeError, csv.Error) as err:
        print(f"refused: {err}", file=sys.stderr)
        return EXIT_REFUSED
    text = render_markdown(campaign, roster, published_roster=not args.roster, scip=scip)
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
