"""Score the full-library MINLPLib campaign: the GM(gap+1) ratio, cbls over SCIP (#300).

#256 reports one headline number for the full-library campaign: the geometric
mean of (gap_to_bks% + 1) over the instances BOTH solvers solved, cbls over
SCIP, which is the measure #228's smoke used. `campaign_report.py` does not
produce it (it scores one instance directory's `comparison.csv` against one
`scip_baseline.csv`), and the campaign's SCIP half is sharded: one CSV per
instance. This module reads the roster, the cbls table and the SCIP shards, and
prints the ratio with the counts and provenance that make it readable. It
solves nothing.

Usage, from any directory:

    .venv/bin/python3 benchmarks/minlplib/gm_ratio.py \\
        --bounds BOUNDS.csv --cbls comparison.csv --scip-shards SCIP_DIR \\
        [--budget 60] --memory-cap "110G cgroup" [--json summary.json]

`--cbls-staging DIR` reads the driver's staging directory instead of the
assembled table. The driver demotes a staged row to `VERIFY-FAILED` in
`verify_staged` (`run_benchmark.py`), after the solve, and a driver killed in
between leaves the row unchecked. So without `--partial`, every staged
`<instance>.csv` must carry the `<instance>.checked` marker `verify_staged`
writes, holding that file's sha256 (#307); an instance whose only row is a
`.rejected` one needs none, as it is scored as no-solution. Any other row is
refused by name. `check_staged.py` checks and marks the rows a driver left
unchecked, whatever commit staged them, and solves nothing.

The #256 rule is one retry and then no-solution, and the data records it (#312).
Without `--partial`, a scored instance left with a lone `.rejected` row or a SCIP
`.failed` marker needs two recorded failed attempts (`exit <rc> <utc>` lines, in
`<instance>.failures` beside the rejected row and in the `.failed` marker itself);
fewer is refused by name. A campaign staged before #312 has no such lines, so it
scores only with `--partial`. Both drivers leave an instance with two recorded
attempts alone on a resume, so how often someone re-runs cannot turn a final
no-solution into solved (#323); `--retry-final NAME` asks for another attempt.

The SCIP shards are `scip_shards.py`'s output: `<instance>.csv`, or an
`<instance>.failed` marker with one `exit <rc> <utc>` line per failed attempt.

THE MEASURE, kept exactly as #228's smoke defined it (`MEASURE_RULE` below
states it in every output):

* gap+1 of a solver's row is `max(gap_to_bks%, 0) + 1`: a row better than the
  published bound scores as matching it. Below |BKS| 1e-12 the runner's gap
  column is an absolute residual (`reference_solve.safe_gap`), used as written.
* `runner.CLAIM_EXCLUDED` instances are out of every count and aggregate (the
  smoke's n=48 of 50), and are listed by name on their own line.
* GM is the geometric mean of gap+1 over the GM POOL: instances both solvers
  solved, with a finite `primal_bks` in `bounds.csv`. The ratio is
  GM_cbls / GM_SCIP; below 1 is cbls ahead.
* Over the same pool, cbls WINS an instance when its gap+1 is more than 1%
  smaller than SCIP's (`(gc+1) < (gs+1) / WIN_FACTOR`), LOSES when SCIP's is more
  than 1% smaller, and ties otherwise. Instances only one solver solved are
  counted beside them in brackets (only-cbls, only-SCIP); those need no gap, so
  instances without a finite BKS are in the brackets.

SOLVED means, the same rule on both sides (decided at the hub, 2026-11-01): a
row with `feasible == true` (for cbls, also a note that is not `VERIFY-FAILED`).
Every other scored instance is that solver's NO SOLUTION, listed by name with
its reason: the row's note (`read-error(...)`, `infeasible`, `no-solution(...)`,
`CHECK-FAILED(...)`, ...), the last exit in a SCIP `<instance>.failed` marker (the
solve process died after its retry), or, in a cbls staging directory,
`rejected: <note>` for an instance whose only row is the driver's
`<instance>.csv.rejected` (its runner exited nonzero). A whole `<instance>.csv`
beside a `.rejected` is a re-solve and wins. `cbls_solved` and `scip_solved` are
the one place that decides it.

PROVENANCE comes from the data: the engine commit and thread count from the cbls
rows, the SCIP version from its rows, and the budget from SCIP's `scip_version`
cell (`... / 60s / ...`) and the cbls side's record (the staging stamp's
`time-limit=`, or the run record beside `comparison.csv`). Two budgets that
disagree are refused, and so is a `--budget` that disagrees with them. The shard
stamp's `seed=` must match the seed in SCIP's `scip_version` cells too: `.failed`
markers carry neither, so the stamp is all that says what they failed under.
`--memory-cap` is free text: the cap is not in the data.

COMPLETENESS (CLAUDE.md, Measuring): the roster is read first, and by default a
roster instance without a cbls row or without a SCIP CSV or failure marker is
refused (exit 2), as is any results file naming an instance outside the roster
or naming one twice, and any scored failure with too few recorded attempts.
`--partial` scores the instances present on both sides instead, and says so on
the first line of every output. Nothing relaxes the unknown-instance and
duplicate refusals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import (  # noqa: E402
    FAILED_ATTEMPTS_REQUIRED,
    atomic_write,
    recorded_attempts,
)
from benchmarks.minlplib.campaign_report import (  # noqa: E402
    Row,
    ScipRow,
    file_sha256,
    load_results,
    load_run_record,
    load_scip,
)
from benchmarks.minlplib.reference_solve import Bound, load_bounds  # noqa: E402
from benchmarks.minlplib.run_benchmark import (  # noqa: E402
    ASSEMBLED_TABLE,
    ASSEMBLED_TRACE,
    FAILURES_SUFFIX,
    REJECTED_SUFFIX,
    SEARCH_CONFIG_THREADS,
    STAMP_NAME,
    checked_digest,
)
from benchmarks.minlplib.runner import CLAIM_EXCLUDED  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: A win or loss needs the loser's gap+1 to exceed the winner's by more than 1%.
WIN_FACTOR = 1.01

#: Exit status of every refusal: an incomplete roster, an unknown or duplicated
#: instance, mixed provenance.
EXIT_REFUSED = 2

#: How many names a refusal or a list prints before it says how many more.
NAMES_SHOWN = 10

#: The suffixes the SCIP shard directory holds beside the per-instance CSV.
SCIP_MERGED_SUFFIX = ".merged.csv"
SCIP_FAILED_SUFFIX = ".failed"

#: Staged per-instance trace files, beside the per-instance rows.
STAGED_TRACE_SUFFIX = ".trace.csv"

#: A staged row the driver set aside after a nonzero runner exit.
STAGED_REJECTED_SUFFIX = ".csv" + REJECTED_SUFFIX

#: The budget field of SCIP's `scip_version` cell: `SCIP 10.0.2 / ... / 60s / seed 0`.
SCIP_BUDGET_FIELD = re.compile(r"/\s*([0-9]+(?:\.[0-9]*)?)s\s*/")

#: The staging stamp's budget line (`run_benchmark.staging_stamp`).
STAMP_BUDGET_KEY = "time-limit"
#: The seed field of SCIP's `scip_version` cell, and the shard stamp's seed line.
SCIP_SEED_FIELD = re.compile(r"/\s*seed\s+([0-9]+)\s*$")
STAMP_SEED_KEY = "seed"

NOT_STATED = "not stated"
NO_NOTE = "no note recorded"

MEASURE_RULE = (
    f"{', '.join(CLAIM_EXCLUDED)} (claim-excluded) out of every count and aggregate. "
    "GM = geometric mean of (max(gap_to_bks%, 0) + 1) over instances both solvers solved, "
    "with a finite BKS; ratio = GM_cbls / GM_SCIP (below 1: cbls ahead). Win/loss: gap+1 "
    "more than 1% smaller than the other's, same pool; brackets: instances only that solver "
    "solved (no BKS needed). Solved = a feasible row (cbls: and not VERIFY-FAILED); anything "
    "else is that solver's no-solution, listed with its reason, on both sides alike: the "
    "row's note, a SCIP .failed marker's exit, or a cbls staged row the driver rejected."
)


class RefusedError(ValueError):
    """The inputs cannot produce a result; the message says why."""


# --------------------------------------------------------------------------
# The measure.
# --------------------------------------------------------------------------


def gap_plus_one(gap_pct: float) -> float:
    """A solved row's gap+1, a better-than-BKS row clamped to 1."""
    return max(gap_pct, 0.0) + 1.0


def gm_gap_plus_one(gaps_pct: Sequence[float]) -> float:
    """Geometric mean of gap+1 over `gaps_pct`; NaN for an empty pool."""
    if not gaps_pct:
        return math.nan
    return math.exp(statistics.fmean(math.log(gap_plus_one(g)) for g in gaps_pct))


def head_to_head(gap_cbls: float, gap_scip: float) -> int:
    """+1 when cbls wins the instance, -1 when SCIP does, 0 for a tie (within 1%)."""
    c, s = gap_plus_one(gap_cbls), gap_plus_one(gap_scip)
    if c < s / WIN_FACTOR:
        return 1
    if s < c / WIN_FACTOR:
        return -1
    return 0


def cbls_solved(name: str, cbls: CblsSide) -> bool:
    """Whether cbls solved `name`: a verified feasible row. A rejected row is no solution."""
    if name in cbls.rejected:
        return False
    row = cbls.rows[name]
    return row.feasible and not row.verification_failed


def scip_solved(name: str, scip: ScipSide) -> bool:
    """Whether SCIP solved `name`. A `.failed` marker is no solution, here and only here."""
    if name in scip.failures:
        return False
    return scip.rows[name].feasible


def cbls_reason(name: str, cbls: CblsSide) -> str:
    """Why cbls has no solution for `name`: the rejection, else the row's note."""
    if name in cbls.rejected:
        return cbls.rejected[name]
    return cbls.rows[name].note or NO_NOTE


def scip_reason(name: str, scip: ScipSide) -> str:
    """Why SCIP has no solution for `name`: the failure marker's exit, else the row's note."""
    if name in scip.failures:
        return scip.failures[name]
    row = scip.rows[name]
    return row.note or (f"status {row.status}" if row.status else NO_NOTE)


def reason_category(reason: str) -> str:
    """A reason's prefix before its first `(` or `:`, e.g. `read-error` or `exit 134`.

    A one-word parenthetical is a solver status and stays, so `no-solution(timelimit)`
    and `no-solution(infeasible)` are two groups; free text, as in
    `read-error(OSError: x)`, is dropped. Annotations `reference_solve.annotate`
    appends after `"; "` (catalogue mismatches) do not change the group.
    """
    reason = reason.split("; ", 1)[0]
    status = re.fullmatch(r"([^(:]+\([\w-]+\))", reason.strip())
    if status:
        return status.group(1).strip()
    return re.split(r"[(:]", reason, maxsplit=1)[0].strip() or reason


# --------------------------------------------------------------------------
# Inputs.
# --------------------------------------------------------------------------


def names_list(names: Iterable[str], shown: int = NAMES_SHOWN) -> str:
    """Up to `shown` names, then how many more there are."""
    ordered = list(names)
    head = ", ".join(ordered[:shown])
    rest = len(ordered) - shown
    return f"{head}, ... (+{rest} more)" if rest > 0 else head


def load_roster(path: Path) -> list[Bound]:
    """`bounds.csv` in file order, refusing an empty roster and a duplicated instance."""
    bounds = load_bounds(path)
    if not bounds:
        raise RefusedError(f"{path}: the roster is empty")
    seen: set[str] = set()
    dups: list[str] = []
    for b in bounds:
        if b.instance in seen:
            dups.append(b.instance)
        seen.add(b.instance)
    if dups:
        raise RefusedError(f"{path}: instance(s) listed twice in the roster: {names_list(dups)}")
    return bounds


def has_whole_row(path: Path) -> bool:
    """Whether a per-instance CSV holds its one whole row; False for a header-only or torn one.

    A solve writes its header before it solves anything, and a killed or running
    one leaves a header-only file or a torn last line: that instance is not there
    yet. More than one row is a malformed file and refused.
    """
    text = path.read_text()
    lines = list(csv.reader(text.splitlines()))
    header, rows = lines[:1], lines[1:]
    if len(rows) > 1:
        raise RefusedError(f"{path}: holds {len(rows)} rows, a per-instance file holds one")
    return (
        bool(header) and len(rows) == 1 and len(rows[0]) == len(header[0]) and text.endswith("\n")
    )


def _check_stem(path: Path, stem: str, recorded: str, roster: Mapping[str, Bound]) -> None:
    if recorded != stem:
        raise RefusedError(f"{path}: the row is for {recorded!r}, the file is named for {stem!r}")
    if stem not in roster:
        raise RefusedError(f"{path}: instance {stem!r} is not in the roster")


def load_cbls_table(path: Path, roster: Mapping[str, Bound]) -> dict[str, Row]:
    """The assembled `comparison.csv`, by instance. Unknown and duplicate rows are refused."""
    try:
        rows = load_results(path, roster)
    except ValueError as exc:
        raise RefusedError(str(exc)) from exc
    return {r.instance: r for r in rows}


@dataclass(frozen=True)
class CblsSide:
    """The cbls rows by instance, and the staged instances whose only row was rejected."""

    rows: dict[str, Row]
    #: instance -> `rejected: <note of the row inside>`.
    rejected: dict[str, str] = field(default_factory=dict)
    #: Rejected instance -> how many failed attempts its record holds.
    attempts: dict[str, int] = field(default_factory=dict)

    def present(self) -> set[str]:
        return set(self.rows) | set(self.rejected)


def _staged_csvs(stage: Path) -> list[Path]:
    """The per-instance row files of a staging directory, without traces or assembled tables.

    `*.rejected` files (a quarantined row) end in `.rejected`, so a `*.csv` glob
    never sees them; `_staged_rejected` reads those.
    """
    skip = {ASSEMBLED_TABLE, ASSEMBLED_TRACE}
    return sorted(
        p
        for p in stage.glob("*.csv")
        if p.name not in skip and not p.name.endswith(STAGED_TRACE_SUFFIX)
    )


def _staged_rejected(stage: Path) -> list[Path]:
    """The quarantined rows (`<instance>.csv.rejected`), without quarantined traces."""
    return sorted(
        p
        for p in stage.glob(f"*{STAGED_REJECTED_SUFFIX}")
        if not p.name.endswith(STAGED_TRACE_SUFFIX + REJECTED_SUFFIX)
    )


def _staged_row(path: Path, stem: str, roster: Mapping[str, Bound]) -> Row | None:
    """The one row of a staged file, None while it is header-only or torn."""
    if stem not in roster:
        raise RefusedError(f"{path}: instance {stem!r} is not in the roster")
    if not has_whole_row(path):
        return None
    (row,) = load_cbls_table(path, roster).values()
    _check_stem(path, stem, row.instance, roster)
    return row


def load_cbls_staging(stage: Path, roster: Mapping[str, Bound]) -> CblsSide:
    """The driver's staging directory, by instance; an in-progress file is absent.

    An instance whose staged row the driver rejected (its runner exited nonzero)
    is cbls no-solution with the rejected row's note, unless a whole
    `<instance>.csv` stands beside it: that is a later re-solve, and it wins.
    """
    if not stage.is_dir():
        raise RefusedError(f"{stage}: not a directory")
    rows: dict[str, Row] = {}
    for path in _staged_csvs(stage):
        stem = path.name.removesuffix(".csv")
        row = _staged_row(path, stem, roster)
        if row is not None:
            rows[stem] = row
    rejected: dict[str, str] = {}
    attempts: dict[str, int] = {}
    for path in _staged_rejected(stage):
        stem = path.name.removesuffix(STAGED_REJECTED_SUFFIX)
        if stem in rows:
            continue
        row = _staged_row(path, stem, roster)
        note = "no whole row" if row is None else row.note or NO_NOTE
        rejected[stem] = f"rejected: {note}"
        attempts[stem] = len(recorded_attempts(stage / f"{stem}{FAILURES_SUFFIX}"))
    return CblsSide(rows=rows, rejected=rejected, attempts=attempts)


@dataclass(frozen=True)
class ScipSide:
    """The SCIP shards: one row per solved instance, the last exit per failed one."""

    rows: dict[str, ScipRow]
    failures: dict[str, str]
    #: Failed instance -> how many failed attempts its marker holds.
    attempts: dict[str, int] = field(default_factory=dict)

    def present(self) -> set[str]:
        return set(self.rows) | set(self.failures)


def load_scip_shards(shards: Path, roster: Mapping[str, Bound]) -> ScipSide:
    """One `<instance>.csv` or `<instance>.failed` per instance; `.merged.csv` and logs ignored."""
    if not shards.is_dir():
        raise RefusedError(f"{shards}: not a directory")
    rows: dict[str, ScipRow] = {}
    for path in sorted(shards.glob("*.csv")):
        if path.name.endswith(SCIP_MERGED_SUFFIX):
            continue
        stem = path.name.removesuffix(".csv")
        if stem not in roster:
            raise RefusedError(f"{path}: instance {stem!r} is not in the roster")
        if not has_whole_row(path):
            continue
        try:
            (row,) = load_scip(path).values()
        except ValueError as exc:
            raise RefusedError(f"{path}: {exc}") from exc
        _check_stem(path, stem, row.instance, roster)
        rows[stem] = row
    failures: dict[str, str] = {}
    attempts: dict[str, int] = {}
    for path in sorted(shards.glob(f"*{SCIP_FAILED_SUFFIX}")):
        stem = path.name.removesuffix(SCIP_FAILED_SUFFIX)
        if stem not in roster:
            raise RefusedError(f"{path}: instance {stem!r} is not in the roster")
        if stem in rows:
            raise RefusedError(f"{path}: {stem!r} has both a result row and a failure marker")
        exits = recorded_attempts(path)
        attempts[stem] = len(exits)
        failures[stem] = exits[-1] if exits else path.read_text().strip() or "no exit recorded"
    return ScipSide(rows=rows, failures=failures, attempts=attempts)


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------


def row_threads(row: Row) -> int:
    """The thread count in `search_config`; a cell without one is a 1-thread row."""
    match = SEARCH_CONFIG_THREADS.search(row.search_config)
    return int(match.group(1)) if match else 1


def _one(values: Iterable[str], what: str) -> str:
    """The single value every row agrees on, or a refusal naming them all."""
    distinct = sorted(set(values))
    if len(distinct) > 1:
        raise RefusedError(f"mixed {what}: {', '.join(repr(v) for v in distinct)}")
    return distinct[0] if distinct else "none"


@dataclass(frozen=True)
class Budget:
    """The per-instance budget and where it was read; `seconds` None when nothing says."""

    seconds: float | None = None
    sources: list[str] = field(default_factory=list)


def scip_budget(scip: ScipSide) -> float | None:
    """The `<N>s` field of SCIP's `scip_version` cells; None when no cell carries one."""
    found: set[float] = set()
    for row in scip.rows.values():
        match = SCIP_BUDGET_FIELD.search(row.version)
        if match:
            found.add(float(match.group(1)))
    if len(found) > 1:
        raise RefusedError(f"mixed SCIP budgets: {', '.join(f'{b:g}s' for b in sorted(found))}")
    return found.pop() if found else None


def _stamp_number(stage: Path, wanted: str) -> float | None:
    """The `wanted=` line of a directory's stamp; None without a stamp, refused without the line."""
    stamp = stage / STAMP_NAME
    if not stamp.exists():
        return None
    for line in stamp.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == wanted:
            try:
                return float(value)
            except ValueError as exc:
                raise RefusedError(f"{stamp}: {wanted}={value!r} is not a number") from exc
    raise RefusedError(f"{stamp}: has no {wanted}= line")


def staging_budget(stage: Path) -> float | None:
    """The `time-limit=` line of a staging directory's stamp; None when it has no stamp."""
    return _stamp_number(stage, STAMP_BUDGET_KEY)


def reconcile_shard_seed(scip: ScipSide, shards: Path) -> None:
    """Refuse a shard stamp whose `seed=` is not the seed in SCIP's rows (#323).

    A `.failed` marker carries no seed, so a `--restart --seed 1` beside a seed 0
    campaign's rows would otherwise read its own failures as the campaign's.
    Mixed row seeds are `mixed SCIP versions`, refused where the versions are read.
    """
    stamped = _stamp_number(shards, STAMP_SEED_KEY)
    matches = (SCIP_SEED_FIELD.search(row.version) for row in scip.rows.values())
    seeds = sorted({float(m.group(1)) for m in matches if m})
    if stamped is not None and seeds and seeds != [stamped]:
        rows = ", ".join(f"{s:g}" for s in seeds)
        raise RefusedError(f"SCIP scip_version seed {rows}, but SCIP shard stamp seed {stamped:g}")


def table_budget(table: Path) -> float | None:
    """The budget in the run record beside an assembled table; None when it has none."""
    try:
        record = load_run_record(table)
    except ValueError as exc:
        raise RefusedError(str(exc)) from exc
    return None if record is None else record.budget_seconds


def reconcile_budget(derived: Mapping[str, float | None], stated: float | None) -> Budget:
    """The one budget the data records, refusing two that differ or a `--budget` that differs.

    `derived` maps a source (`SCIP scip_version`, `cbls staging stamp`, ...) to
    what it records. A stated budget with nothing in the data to check it is
    reported as stated.
    """
    known = {src: b for src, b in derived.items() if b is not None}
    if len(set(known.values())) > 1:
        sides = ", ".join(f"{src} {b:g}s" for src, b in known.items())
        raise RefusedError(f"the two sides' budgets differ: {sides}")
    if known:
        seconds = next(iter(known.values()))
        if stated is not None and stated != seconds:
            raise RefusedError(
                f"--budget {stated:g}s, but the data records {seconds:g}s ({', '.join(known)})"
            )
        return Budget(seconds=seconds, sources=list(known))
    if stated is not None:
        return Budget(seconds=stated, sources=["--budget, not in the data"])
    return Budget()


@dataclass(frozen=True)
class Provenance:
    commit_sha: str
    cbls_threads: int | None
    scip_version: str
    budget: Budget
    memory_cap: str
    #: "on", "on-no-aggregation", "off" or "none" (no rows), as `_one` reads it: a
    #: presolved run's rows record `presolve` in `search_config` (#343), and its
    #: aggregation off arm `no-presolve-aggregation` too (#347).
    cbls_presolve: str = "off"


def provenance(
    cbls: CblsSide, scip: ScipSide, budget: Budget | None, memory_cap: str | None
) -> Provenance:
    """Read from every row loaded, scored or not: a stray row is still a stray.

    A rejected staged row is not a result and is not read: an exit-3 row can
    lack the search configuration a solved row records.
    """
    rows = cbls.rows.values()
    threads = _one((str(row_threads(r)) for r in rows), "cbls thread counts")
    return Provenance(
        commit_sha=_one((r.commit_sha for r in rows), "engine commits"),
        cbls_threads=None if threads == "none" else int(threads),
        scip_version=_one((r.version for r in scip.rows.values()), "SCIP versions"),
        budget=budget or Budget(),
        memory_cap=memory_cap or NOT_STATED,
        cbls_presolve=_one(
            (presolve_setting(r.search_config) for r in rows), "cbls presolve settings"
        ),
    )


def presolve_setting(search_config: str) -> str:
    """`on`, `on-no-aggregation` (#347) or `off`, as a row's `search_config` records it.

    Aggregation's off arm is its own setting, so a table mixing it with the on
    arm is refused like one mixing presolve with the plain engine.
    """
    tokens = search_config.split(";")
    if "presolve" not in tokens:
        return "off"
    return "on-no-aggregation" if "no-presolve-aggregation" in tokens else "on"


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    n_roster: int
    #: The roster less its claim-excluded instances: what a complete campaign scores.
    n_counted: int
    #: Instances present on both sides, claim-excluded out.
    n_scored: int
    partial: bool
    cbls_feasible: int
    scip_feasible: int
    both_feasible: int
    gm_pool: int
    gm_cbls: float
    gm_scip: float
    ratio: float
    wins: int
    losses: int
    ties: int
    only_cbls: int
    only_scip: int
    provenance: Provenance
    claim_excluded: list[str] = field(default_factory=list)
    no_bks: list[str] = field(default_factory=list)
    #: Scored instance -> why that solver has no solution, in roster order.
    cbls_no_solution: dict[str, str] = field(default_factory=dict)
    scip_no_solution: dict[str, str] = field(default_factory=dict)
    missing_cbls: list[str] = field(default_factory=list)
    missing_scip: list[str] = field(default_factory=list)


def _missing(roster: Sequence[Bound], present: set[str]) -> list[str]:
    return [b.instance for b in roster if b.instance not in present]


def completeness_refusal(missing_cbls: Sequence[str], missing_scip: Sequence[str]) -> str | None:
    """The refusal for an incomplete campaign, or None when every roster instance is there."""
    parts = []
    if missing_cbls:
        parts.append(f"{len(missing_cbls)} without a cbls row: {names_list(missing_cbls)}")
    if missing_scip:
        parts.append(
            f"{len(missing_scip)} without a SCIP row or failure marker: {names_list(missing_scip)}"
        )
    if not parts:
        return None
    return "incomplete campaign (pass --partial for an interim score): " + "; ".join(parts)


def short_of_attempts(
    scored: Sequence[str], failed: Iterable[str], attempts: Mapping[str, int]
) -> dict[str, int]:
    """The scored `failed` instances with fewer than the required attempts, and their count.

    Keyed on the failures, not on `attempts`: a failure the count map does not
    know has no recorded attempt, so it is refused rather than let through.
    """
    failed = set(failed)
    return {
        name: attempts.get(name, 0)
        for name in scored
        if name in failed and attempts.get(name, 0) < FAILED_ATTEMPTS_REQUIRED
    }


def attempts_refusal(cbls_short: Mapping[str, int], scip_short: Mapping[str, int]) -> str | None:
    """The refusal for a failure not yet retried, or None when each scored one has been (#312)."""
    parts = []
    advice = []
    if cbls_short:
        parts.append(
            f"cbls {len(cbls_short)}: {names_list(f'{n} ({k})' for n, k in cbls_short.items())}"
        )
        advice.append(
            "run_benchmark.py --instances NAME at the commit stamp.txt names, resuming (not "
            "--no-resume, which starts over)"
        )
    if scip_short:
        parts.append(
            f"SCIP {len(scip_short)}: {names_list(f'{n} ({k})' for n, k in scip_short.items())}"
        )
        advice.append("scip_shards.py --instances NAME at the campaign's --time-limit and --seed")
    if not parts:
        return None
    return (
        f"failed instance(s) with fewer than {FAILED_ATTEMPTS_REQUIRED} recorded attempts, so "
        "not yet no-solution under the one-retry rule (pass --partial for an interim score): "
        + "; ".join(parts)
        + ". Solve them again: "
        + "; ".join(advice)
    )


def _pool_gaps(name: str, cbls: Row, scip: ScipRow) -> tuple[float, float]:
    gaps = (cbls.gap_pct, scip.gap_pct)
    if not all(math.isfinite(g) for g in gaps):
        raise RefusedError(f"{name}: solved by both with a finite BKS, but a gap is {gaps}")
    return gaps


def score(
    roster: Sequence[Bound],
    cbls: CblsSide,
    scip: ScipSide,
    *,
    partial: bool = False,
    budget: Budget | None = None,
    memory_cap: str | None = None,
) -> Score:
    """Score every roster instance present on both sides; refuse a gap unless `partial`.

    Claim-excluded instances are out of every count, as in #228's smoke (n=48
    of 50), and are listed by name instead.
    """
    cbls_present = cbls.present()
    scip_present = scip.present()
    missing_cbls = _missing(roster, cbls_present)
    missing_scip = _missing(roster, scip_present)
    refusal = completeness_refusal(missing_cbls, missing_scip)
    if refusal is not None and not partial:
        raise RefusedError(refusal)
    prov = provenance(cbls, scip, budget, memory_cap)
    both = [b for b in roster if b.instance in cbls_present and b.instance in scip_present]
    scored = [b for b in both if b.instance not in CLAIM_EXCLUDED]
    if not partial:
        names = [b.instance for b in scored]
        refusal = attempts_refusal(
            short_of_attempts(names, cbls.rejected, cbls.attempts),
            short_of_attempts(names, scip.failures, scip.attempts),
        )
        if refusal is not None:
            raise RefusedError(refusal)

    solved_c = {b.instance for b in scored if cbls_solved(b.instance, cbls)}
    solved_s = {b.instance for b in scored if scip_solved(b.instance, scip)}
    pool = [
        b
        for b in scored
        if b.instance in solved_c and b.instance in solved_s and math.isfinite(b.primal_bks)
    ]
    gaps = [_pool_gaps(b.instance, cbls.rows[b.instance], scip.rows[b.instance]) for b in pool]
    verdicts = [head_to_head(c, s) for c, s in gaps]
    gm_c = gm_gap_plus_one([c for c, _ in gaps])
    gm_s = gm_gap_plus_one([s for _, s in gaps])
    return Score(
        n_roster=len(roster),
        n_counted=sum(1 for b in roster if b.instance not in CLAIM_EXCLUDED),
        n_scored=len(scored),
        partial=partial,
        cbls_feasible=len(solved_c),
        scip_feasible=len(solved_s),
        both_feasible=len(solved_c & solved_s),
        gm_pool=len(pool),
        gm_cbls=gm_c,
        gm_scip=gm_s,
        ratio=gm_c / gm_s if pool else math.nan,
        wins=verdicts.count(1),
        losses=verdicts.count(-1),
        ties=verdicts.count(0),
        only_cbls=len(solved_c - solved_s),
        only_scip=len(solved_s - solved_c),
        provenance=prov,
        claim_excluded=[b.instance for b in both if b.instance in CLAIM_EXCLUDED],
        no_bks=[b.instance for b in scored if not math.isfinite(b.primal_bks)],
        cbls_no_solution={
            b.instance: cbls_reason(b.instance, cbls) for b in scored if b.instance not in solved_c
        },
        scip_no_solution={
            b.instance: scip_reason(b.instance, scip) for b in scored if b.instance not in solved_s
        },
        missing_cbls=missing_cbls,
        missing_scip=missing_scip,
    )


# --------------------------------------------------------------------------
# Output.
# --------------------------------------------------------------------------


def partial_banner(s: Score) -> str:
    return f"PARTIAL: {s.n_scored} of {s.n_counted} instances, not a result"


def _f(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _listed(label: str, names: Sequence[str]) -> str:
    return f"- {label}: {len(names)}" + (f" ({', '.join(names)})" if names else "")


def _no_solution_lines(label: str, reasons: Mapping[str, str]) -> list[str]:
    """`reasons` grouped by category, largest group first, every name shown."""
    groups: dict[str, list[str]] = {}
    for name, why in reasons.items():
        groups.setdefault(reason_category(why), []).append(name)
    ordered = sorted(groups, key=lambda cat: (-len(groups[cat]), cat))
    return [f"- {label}: {len(reasons)}"] + [
        f"  - {cat}: {len(groups[cat])} ({', '.join(groups[cat])})" for cat in ordered
    ]


def _budget_line(b: Budget) -> str:
    if b.seconds is None:
        return f"- budget: {NOT_STATED} (no SCIP budget field, stamp or run record)"
    return f"- budget: {b.seconds:g}s per instance (from {', '.join(b.sources)})"


def render_markdown(s: Score) -> str:
    p = s.provenance
    lines = [partial_banner(s), ""] if s.partial else []
    threads = "none" if p.cbls_threads is None else str(p.cbls_threads)
    lines += [
        "## MINLPLib GM(gap+1): cbls / SCIP",
        "",
        "| measure | value |",
        "|---|---|",
        f"| roster (bounds.csv) | {s.n_roster} |",
        f"| counted (roster less claim-excluded) | {s.n_counted} |",
        f"| scored | {s.n_scored} |",
        f"| cbls feasible | {s.cbls_feasible} |",
        f"| SCIP feasible | {s.scip_feasible} |",
        f"| both feasible | {s.both_feasible} |",
        f"| GM pool (both, finite BKS) | {s.gm_pool} |",
        f"| GM(gap+1) cbls | {_f(s.gm_cbls)} |",
        f"| GM(gap+1) SCIP | {_f(s.gm_scip)} |",
        f"| ratio cbls/SCIP | {_f(s.ratio)} |",
        f"| cbls wins (only-cbls) | {s.wins} ({s.only_cbls}) |",
        f"| cbls losses (only-SCIP) | {s.losses} ({s.only_scip}) |",
        f"| ties (within 1%) | {s.ties} |",
        "",
        f"- engine commit: {p.commit_sha} (commit_sha column)",
        f"- cbls threads: {threads} (search_config)",
        f"- cbls presolve: {p.cbls_presolve} (search_config)",
        f"- SCIP: {p.scip_version} (scip_version column), at its defaults",
        _budget_line(p.budget),
        f"- memory cap: {p.memory_cap}",
        f"- rule: {MEASURE_RULE}",
        _listed("claim-excluded (out of every count)", s.claim_excluded),
        _listed("no finite BKS (out of GM and both-solved win/loss)", s.no_bks),
        *_no_solution_lines("cbls no solution", s.cbls_no_solution),
        *_no_solution_lines("SCIP no solution", s.scip_no_solution),
    ]
    if s.partial:
        lines += [
            f"- missing cbls rows: {len(s.missing_cbls)}",
            f"- missing SCIP rows: {len(s.missing_scip)}",
        ]
    return "\n".join(lines) + "\n"


def _jsonable(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def to_json(s: Score) -> str:
    return json.dumps(_jsonable(asdict(s)), indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def budget_seconds(text: str) -> float:
    """`--budget`: seconds, as `60`, `60s` or `60 s`."""
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]*)?)\s*s?\s*", text)
    if match is None:
        raise argparse.ArgumentTypeError(f"not a budget in seconds: {text!r}")
    return float(match.group(1))


def staged_digests(stage: Path, roster: Sequence[Bound]) -> dict[str, str]:
    """The sha256 of every roster instance's staged `<instance>.csv`, by instance."""
    return {
        b.instance: file_sha256(path)
        for b in roster
        if (path := stage / f"{b.instance}.csv").exists()
    }


def unchecked_staged(stage: Path, digests: Mapping[str, str]) -> list[str]:
    """The instances in `digests` whose `.checked` marker does not vouch for that digest.

    Header-only and torn files count: a re-solve that is running or was killed
    has not been checked either, even with an older `.rejected` row beside it.
    An instance with no `<instance>.csv` is not in `digests`: a lone `.rejected`
    row is its no-solution, and no row at all is the completeness refusal's.
    """
    return [name for name, digest in digests.items() if checked_digest(stage, name) != digest]


def unchecked_refusal(unchecked: Sequence[str]) -> str:
    return (
        f"{len(unchecked)} staged cbls row(s) without a matching .checked marker, so not "
        f"known to have passed the independent SCIP check: {names_list(unchecked)}. Run "
        "check_staged.py --inst-dir NL --staging-dir STAGE: it checks and marks every "
        "whole row, whatever commit staged it, and solves nothing. It leaves header-only, "
        "torn and unstageable rows unmarked: re-solve such a row with run_benchmark.py at "
        "the commit stamp.txt names (--instances NAME). Or pass --partial for an interim "
        "score"
    )


def load_checked_staging(stage: Path, bounds: Sequence[Bound]) -> CblsSide:
    """`load_cbls_staging`, refused unless every staged row it read is marked as checked.

    The rows are hashed before they are parsed and again after: a driver that
    rewrites a row in between (its check demoting it, then marking the new
    bytes) would otherwise pair a parsed, unchecked row with a marker that
    matches the file as it is now. A row or marker that vanishes between a
    test for it and its read (a driver quarantining a row, or deleting a marker
    before a re-solve) is the same race and gets the same refusal.
    """
    changed = RefusedError(
        f"{stage}: staged rows changed while they were read (a driver is writing "
        "there); score again once it has stopped"
    )
    try:
        before = staged_digests(stage, bounds)
        cbls = load_cbls_staging(stage, {b.instance: b for b in bounds})
        if staged_digests(stage, bounds) != before:
            raise changed
        unchecked = unchecked_staged(stage, before)
    except FileNotFoundError as exc:
        raise changed from exc
    if unchecked:
        raise RefusedError(unchecked_refusal(unchecked))
    return cbls


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bounds", type=Path, required=True, help="the roster, bounds.csv")
    cbls = parser.add_mutually_exclusive_group(required=True)
    cbls.add_argument("--cbls", type=Path, help="the assembled comparison.csv")
    cbls.add_argument(
        "--cbls-staging",
        type=Path,
        help="the driver's staging directory (every row checked, or --partial)",
    )
    parser.add_argument(
        "--scip-shards", type=Path, required=True, help="one <instance>.csv or .failed each"
    )
    parser.add_argument("--partial", action="store_true", help="score what is there; no result")
    parser.add_argument(
        "--budget",
        type=budget_seconds,
        help="the campaign's budget in seconds; refused unless the data records the same",
    )
    parser.add_argument("--memory-cap", help="the campaign's memory cap, as stated")
    parser.add_argument("--json", type=Path, help="also write the score as JSON here")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Score:
    bounds = load_roster(args.bounds)
    roster = {b.instance: b for b in bounds}
    if args.cbls is not None:
        cbls = CblsSide(rows=load_cbls_table(args.cbls, roster))
        derived = {"cbls run record": table_budget(args.cbls)}
    else:
        cbls = (
            load_cbls_staging(args.cbls_staging, roster)
            if args.partial
            else load_checked_staging(args.cbls_staging, bounds)
        )
        derived = {"cbls staging stamp": staging_budget(args.cbls_staging)}
    scip = load_scip_shards(args.scip_shards, roster)
    # The shard stamp (`scip_shards.shard_stamp`, same name and `time-limit=` key
    # as the staging stamp) records the budget the `.failed` attempts ran under,
    # which the markers themselves do not (#312).
    shard_stamp = {"SCIP shard stamp": staging_budget(args.scip_shards)}
    budget = reconcile_budget(
        {"SCIP scip_version": scip_budget(scip), **shard_stamp, **derived}, args.budget
    )
    reconcile_shard_seed(scip, args.scip_shards)
    return score(
        bounds,
        cbls,
        scip,
        partial=args.partial,
        budget=budget,
        memory_cap=args.memory_cap,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except RefusedError as exc:
        print(f"gm_ratio: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.write(render_markdown(result))
    if args.json is not None:
        atomic_write(args.json, to_json(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
