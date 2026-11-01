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
        --budget "60 s" --memory-cap "110G cgroup" [--json summary.json]

`--cbls-staging DIR` reads the driver's staging directory instead of the
assembled table, for interim scoring with `--partial`.

THE MEASURE, kept exactly as #228's smoke defined it (`MEASURE_RULE` below
states it in every output):

* gap+1 of a solver's row is `max(gap_to_bks%, 0) + 1`: a row better than the
  published bound scores as matching it. Below |BKS| 1e-12 the runner's gap
  column is an absolute residual (`reference_solve.safe_gap`), used as written.
* GM is the geometric mean of gap+1 over the GM POOL: instances both solvers
  solved, with a finite `primal_bks` in `bounds.csv`, not in
  `runner.CLAIM_EXCLUDED`. The ratio is GM_cbls / GM_SCIP; below 1 is cbls ahead.
* Over the same pool, cbls WINS an instance when its gap+1 is more than 1%
  smaller than SCIP's (`(gc+1) < (gs+1) / WIN_FACTOR`), LOSES when SCIP's is more
  than 1% smaller, and ties otherwise. Instances only one solver solved are
  counted beside them in brackets (only-cbls, only-SCIP); those need no gap, so
  instances without a finite BKS are in the brackets, and claim-excluded ones
  are not.

SOLVED means: a cbls row with `feasible == true` whose note is not
`VERIFY-FAILED` (the runner rejected its own incumbent), and a SCIP shard row
with `feasible == true`. A SCIP `<instance>.failed` marker (the solve process
died; it holds e.g. `exit 134`) is SCIP no-solution, and is listed by name with
its exit -- `scip_solved` is the one place that decides it.

COMPLETENESS (CLAUDE.md, Measuring): the roster is read first, and by default a
roster instance without a cbls row or without a SCIP CSV or failure marker is
refused (exit 2), as is any results file naming an instance outside the roster
or naming one twice. `--partial` scores the instances present on both sides
instead, and says so on the first line of every output. Nothing relaxes the
unknown-instance and duplicate refusals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write  # noqa: E402
from benchmarks.minlplib.campaign_report import (  # noqa: E402
    Row,
    ScipRow,
    load_results,
    load_scip,
)
from benchmarks.minlplib.reference_solve import Bound, load_bounds  # noqa: E402
from benchmarks.minlplib.run_benchmark import (  # noqa: E402
    ASSEMBLED_TABLE,
    ASSEMBLED_TRACE,
    SEARCH_CONFIG_THREADS,
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

NOT_STATED = "not stated"

MEASURE_RULE = (
    "GM = geometric mean of (max(gap_to_bks%, 0) + 1) over instances both solvers solved, "
    f"with a finite BKS, excluding {', '.join(CLAIM_EXCLUDED)}; ratio = GM_cbls / GM_SCIP "
    "(below 1: cbls ahead). Win/loss: gap+1 more than 1% smaller than the other's, same "
    "pool; brackets: instances only that solver solved (no BKS needed, claim-excluded out). "
    "cbls solved = feasible and not VERIFY-FAILED; a SCIP .failed marker is SCIP no-solution."
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


def cbls_solved(row: Row) -> bool:
    """A cbls row with a verified feasible incumbent."""
    return row.feasible and not row.verification_failed


def scip_solved(name: str, scip: ScipSide) -> bool:
    """Whether SCIP solved `name`. A `.failed` marker is no solution, here and only here."""
    if name in scip.failures:
        return False
    return scip.rows[name].feasible


# --------------------------------------------------------------------------
# Inputs.
# --------------------------------------------------------------------------


def _names(names: Iterable[str], shown: int = NAMES_SHOWN) -> str:
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
        raise RefusedError(f"{path}: instance(s) listed twice in the roster: {_names(dups)}")
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


def _staged_csvs(stage: Path) -> list[Path]:
    """The per-instance row files of a staging directory, without traces or assembled tables.

    `*.rejected` files (a quarantined row) end in `.rejected`, so a `*.csv` glob
    never sees them.
    """
    skip = {ASSEMBLED_TABLE, ASSEMBLED_TRACE}
    return sorted(
        p
        for p in stage.glob("*.csv")
        if p.name not in skip and not p.name.endswith(STAGED_TRACE_SUFFIX)
    )


def load_cbls_staging(stage: Path, roster: Mapping[str, Bound]) -> dict[str, Row]:
    """The driver's staging directory, by instance; an in-progress file is absent."""
    if not stage.is_dir():
        raise RefusedError(f"{stage}: not a directory")
    rows: dict[str, Row] = {}
    for path in _staged_csvs(stage):
        stem = path.name.removesuffix(".csv")
        if stem not in roster:
            raise RefusedError(f"{path}: instance {stem!r} is not in the roster")
        if not has_whole_row(path):
            continue
        (row,) = load_cbls_table(path, roster).values()
        _check_stem(path, stem, row.instance, roster)
        rows[stem] = row
    return rows


@dataclass(frozen=True)
class ScipSide:
    """The SCIP shards: one row per solved instance, one exit per failed process."""

    rows: dict[str, ScipRow]
    failures: dict[str, str]

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
    for path in sorted(shards.glob(f"*{SCIP_FAILED_SUFFIX}")):
        stem = path.name.removesuffix(SCIP_FAILED_SUFFIX)
        if stem not in roster:
            raise RefusedError(f"{path}: instance {stem!r} is not in the roster")
        if stem in rows:
            raise RefusedError(f"{path}: {stem!r} has both a result row and a failure marker")
        failures[stem] = path.read_text().strip() or "no exit recorded"
    return ScipSide(rows=rows, failures=failures)


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
class Provenance:
    commit_sha: str
    cbls_threads: int | None
    scip_version: str
    budget: str
    memory_cap: str


def provenance(
    cbls: Mapping[str, Row], scip: ScipSide, budget: str | None, memory_cap: str | None
) -> Provenance:
    """Read from every row loaded, scored or not: a stray row is still a stray."""
    threads = _one((str(row_threads(r)) for r in cbls.values()), "cbls thread counts")
    return Provenance(
        commit_sha=_one((r.commit_sha for r in cbls.values()), "engine commits"),
        cbls_threads=None if threads == "none" else int(threads),
        scip_version=_one((r.version for r in scip.rows.values()), "SCIP versions"),
        budget=budget or NOT_STATED,
        memory_cap=memory_cap or NOT_STATED,
    )


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    n_roster: int
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
    scip_failures: dict[str, str] = field(default_factory=dict)
    cbls_verify_failed: list[str] = field(default_factory=list)
    missing_cbls: list[str] = field(default_factory=list)
    missing_scip: list[str] = field(default_factory=list)


def _missing(roster: Sequence[Bound], present: set[str]) -> list[str]:
    return [b.instance for b in roster if b.instance not in present]


def completeness_refusal(missing_cbls: Sequence[str], missing_scip: Sequence[str]) -> str | None:
    """The refusal for an incomplete campaign, or None when every roster instance is there."""
    parts = []
    if missing_cbls:
        parts.append(f"{len(missing_cbls)} without a cbls row: {_names(missing_cbls)}")
    if missing_scip:
        parts.append(
            f"{len(missing_scip)} without a SCIP row or failure marker: {_names(missing_scip)}"
        )
    if not parts:
        return None
    return "incomplete campaign (pass --partial for an interim score): " + "; ".join(parts)


def _pool_gaps(name: str, cbls: Row, scip: ScipRow) -> tuple[float, float]:
    gaps = (cbls.gap_pct, scip.gap_pct)
    if not all(math.isfinite(g) for g in gaps):
        raise RefusedError(f"{name}: solved by both with a finite BKS, but a gap is {gaps}")
    return gaps


def score(
    roster: Sequence[Bound],
    cbls: Mapping[str, Row],
    scip: ScipSide,
    *,
    partial: bool = False,
    budget: str | None = None,
    memory_cap: str | None = None,
) -> Score:
    """Score every roster instance present on both sides; refuse a gap unless `partial`."""
    missing_cbls = _missing(roster, set(cbls))
    scip_present = scip.present()
    missing_scip = _missing(roster, scip_present)
    refusal = completeness_refusal(missing_cbls, missing_scip)
    if refusal is not None and not partial:
        raise RefusedError(refusal)
    prov = provenance(cbls, scip, budget, memory_cap)
    scored = [b for b in roster if b.instance in cbls and b.instance in scip_present]

    solved_c = {b.instance for b in scored if cbls_solved(cbls[b.instance])}
    solved_s = {b.instance for b in scored if scip_solved(b.instance, scip)}
    counted = [b for b in scored if b.instance not in CLAIM_EXCLUDED]
    pool = [
        b
        for b in counted
        if b.instance in solved_c and b.instance in solved_s and math.isfinite(b.primal_bks)
    ]
    gaps = [_pool_gaps(b.instance, cbls[b.instance], scip.rows[b.instance]) for b in pool]
    verdicts = [head_to_head(c, s) for c, s in gaps]
    gm_c = gm_gap_plus_one([c for c, _ in gaps])
    gm_s = gm_gap_plus_one([s for _, s in gaps])
    return Score(
        n_roster=len(roster),
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
        only_cbls=sum(1 for b in counted if b.instance in solved_c - solved_s),
        only_scip=sum(1 for b in counted if b.instance in solved_s - solved_c),
        provenance=prov,
        claim_excluded=[b.instance for b in scored if b.instance in CLAIM_EXCLUDED],
        no_bks=[b.instance for b in scored if not math.isfinite(b.primal_bks)],
        scip_failures={
            b.instance: scip.failures[b.instance] for b in scored if b.instance in scip.failures
        },
        cbls_verify_failed=[b.instance for b in scored if cbls[b.instance].verification_failed],
        missing_cbls=missing_cbls,
        missing_scip=missing_scip,
    )


# --------------------------------------------------------------------------
# Output.
# --------------------------------------------------------------------------


def partial_banner(s: Score) -> str:
    return f"PARTIAL: {s.n_scored} of {s.n_roster} instances, not a result"


def _f(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _listed(label: str, names: Sequence[str]) -> str:
    return f"- {label}: {len(names)}" + (f" ({', '.join(names)})" if names else "")


def render_markdown(s: Score) -> str:
    p = s.provenance
    lines = [partial_banner(s), ""] if s.partial else []
    threads = "none" if p.cbls_threads is None else str(p.cbls_threads)
    failures = [f"{n} ({why})" for n, why in s.scip_failures.items()]
    lines += [
        "## MINLPLib GM(gap+1): cbls / SCIP",
        "",
        "| measure | value |",
        "|---|---|",
        f"| roster (bounds.csv) | {s.n_roster} |",
        f"| scored | {s.n_scored} |",
        f"| cbls feasible | {s.cbls_feasible} |",
        f"| SCIP feasible | {s.scip_feasible} |",
        f"| both feasible | {s.both_feasible} |",
        f"| GM pool (both, finite BKS, not claim-excluded) | {s.gm_pool} |",
        f"| GM(gap+1) cbls | {_f(s.gm_cbls)} |",
        f"| GM(gap+1) SCIP | {_f(s.gm_scip)} |",
        f"| ratio cbls/SCIP | {_f(s.ratio)} |",
        f"| cbls wins (only-cbls) | {s.wins} ({s.only_cbls}) |",
        f"| cbls losses (only-SCIP) | {s.losses} ({s.only_scip}) |",
        f"| ties (within 1%) | {s.ties} |",
        "",
        f"- engine commit: {p.commit_sha} (commit_sha column)",
        f"- cbls threads: {threads} (search_config)",
        f"- SCIP: {p.scip_version} (scip_version column), at its defaults",
        f"- budget: {p.budget}",
        f"- memory cap: {p.memory_cap}",
        f"- rule: {MEASURE_RULE}",
        _listed("claim-excluded (counted, out of GM and win/loss)", s.claim_excluded),
        _listed("no finite BKS (out of GM and both-solved win/loss)", s.no_bks),
        _listed("SCIP failures (no solution)", failures),
        _listed("cbls VERIFY-FAILED (no solution)", s.cbls_verify_failed),
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bounds", type=Path, required=True, help="the roster, bounds.csv")
    cbls = parser.add_mutually_exclusive_group(required=True)
    cbls.add_argument("--cbls", type=Path, help="the assembled comparison.csv")
    cbls.add_argument("--cbls-staging", type=Path, help="the driver's staging directory")
    parser.add_argument(
        "--scip-shards", type=Path, required=True, help="one <instance>.csv or .failed each"
    )
    parser.add_argument("--partial", action="store_true", help="score what is there; no result")
    parser.add_argument("--budget", help="the campaign's budget, as stated in the report")
    parser.add_argument("--memory-cap", help="the campaign's memory cap, as stated")
    parser.add_argument("--json", type=Path, help="also write the score as JSON here")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Score:
    bounds = load_roster(args.bounds)
    roster = {b.instance: b for b in bounds}
    if args.cbls is not None:
        cbls = load_cbls_table(args.cbls, roster)
    else:
        cbls = load_cbls_staging(args.cbls_staging, roster)
    scip = load_scip_shards(args.scip_shards, roster)
    return score(
        bounds,
        cbls,
        scip,
        partial=args.partial,
        budget=args.budget,
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
