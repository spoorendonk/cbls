"""Score a CVRP results directory: gap to BKS, size bands, paired A/B (#199).

A table here is an OUTPUT (docs/measuring.md, "A comparison table is an OUTPUT"): it is
a function of a results directory, a roster and a budget, and is re-derived from
the per-job records whenever the question is asked. The reference values are the
pinned BKS in the roster tables of `benchmarks/instances/cvrp/`.

THE RECORD SHAPE, for every engine. `run_benchmark.py` writes one JSON record per
job at `<results>/<engine>/<arm>/t<threads>/b<budget>/<instance>.s<seed>.json`,
its `.sol` beside it; the PyVRP baseline (#200) writes the same shape. A record
must carry `RECORD_FIELDS`; `workers_completed` is read when present (#170).
`(engine, arm, threads)` names one row group.

What is scored, per group:

* per instance: runs, feasible runs, best / mean / median gap over the feasible
  runs, mean time to first feasible;
* per size band (Hexaly's, `SIZE_BANDS`): the mean of the per-instance mean
  gaps, beside Hexaly's published per-band averages: another machine always, and
  another budget or thread count wherever the row's differ, labelled so;
* with `--ab A B`: the per-(instance, seed) difference in gap, A minus B, with a
  95% t interval over the pairs and over the per-instance means, plus a sign test
  over instances -- the instance level is the one to read, since seeds are nested
  in instances. Only pairs where both runs are feasible differ; the rest are
  counted;
* for OR-Tools: the rows whose `routing_status` is failing, by name, or a line
  saying there are none (#204).

What is refused, rather than scored:

* a record missing a `RECORD_FIELDS` entry;
* an OR-Tools row with no `routing_status`;
* more than one budget, unless `--budget` picks one;
* a row whose `status` is not `ok` -- a timeout or a failed run is a defect to
  re-run (`run_benchmark.py --retry-failed`), not an infeasible result;
* a row with fewer completed portfolio workers than `threads` (#170);
* a feasible row whose `.sol` fails `verify_solution.py`, or that has none;
* a group missing roster instances, or whose instances ran different seed sets,
  unless `--allow-partial`, which banners the table with what is uneven;
* an A/B between different thread counts: PyVRP is single-threaded, and a
  multi-thread CBLS row set beside it would not be the same comparison;
* writing `PUBLISHED_TABLE` from a directory stamped not comparable (`--jobs`
  above 1, more threads than cores, or a modified tree), from rows of a modified
  tree, from a partial roster, on a roster outside `PUBLISHABLE_ROSTERS`, with a group outside
  `PUBLISHABLE_GROUPS`, with groups on different seed sets, or with a group above
  one thread other than a `cbls` group at `HEXALY_THREADS` and `HEXALY_BUDGET`
  beside its own single-thread group.

    .venv/bin/python benchmarks/cvrp/score.py /tmp/cvrp-smoke --roster smoke
    .venv/bin/python benchmarks/cvrp/score.py /tmp/cvrp-ab --roster dev --ab guided default
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.provenance import REPO_ROOT  # noqa: E402
from benchmarks.common.records import atomic_write, read_json_object  # noqa: E402
from benchmarks.cvrp.run_benchmark import (  # noqa: E402
    DEFAULT_INST_DIR,
    MACHINE_NAME,
    ROSTERS,
    STATUS_OK,
    stamp_comparable,
)
from benchmarks.cvrp.verify_solution import verify  # noqa: E402
from benchmarks.instances.cvrp.download import (  # noqa: E402
    SIZE_BANDS,
    read_manifest,
    read_roster,
)
from benchmarks.minlplib.portfolio_ab import t975  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Every record, of every engine, carries these.
RECORD_FIELDS: tuple[str, ...] = (
    "engine",
    "instance",
    "arm",
    "budget_seconds",
    "seed",
    "threads",
    "status",
    "feasible",
    "objective",
    "time_to_first_feasible",
    "commit_sha",
)

#: The published table, once #204 writes one. Refused for anything but a
#: comparable, complete run of the arms below.
PUBLISHED_TABLE = REPO_ROOT / "benchmarks" / "instances" / "cvrp" / "results.md"

#: The (engine, arm) pairs a published table may carry. `cbls/default` is the
#: engine's own configuration over the runner's neighbour list; #204 decides
#: which arm it publishes and edits this.
#: `pyvrp/pyvrp` is the reference baseline of #200 and `ortools/ortools` the
#: second baseline of #204, both published beside it.
PUBLISHABLE_GROUPS: frozenset[tuple[str, str]] = frozenset(
    {("cbls", "default"), ("pyvrp", "pyvrp"), ("ortools", "ortools")}
)

#: The rosters a published table may be scored on: results are reported on
#: `report.csv` (README, "Dev and report rosters"). Not on `full`, whose table
#: would carry the dev instances unflagged, nor on the smoke or dev subsets. A
#: run over all 100 scores on `report`: rows off the roster are left out.
PUBLISHABLE_ROSTERS: frozenset[str] = frozenset({"report"})

#: Hexaly's published X-set figures (60 s, AMD Ryzen 7 7700, 8 cores): average
#: gap to BKS per size band, in `SIZE_BANDS` order. Quoted, not re-run.
HEXALY_BAND_GAPS: tuple[float, ...] = (0.1, 0.5, 0.6, 0.8, 0.9)

#: The one thread count above 1 a published table may carry: a `cbls` row set to
#: read beside Hexaly's 8-core figures (#204), labelled by `thread_labels`.
HEXALY_THREADS = 8

#: The budget of those figures; a row set at any other is not read beside them.
HEXALY_BUDGET = 60.0


#: OR-Tools' `RoutingSearchStatus`, by number: this module cannot import ortools,
#: so `tests/python/test_cvrp_ortools.py` pins these against the enum.
ROUTING_STATUS_NAMES: dict[int, str] = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
    3: "ROUTING_FAIL",
    4: "ROUTING_FAIL_TIMEOUT",
    5: "ROUTING_INVALID",
    6: "ROUTING_INFEASIBLE",
    7: "ROUTING_OPTIMAL",
}

#: The statuses of a search that ran as asked: a budget that ends the search
#: before a local optimum is the normal end of a timed run. Anything else,
#: a number not listed included, is called out in the table (#204).
ROUTING_STATUS_GOOD: frozenset[int] = frozenset({1, 2, 7})


class ScoreError(Exception):
    """A refusal; the message says what and why."""


@dataclass(frozen=True)
class Run:
    record: dict[str, object]
    path: Path
    gap: float  # NaN when infeasible

    @property
    def group(self) -> tuple[str, str, int]:
        return (str(self.record["engine"]), str(self.record["arm"]), _int(self.record["threads"]))

    @property
    def instance(self) -> str:
        return str(self.record["instance"])

    @property
    def seed(self) -> int:
        return _int(self.record["seed"])

    @property
    def feasible(self) -> bool:
        return self.record["feasible"] is True and self.record["objective"] is not None


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoreError(f"expected an integer, got {value!r}")
    return value


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def load_records(results: Path) -> list[tuple[Path, dict[str, object]]]:
    """Every job record under `results`, refusing one that lacks a field."""
    records: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(results.rglob("*.json")):
        if path.name.endswith(".runner.json") or path == results / MACHINE_NAME:
            continue
        record = read_json_object(path)
        if record is None:
            raise ScoreError(f"{path} is not a readable record; re-run its job")
        missing = [f for f in RECORD_FIELDS if f not in record]
        if missing:
            raise ScoreError(f"{path} lacks {missing}")
        records.append((path, record))
    return records


def select_budget(
    records: list[tuple[Path, dict[str, object]]], budget: float | None
) -> tuple[float, list[tuple[Path, dict[str, object]]]]:
    budgets = sorted({_float(r["budget_seconds"]) for _, r in records})
    if budget is None:
        if len(budgets) > 1:
            raise ScoreError(
                f"the directory holds budgets {budgets}; a table mixing them would average "
                "different measurements. Pick one with --budget."
            )
        if not budgets:
            raise ScoreError("no records found")
        budget = budgets[0]
    return budget, [(p, r) for p, r in records if _float(r["budget_seconds"]) == budget]


def _row_refusal(path: Path, record: dict[str, object], inst_dir: Path) -> str | None:
    if record["status"] != STATUS_OK:
        return f"{path}: status {record['status']!r}; re-run it (--retry-failed)"
    completed = record.get("workers_completed")
    if isinstance(completed, int) and completed < _int(record["threads"]):
        return f"{path}: {completed} of {record['threads']} portfolio workers completed (#170)"
    if record["engine"] == "ortools" and type(record.get("routing_status")) is not int:
        return f"{path}: no routing_status, so #204's check on it cannot be made"
    if record["feasible"] is True and record["objective"] is not None:
        sol = path.with_name(path.name.removesuffix(".json") + ".sol")
        if not sol.exists():
            return f"{path}: feasible, but no {sol.name} to verify"
        vrp = inst_dir / "X" / f"{record['instance']}.vrp"
        try:
            verdict = verify(vrp, sol, _float(record["objective"]))
        except (OSError, ValueError) as exc:
            return f"{path}: cannot read {vrp}: {exc}"
        if not verdict.ok:
            return f"{sol}: failed verification: {'; '.join(verdict.errors)}"
    return None


def build_runs(
    records: list[tuple[Path, dict[str, object]]], bks: dict[str, float], inst_dir: Path
) -> list[Run]:
    """Runs on the roster, every one checked; refuses with every problem listed."""
    problems: list[str] = []
    runs: list[Run] = []
    for path, record in records:
        if record["instance"] not in bks:
            continue
        refusal = _row_refusal(path, record, inst_dir)
        if refusal is not None:
            problems.append(refusal)
            continue
        ref = bks[str(record["instance"])]
        run = Run(record, path, math.nan)
        if run.feasible:
            run = Run(record, path, 100.0 * (_float(record["objective"]) - ref) / ref)
        runs.append(run)
    if problems:
        raise ScoreError("refusing rows:\n  " + "\n  ".join(problems))
    return runs


def _fmt(value: float, spec: str = ".2f") -> str:
    return "-" if math.isnan(value) else format(value, spec)


def _group_label(group: tuple[str, str, int]) -> str:
    engine, arm, threads = group
    return f"{engine}/{arm} t{threads}"


def instance_table(runs: list[Run], instances: list[str]) -> list[str]:
    lines = [
        "| group | instance | runs | feasible | best gap % | mean gap % | median gap % "
        "| mean TTFF s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_key: dict[tuple[tuple[str, str, int], str], list[Run]] = defaultdict(list)
    for run in runs:
        by_key[(run.group, run.instance)].append(run)
    for group in sorted({r.group for r in runs}):
        for instance in instances:
            cell = by_key.get((group, instance), [])
            if not cell:
                continue
            gaps = [r.gap for r in cell if r.feasible]
            ttff = [_float(r.record["time_to_first_feasible"]) for r in cell if r.feasible]
            lines.append(
                f"| {_group_label(group)} | {instance} | {len(cell)} | {len(gaps)} "
                f"| {_fmt(min(gaps) if gaps else math.nan)} "
                f"| {_fmt(statistics.fmean(gaps) if gaps else math.nan)} "
                f"| {_fmt(statistics.median(gaps) if gaps else math.nan)} "
                f"| {_fmt(statistics.fmean(ttff) if ttff else math.nan)} |"
            )
    return lines


def band_of(customers: int) -> int | None:
    for i, (lo, hi) in enumerate(SIZE_BANDS):
        if lo <= customers < hi:
            return i
    return None


def band_table(runs: list[Run], customers: dict[str, int]) -> list[str]:
    lines = [
        f"Hexaly: published figures, {HEXALY_BUDGET:g} s on an AMD Ryzen 7 7700 "
        f"({HEXALY_THREADS} cores) -- different hardware, and a different budget or thread "
        "count wherever a row's differ; quoted for context, not a paired comparison.",
        "",
        "| group | band | instances | with a feasible run | mean of instance mean gaps % "
        "| Hexaly % |",
        "|---|---|---|---|---|---|",
    ]
    for group in sorted({r.group for r in runs}):
        per_instance: dict[str, list[Run]] = defaultdict(list)
        for run in runs:
            if run.group == group:
                per_instance[run.instance].append(run)
        for band, (lo, hi) in enumerate(SIZE_BANDS):
            members = {i: rs for i, rs in per_instance.items() if band_of(customers[i]) == band}
            if not members:
                continue
            means = [
                statistics.fmean([r.gap for r in rs if r.feasible])
                for rs in members.values()
                if any(r.feasible for r in rs)
            ]
            lines.append(
                f"| {_group_label(group)} | {lo}-{min(hi, 1000)} | {len(members)} | {len(means)} "
                f"| {_fmt(statistics.fmean(means) if means else math.nan)} "
                f"| {HEXALY_BAND_GAPS[band]:.1f} |"
            )
    return lines


def routing_callout(runs: list[Run]) -> list[str]:
    """OR-Tools rows whose search did not run as asked (#204): called out rather
    than refused, because a failed construction can be a result of the fleet it
    was given, which re-running would repeat. The status cannot say which: a
    fleet too tight and a budget too short both give ROUTING_FAIL_TIMEOUT
    (`ortools_solve.py`), so the section says to read the rows. Empty without
    OR-Tools rows."""
    rows = [r for r in runs if r.group[0] == "ortools"]
    if not rows:
        return []
    failing = [r for r in rows if _int(r.record["routing_status"]) not in ROUTING_STATUS_GOOD]
    lines = ["", "## OR-Tools rows with a failing routing_status", ""]
    if not failing:
        return [*lines, f"None of the {len(rows)} OR-Tools rows has a failing routing_status."]
    lines += [
        "The status does not separate a fleet too tight for the construction from a budget "
        "too short for it: read these rows before publishing them.",
        "",
        "| instance | seed | routing_status |",
        "|---|---|---|",
    ]
    for run in sorted(failing, key=lambda r: (r.instance, r.seed)):
        status = _int(run.record["routing_status"])
        name = ROUTING_STATUS_NAMES.get(status, f"unknown status {status}")
        lines.append(f"| {run.instance} | {run.seed} | {name} |")
    return lines


def sign_test_p(better: int, worse: int) -> float:
    """Two-sided exact sign test; ties are excluded by the caller."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / float(1 << n)
    return min(1.0, 2 * tail)


def _interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    half = t975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def parse_group(label: str, groups: Iterable[tuple[str, str, int]]) -> tuple[str, str, int]:
    """`[engine/]arm[@threads]` to its group; the parts left out must not be needed."""
    name, _, threads = label.partition("@")
    engine, _, arm = name.rpartition("/")
    matches = [
        g
        for g in groups
        if g[1] == arm and (not engine or g[0] == engine) and (not threads or str(g[2]) == threads)
    ]
    if len(matches) != 1:
        raise ScoreError(
            f"--ab {label!r} matches {len(matches)} group(s); name one as engine/arm@threads"
        )
    return matches[0]


def paired_ab(runs: list[Run], label_a: str, label_b: str) -> list[str]:
    groups = {r.group for r in runs}
    a, b = parse_group(label_a, groups), parse_group(label_b, groups)
    if a[2] != b[2]:
        raise ScoreError(
            f"{_group_label(a)} and {_group_label(b)} ran at different thread counts; "
            "a paired A/B must compare like with like"
        )
    by_pair: dict[tuple[str, int], dict[tuple[str, str, int], Run]] = defaultdict(dict)
    for run in runs:
        if run.group in (a, b):
            by_pair[(run.instance, run.seed)][run.group] = run
    diffs: dict[str, list[float]] = defaultdict(list)
    unpaired = 0
    one_sided = 0
    for (instance, _seed), arms in sorted(by_pair.items()):
        if a not in arms or b not in arms:
            unpaired += 1
        elif arms[a].feasible and arms[b].feasible:
            diffs[instance].append(arms[a].gap - arms[b].gap)
        else:
            one_sided += 1
    lines = [
        f"Paired A/B: {_group_label(a)} minus {_group_label(b)}, gap percentage points "
        "(negative favours A).",
        f"{unpaired} (instance, seed) run(s) without a partner; {one_sided} pair(s) where "
        "at least one side was infeasible (not differenced).",
    ]
    flat = [d for ds in diffs.values() for d in ds]
    if len(flat) < 2:
        return [*lines, "too few feasible pairs for an interval"]
    mean, half = _interval(flat)
    lines.append(
        f"pairs: n={len(flat)} mean={mean:+.3f} 95% CI [{mean - half:+.3f}, {mean + half:+.3f}]"
    )
    means = [statistics.fmean(ds) for ds in diffs.values()]
    if len(means) >= 2:
        imean, ihalf = _interval(means)
        better = sum(m < 0 for m in means)
        worse = sum(m > 0 for m in means)
        lines.append(
            f"instances: n={len(means)} mean={imean:+.3f} "
            f"95% CI [{imean - ihalf:+.3f}, {imean + ihalf:+.3f}] "
            f"A better={better} worse={worse} tied={len(means) - better - worse} "
            f"sign-test p={sign_test_p(better, worse):.3f}"
        )
    return lines


def uneven_groups(runs: list[Run], instances: list[str]) -> dict[str, list[str]]:
    """Per group, what makes it partial: roster instances it has no run on, and
    instances whose seed set differs from the group's (a run killed between
    seeds), which would average different seed counts per instance."""
    seeds: dict[tuple[str, str, int], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for run in runs:
        seeds[run.group][run.instance].add(run.seed)
    out: dict[str, list[str]] = {}
    for group, by_instance in sorted(seeds.items()):
        every = set().union(*by_instance.values())
        problems = [f"no run on {i}" for i in instances if i not in by_instance]
        problems += [
            f"{i} lacks seed(s) {sorted(every - s)}" for i, s in by_instance.items() if s != every
        ]
        if problems:
            out[_group_label(group)] = problems
    return out


def check_published_seeds(runs: list[Run]) -> None:
    """Every published group ran the same seeds: a mean over one seed beside a mean
    over five reads as the same measurement. `uneven_groups` checks within a group."""
    seeds: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for run in runs:
        seeds[run.group].add(run.seed)
    if len({frozenset(s) for s in seeds.values()}) > 1:
        listed = {_group_label(g): sorted(s) for g, s in sorted(seeds.items())}
        raise ScoreError(
            f"refusing to write the published table {PUBLISHED_TABLE}: "
            f"groups ran different seeds: {listed}"
        )


def check_published_threads(groups: set[tuple[str, str, int]], budget: float) -> None:
    """Every published group is at one thread, the baselines' count (#200), except
    a `cbls` group at `HEXALY_THREADS` and `HEXALY_BUDGET` (#204), whose only reason
    to be published is Hexaly's figures. That one sits beside the same arm at one
    thread: alone, it would have nothing the baselines can be read against."""
    if budget != HEXALY_BUDGET and any(g[2] != 1 for g in groups):
        raise ScoreError(
            f"refusing to write the published table {PUBLISHED_TABLE}: multi-thread rows at "
            f"{budget:g} s; they are published only at Hexaly's {HEXALY_BUDGET:g} s"
        )
    stray = sorted(
        g for g in groups if g[2] != 1 and not (g[0] == "cbls" and g[2] == HEXALY_THREADS)
    )
    if stray:
        raise ScoreError(
            f"refusing to write the published table {PUBLISHED_TABLE}: "
            f"{[_group_label(g) for g in stray]}; a published group runs at 1 thread, "
            f"or is a cbls group at {HEXALY_THREADS} threads"
        )
    alone = sorted(g for g in groups if g[2] != 1 and (g[0], g[1], 1) not in groups)
    if alone:
        raise ScoreError(
            f"refusing to write the published table {PUBLISHED_TABLE}: "
            f"{[_group_label(g) for g in alone]} without the same arm at 1 thread"
        )


def thread_labels(runs: list[Run], budget: float) -> list[str]:
    """One line per group above one thread, so its rows are not read as the
    single-threaded comparison (`paired_ab` refuses that pairing as well)."""
    lines = []
    for group in sorted(g for g in {r.group for r in runs} if g[2] > 1):
        beside = ""
        if group[2] == HEXALY_THREADS and budget == HEXALY_BUDGET:
            beside = f"; to read beside Hexaly's {HEXALY_THREADS}-core figures"
        lines.append(
            f"{_group_label(group)} is a portfolio of {group[2]} threads{beside}; "
            "not a paired comparison with the single-threaded baselines."
        )
    return lines


def check_publishable(runs: list[Run], banners: list[str], roster: str, budget: float) -> None:
    """Refuse a table that may not become `PUBLISHED_TABLE`: one carrying a
    group outside `PUBLISHABLE_GROUPS`, thread counts or seed sets the checks above
    refuse, any banner, or the wrong roster."""
    check_published_threads({r.group for r in runs}, budget)
    check_published_seeds(runs)
    groups = sorted({r.group[:2] for r in runs} - PUBLISHABLE_GROUPS)
    if groups or banners:
        raise ScoreError(
            f"refusing to write the published table {PUBLISHED_TABLE}: "
            + (
                f"non-publishable group(s) {['/'.join(g) for g in groups]}"
                if groups
                else "the table carries banners"
            )
        )
    if roster not in PUBLISHABLE_ROSTERS:
        raise ScoreError(
            f"refusing to write the published table {PUBLISHED_TABLE} from the "
            f"{roster} roster; publish on {sorted(PUBLISHABLE_ROSTERS)}"
        )


def score(args: argparse.Namespace) -> str:
    results: Path = args.results
    roster = read_roster(args.inst_dir / ROSTERS[args.roster])
    instances = [e.instance for e in roster]
    bks = {e.instance: e.bks for e in roster}
    customers = {
        name: int(row["n"]) for name, row in read_manifest(args.inst_dir / "manifest.csv").items()
    }
    unknown = [i for i in instances if i not in customers]
    if unknown:
        raise ScoreError(f"{args.inst_dir / 'manifest.csv'} has no row for {unknown[:5]}")
    budget, records = select_budget(load_records(results), args.budget)
    runs = build_runs(records, bks, args.inst_dir)
    if not runs:
        raise ScoreError(f"no record in {results} is on the {args.roster} roster")
    comparable = stamp_comparable(results)
    uneven = uneven_groups(runs, instances)
    if uneven and not args.allow_partial:
        raise ScoreError(
            "partial roster: "
            + "; ".join(f"{g}: {', '.join(p[:3])}" for g, p in uneven.items())
            + ". Finish the run, or pass --allow-partial to banner the table."
        )
    commits = sorted({str(r.record["commit_sha"]) for r in runs})
    banners: list[str] = []
    if not comparable:
        banners.append(
            "**NOT COMPARABLE**: this directory was run with jobs sharing the machine, "
            "more threads than cores, from a modified tree, or carries no stamp. Its numbers "
            "are a tooling check, "
            "not a measurement."
        )
    dirty = [c for c in commits if c.endswith("-dirty")]
    if dirty:
        banners.append(f"**MODIFIED TREE**: rows from {', '.join(dirty)} ran code no commit names.")
    for group, problems in uneven.items():
        banners.append(f"**PARTIAL**: {group}: {'; '.join(problems)}")
    if args.out is not None and args.out.resolve() == PUBLISHED_TABLE.resolve():
        check_publishable(runs, banners, args.roster, budget)
    machine = read_json_object(results / MACHINE_NAME) or {}
    host = machine.get("machine")
    cpu = host.get("cpu_model") if isinstance(host, dict) else None
    versions = sorted(
        {
            f"{label} {r.record[key]}"
            for r in runs
            for key, label in (("pyvrp_version", "PyVRP"), ("ortools_version", "OR-Tools"))
            if key in r.record
        }
    )
    lines = [
        f"# CVRP results: {results.name}",
        "",
        *[f"> {b}" for b in banners],
        *([""] if banners else []),
        f"Budget {budget:g} s; roster `{args.roster}` ({len(instances)} instances); "
        f"commit(s) {', '.join(commits)}; machine {cpu or 'unrecorded'}"
        + "".join(f"; {v}" for v in versions)
        + ".",
        "",
        *[line for label in thread_labels(runs, budget) for line in (label, "")],
        "## Per instance",
        "",
        *instance_table(runs, instances),
        "",
        "## Per size band",
        "",
        *band_table(runs, customers),
        *routing_callout(runs),
    ]
    if args.ab:
        lines += ["", "## Paired A/B", "", *paired_ab(runs, args.ab[0], args.ab[1])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--roster", choices=sorted(ROSTERS), default="smoke")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--ab", nargs=2, metavar=("A", "B"), default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--out", type=Path, default=None, help="write here instead of stdout")
    args = parser.parse_args(argv)
    try:
        text = score(args)
    except ScoreError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    if args.out is None:
        sys.stdout.write(text)
    else:
        atomic_write(args.out, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
