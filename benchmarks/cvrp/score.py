"""Score a CVRP results directory: gap to BKS, size bands, paired A/B (#199).

A table here is an OUTPUT (CLAUDE.md, "A comparison table is an OUTPUT"): it is
a function of a results directory, a roster and a budget, and is re-derived from
the per-job records whenever the question is asked. The reference values are the
pinned BKS in the roster tables of `benchmarks/instances/cvrp/`.

THE RECORD SHAPE, for every engine. `run_benchmark.py` writes one JSON record per
job at `<results>/<arm>/t<threads>/b<budget>/<instance>.s<seed>.json`, its `.sol`
beside it; the PyVRP baseline (#200) writes the same shape. A record must carry
`RECORD_FIELDS`; `workers_completed` is read when present (#170). Arm names are
unique across engines, so `(engine, arm, threads)` names one row group.

What is scored, per group:

* per instance: runs, feasible runs, best / mean / median gap over the feasible
  runs, mean time to first feasible;
* per size band (Hexaly's, `SIZE_BANDS`): the mean of the per-instance mean
  gaps, beside Hexaly's published per-band averages, which are a different
  machine, budget and thread count and are labelled so;
* with `--ab A B`: the per-(instance, seed) difference in gap, A minus B, with a
  95% t interval over the pairs and over the per-instance means, plus a sign test
  over instances -- the instance level is the one to read, since seeds are nested
  in instances. Only pairs where both runs are feasible differ; the rest are
  counted.

What is refused, rather than scored:

* a record missing a `RECORD_FIELDS` entry;
* more than one budget, unless `--budget` picks one;
* a row whose `status` is not `ok` -- a timeout or a failed run is a defect to
  re-run (`run_benchmark.py --retry-failed`), not an infeasible result;
* a row with fewer completed portfolio workers than `threads` (#170);
* a feasible row whose `.sol` fails `verify_solution.py`, or that has none;
* a group missing roster instances, unless `--allow-partial`, which banners the
  table with what is missing;
* an A/B between different thread counts: PyVRP is single-threaded, and a
  multi-thread CBLS row set beside it would not be the same comparison;
* writing `PUBLISHED_TABLE` from a directory run `--not-comparable`, from a
  partial roster, or with an arm outside `PUBLISHABLE_ARMS`.

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

#: The arms a published table may carry. `default` is the engine's own
#: configuration; #204 decides which arm it publishes and edits this.
PUBLISHABLE_ARMS: frozenset[str] = frozenset({"default"})

#: Hexaly's published X-set figures (60 s, AMD Ryzen 7 7700, 8 cores): average
#: gap to BKS per size band, in `SIZE_BANDS` order. Quoted, not re-run.
HEXALY_BAND_GAPS: tuple[float, ...] = (0.1, 0.5, 0.6, 0.8, 0.9)


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
    if record["feasible"] is True and record["objective"] is not None:
        sol = path.with_name(path.name.removesuffix(".json") + ".sol")
        if not sol.exists():
            return f"{path}: feasible, but no {sol.name} to verify"
        vrp = inst_dir / "X" / f"{record['instance']}.vrp"
        verdict = verify(vrp, sol, _float(record["objective"]))
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
        "Hexaly: published figures, 60 s on an AMD Ryzen 7 7700 (8 cores) -- different "
        "hardware, budget and threads; quoted for context, not a paired comparison.",
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
    """`arm` (one thread count present) or `arm@threads`, to its group."""
    arm, _, threads = label.partition("@")
    matches = [g for g in groups if g[1] == arm and (not threads or g[2] == int(threads))]
    if len(matches) != 1:
        raise ScoreError(f"--ab {label!r} matches {len(matches)} group(s); name one as arm@threads")
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


def missing_instances(runs: list[Run], instances: list[str]) -> dict[str, list[str]]:
    present: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for run in runs:
        present[run.group].add(run.instance)
    return {
        _group_label(g): [i for i in instances if i not in seen]
        for g, seen in sorted(present.items())
        if any(i not in seen for i in instances)
    }


def score(args: argparse.Namespace) -> str:
    results: Path = args.results
    roster = read_roster(args.inst_dir / ROSTERS[args.roster])
    instances = [e.instance for e in roster]
    bks = {e.instance: e.bks for e in roster}
    customers = {
        name: int(row["n"]) for name, row in read_manifest(args.inst_dir / "manifest.csv").items()
    }
    budget, records = select_budget(load_records(results), args.budget)
    runs = build_runs(records, bks, args.inst_dir)
    if not runs:
        raise ScoreError(f"no record in {results} is on the {args.roster} roster")
    comparable = stamp_comparable(results)
    missing = missing_instances(runs, instances)
    if missing and not args.allow_partial:
        raise ScoreError(
            "partial roster: "
            + "; ".join(f"{g} lacks {len(m)} ({', '.join(m[:3])}...)" for g, m in missing.items())
            + ". Finish the run, or pass --allow-partial to banner the table."
        )
    banners: list[str] = []
    if not comparable:
        banners.append(
            "**NOT COMPARABLE**: this directory was run with jobs sharing the machine, or "
            "carries no stamp. Its numbers are a tooling check, not a measurement."
        )
    for group, names in missing.items():
        banners.append(f"**PARTIAL**: {group} has no run on {len(names)}: {', '.join(names)}")
    if args.out is not None and args.out.resolve() == PUBLISHED_TABLE.resolve():
        arms = sorted({r.group[1] for r in runs} - PUBLISHABLE_ARMS)
        if arms or banners:
            raise ScoreError(
                f"refusing to write the published table {PUBLISHED_TABLE}: "
                + (f"non-publishable arm(s) {arms}" if arms else "the table carries banners")
            )
    machine = read_json_object(results / MACHINE_NAME) or {}
    host = machine.get("machine")
    cpu = host.get("cpu_model") if isinstance(host, dict) else None
    commits = sorted({str(r.record["commit_sha"]) for r in runs})
    lines = [
        f"# CVRP results: {results.name}",
        "",
        *[f"> {b}" for b in banners],
        *([""] if banners else []),
        f"Budget {budget:g} s; roster `{args.roster}` ({len(instances)} instances); "
        f"commit(s) {', '.join(commits)}; machine {cpu or 'unrecorded'}.",
        "",
        "## Per instance",
        "",
        *instance_table(runs, instances),
        "",
        "## Per size band",
        "",
        *band_table(runs, customers),
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
