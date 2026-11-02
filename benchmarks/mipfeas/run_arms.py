"""Run MIPfeas arms -- one engine, several flag sets -- interleaved, and score them.

Issue #331 asks whether load-time presolve (#326-#328) and propagation inside
compound moves (#339, #340) should be on by default. That is a CBLS-against-CBLS
comparison: the same instances and seeds under four flag sets, scored against
each other and against the control's own seed-to-seed spread, which is the
noise floor any difference has to clear.

This driver adds the arms and nothing else. Every solve is one
`run_benchmark.py --engines cbls` call on a one-instance roster, so the
verification against the original file (#138), the pinned-bytes check (#137),
the resume and the run record are that driver's, unchanged. What this one adds:

* the arm sets (`--campaign`), each arm a fixed list of runner flags;
* the order: instance-major, then seed, then arm, so the arms of one
  (instance, seed) run back to back and drift in the machine's state lands on
  all of them alike rather than on whichever arm ran last;
* the machine lock per solve (`--lock-command`, `xk lock --` by default), so a
  build in another session cannot share the CPUs with a timed solve, and
  nothing holds the lock between solves;
* `campaign.json`, the stamp a resume must match: two invocations with
  different flags, seeds or budget would otherwise fill one directory with two
  campaigns;
* the report (`arms_report.md`): per arm and seed, and pooled per arm, the
  feasible count and the shifted geometric mean of the primal integral, each
  as a ratio to the control, beside the control's seed spread.

Usage:
    python benchmarks/mipfeas/run_arms.py --out-dir ~/scratch/arms --budget 60 \\
        -- --cbls-threads 4 --cpuset 0,2,4,6
    python benchmarks/mipfeas/run_arms.py --out-dir ~/scratch/arms --budget 60 --report-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write, read_json_object  # noqa: E402
from benchmarks.mipfeas.primal_integral import (  # noqa: E402
    Scored,
    check_uniform_configuration,
    read_held_out,
    score_instance,
    shifted_geometric_mean,
)
from benchmarks.mipfeas.primal_integral import read_roster as read_reference_roster  # noqa: E402
from benchmarks.mipfeas.run_benchmark import (  # noqa: E402
    DEFAULT_CBLS_BIN,
    DEFAULT_INSTANCE_DIR,
    Job,
    engine_commit,
    has_usable_result,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

RUN_BENCHMARK = Path(__file__).resolve().parent / "run_benchmark.py"
DEFAULT_ROSTER = DEFAULT_INSTANCE_DIR / "tuned.txt"
STAMP_NAME = "campaign.json"
REPORT_NAME = "arms_report.md"
ROSTERS_DIR = "rosters"
DEFAULT_LOCK_COMMAND = "xk lock --"


class Arm(NamedTuple):
    name: str
    flags: tuple[str, ...]


#: The control runs the runner's defaults, which are presolve off and compound
#: propagation off: #331 requires the fences to show that equals main.
CONTROL = "control"

#: The arm sets, selected with `--campaign`. `presolve` is A1 + A2 + A3: the
#: runner's `--presolve` with coefficient tightening and aggregation at their
#: defaults (on under presolve).
CAMPAIGNS: dict[str, tuple[Arm, ...]] = {
    "presolve-322": (
        Arm(CONTROL, ()),
        Arm("presolve", ("--presolve",)),
        Arm("compound-propagation", ("--compound-propagation",)),
        Arm("both", ("--presolve", "--compound-propagation")),
    ),
}

#: Runner flags the arms or the plan own. Passed through, any of them would make
#: an arm's name lie about its flags, or run a cell other than the planned one.
OWNED_FLAGS = frozenset(
    {
        "--presolve",
        "--no-presolve-aggregation",
        "--no-presolve-coef-tightening",
        "--compound-propagation",
        "--no-compound-moves",
        "--seed",
        "--engines",
        "--roster",
        "--results-dir",
        "--budget",
        "--inst-dir",
        "--force",
        "--no-verify",
        "--skip-preconditions",
        "--jobs",
    }
)

#: Pass-through flags the driver reads itself (the binary it stamps, the commit
#: it records), beside the owned ones. `run_benchmark.py`'s argparse accepts a
#: unique prefix of any of them, so a prefix must be refused as the full name
#: would be: `--pres` is `--presolve`, `--se 5` overrides the planned seed, and
#: `--cbls-b X` runs a binary the stamp does not describe.
READ_FLAGS = frozenset({"--cbls-bin", "--engine-commit"})


class Cell(NamedTuple):
    instance: str
    seed: int
    arm: Arm


def plan(instances: Sequence[str], seeds: Sequence[int], arms: Sequence[Arm]) -> list[Cell]:
    """Instance-major, then seed, then arm: one (instance, seed)'s arms run back to back."""
    return [Cell(i, s, a) for i in instances for s in seeds for a in arms]


def arm_dir(out_dir: Path, arm: str, seed: int) -> Path:
    """One `run_benchmark.py` results directory per (arm, seed): its config is uniform."""
    return out_dir / arm / f"seed{seed}"


def owned_flag_in(passthrough: Sequence[str]) -> str | None:
    """The first pass-through argument the driver owns, or an abbreviation of one.

    Matched in `--flag` or `--flag=value` form, and as any strict prefix of an
    owned or read flag, since argparse would expand it. No full runner flag is
    a strict prefix of one of these, so nothing legitimate is refused.
    """
    for arg in passthrough:
        name = arg.split("=", 1)[0]
        if name in OWNED_FLAGS:
            return arg
        if (
            name.startswith("--")
            and len(name) > 2
            and any(f != name and f.startswith(name) for f in OWNED_FLAGS | READ_FLAGS)
        ):
            return arg
    return None


def passthrough_value(passthrough: Sequence[str], flag: str) -> str | None:
    """The value a pass-through flag is given (`--flag V` or `--flag=V`), last one wins."""
    value: str | None = None
    for k, arg in enumerate(passthrough):
        if arg == flag and k + 1 < len(passthrough):
            value = passthrough[k + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def write_cell_roster(out_dir: Path, reference_roster: Path, instance: str) -> Path:
    """A one-instance roster CSV for `run_benchmark.py --roster`, its row copied verbatim."""
    with reference_roster.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("instance") == instance]
    if len(rows) != 1:
        raise ValueError(f"{reference_roster} has {len(rows)} rows for {instance}, not 1")
    path = out_dir / ROSTERS_DIR / f"{instance}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(rows[0])
    return path


def runner_command(
    cell: Cell,
    roster_file: Path,
    out_dir: Path,
    budget: float,
    inst_dir: Path,
    passthrough: Sequence[str],
    lock_command: Sequence[str],
) -> list[str]:
    """The locked `run_benchmark.py` call for one cell."""
    return [
        *lock_command,
        sys.executable,
        str(RUN_BENCHMARK),
        "--engines",
        "cbls",
        "--roster",
        str(roster_file),
        "--inst-dir",
        str(inst_dir),
        "--results-dir",
        str(arm_dir(out_dir, cell.arm.name, cell.seed)),
        "--budget",
        # repr, not :g: every row records the budget it ran at, and the scorer
        # refuses a row whose budget differs from the one it is scored at.
        repr(budget),
        "--seed",
        str(cell.seed),
        *cell.arm.flags,
        *passthrough,
    ]


def cell_done(out_dir: Path, cell: Cell) -> bool:
    """Solved and verified: skipped on resume without taking the lock."""
    return has_usable_result(
        Job("cbls", cell.instance), arm_dir(out_dir, cell.arm.name, cell.seed), verify=True
    )


def binary_digest(path: Path) -> str | None:
    """sha256 of the runner binary, or None when there is none to hash.

    The commit names the source, not the build: a rebuild between two solves
    (the lock is released between them) would change the binary under one
    commit. Hashing a few-MiB file costs milliseconds against a solve of seconds
    to minutes, so it is re-checked before every cell.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def campaign_stamp(
    campaign: str,
    instances: Sequence[str],
    seeds: Sequence[int],
    budget: float,
    passthrough: Sequence[str],
    commit: str,
    digest: str | None = None,
) -> dict[str, object]:
    """What a resume must match. The arms' flags are in it, not only their names."""
    return {
        "campaign": campaign,
        "arms": {arm.name: list(arm.flags) for arm in CAMPAIGNS[campaign]},
        "instances": list(instances),
        "seeds": list(seeds),
        "budget": budget,
        "passthrough": list(passthrough),
        "engine_commit": commit,
        "cbls_bin_sha256": digest,
    }


#: What a report must agree with the stamp on: the pass-through and the commit
#: are the run's business, and `score_campaign` checks the commit from the rows.
REPORT_KEYS = ("campaign", "arms", "instances", "seeds", "budget")


def check_stamp(
    out_dir: Path, stamp: dict[str, object], *, report_only: bool = False
) -> str | None:
    """Write the stamp into a fresh out-dir, or the reason to refuse this one.

    A run compares every key; a report only `REPORT_KEYS`, and needs a stamp.
    """
    path = out_dir / STAMP_NAME
    if not path.exists():
        if report_only:
            return f"{path} is absent: nothing says which campaign {out_dir} holds"
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(stamp, indent=2) + "\n")
        return None
    recorded = read_json_object(path)
    if recorded is None:
        return f"{path} is unreadable; it cannot say which campaign this out-dir holds"
    keys = REPORT_KEYS if report_only else tuple(stamp.keys() | recorded.keys())
    differing = sorted(k for k in keys if stamp.get(k) != recorded.get(k))
    if differing:
        return f"{path} records another campaign ({', '.join(differing)} differ): " + (
            "score it with the arguments it was run with."
            if report_only
            else "resuming would mix two configurations under one arm name. Use a fresh --out-dir."
        )
    return None


def run_cells(
    cells: Sequence[Cell],
    out_dir: Path,
    reference_roster: Path,
    budget: float,
    inst_dir: Path,
    passthrough: Sequence[str],
    lock_command: Sequence[str],
    cbls_bin: Path,
    digest: str | None,
) -> tuple[list[Cell], str | None]:
    """Run every cell not already done, in order; return the failed ones and a stop reason.

    A failed cell does not stop the campaign: its row (or its absence) is what
    the report counts, and the remaining cells are still worth their hours. A
    runner binary that no longer matches the stamped digest does: every later
    cell would run another build under the same arm names.
    """
    failed: list[Cell] = []
    pending = [c for c in cells if not cell_done(out_dir, c)]
    print(f"{len(cells) - len(pending)} of {len(cells)} cell(s) done; {len(pending)} to run.")
    for k, cell in enumerate(pending, 1):
        if binary_digest(cbls_bin) != digest:
            return failed, (
                f"{cbls_bin} changed since the campaign was stamped (sha256 differs); "
                f"stopped before {cell.instance} seed {cell.seed} {cell.arm.name}. "
                "Rebuild the stamped commit, or start a fresh --out-dir."
            )
        roster_file = write_cell_roster(out_dir, reference_roster, cell.instance)
        cmd = runner_command(
            cell, roster_file, out_dir, budget, inst_dir, passthrough, lock_command
        )
        print(f"[{k}/{len(pending)}] {cell.instance} seed {cell.seed} {cell.arm.name}", flush=True)
        if subprocess.run(cmd, check=False).returncode != 0:
            failed.append(cell)
    return failed, None


class ArmSeed(NamedTuple):
    """One (arm, seed) of the report: its scored rows' aggregates."""

    arm: str
    seed: int
    scored: int
    not_run: int
    feasible: int
    verification_failed: int
    unverified: int
    pi_sgm: float
    median_setup_seconds: float | None
    max_peak_rss_mib: float | None


def _published(rows: Sequence[Scored]) -> list[Scored]:
    """Rows that ran and are not withheld (rejected or unverified): those a PI mean may use."""
    return [r for r in rows if r.status != "not_run" and not r.withheld]


def compared_instances(
    scored: dict[tuple[str, int], list[Scored]], arms: Sequence[Arm], seed: int
) -> set[str]:
    """The instances published in EVERY arm at this seed.

    The PI means are taken over this set alone. An arm whose hard instance is
    withheld or missing would otherwise drop it from its own mean but not from
    the control's, and read as better for a row it does not have.
    """
    sets = [{r.instance for r in _published(scored[(arm.name, seed)])} for arm in arms]
    return set.intersection(*sets) if sets else set()


def summarize_arm_seed(arm: str, seed: int, rows: Sequence[Scored], compared: set[str]) -> ArmSeed:
    """The aggregates of one (arm, seed); the PI mean is over `compared` only."""
    ran = [r for r in rows if r.status != "not_run"]
    published = _published(rows)
    in_mean = [r.primal_integral for r in published if r.instance in compared]
    setups = [r.setup_seconds for r in ran if r.setup_seconds is not None]
    rss = [r.peak_rss_kib for r in ran if r.peak_rss_kib is not None]
    return ArmSeed(
        arm=arm,
        seed=seed,
        scored=len(rows),
        not_run=len(rows) - len(ran),
        feasible=sum(1 for r in published if r.status == "feasible"),
        verification_failed=sum(1 for r in ran if r.verification == "fail"),
        unverified=sum(1 for r in ran if r.verification == "unverified" and r.withheld),
        pi_sgm=shifted_geometric_mean(in_mean) if in_mean else float("nan"),
        median_setup_seconds=statistics.median(setups) if setups else None,
        max_peak_rss_mib=max(rss) / 1024.0 if rss else None,
    )


#: Provenance values that name no build: a row the driver wrote for a solve it
#: killed (timeout, memory cap) carries no commit. The scorer leaves them out
#: of its commit line for the same reason (`primal_integral.py`).
_NO_COMMIT = ("unknown", "n/a")


def engine_commits(scored: dict[tuple[str, int], list[Scored]]) -> list[str]:
    """The engine commits the rows ran at, leaving out rows that record none."""
    return sorted(
        {
            r.provenance
            for rows in scored.values()
            for r in rows
            if r.status != "not_run" and r.provenance not in _NO_COMMIT
        }
    )


def score_campaign(
    out_dir: Path,
    arms: Sequence[Arm],
    seeds: Sequence[int],
    entries: Sequence[tuple[str, float, str]],
    budget: float,
) -> dict[tuple[str, int], list[Scored]]:
    """Every (arm, seed)'s cbls rows, refusing a mixed configuration or engine commit."""
    scored: dict[tuple[str, int], list[Scored]] = {}
    for arm in arms:
        for seed in seeds:
            rows = [
                score_instance(name, "cbls", value, kind, arm_dir(out_dir, arm.name, seed), budget)
                for name, value, kind in entries
            ]
            check_uniform_configuration(rows)
            scored[(arm.name, seed)] = rows
    commits = engine_commits(scored)
    if len(commits) > 1:
        raise ValueError(
            f"the arms ran at {len(commits)} engine commits ({', '.join(commits)}); "
            f"an arm comparison is one commit's flags against each other"
        )
    return scored


def _ratio(value: float, control: float) -> float:
    return value / control if control > 0.0 else float("nan")


def _cell(value: float | None, fmt: str) -> str:
    return "" if value is None or value != value else format(value, fmt)


def render_report(
    campaign: str,
    arms: Sequence[Arm],
    seeds: Sequence[int],
    budget: float,
    scored: dict[tuple[str, int], list[Scored]],
) -> str:
    """The arms report: per (arm, seed), pooled per arm, and the control's noise floor."""
    compared = {s: compared_instances(scored, arms, s) for s in seeds}
    per = {
        key: summarize_arm_seed(key[0], key[1], rows, compared[key[1]])
        for key, rows in scored.items()
    }
    commits = engine_commits(scored)
    n_instances = len(next(iter(scored.values()))) if scored else 0
    control_sgm = {s: per[(CONTROL, s)].pi_sgm for s in seeds}
    floor_seeds = [s for s, v in control_sgm.items() if v == v and v > 0.0]
    finite = [control_sgm[s] for s in floor_seeds]
    noise = max(finite) / min(finite) if len(finite) > 1 else float("nan")
    lines = [
        f"# MIPfeas arms: campaign {campaign}",
        "",
        f"- Engine commit: {', '.join(commits) or 'none recorded'}",
        f"- Budget: {budget:g} s; seeds: {', '.join(map(str, seeds))}; instances: {n_instances}",
        "- Arms: "
        + "; ".join(f"`{a.name}` = `{' '.join(a.flags) or '(runner defaults)'}`" for a in arms),
        "- PI SGM: shifted geometric mean of the primal integral (lower is better) over the "
        "instances published in every arm at that seed (`compared`); ratio = arm / control, "
        "so below 1 is better. Feasible counts are over all rows.",
        f"- Noise floor: the control's PI SGM, max / min over seeds "
        f"{', '.join(map(str, floor_seeds)) or 'none'} = {_cell(noise, '.4f')}. "
        "A pooled ratio inside [1/floor, floor] is not a difference.",
        "",
        "## Per arm and seed",
        "",
        "| arm | seed | feasible | not run | verify failed | unverified | compared | PI SGM | "
        "ratio to control | median setup s | max peak RSS MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        for seed in seeds:
            p = per[(arm.name, seed)]
            lines.append(
                f"| {arm.name} | {seed} | {p.feasible}/{p.scored} | {p.not_run} | "
                f"{p.verification_failed} | {p.unverified} | {len(compared[seed])} | "
                f"{_cell(p.pi_sgm, '.4f')} | "
                f"{_cell(_ratio(p.pi_sgm, control_sgm[seed]), '.4f')} | "
                f"{_cell(p.median_setup_seconds, '.3f')} | {_cell(p.max_peak_rss_mib, '.1f')} |"
            )
    lines += [
        "",
        "## Pooled over seeds",
        "",
        "| arm | feasible | feasible lower than control on a seed | PI SGM | ratio to control "
        "| per-seed ratio range |",
        "|---|---|---|---|---|---|",
    ]
    pooled_control = _pooled_sgm(scored, CONTROL, seeds, compared)
    for arm in arms:
        feasible = sum(per[(arm.name, s)].feasible for s in seeds)
        total = sum(per[(arm.name, s)].scored for s in seeds)
        lower = [s for s in seeds if per[(arm.name, s)].feasible < per[(CONTROL, s)].feasible]
        pooled = _pooled_sgm(scored, arm.name, seeds, compared)
        ratios = [_ratio(per[(arm.name, s)].pi_sgm, control_sgm[s]) for s in seeds]
        ratios = [r for r in ratios if r == r]
        spread = f"{min(ratios):.4f}-{max(ratios):.4f}" if ratios else ""
        lines.append(
            f"| {arm.name} | {feasible}/{total} | {', '.join(map(str, lower)) or 'none'} | "
            f"{_cell(pooled, '.4f')} | {_cell(_ratio(pooled, pooled_control), '.4f')} | {spread} |"
        )
    return "\n".join(lines) + "\n"


def _pooled_sgm(
    scored: dict[tuple[str, int], list[Scored]],
    arm: str,
    seeds: Sequence[int],
    compared: dict[int, set[str]],
) -> float:
    """PI SGM over every (instance, seed) published in all arms."""
    values = [
        r.primal_integral
        for s in seeds
        for r in _published(scored[(arm, s)])
        if r.instance in compared[s]
    ]
    return shifted_geometric_mean(values) if values else float("nan")


def split_passthrough(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """The driver's own arguments, and those after `--` for `run_benchmark.py`."""
    if "--" in argv:
        k = list(argv).index("--")
        return list(argv[:k]), list(argv[k + 1 :])
    return list(argv), []


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    own, passthrough = split_passthrough(argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), default="presolve-322")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=60.0, help="seconds per solve")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--roster",
        type=Path,
        default=DEFAULT_ROSTER,
        help="instance names, one per line, # comments (default: #255's tuned.txt)",
    )
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    parser.add_argument(
        "--lock-command",
        default=DEFAULT_LOCK_COMMAND,
        help="prefix taking the machine lock for one solve ('' for none)",
    )
    parser.add_argument("--report-only", action="store_true", help="score what is there")
    return parser.parse_args(own), passthrough


def main(argv: Sequence[str] | None = None) -> int:
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)
    if (owned := owned_flag_in(passthrough)) is not None:
        print(
            f"{owned} is set by the arms or the plan, or abbreviates such a flag; "
            "it cannot be passed through",
            file=sys.stderr,
        )
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        print("--seeds repeats a seed", file=sys.stderr)
        return 2
    arms = CAMPAIGNS[args.campaign]
    instances = list(read_held_out(args.roster).names)
    reference_roster = args.inst_dir / "roster.csv"
    entries = {
        name: (name, value, kind) for name, value, kind in read_reference_roster(reference_roster)
    }
    missing = [i for i in instances if i not in entries]
    if missing:
        print(f"{reference_roster} lacks {', '.join(missing)}", file=sys.stderr)
        return 2
    out_dir: Path = args.out_dir

    cbls_bin = Path(passthrough_value(passthrough, "--cbls-bin") or DEFAULT_CBLS_BIN)
    commit = engine_commit(cbls_bin, passthrough_value(passthrough, "--engine-commit"))
    lock_command = shlex.split(args.lock_command)
    if not args.report_only and lock_command and shutil.which(lock_command[0]) is None:
        print(
            f"--lock-command {args.lock_command!r}: {lock_command[0]} is not on PATH "
            "(pass '' to run without the machine lock)",
            file=sys.stderr,
        )
        return 2
    digest = binary_digest(cbls_bin)
    stamp = campaign_stamp(
        args.campaign, instances, args.seeds, args.budget, passthrough, commit, digest
    )
    if (refusal := check_stamp(out_dir, stamp, report_only=args.report_only)) is not None:
        print(refusal, file=sys.stderr)
        return 2
    failed: list[Cell] = []
    stopped: str | None = None
    if not args.report_only:
        failed, stopped = run_cells(
            plan(instances, args.seeds, arms),
            out_dir,
            reference_roster,
            args.budget,
            args.inst_dir,
            passthrough,
            lock_command,
            cbls_bin,
            digest,
        )

    scored = score_campaign(out_dir, arms, args.seeds, [entries[i] for i in instances], args.budget)
    report = render_report(args.campaign, arms, args.seeds, args.budget, scored)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(out_dir / REPORT_NAME, report)
    print(f"Report -> {out_dir / REPORT_NAME}")
    rejected = sum(r.verification == "fail" for rows in scored.values() for r in rows)
    if failed:
        print(f"{len(failed)} cell(s) failed; see the lines above.", file=sys.stderr)
    if rejected:
        print(f"DEFECT: {rejected} solution(s) rejected by the independent check", file=sys.stderr)
    if stopped is not None:
        print(stopped, file=sys.stderr)
        return 2
    return 1 if failed or rejected else 0


if __name__ == "__main__":
    sys.exit(main())
