"""One `xk tune` evaluation: run one benchmark runner once and score it (#262).

`xk tune` reads a run's cost as a JSON metric from a file in the run directory.
cbls's runners write CSV traces and their own result files, not that, so this
wrapper runs one (runner, instance, seed) and writes `<out-dir>/cost.json`:

    {"cost": <float or null>, "runner": ..., "instance": ..., "returncode": ...}

A study reads it with `[cost] json = "cost"`, `file = "{rundir}/cost.json"`;
`null` is a missing value, which xk scores as `failed`. The value is the one
each class's own table scores, computed by that table's own code, so a tuned
configuration is chosen on the measure it will be reported on:

- `mipfeas`: the Primal Integral of `primal_integral.score_instance` against
  the roster's reference value, without the table's verification gate (the
  runner writes no `.verify.json`; the driver's checker does). A no-solution
  run scores its 2.0; an errored or mismatched one is null.
- `minlplib`: the Primal Integral of `campaign_report.anytime_scores` against
  `bounds.csv`'s primal BKS (sign-normalised on a maximize row).
- `uc`: the objective of the one measured row, null unless the engine and the
  independent verifier (`--verify`) both pass it. No bound exists for the
  constructed rows, so the study scores it with `relative = "reference"`.
- `cvrp`: the runner's `gap_percent` to the pinned BKS, null unless verified.

A UC or CVRP run that ends infeasible is null too: those tables have no
"never feasible" score, and xk gives a missing value the study's `[cost]
failed`, a cost worse than any run's, so the tuner ranks it last.

Flags after `--` go to the runner unchanged: that is where a study's
`{config}` lands. A flag this script sets itself (the seed, threads, budget,
outputs) is refused there, since the runner keeps the last value given and the
cost would be scored under the wrong one. The runner's exit code is this
script's, after `cost.json` is written; a runner refusal (exit 2) is a
configuration the runner rejects. A run the scorer cannot read leaves a null
cost and exits 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

# Run as a script (the form a study's command uses), only this file's own
# directory is on sys.path; the repository root makes `benchmarks` importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write  # noqa: E402
from benchmarks.minlplib import campaign_report  # noqa: E402
from benchmarks.mipfeas import primal_integral  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNNERS = ("mipfeas", "minlplib", "uc", "cvrp")
BINARIES = {
    "mipfeas": "cbls_mipfeas",
    "minlplib": "cbls_minlplib",
    "uc": "cbls_uc_chped",
    "cvrp": "cbls_cvrp",
}
# Each runner's own default, anchored at the repository: the runners resolve
# theirs against the working directory, which under xk is not the checkout.
DEFAULT_INST_DIRS = {
    "mipfeas": REPO / "benchmarks" / "instances" / "mipfeas",
    "minlplib": REPO / "benchmarks" / "instances" / "minlplib",
    "uc": REPO / "benchmarks" / "instances" / "uc-chped",
    "cvrp": REPO / "benchmarks" / "instances" / "cvrp",
}
DEFAULT_MIPFEAS_ROSTER = REPO / "benchmarks" / "instances" / "mipfeas" / "roster.csv"
# The engine directory `score_instance` reads a result from.
MIPFEAS_ENGINE = "cbls"
# Statuses whose Primal Integral is a cost. Any other status is the harness or
# the engine failing (a read error, a checker mismatch), not the configuration
# searching badly, and scoring it 2.0 would teach the tuner the wrong thing.
MIPFEAS_SCORED_STATUSES = ("feasible", "no_solution")
UC_MEASURED_METHOD = "CBLS ViolationLS"
# What this script passes itself, and so what a study's flags may not repeat.
OWNED_FLAGS = frozenset(
    {
        "--seed",
        "--threads",
        "--budget",
        "--time-limit",
        "--no-time-limit",
        "--instance",
        "--inst-dir",
        "--out",
        "--out-dir",
        "--trace",
        "--periods",
        "--verify",
        # Not set here, but the MINLPLib residual check and the MIPfeas status
        # judge feasibility at it: a loosened tolerance would score as a better
        # configuration and not reproduce under the published 1e-6.
        "--feas-tol",
    }
)


def finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def mipfeas_reference(roster: Path, instance: str) -> tuple[float, str]:
    for name, value, kind in primal_integral.read_roster(roster):
        if name == instance:
            return value, kind
    raise ValueError(f"{roster} has no instance {instance!r}")


def score_mipfeas(results_dir: Path, instance: str, roster: Path, budget: float) -> float | None:
    reference, kind = mipfeas_reference(roster, instance)
    scored = primal_integral.score_instance(
        instance,
        MIPFEAS_ENGINE,
        reference,
        kind,
        results_dir,
        budget,
        require_verification=False,
    )
    if scored.status not in MIPFEAS_SCORED_STATUSES:
        return None
    return finite_or_none(scored.primal_integral)


def score_minlplib(table: Path, trace: Path, bounds: Path, budget: float) -> float | None:
    if not table.exists():
        return None
    rows = campaign_report.load_results(table, campaign_report.load_bounds_index(bounds))
    if len(rows) != 1:
        raise ValueError(f"{table} has {len(rows)} rows; one run writes exactly one")
    points = campaign_report.load_trace(trace) if trace.exists() else {}
    scores = campaign_report.anytime_scores(rows, points, budget)
    return finite_or_none(scores.per_instance[0].primal_integral)


def read_uc_rows(table: Path) -> list[dict[str, str]]:
    with table.open(newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def score_uc(table: Path) -> float | None:
    if not table.exists():
        return None
    measured = [r for r in read_uc_rows(table) if r["method"] == UC_MEASURED_METHOD]
    if len(measured) != 1:
        raise ValueError(f"{table} has {len(measured)} measured rows; one run writes exactly one")
    row = measured[0]
    if row["feasible"] != "true" or row["verified"] != "true" or row["objective"] == "":
        return None
    return finite_or_none(float(row["objective"]))


def score_cvrp(result: Path) -> float | None:
    if not result.exists():
        return None
    record = json.loads(result.read_text())
    gap = record.get("gap_percent")
    if not isinstance(gap, (int, float)) or record.get("verified") is not True:
        return None
    return finite_or_none(float(gap))


def runner_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    binary = str(args.build_dir / BINARIES[args.runner])
    common = ["--seed", str(args.seed), "--threads", str(args.threads)]
    if args.runner == "mipfeas":
        cmd = [binary, "--instance", args.instance, "--out-dir", str(out_dir / MIPFEAS_ENGINE)]
        cmd += ["--budget", str(args.budget), "--inst-dir", str(inst_dir(args)), *common]
    elif args.runner == "minlplib":
        cmd = [binary, str(inst_dir(args)), "--instance", args.instance]
        cmd += ["--time-limit", str(args.budget), *common]
        cmd += ["--out", str(out_dir / "table.csv"), "--trace", str(out_dir / "trace.csv")]
    elif args.runner == "uc":
        cmd = [binary, str(inst_dir(args)), "--instance", args.instance]
        cmd += ["--periods", str(args.periods), "--verify", *common]
        cmd += ["--out", str(out_dir / "table.csv")]
        if args.budget is not None:
            cmd += ["--time-limit", str(args.budget)]
    else:
        cmd = [binary, "--instance", args.instance, "--budget", str(args.budget), *common]
        cmd += ["--inst-dir", str(inst_dir(args)), "--out", str(out_dir / "result.json")]
    return cmd + list(args.runner_flags)


def inst_dir(args: argparse.Namespace) -> Path:
    return args.inst_dir if args.inst_dir is not None else DEFAULT_INST_DIRS[args.runner]


def score(args: argparse.Namespace, out_dir: Path) -> float | None:
    if args.runner == "mipfeas":
        return score_mipfeas(out_dir, args.instance, args.roster, args.budget)
    if args.runner == "minlplib":
        bounds = inst_dir(args) / "bounds.csv"
        return score_minlplib(out_dir / "table.csv", out_dir / "trace.csv", bounds, args.budget)
    if args.runner == "uc":
        return score_uc(out_dir / "table.csv")
    return score_cvrp(out_dir / "result.json")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runner", choices=RUNNERS)
    parser.add_argument("--build-dir", type=Path, required=True, help="where the runners are")
    parser.add_argument("--out-dir", type=Path, required=True, help="the run directory")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--budget", type=float, help="seconds (uc: default the horizon's)")
    parser.add_argument("--inst-dir", type=Path)
    parser.add_argument("--periods", type=int, help="uc: the one horizon this run measures")
    parser.add_argument("--roster", type=Path, default=DEFAULT_MIPFEAS_ROSTER)
    parser.add_argument("runner_flags", nargs="*", help="after --: passed to the runner")
    args = parser.parse_args(argv)
    if args.runner != "uc" and args.budget is None:
        parser.error(f"{args.runner} needs --budget")
    if args.budget is not None and not (math.isfinite(args.budget) and args.budget > 0):
        parser.error("--budget must be a positive number of seconds")
    if args.runner == "uc" and args.periods is None:
        parser.error("uc needs --periods: a run scores one (instance, horizon)")
    if args.runner != "uc" and args.periods is not None:
        parser.error("--periods is uc's")
    owned = sorted({flag.split("=", 1)[0] for flag in args.runner_flags} & OWNED_FLAGS)
    if owned:
        parser.error(f"{', '.join(owned)} after -- would override this script's own")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.runner == "mipfeas":
        # Looked up before the runner starts too: an instance the roster lacks
        # is a broken call, refused before it spends the budget.
        try:
            mipfeas_reference(args.roster, args.instance)
        except ValueError as exc:
            print(f"tune_trial: {exc}", file=sys.stderr)
            return 2
        (out_dir / MIPFEAS_ENGINE).mkdir(exist_ok=True)
    # A reused run directory must not show the last trial's cost if this one
    # dies before writing its own.
    (out_dir / "cost.json").unlink(missing_ok=True)
    cmd = runner_command(args, out_dir)
    try:
        returncode = subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        print(f"tune_trial: cannot run {cmd[0]}: {exc}", file=sys.stderr)
        return 2
    cost: float | None = None
    error = ""
    if returncode == 0:
        try:
            cost = score(args, out_dir)
        except (ValueError, KeyError, OSError, csv.Error) as exc:
            # Still a cost.json: a run with no readable score is a failed
            # trial, and the reason belongs beside it.
            error = str(exc)
            returncode = 1
    record = {
        "cost": cost,
        "runner": args.runner,
        "instance": args.instance,
        "seed": args.seed,
        "returncode": returncode,
        "error": error,
        "command": cmd,
    }
    atomic_write(out_dir / "cost.json", json.dumps(record, indent=2) + "\n")
    if error:
        print(f"tune_trial: {error}", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
