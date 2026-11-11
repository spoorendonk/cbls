"""Check and mark a staging directory's rows in SCIP, solving nothing (#307).

`gm_ratio.py --cbls-staging` scores without `--partial` only when every staged
row has an `<instance>.checked` marker holding its sha256. Rows staged by a
driver older than the marker, or left by a driver killed between solve and
check, have none, and a later `run_benchmark.py` refuses the commit-stamped
directory (or re-solves everything with `--no-resume`). This tool runs
`run_benchmark.check_and_mark` on every whole, stageable row, whatever commit
staged it: a row SCIP rejects is demoted to `VERIFY-FAILED` and marked, an
unchanged marked row is skipped, and header-only, torn or unstageable rows
and missing `<instance>.csv` are left alone and listed.

An unmarked row keeps a non-partial score refused; clear it by re-solving at
the stamp's commit (`run_benchmark.py --instances NAME`). Renaming it
`.rejected` by hand is not a no-solution: the scorer wants two failed attempts
in the driver's record (#312).

Refuses a stamp naming another roster (`bounds.csv` fingerprint) or host:
SCIP would check against the wrong models, and a demotion is permanent. Holds
the wall-clock lock so it never reads a row a driver here is writing. A row
whose check raises (no `.sol`, a `.sol` for another instance, no `.nl`, a row
changed under it) is left unmarked and listed, and the exit is nonzero.

It cannot see a runner's exit status; only a driver killed between a nonzero
runner exit and its `.rejected` rename leaves such a row in place.

Usage:
    .venv/bin/python3 benchmarks/minlplib/check_staged.py \\
        --inst-dir NL_DIR --staging-dir STAGE [--instances A B ...]
"""

from __future__ import annotations

import argparse
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.jobs import LockHeldError, wallclock_lock  # noqa: E402
from benchmarks.minlplib.gm_ratio import names_list  # noqa: E402
from benchmarks.minlplib.run_benchmark import (  # noqa: E402
    STAMP_NAME,
    check_and_mark,
    roster_fingerprint,
    roster_from_bounds,
    staged_checked,
    staged_row_whole,
    staged_unpublishable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Exit status of a refusal: no staging directory, no roster, a stamp for
#: another roster or host, or another wall-clock driver holds the lock.
EXIT_REFUSED = 2

#: Exit status when some row's check raised: the others are checked and marked.
EXIT_CHECK_ERRORS = 1

#: What clears a row this tool leaves unmarked; see the module docstring.
REMEDY = "re-solve it with run_benchmark.py at the commit stamp.txt names (--instances NAME)"


@dataclass
class CheckReport:
    checked: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    #: Instances with an `<instance>.csv` this tool will not mark, and why.
    skipped: dict[str, str] = field(default_factory=dict)
    #: Instances whose check raised, and the error.
    errors: dict[str, str] = field(default_factory=dict)
    absent: list[str] = field(default_factory=list)


def stamp_mismatch(inst_dir: Path, stage: Path) -> str | None:
    """Why `stage`'s rows must not be checked against `inst_dir` on this host; None when fine."""
    if not (inst_dir / "bounds.csv").is_file():
        return f"{inst_dir}: no bounds.csv, so no roster to check against"
    stamp = stage / STAMP_NAME
    if not stamp.is_file():
        return f"{stamp}: missing, so the roster the rows were staged for is unknown"
    fields = dict(line.split("=", 1) for line in stamp.read_text().splitlines() if "=" in line)
    want = {"roster": roster_fingerprint(inst_dir), "host": socket.gethostname()}
    for key, value in want.items():
        if fields.get(key) != value:
            return (
                f"{stamp}: {key}={fields.get(key)}, but here it is {value}; checking the "
                "rows against another roster's models, or beside another host's driver, "
                "could demote valid rows for good"
            )
    return None


def skip_reason(stage: Path, name: str) -> str | None:
    """Why `name`'s staged row cannot be checked and marked; None when it can."""
    path = stage / f"{name}.csv"
    if not staged_row_whole(path):
        return "no whole row (header-only or torn)"
    if staged_unpublishable(path):
        return "unstageable note"
    return None


def check_staged(inst_dir: Path, stage: Path, names: Sequence[str]) -> CheckReport:
    report = CheckReport()
    for name in names:
        if not (stage / f"{name}.csv").exists():
            report.absent.append(name)
        elif staged_checked(stage, name):
            report.already.append(name)
        elif (reason := skip_reason(stage, name)) is not None:
            report.skipped[name] = reason
        else:
            # Any error: SCIP's reader and the solution parser raise their own
            # kinds, and one bad row must not hide the rest of the report.
            try:
                report.demoted += check_and_mark(inst_dir, stage, name)
            except Exception as exc:  # noqa: BLE001
                report.errors[name] = f"{type(exc).__name__}: {exc}"
            else:
                report.checked.append(name)
    return report


def render(report: CheckReport) -> str:
    lines = [
        f"checked and marked: {len(report.checked)}",
        f"  of them demoted to VERIFY-FAILED: {len(report.demoted)}"
        + (f" ({names_list(report.demoted)})" if report.demoted else ""),
        f"already marked: {len(report.already)}",
        f"no <instance>.csv (rejected-only or not staged): {len(report.absent)}",
        f"left unmarked: {len(report.skipped)}",
    ]
    lines += [f"  {name}: {why}" for name, why in report.skipped.items()]
    lines.append(f"check raised, left unmarked: {len(report.errors)}")
    lines += [f"  {name}: {why}" for name, why in report.errors.items()]
    if report.skipped:
        lines.append(f"to clear a row left unmarked: {REMEDY}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--inst-dir", type=Path, required=True, help="the .nl files and bounds.csv")
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--instances", nargs="+", help="default: every instance in bounds.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.staging_dir.is_dir():
        print(f"check_staged: {args.staging_dir}: not a directory", file=sys.stderr)
        return EXIT_REFUSED
    mismatch = stamp_mismatch(args.inst_dir, args.staging_dir)
    if mismatch is not None:
        print(f"check_staged: {mismatch}", file=sys.stderr)
        return EXIT_REFUSED
    names = args.instances or roster_from_bounds(args.inst_dir / "bounds.csv")
    try:
        with wallclock_lock("check_staged.py"):
            report = check_staged(args.inst_dir, args.staging_dir, names)
    except LockHeldError as exc:
        print(f"check_staged: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.write(render(report))
    return EXIT_CHECK_ERRORS if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
