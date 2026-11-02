"""Check and mark a staging directory's rows in SCIP, solving nothing (#307).

`run_benchmark.py` checks every staged verified row in SCIP after its solves
(`verify_staged`) and leaves `<instance>.checked` beside it, holding the row
file's sha256; `gm_ratio.py --cbls-staging` scores a staging directory without
`--partial` only when every staged row carries a matching marker. Two kinds of
row have none:

* rows staged by a driver older than the marker, and
* rows a driver killed between its solve and its check left behind.

A re-run of `run_benchmark.py` cannot mark either kind when the checkout has
moved on: its staging stamp records the commit, so a later driver refuses the
directory, or with `--no-resume` solves every row again. This tool runs the
same check (`run_benchmark.check_and_mark`) on every staged row that is whole
and stageable, whatever commit staged it, and marks it. A row SCIP rejects is
demoted to `VERIFY-FAILED` and marked, as the driver does. A row already
marked and unchanged is skipped. Header-only, torn and unstageable rows, and
instances with no `<instance>.csv`, are left alone and listed.

It takes the machine's wall-clock lock, as the driver does, so it never reads
a row while a driver is writing one. It cannot see a runner's exit status: a
row whose runner exited nonzero is set aside as `.rejected` by the driver at
once, and only a driver killed between the runner's exit and that rename
leaves such a row in place.

Usage:
    .venv/bin/python3 benchmarks/minlplib/check_staged.py \\
        --inst-dir NL_DIR --staging-dir STAGE [--instances A B ...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.jobs import LockHeldError, wallclock_lock  # noqa: E402
from benchmarks.minlplib.run_benchmark import (  # noqa: E402
    check_and_mark,
    roster_from_bounds,
    staged_checked,
    staged_row_whole,
    staged_unpublishable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Exit status of a refusal: no staging directory, or another wall-clock driver holds the lock.
EXIT_REFUSED = 2

#: How many names a summary line prints before it says how many more.
NAMES_SHOWN = 10


@dataclass
class CheckReport:
    checked: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    #: Instances with an `<instance>.csv` this tool will not mark, and why.
    skipped: dict[str, str] = field(default_factory=dict)
    absent: list[str] = field(default_factory=list)


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
            report.demoted += check_and_mark(inst_dir, stage, name)
            report.checked.append(name)
    return report


def _names(names: Sequence[str]) -> str:
    shown = ", ".join(names[:NAMES_SHOWN])
    more = len(names) - NAMES_SHOWN
    return f"{shown}, and {more} more" if more > 0 else shown


def render(report: CheckReport) -> str:
    lines = [
        f"checked and marked: {len(report.checked)}",
        f"  of them demoted to VERIFY-FAILED: {len(report.demoted)}"
        + (f" ({_names(report.demoted)})" if report.demoted else ""),
        f"already marked: {len(report.already)}",
        f"no <instance>.csv (rejected-only or not staged): {len(report.absent)}",
        f"left unmarked: {len(report.skipped)}",
    ]
    lines += [f"  {name}: {why}" for name, why in report.skipped.items()]
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
    names = args.instances or roster_from_bounds(args.inst_dir / "bounds.csv")
    try:
        with wallclock_lock("check_staged.py"):
            report = check_staged(args.inst_dir, args.staging_dir, names)
    except LockHeldError as exc:
        print(f"check_staged: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
