"""Give every MINLPLib catalogue instance one status: modelled, or why not (#283).

Reads the catalogue (``instancedata.csv``) and a directory of NL files (the
unpacked ``https://www.minlplib.org/minlplib_nl.zip``), and runs the C++
checker ``cbls_minlplib_nl_check`` over every file the catalogue does not
already exclude. Each instance gets exactly one status, the first that applies:

  unsupported-op        it uses an operator outside the full-library set
                        (``download.FULL_EXCLUDED_OP_COLUMNS``)
  missing-nl            no ``<name>.nl`` in the directory
  no-primal-bound       the catalogue has no finite primal bound
  reader-error          the NL reader refused the file (its message)
  adapter-unsupported   the NL-to-model adapter refused it (its first reason)
  checker-failed        the checker crashed, was killed or timed out on it
  modelled              read and modelled

This is the full-library selection: every problem type, convex and
non-convex, no size budget. The published and held-out rosters are a separate,
pinned selection (``download.row_to_instance``) and are not touched here.

    .venv/bin/python3 benchmarks/instances/minlplib/status.py \\
        --catalogue instancedata.csv --nl-dir minlplib/nl \\
        --out status.csv --bounds minlplib/nl/bounds.csv

``--bounds`` writes the modelled instances as a ``bounds.csv`` the MINLPLib
runner reads; it belongs next to the NL files, since the runner takes one
instance directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.common.provenance import REPO_ROOT  # noqa: E402
from benchmarks.common.records import atomic_write  # noqa: E402
from benchmarks.instances.minlplib import download  # noqa: E402

DEFAULT_CHECKER = REPO_ROOT / "build" / "cbls_minlplib_nl_check"
# Files per checker process: amortises start-up without letting one crash
# cost more than a batch, which is then re-run a file at a time.
BATCH = 64
# Seconds per file before the checker counts as hung (scaled by batch size).
DEFAULT_TIMEOUT = 600.0

STATUSES: tuple[str, ...] = (
    "unsupported-op",
    "missing-nl",
    "no-primal-bound",
    "reader-error",
    "adapter-unsupported",
    "checker-failed",
    "modelled",
)


# The statuses the checker decides (the rest come from the catalogue).
CHECKER_STATUSES: frozenset[str] = frozenset(
    {"reader-error", "adapter-unsupported", "checker-failed", "modelled"}
)


@dataclass(frozen=True)
class Status:
    name: str
    status: str
    detail: str


def catalogue_status(row: dict[str, str], nl_dir: Path) -> Status | None:
    """The status decided before the checker runs, or None if the file must be checked."""
    name = row["name"].strip()
    ops = download.full_excluded_ops(row)
    if ops:
        return Status(name, "unsupported-op", ",".join(ops))
    if not (nl_dir / f"{name}.nl").is_file():
        return Status(name, "missing-nl", "")
    primal = download.finite_primal(row)
    if primal is None:
        return Status(name, "no-primal-bound", row.get("primalbound", "").strip())
    if download.full_instance(row) is None:
        # Not a status: the catalogue row itself is malformed (no nvars/ncons).
        raise ValueError(f"{name}: catalogue row has no usable nvars/ncons")
    return None


def parse_checker_line(line: str) -> tuple[str, str, str]:
    """(path, status, detail) from one checker line."""
    parts = line.rstrip("\n").split("\t", 2)
    if len(parts) < 2 or parts[1] not in ("modelled", "reader-error", "adapter-unsupported"):
        raise ValueError(f"unexpected checker line: {line!r}")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else ""


def _run_checker(
    checker: Path, paths: list[Path], timeout: float
) -> dict[str, tuple[str, str]] | str:
    """Status per path for one batch, or why the checker did not answer for all.

    Output is decoded leniently: reader messages can quote raw file bytes,
    and one undecodable byte must not take the whole run down.
    """
    try:
        proc = subprocess.run(
            [str(checker), *map(str, paths)],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout * len(paths),
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout * len(paths):g}s"
    if proc.returncode < 0:
        return f"killed by signal {-proc.returncode}"
    if proc.returncode != 0:
        return f"exit code {proc.returncode}"
    out: dict[str, tuple[str, str]] = {}
    for line in proc.stdout.splitlines():
        path, status, detail = parse_checker_line(line)
        out[path] = (status, detail)
    if set(out) != {str(p) for p in paths}:
        return "no answer for every file"
    return out


def check_files(
    checker: Path, paths: list[Path], timeout: float = DEFAULT_TIMEOUT
) -> dict[str, tuple[str, str]]:
    """Checker status per path; a file the checker fails on is `checker-failed`.

    A failed batch is re-run a file at a time, so only the file that crashes,
    hangs or is killed (an adapter crash, the OOM killer) is charged, and
    with its own status rather than as a reader refusal.
    """
    result: dict[str, tuple[str, str]] = {}
    for start in range(0, len(paths), BATCH):
        batch = paths[start : start + BATCH]
        answered = _run_checker(checker, batch, timeout)
        if isinstance(answered, str):
            answered = {}
            for path in batch:
                one = _run_checker(checker, [path], timeout)
                answered[str(path)] = (
                    ("checker-failed", one) if isinstance(one, str) else one[str(path)]
                )
        result.update(answered)
    return result


def statuses(
    rows: list[dict[str, str]], nl_dir: Path, checker: Path, timeout: float = DEFAULT_TIMEOUT
) -> list[Status]:
    """One Status per catalogue row, in catalogue order."""
    decided: list[Status | None] = [catalogue_status(row, nl_dir) for row in rows]
    pending = [
        nl_dir / f"{row['name'].strip()}.nl"
        for row, s in zip(rows, decided, strict=True)
        if s is None
    ]
    checked = check_files(checker, pending, timeout)
    out: list[Status] = []
    for row, s in zip(rows, decided, strict=True):
        if s is None:
            name = row["name"].strip()
            status, detail = checked[str(nl_dir / f"{name}.nl")]
            s = Status(name, status, detail)
        out.append(s)
    return out


def status_text(results: list[Status]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["instance", "status", "detail"])
    for s in results:
        writer.writerow([s.name, s.status, s.detail])
    return buf.getvalue()


def counts(results: list[Status]) -> dict[str, int]:
    c = Counter(s.status for s in results)
    return {k: c.get(k, 0) for k in STATUSES}


def modelled_instances(
    rows: list[dict[str, str]], results: list[Status]
) -> list[download.Instance]:
    by_name = {row["name"].strip(): row for row in rows}
    out: list[download.Instance] = []
    for s in results:
        if s.status != "modelled":
            continue
        inst = download.full_instance(by_name[s.name])
        if inst is None:  # catalogue_status already admitted it
            raise AssertionError(f"{s.name}: modelled but not in the full-library selection")
        out.append(inst)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True, help="instancedata.csv")
    parser.add_argument("--nl-dir", type=Path, required=True, help="directory of <name>.nl")
    parser.add_argument("--checker", type=Path, default=DEFAULT_CHECKER)
    parser.add_argument("--out", type=Path, required=True, help="status CSV to write")
    parser.add_argument(
        "--bounds", type=Path, default=None, help="also write the modelled set as bounds.csv"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds per file before the checker counts as hung (default {DEFAULT_TIMEOUT:g})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.checker.is_file():
        print(f"checker not found: {args.checker} (build the cbls_minlplib_nl_check target)")
        return 2
    rows = download.parse_csv(args.catalogue.read_bytes())
    results = statuses(rows, args.nl_dir, args.checker, args.timeout)
    atomic_write(args.out, status_text(results))
    if args.bounds is not None:
        atomic_write(args.bounds, download.bounds_text(modelled_instances(rows, results)))
    print(f"{len(results)} catalogue instances")
    for status, n in counts(results).items():
        print(f"  {status:20s} {n}")
    checked = [s for s in results if s.status in CHECKER_STATUSES]
    if checked and all(s.status == "checker-failed" for s in checked):
        # A wrong, stale or broken binary, not a property of the library.
        print("the checker failed on every file it was given; the counts are not a status")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
