"""The files a run leaves behind: how they are written, read, and where their shapes live.

A kill at any instant (OOM, reboot, Ctrl-C) leaves a whole-file record either as
it was or as it was meant to be, and an append-only record with at most a torn
final line, which `repair_torn_tail` removes. Readers treat a file such a kill
could have left as what it is, not as a result.

THE RECORD SCHEMAS. No shared per-job record (#160): the runners (C++ and the
CP-SAT baseline) write them, published table headers are pinned against them,
and each resume rule keys on its own, so each is declared once beside the code
that owns it and listed here:

=====================  =================================  ==============================
record                 declared in                        resume key
=====================  =================================  ==============================
mipfeas result         `mipfeas.run_benchmark.Job`        `<engine>/<instance>.json`
                       (paths) and the runner/CP-SAT      exists and, when verifying,
                       `.json` it writes; the driver's    carries its `.sol` and a
                       own `write_failure_result`         verdict (`needs_solve`,
                                                          `needs_verification`)
mipfeas verdict        `verify_solution.Verification`;    `DRIVER_WRITTEN_VERDICT_REASONS`
                       driver's `write_failure_verdict`   retried `MAX_VERIFY_ATTEMPTS`
mipfeas machine        `mipfeas.run_benchmark.            none: one entry per invocation
record                 build_run_record`                  appended to `run_record.json`
minlplib runner row    `minlplib.runner.RUNNER_COLUMNS`,  --
                       pinned against `minlplib.cpp`
minlplib staged row    one runner row per                 `run_benchmark.staged_complete`
                       `<stage>/<instance>.csv`           under a matching `stamp.txt`
minlplib per-seed row  `run_benchmark.                    none: `publish_seed_rows`
(#141)                 SEEDS_TABLE_COLUMNS` (`seed` +     replaces the run's seed block
                       the runner row), in                and keeps every other seed's
                       `comparison_seeds.csv`
minlplib run record    `campaign_report.RunRecord`, at    none: rewritten at each publish
(#141)                 `campaign_report.run_record_path`  (per seed, in the per-seed
                       beside each published table        table's record)
ablation campaign row  `run_ablation.RESULT_COLUMNS`      `(instance, arm, seed)` in
                       (provenance + the runner row)      `results.csv`, less the trailing
                                                          partial block, under a matching
                                                          `stamp.txt`
cvrp job record        `cvrp.score.RECORD_FIELDS` (both   `<arm>/t<threads>/b<budget>/
(#199)                 engines); written by               <instance>.s<seed>.json` under
                       `cvrp.run_benchmark.job_record`    `<engine>/` holds a readable
                                                          record with that key
                                                          (`run_benchmark.needs_run`)
=====================  =================================  ==============================
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def atomic_write(path: Path, data: str | bytes) -> None:
    """Replace `path`'s contents without ever leaving it truncated.

    Written to a sibling temporary file, fsynced, then renamed over `path`:
    `open(path, "w")` truncates first, so a kill in that window would leave an
    empty file at exit 0. No newline translation: the bytes are the string's.
    """
    payload = data.encode() if isinstance(data, str) else data
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json(path: Path, record: object) -> None:
    """Write one JSON record the way every driver-written record is formatted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")


def read_json_object(path: Path) -> dict[str, object] | None:
    """The object at `path`, or None when it is absent, unreadable or not an object.

    A file truncated mid-write reads as absent, so the driver redoes it rather
    than keeping it for scoring to abort on. `ValueError`, not only
    `JSONDecodeError`: a binary-corrupted file raises `UnicodeDecodeError` from
    `read_text()`.
    """
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    return parsed if isinstance(parsed, dict) else None


def csv_text(rows: Iterable[Iterable[object]]) -> str:
    """Rows rendered exactly as `csv.writer` renders them to a file."""
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def csv_header(path: Path) -> list[str]:
    """The first row of a CSV file, empty when the file is."""
    with path.open(newline="") as fh:
        return next(csv.reader(fh), [])


def append_csv_row(path: Path, values: Iterable[object]) -> None:
    """Append one row, durably, before anything else happens.

    `fsync`, not just a flush: the machine going away is what this guards
    against, and a write still in the page cache does not survive that.
    """
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow(values)
        fh.flush()
        os.fsync(fh.fileno())


#: One failed attempt on an instance: `exit <rc> <utc>`.
ATTEMPT_LINE = re.compile(r"(exit [0-9]+) [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z")

#: Failed attempts a final no-solution needs: the first and its one retry (#256).
FAILED_ATTEMPTS_REQUIRED = 2


def shell_exit(returncode: int) -> int:
    """`returncode` as `sh` reports it: a death by signal N is 128+N, not -N.

    `subprocess` reports a death by signal as negative. The #256 markers were
    written by `sh` (`exit 134` for an abort), so every attempt record writes
    the same failure the same way, and an OOM SIGKILL reads as 137 on both sides.
    """
    return 128 - returncode if returncode < 0 else returncode


def append_attempt(path: Path, returncode: int) -> None:
    """Record one failed attempt, durably, as `append_csv_row` does a row (#312).

    A torn last line (a kill mid-append) is dropped first, so it cannot glue
    onto this line and spoil both: the lost attempt can only ask for a retry.
    """
    repair_torn_tail(path)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a") as fh:
        fh.write(f"exit {shell_exit(returncode)} {stamp}\n")
        fh.flush()
        os.fsync(fh.fileno())


def recorded_attempts(path: Path) -> list[str]:
    """The exits (`exit 134`) of `path`'s attempt lines, oldest first; [] without the file.

    A line that is not a whole attempt does not count: a score refuses too few
    attempts, so a torn or hand-written line can only ask for another retry.
    """
    if not path.exists():
        return []
    matches = (ATTEMPT_LINE.fullmatch(line) for line in path.read_text().splitlines())
    return [m.group(1) for m in matches if m]


def attempts_exhausted(path: Path) -> bool:
    """Whether `path` records the failed attempts a final no-solution needs (#323).

    The drivers' test for leaving an instance alone and the scorer's for accepting
    its failure are this one count, so a re-run cannot outvote the #256 rule.
    """
    return len(recorded_attempts(path)) >= FAILED_ATTEMPTS_REQUIRED


def repair_torn_tail(path: Path) -> bool:
    """Drop an unterminated final line from an append-only file. True if one was dropped.

    A kill mid-append leaves a partial row whose key is missing from a resume
    set while its bytes stay, so the redone run would wedge a half-row between
    two whole ones.
    """
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n")
    # Write-then-rename: `write_bytes` truncates first, so a kill would destroy
    # the whole record.
    atomic_write(path, data[: cut + 1] if cut >= 0 else b"")
    return True


def stamp_refusal(path: Path, stamp: str, *, resume: bool, label: str, advice: str) -> str | None:
    """The refusal to resume into a directory stamped with another configuration.

    Resume keys on a record being there, not on what produced it, so a directory
    carries a stamp (commit, budget, seeds: whatever makes two runs comparable).
    Resuming under another stamp is refused with both quoted (the recorded one
    under `label`) and then `advice`, leaving the recorded stamp in place.
    Otherwise (fresh directory, matching stamp, or `resume=False`) the stamp is
    (re)written and None returned.
    """
    if resume and path.exists():
        recorded = path.read_text()
        if recorded != stamp:
            return (
                f"{path} was written by a different configuration:\n"
                f"--- {label} ---\n{recorded}--- now ---\n{stamp}{advice}"
            )
    path.write_text(stamp)
    return None


def csv_number(text: str | None) -> float:
    """A CSV cell as a float, with anything unparseable reading as NaN.

    "NaN" is what the runners write for a cell they have no value for, and an
    empty cell means the same thing; neither may become a 0 that an aggregate
    would then treat as a measurement.
    """
    try:
        return float(text) if text is not None and text.strip() else math.nan
    except ValueError:
        return math.nan
