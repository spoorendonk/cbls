"""The files a run leaves behind: how they are written, read, and where their shapes live.

Every writer here survives a kill at any instant -- an OOM kill, a reboot, a
Ctrl-C at hour nine -- by leaving either the previous file or the new one. Every
reader treats a file such a kill could have left as what it is, rather than as a
result.

THE RECORD SCHEMAS. Issue #160 asked for one per-job record both benchmarks
emit. They cannot share one without changing the records themselves, which the
same issue forbids: the runners write them (C++ and the CP-SAT baseline), a
published table's header is pinned against them, and each benchmark's resume
rule keys on its own. So each is declared once, beside the code that owns it,
and listed here:

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
ablation campaign row  `run_ablation.RESULT_COLUMNS`      `(instance, arm, seed)` in
                       (provenance + the runner row)      `results.csv`, less the trailing
                                                          partial block, under a matching
                                                          `stamp.txt`
=====================  =================================  ==============================
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def atomic_write(path: Path, data: str | bytes) -> None:
    """Replace `path`'s contents without ever leaving it truncated.

    Written to a sibling temporary file, flushed to disk, then renamed over
    `path`. `open(path, "w")` truncates before the first byte is written, so a
    kill inside that window replaces a published table -- or thirteen hours of
    campaign rows -- with nothing, at exit 0. Text is written with no newline
    translation: the bytes on disk are exactly the string's.
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

    A file truncated by an OOM kill or a reboot mid-write reads as absent: a
    driver would otherwise make the damage permanent, and scoring would later
    abort on the unparseable file. `ValueError` rather than only
    `JSONDecodeError`, because a file corrupted to binary raises
    `UnicodeDecodeError` out of `read_text()` before JSON parsing starts.
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

    `fsync` and not just a flush: a run lasts hours and the thing it is protected
    from is the machine going away, which is exactly the case a buffered write in
    the kernel's page cache does not survive.
    """
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow(values)
        fh.flush()
        os.fsync(fh.fileno())


def repair_torn_tail(path: Path) -> bool:
    """Drop an unterminated final line from an append-only file. True if one was dropped.

    A process killed mid-append leaves a partial row. It is a row nobody can read
    and, worse, one whose key would be missing from a resume set while its bytes
    stay in the file, so the run would be redone and the file would then hold a
    half-row wedged between two whole ones.
    """
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n")
    # Write-then-rename: this is the one function whose job is protecting an
    # append-only record against a kill, and `write_bytes` truncates to zero
    # before writing, so a kill inside that window destroys the whole record.
    atomic_write(path, data[: cut + 1] if cut >= 0 else b"")
    return True


def stamp_mismatch(path: Path, stamp: str, *, resume: bool) -> str | None:
    """The configuration a directory's records were written under, if it is not `stamp`.

    Resume keys on a record being *there*, which says nothing about what produced
    it. So a directory carries a stamp -- commit, budget, seeds, whatever makes
    two runs comparable -- and resuming into one written under another is refused
    by the caller, which is handed the recorded stamp to quote. Otherwise (a
    fresh directory, a matching stamp, or `resume=False`, which is starting over)
    the stamp is (re)written and None returned.
    """
    if resume and path.exists():
        recorded = path.read_text()
        if recorded != stamp:
            return recorded
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
