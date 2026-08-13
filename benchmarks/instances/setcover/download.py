"""Download the OR-Library set-covering instances used for Set-variable coverage.

The roster is deliberately small (issue #93): ten instances from J.E. Beasley's
OR-Library set-covering collection
(https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html), chosen to cover
both cost regimes with the smallest standard files available:

  * scp41..scp45 - 200 rows x 1000 columns, integer costs in [1, 100]
    (Balas & Ho instances, distributed via OR-Library)
  * scpe1..scpe5 -  50 rows x  500 columns, unit costs (unicost)

Every one of the ten has a *proven* optimum (see OPTIMUM below), so no reference
solver run is needed to score a result.

The files are vendored next to this script (~200 KiB total) so the C++ tests and
the benchmark runner work with no network. This script refreshes them and
verifies that what the server returns still parses to the same instance:

    python download.py            # fetch missing files, verify all
    python download.py --force    # re-fetch every file
    python download.py --check    # verify vendored files only, no network

`--check` is the offline gate: it re-parses each vendored file and asserts the
declared dimensions, the column-index range, and the recorded SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

URL_TEMPLATE = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}.txt"
MANIFEST_FILENAME = "manifest.csv"

HERE = Path(__file__).resolve().parent


class Expected(NamedTuple):
    """Published facts about an instance, asserted after every download."""

    rows: int
    cols: int
    optimum: int


# Optima are proven optimal values. Primary source: J.E. Beasley, "An algorithm
# for set covering problems", EJOR 31 (1987) 85-93, which OR-Library names as the
# source of the optimal values for problem sets 4-6 and A-E. Cross-checked
# against the public machine-readable table in
# https://github.com/fontanf/setcoveringsolver/blob/master/data/data.csv, whose
# "Best known solution value" and "Best known bound" columns agree for all ten
# (i.e. optimality is proven, not merely best-known).
ROSTER: dict[str, Expected] = {
    "scp41": Expected(rows=200, cols=1000, optimum=429),
    "scp42": Expected(rows=200, cols=1000, optimum=512),
    "scp43": Expected(rows=200, cols=1000, optimum=516),
    "scp44": Expected(rows=200, cols=1000, optimum=494),
    "scp45": Expected(rows=200, cols=1000, optimum=512),
    "scpe1": Expected(rows=50, cols=500, optimum=5),
    "scpe2": Expected(rows=50, cols=500, optimum=5),
    "scpe3": Expected(rows=50, cols=500, optimum=5),
    "scpe4": Expected(rows=50, cols=500, optimum=5),
    "scpe5": Expected(rows=50, cols=500, optimum=5),
}


class Instance(NamedTuple):
    """A parsed OR-Library set-covering file."""

    rows: int
    cols: int
    cost: list[int]
    row_cols: list[list[int]]  # 0-based column indices covering each row

    @property
    def nonzeros(self) -> int:
        return sum(len(cols) for cols in self.row_cols)


def parse(text: str, name: str) -> Instance:
    """Parse the OR-Library set-covering format.

    Layout (whitespace-separated, line breaks are not significant):
        m n
        c(1) ... c(n)
        for each row i: k(i) followed by k(i) 1-based column indices
    """
    tokens = text.split()
    pos = 0

    def take() -> int:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"{name}: file ended early after {pos} tokens")
        value = int(tokens[pos])
        pos += 1
        return value

    rows = take()
    cols = take()
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name}: nonsensical dimensions {rows}x{cols}")
    cost = [take() for _ in range(cols)]
    row_cols: list[list[int]] = []
    for i in range(rows):
        count = take()
        if count <= 0:
            raise ValueError(f"{name}: row {i} is covered by no column - infeasible instance")
        covering: list[int] = []
        for _ in range(count):
            col = take()
            if not 1 <= col <= cols:
                raise ValueError(f"{name}: row {i} references column {col} outside 1..{cols}")
            covering.append(col - 1)
        row_cols.append(covering)
    if pos != len(tokens):
        raise ValueError(f"{name}: {len(tokens) - pos} trailing tokens after the last row")
    return Instance(rows=rows, cols=cols, cost=cost, row_cols=row_cols)


def verify(name: str, text: str) -> Instance:
    """Parse and check the instance against its published dimensions."""
    inst = parse(text, name)
    expected = ROSTER[name]
    if (inst.rows, inst.cols) != (expected.rows, expected.cols):
        raise ValueError(
            f"{name}: got {inst.rows}x{inst.cols}, expected {expected.rows}x{expected.cols}"
        )
    return inst


def fetch(name: str) -> str:
    url = URL_TEMPLATE.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
            return response.read().decode("ascii")
    except urllib.error.URLError as exc:
        raise SystemExit(f"{name}: download failed ({url}): {exc}") from exc


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def read_manifest() -> dict[str, str]:
    path = HERE / MANIFEST_FILENAME
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["instance"]: row["sha256"] for row in csv.DictReader(handle)}


def write_manifest(rows: list[tuple[str, Instance, str]]) -> None:
    path = HERE / MANIFEST_FILENAME
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "rows", "cols", "nonzeros", "optimum", "sha256"])
        for name, inst, digest in rows:
            writer.writerow(
                [name, inst.rows, inst.cols, inst.nonzeros, ROSTER[name].optimum, digest]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that exist")
    parser.add_argument("--check", action="store_true", help="verify vendored files, no network")
    args = parser.parse_args()

    recorded = read_manifest()
    manifest_rows: list[tuple[str, Instance, str]] = []
    failures: list[str] = []

    for name in ROSTER:
        path = HERE / f"{name}.txt"
        if args.check or (path.exists() and not args.force):
            if not path.exists():
                failures.append(f"{name}: missing (run without --check to download)")
                continue
            text = path.read_text()
            source = "vendored"
        else:
            text = fetch(name)
            path.write_text(text)
            source = "downloaded"

        try:
            inst = verify(name, text)
        except ValueError as exc:
            failures.append(str(exc))
            continue

        digest = sha256(text)
        if name in recorded and recorded[name] != digest:
            failures.append(
                f"{name}: sha256 {digest} does not match the recorded {recorded[name]} - "
                "the upstream file changed, or the vendored copy was edited"
            )
            continue
        manifest_rows.append((name, inst, digest))
        print(
            f"{name:8s} {source:10s} {inst.rows:4d} rows x {inst.cols:5d} cols, "
            f"{inst.nonzeros:6d} nonzeros, optimum {ROSTER[name].optimum}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1

    if not args.check:
        write_manifest(manifest_rows)
    print(f"{len(manifest_rows)}/{len(ROSTER)} instances verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
