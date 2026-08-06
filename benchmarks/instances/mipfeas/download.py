"""Download the MIPfeas benchmark roster.

MIPfeas (https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/)
scores solvers on 233 instances: the MIPLIB 2017 *benchmark set* (240) minus the
instances known to be infeasible, which are excluded so the Primal Integral stays
well defined.

The roster is not published as a name list, so it is derived here and the derivation
is asserted: benchmark-v2.test minus the `=inf=`-tagged names in the solution file
must come to exactly 233. A MIPLIB revision that changes either input therefore fails
loudly instead of silently redefining the benchmark.

Usage:
    python download.py                 # full 233 roster (~317 MB via benchmark.zip)
    python download.py --subset smoke  # the 11-instance smoke roster only
    python download.py --force         # re-download even if files exist
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

BENCHMARK_TEST_URL = "https://miplib.zib.de/downloads/benchmark-v2.test"
SOLU_URL = "https://miplib.zib.de/downloads/miplib2017-v36.solu"
BENCHMARK_ZIP_URL = "https://miplib.zib.de/downloads/benchmark.zip"
INSTANCE_URL_TEMPLATE = "https://miplib.zib.de/WebData/instances/{name}.mps.gz"

SOLU_FILENAME = "miplib2017-v36.solu"
ROSTER_FILENAME = "roster.csv"
MANIFEST_FILENAME = "manifest.csv"
SMOKE_FILENAME = "smoke.csv"

# MIPfeas: 240 benchmark-set instances minus the 7 known-infeasible ones.
EXPECTED_ROSTER_SIZE = 233

# Smoke roster: a genuine subset of the 233, small enough to run in minutes.
# Nine are also vendored under benchmarks/instances/miplib-fj/ and span pure
# binary, general integer and mixed binary/continuous structures; `atlanta-ip`
# adds a mid-size model and `neos-5114902-kasavu` (~4.2M nonzeros) is the
# memory probe used to size job concurrency for the full run.
SMOKE_INSTANCES: tuple[str, ...] = (
    "enlight_hard",
    "markshare2",
    "gen-ip054",
    "gen-ip002",
    "pk1",
    "mas76",
    "neos5",
    "mad",
    "binkar10_1",
    "atlanta-ip",
    "neos-5114902-kasavu",
)

USER_AGENT = "cbls-mipfeas/0.1"


class RosterEntry(NamedTuple):
    """One roster instance and the reference value the Primal Integral scores against."""

    instance: str
    reference_value: float
    #: `opt` when the value is a proven optimum, `best` when it is best-known.
    reference_kind: str


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data: bytes = resp.read()
    return data


def _looks_like_html(data: bytes) -> bool:
    head = data[:128].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def parse_test_file(text: str) -> list[str]:
    """Instance names from a MIPLIB `.test` file (one `<name>.mps.gz` per line)."""
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.removesuffix(".gz").removesuffix(".mps"))
    return names


def parse_solu(text: str) -> dict[str, tuple[str, float | None]]:
    """Map instance name -> (tag, value). Tags: =opt=, =best=, =inf=, =unkn=, =unbd=.

    `=inf=` and `=unbd=` lines carry no value.
    """
    entries: dict[str, tuple[str, float | None]] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 2 or not parts[0].startswith("="):
            continue
        tag, name = parts[0], parts[1]
        value = float(parts[2]) if len(parts) > 2 else None
        entries[name] = (tag, value)
    return entries


def build_roster(test_text: str, solu_text: str) -> list[RosterEntry]:
    """Derive the MIPfeas roster, raising if it does not come to exactly 233."""
    names = parse_test_file(test_text)
    solu = parse_solu(solu_text)

    missing = [n for n in names if n not in solu]
    if missing:
        raise RuntimeError(
            f"{len(missing)} benchmark-set instances have no solution-file entry "
            f"(e.g. {missing[:3]}); the reference value the Primal Integral needs is "
            f"unavailable, so the roster cannot be built."
        )

    roster: list[RosterEntry] = []
    unusable: list[str] = []
    for name in names:
        tag, value = solu[name]
        if tag == "=inf=":
            continue  # excluded by MIPfeas: no primal value to measure against
        if tag == "=opt=" and value is not None:
            roster.append(RosterEntry(name, value, "opt"))
        elif tag == "=best=" and value is not None:
            roster.append(RosterEntry(name, value, "best"))
        else:
            unusable.append(f"{name} ({tag})")

    if unusable:
        raise RuntimeError(
            f"{len(unusable)} benchmark-set instances carry no usable reference value "
            f"(e.g. {unusable[:3]}). The Primal Integral needs one per instance."
        )
    if len(roster) != EXPECTED_ROSTER_SIZE:
        raise RuntimeError(
            f"Derived roster has {len(roster)} instances, expected {EXPECTED_ROSTER_SIZE}. "
            f"MIPLIB's benchmark set or solution file has changed; re-check the roster "
            f"against the MIPfeas methodology before publishing any comparison."
        )
    return roster


def write_roster_csv(roster: list[RosterEntry], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "reference_value", "reference_kind"])
        for entry in roster:
            writer.writerow([entry.instance, repr(entry.reference_value), entry.reference_kind])


def read_roster_csv(path: Path) -> list[RosterEntry]:
    with open(path, newline="") as fh:
        return [
            RosterEntry(row["instance"], float(row["reference_value"]), row["reference_kind"])
            for row in csv.DictReader(fh)
        ]


def write_smoke_csv(roster: list[RosterEntry], path: Path) -> None:
    by_name = {entry.instance: entry for entry in roster}
    unknown = [n for n in SMOKE_INSTANCES if n not in by_name]
    if unknown:
        raise RuntimeError(f"Smoke instances not in the roster: {unknown}")
    write_roster_csv([by_name[n] for n in SMOKE_INSTANCES], path)


def fetch_instance(name: str, dest: Path, force: bool = False) -> bool:
    """Fetch one `<name>.mps.gz`. Returns True on success (already-present counts)."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        return True
    url = INSTANCE_URL_TEMPLATE.format(name=name)
    try:
        data = _http_get(url)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"[fail]  {url}: {exc}")
        return False
    # A 404 disguised as a 200 landing page must not be written out as an instance.
    if not data.startswith(b"\x1f\x8b") or _looks_like_html(data):
        print(f"[fail]  {url}: server returned HTML / non-gzip")
        return False
    dest.write_bytes(data)
    return True


def fetch_via_zip(names: list[str], target_dir: Path, force: bool) -> list[str]:
    """Fetch the whole benchmark set in one request and extract the roster from it.

    233 individual requests to miplib.zib.de is antisocial; `benchmark.zip` is a
    single ~317 MB download covering all of them.
    """
    wanted = {f"{name}.mps.gz": name for name in names}
    missing = [n for n in names if force or not (target_dir / f"{n}.mps.gz").exists()]
    if not missing:
        print("[skip]  all roster instances already present")
        return []

    print(f"[fetch] {BENCHMARK_ZIP_URL} ({len(missing)} of {len(names)} instances missing)")
    archive = _http_get(BENCHMARK_ZIP_URL, timeout=1800)
    print(f"        -> {len(archive)} bytes; extracting")

    failed: list[str] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        present = {Path(info.filename).name for info in zf.infolist()}
        for member, name in wanted.items():
            dest = target_dir / member
            if dest.exists() and not force:
                continue
            if member not in present:
                print(f"[fail]  {member} not in {BENCHMARK_ZIP_URL}")
                failed.append(name)
                continue
            match = next(i for i in zf.infolist() if Path(i.filename).name == member)
            dest.write_bytes(zf.read(match))
    return failed


def write_manifest(names: list[str], target_dir: Path, path: Path) -> None:
    """Record sha256 + byte size per instance, so a re-fetch is verifiable.

    `bytes` doubles as the size proxy the run driver schedules on: the largest
    instances are run serially rather than alongside others.
    """
    rows: list[tuple[str, str, int]] = []
    for name in names:
        src = target_dir / f"{name}.mps.gz"
        if not src.exists():
            continue
        data = src.read_bytes()
        rows.append((name, hashlib.sha256(data).hexdigest(), len(data)))
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "sha256", "bytes"])
        writer.writerows(rows)
    print(f"[write] {path.name} ({len(rows)} instances)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("full", "smoke"),
        default="full",
        help="full = all 233 instances via benchmark.zip; smoke = the 11-instance subset",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--roster-only",
        action="store_true",
        help="derive roster.csv / smoke.csv without downloading any instance",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    os.chdir(here)

    print("=== MIPfeas roster ===")
    print(f"[fetch] {BENCHMARK_TEST_URL}")
    test_text = _http_get(BENCHMARK_TEST_URL).decode()
    print(f"[fetch] {SOLU_URL}")
    solu_bytes = _http_get(SOLU_URL)
    (here / SOLU_FILENAME).write_bytes(solu_bytes)

    roster = build_roster(test_text, solu_bytes.decode())
    n_opt = sum(1 for e in roster if e.reference_kind == "opt")
    print(f"        roster: {len(roster)} instances ({n_opt} proven optimal)")

    write_roster_csv(roster, here / ROSTER_FILENAME)
    write_smoke_csv(roster, here / SMOKE_FILENAME)
    print(f"[write] {ROSTER_FILENAME}, {SMOKE_FILENAME}")

    if args.roster_only:
        return 0

    names = (
        list(SMOKE_INSTANCES) if args.subset == "smoke" else [entry.instance for entry in roster]
    )
    if args.subset == "smoke":
        failed = [n for n in names if not fetch_instance(n, here / f"{n}.mps.gz", args.force)]
    else:
        failed = fetch_via_zip(names, here, args.force)

    write_manifest(names, here, here / MANIFEST_FILENAME)

    have = sum(1 for n in names if (here / f"{n}.mps.gz").exists())
    print(f"\nDone. {have}/{len(names)} instances present, {len(failed)} failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
