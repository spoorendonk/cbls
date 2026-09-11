"""Download the MIPfeas benchmark roster.

MIPfeas (https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/)
scores solvers on 233 instances: the MIPLIB 2017 *benchmark set* (240) minus the
instances known to be infeasible, which are excluded so the Primal Integral stays
well defined.

The roster is not published as a name list, so it is derived here and the derivation
is asserted: benchmark-v2.test minus the `=inf=`-tagged names in the solution file
must come to exactly 233. A MIPLIB revision that changes either input therefore fails
loudly instead of silently redefining the benchmark.

The bytes are pinned as well as derived. `manifest.csv` records a sha256 per
instance and `references.csv` records one per *yardstick* file -- the MIPLIB
solution file and the two roster tables scored against it. Both are verified
rather than merely written:

* `--verify` checks every pinned file against what is on disk, offline, and exits
  non-zero naming each mismatch. The run driver does the same check before it
  starts, so a corrupted or substituted instance stops the run instead of
  silently changing what a published row measured.
* A download whose solution file no longer hashes to the pinned value **fails**
  and writes nothing. Replacing the yardstick needs `--update-references`, which
  prints the per-instance value and kind changes it is about to make -- MIPLIB
  does revise its solution file, and a silent revision moves every gap in the
  published table at once.
* An instance whose bytes changed under a re-fetch likewise fails, and needs
  `--update-manifest`. An instance the manifest has never seen is simply added:
  that is an acquisition, not an overwrite.

Usage:
    python download.py                 # full 233 roster (~546 MiB via benchmark.zip)
    python download.py --subset smoke  # the 11-instance smoke roster only
    python download.py --force         # re-download even if files exist
    python download.py --verify        # check pinned bytes, no network
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
REFERENCES_FILENAME = "references.csv"

#: The files every published gap is measured against, pinned by hash in
#: `references.csv`. The solution file is the yardstick itself; the two roster
#: tables are the derivation of it this benchmark actually reads, and pinning the
#: derivation as well as its input is what makes a hand-edited roster visible.
PINNED_REFERENCE_FILES: tuple[str, ...] = (SOLU_FILENAME, ROSTER_FILENAME, SMOKE_FILENAME)

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


def write_smoke_csv(roster: list[RosterEntry], path: Path) -> None:
    by_name = {entry.instance: entry for entry in roster}
    unknown = [n for n in SMOKE_INSTANCES if n not in by_name]
    if unknown:
        raise RuntimeError(f"Smoke instances not in the roster: {unknown}")
    write_roster_csv([by_name[n] for n in SMOKE_INSTANCES], path)


def read_roster_csv(path: Path) -> list[RosterEntry]:
    """The roster as committed, so a freshly derived one can be diffed against it."""
    with open(path, newline="") as fh:
        return [
            RosterEntry(row["instance"], float(row["reference_value"]), row["reference_kind"])
            for row in csv.DictReader(fh)
        ]


def reference_changes(old: list[RosterEntry], new: list[RosterEntry]) -> list[str]:
    """One line per instance whose reference value or kind moved, plus arrivals/departures.

    This is the visible diff an update has to print. A changed *value* moves the
    gap a published row reports; a changed *kind* moves what the gap means, since
    a run below a proven optimum is a defect and a run below a best-known value is
    a new record. Both are reported, and neither is allowed to happen as a side
    effect of re-running acquisition.
    """
    before = {entry.instance: entry for entry in old}
    after = {entry.instance: entry for entry in new}
    lines: list[str] = []
    for name in sorted(set(before) - set(after)):
        lines.append(
            f"  - dropped  {name} ({before[name].reference_value!r} {before[name].reference_kind})"
        )
    for name in sorted(set(after) - set(before)):
        lines.append(
            f"  + added    {name} ({after[name].reference_value!r} {after[name].reference_kind})"
        )
    for name in sorted(set(before) & set(after)):
        was, now = before[name], after[name]
        if was.reference_value != now.reference_value or was.reference_kind != now.reference_kind:
            lines.append(
                f"  ~ changed  {name}: {was.reference_value!r} ({was.reference_kind}) "
                f"-> {now.reference_value!r} ({now.reference_kind})"
            )
    return lines


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pin_table(path: Path, key_column: str) -> dict[str, tuple[str, int]]:
    """`{key: (sha256, bytes)}` from a two-column pin table, empty when absent."""
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {row[key_column]: (row["sha256"], int(row["bytes"])) for row in csv.DictReader(fh)}


def write_pin_table(path: Path, key_column: str, rows: dict[str, tuple[str, int]]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([key_column, "sha256", "bytes"])
        writer.writerows((key, sha, size) for key, (sha, size) in sorted(rows.items()))


def pin_mismatch(label: str, data: bytes, pinned: tuple[str, int] | None) -> str | None:
    """The complaint about `data` against its pin, or None when it matches.

    A size-only comparison would pass a byte-for-byte substitution of the same
    length, and a hash-only one loses the cheapest thing to say about a truncated
    download, so both are reported.
    """
    if pinned is None:
        return f"{label}: present but not pinned (no recorded hash)"
    sha, size = pinned
    actual = sha256_of(data)
    if actual == sha and len(data) == size:
        return None
    return f"{label}: pinned sha256 {sha} ({size} bytes), found {actual} ({len(data)} bytes)"


def verify_references(here: Path) -> list[str]:
    """Check the yardstick files against `references.csv`. Empty list means clean."""
    pins = read_pin_table(here / REFERENCES_FILENAME, "file")
    if not pins:
        return [
            f"{REFERENCES_FILENAME} is absent, so the reference files are not pinned at all. "
            f"Re-create it with --update-references."
        ]
    problems: list[str] = []
    for name in PINNED_REFERENCE_FILES:
        path = here / name
        if not path.exists():
            problems.append(f"{name}: pinned but absent")
            continue
        complaint = pin_mismatch(name, path.read_bytes(), pins.get(name))
        if complaint is not None:
            problems.append(complaint)
    return problems


def verify_instances(
    names: list[str], target_dir: Path, manifest_path: Path
) -> tuple[list[str], list[str]]:
    """Check each present instance against `manifest.csv`.

    Returns `(problems, absent)`. Absent is not a problem here: a checkout that
    fetched only the smoke subset is a legitimate state, and the run driver
    refuses a roster with missing instances on its own. A file that is *present*
    and does not hash to its pin is the failure this exists for, and so is one
    the manifest has never heard of -- an unpinned instance is indistinguishable
    from a substituted one.
    """
    pins = read_pin_table(manifest_path, "instance")
    problems: list[str] = []
    absent: list[str] = []
    if not pins:
        return (
            [f"{manifest_path.name} is absent or empty, so no instance bytes are pinned."],
            list(names),
        )
    for name in names:
        path = target_dir / f"{name}.mps.gz"
        if not path.exists():
            absent.append(name)
            continue
        complaint = pin_mismatch(f"{name}.mps.gz", path.read_bytes(), pins.get(name))
        if complaint is not None:
            problems.append(complaint)
    return problems, absent


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
    single ~546 MiB download covering all of them.
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


def write_manifest(names: list[str], target_dir: Path, path: Path, *, update: bool) -> list[str]:
    """Record sha256 + byte size per instance, so a re-fetch is verifiable.

    `bytes` doubles as the size proxy the run driver schedules on: the largest
    instances are run serially rather than alongside others.

    Merged into what is already committed rather than replacing it. `--subset smoke`
    fetches 11 of the 233 names, and rewriting the file from those alone drops the
    other 222 rows -- after which the run driver sees no sizes and schedules the
    largest instances alongside everything else instead of alone. That is the #103
    failure mode: running a tool over a subset silently destroys committed data.

    An instance the manifest has never seen is added; one whose bytes have *moved*
    is a substitution, and is refused unless `update` says so. Returns the names
    whose pinned bytes changed -- empty when the manifest merely grew.
    """
    rows = read_pin_table(path, "instance")
    changed: list[str] = []
    refreshed = 0
    for name in names:
        src_path = target_dir / f"{name}.mps.gz"
        if not src_path.exists():
            continue
        data = src_path.read_bytes()
        pin = (sha256_of(data), len(data))
        previous = rows.get(name)
        if previous is not None and previous != pin:
            changed.append(name)
            if not update:
                continue
        rows[name] = pin
        refreshed += 1
    if changed and not update:
        return changed
    write_pin_table(path, "instance", rows)
    print(f"[write] {path.name} ({len(rows)} instances, {refreshed} refreshed)")
    return changed


def verify_mode(here: Path, names: list[str]) -> int:
    """`--verify`: check every pin against what is on disk. No network."""
    problems = verify_references(here)
    instance_problems, absent = verify_instances(names, here, here / MANIFEST_FILENAME)
    problems += instance_problems
    checked = len(names) - len(absent)
    print(f"[check] {len(PINNED_REFERENCE_FILES)} reference files, {checked} instances present")
    if absent:
        print(f"[note]  {len(absent)} instance(s) absent, e.g. {absent[:3]}; not checked")
    if problems:
        print(
            f"\nFAILED: {len(problems)} pinned file(s) do not match what is recorded:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nA published table measured the pinned bytes, not these. Restore the files, "
            "or re-pin deliberately with --update-references / --update-manifest.",
            file=sys.stderr,
        )
        return 1
    print("OK: every pinned file matches.")
    return 0


def update_references(here: Path, solu_bytes: bytes, roster: list[RosterEntry]) -> None:
    """Write the yardstick files and re-pin them, printing the value diff first."""
    roster_path = here / ROSTER_FILENAME
    previous = read_roster_csv(roster_path) if roster_path.exists() else []
    changes = reference_changes(previous, roster)
    print(f"[update] reference values: {len(changes)} change(s)")
    for line in changes:
        print(line)

    (here / SOLU_FILENAME).write_bytes(solu_bytes)
    write_roster_csv(roster, roster_path)
    write_smoke_csv(roster, here / SMOKE_FILENAME)
    pins = {
        name: (sha256_of((here / name).read_bytes()), (here / name).stat().st_size)
        for name in PINNED_REFERENCE_FILES
    }
    write_pin_table(here / REFERENCES_FILENAME, "file", pins)
    print(f"[write] {SOLU_FILENAME}, {ROSTER_FILENAME}, {SMOKE_FILENAME}, {REFERENCES_FILENAME}")


def check_reference_pins(here: Path, solu_bytes: bytes, roster: list[RosterEntry]) -> list[str]:
    """Complaints about the freshly fetched yardstick against the committed pins."""
    pins = read_pin_table(here / REFERENCES_FILENAME, "file")
    if not pins:
        return [
            f"{REFERENCES_FILENAME} is absent: the reference files are not pinned. "
            f"Re-run with --update-references to create it."
        ]
    problems: list[str] = []
    upstream = pin_mismatch(f"{SOLU_FILENAME} (upstream)", solu_bytes, pins.get(SOLU_FILENAME))
    if upstream is not None:
        problems.append(upstream)
        problems += reference_changes(
            read_roster_csv(here / ROSTER_FILENAME) if (here / ROSTER_FILENAME).exists() else [],
            roster,
        )
    # The committed derivation is checked too: a hand-edited roster.csv is the same
    # defect as a revised upstream, and only this catches it.
    problems += [p for p in verify_references(here) if not p.startswith(SOLU_FILENAME + ":")]
    return problems


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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every pinned file against what is on disk and exit; no network",
    )
    parser.add_argument(
        "--update-references",
        action="store_true",
        help="accept a moved solution file and re-pin it, rewriting roster.csv / smoke.csv. "
        "Prints the per-instance reference value and kind changes first. Without this "
        "flag a moved yardstick fails the download and nothing is written",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="accept instance bytes that no longer match manifest.csv and re-pin them. "
        "Without it a substituted instance fails the download",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    os.chdir(here)

    if args.verify:
        roster_path = here / (SMOKE_FILENAME if args.subset == "smoke" else ROSTER_FILENAME)
        if not roster_path.exists():
            print(f"{roster_path.name} is absent; nothing to verify against.", file=sys.stderr)
            return 2
        return verify_mode(here, [entry.instance for entry in read_roster_csv(roster_path)])

    print("=== MIPfeas roster ===")
    print(f"[fetch] {BENCHMARK_TEST_URL}")
    test_text = _http_get(BENCHMARK_TEST_URL).decode()
    print(f"[fetch] {SOLU_URL}")
    solu_bytes = _http_get(SOLU_URL)

    roster = build_roster(test_text, solu_bytes.decode())
    n_opt = sum(1 for e in roster if e.reference_kind == "opt")
    print(f"        roster: {len(roster)} instances ({n_opt} proven optimal)")

    if args.update_references:
        update_references(here, solu_bytes, roster)
    else:
        problems = check_reference_pins(here, solu_bytes, roster)
        if problems:
            print(
                "\nFAILED: the reference files this benchmark is scored against have moved.",
                file=sys.stderr,
            )
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "\nEvery published gap is measured against these, so they are not replaced as "
                "a side effect of re-running acquisition. Re-run with --update-references to "
                "accept the change as a reviewable diff.",
                file=sys.stderr,
            )
            return 3
        print(f"[check] {SOLU_FILENAME} and the roster tables match their pins")

    if args.roster_only:
        return 0

    names = (
        list(SMOKE_INSTANCES) if args.subset == "smoke" else [entry.instance for entry in roster]
    )
    if args.subset == "smoke":
        failed = [n for n in names if not fetch_instance(n, here / f"{n}.mps.gz", args.force)]
    else:
        failed = fetch_via_zip(names, here, args.force)

    moved = write_manifest(names, here, here / MANIFEST_FILENAME, update=args.update_manifest)
    if moved and not args.update_manifest:
        print(
            f"\nFAILED: {len(moved)} instance file(s) no longer match their pinned bytes "
            f"(e.g. {moved[:3]}). manifest.csv was left untouched.\n"
            "A published row measured the pinned bytes. Re-run with --update-manifest to "
            "accept the new ones.",
            file=sys.stderr,
        )
        return 3

    have = sum(1 for n in names if (here / f"{n}.mps.gz").exists())
    print(f"\nDone. {have}/{len(names)} instances present, {len(failed)} failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
