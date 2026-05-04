"""Download the MIPLIB-FJ instance subset.

Fetches each `.mps.gz` from the MIPLIB 2017 server and the matching
`.solu` file with reference optima.

Usage:
    python download.py            # fetch all instances in INSTANCES
    python download.py --force    # re-download even if file exists
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# MIPLIB 2017 instance URL pattern. Each instance lives at
#   https://miplib.zib.de/WebData/instances/<name>.mps.gz
INSTANCE_URL_TEMPLATE = "https://miplib.zib.de/WebData/instances/{name}.mps.gz"

# Reference solution file (verified =opt= / =inf= / =best= tags).
SOLU_URL = "https://miplib.zib.de/downloads/miplib2017-v22.solu"
SOLU_FILENAME = "miplib2017-v22.solu"

# Initial FJ subset. See README.md for selection rationale.
# Small, structurally diverse, all should have an `=opt=` in miplib2017-v22.solu.
INSTANCES: list[str] = [
    # Binary combinatorial / small puzzles
    "enlight_hard",
    "markshare1",
    "markshare2",
    # General-integer small MILPs
    "gen-ip054",
    "gen-ip002",
    # Set covering / knapsack-like
    "pk1",
    "mas76",
    # Mixed binary / continuous, small
    "neos5",
    "flugpl",
    "mad",
    # Mixed binary, slightly larger
    "binkar10_1",
]


def _looks_like_html(data: bytes) -> bool:
    head = data[:128].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def fetch(url: str, dest: Path, force: bool = False) -> bool:
    """Download `url` to `dest`. Returns True on success (already-present counts).

    Validates content for `.mps.gz` (magic bytes 1F 8B) so a 200-OK that
    redirects to an HTML "not found" landing page is rejected.
    """
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"[skip]  {dest.name} (exists, {dest.stat().st_size} bytes)")
        return True
    try:
        print(f"[fetch] {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "cbls-miplib-fj/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        # Content validation: .mps.gz must start with the gzip magic.
        if dest.suffix == ".gz":
            if not data.startswith(b"\x1f\x8b") or _looks_like_html(data):
                print(f"[fail]  {url}: server returned HTML / non-gzip "
                      f"(probably 404 disguised as 200)")
                return False
        with open(dest, "wb") as out:
            out.write(data)
        print(f"        -> {dest.name} ({len(data)} bytes, "
              f"sha256 {hashlib.sha256(data).hexdigest()[:12]}...)")
        return True
    except urllib.error.HTTPError as e:
        print(f"[fail]  {url}: HTTP {e.code}")
    except urllib.error.URLError as e:
        print(f"[fail]  {url}: {e.reason}")
    except Exception as e:  # noqa: BLE001
        print(f"[fail]  {url}: {e}")
    if dest.exists():
        try:
            dest.unlink()
        except OSError:
            pass
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-download even if file exists")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    os.chdir(here)

    ok = 0
    fail = 0

    print("=== MIPLIB-FJ download ===")
    print(f"target dir: {here}")

    # Solu first (small).
    solu_path = here / SOLU_FILENAME
    if fetch(SOLU_URL, solu_path, force=args.force):
        ok += 1
    else:
        fail += 1
        print("WARNING: .solu fetch failed — gap-to-opt computation will skip.")

    for name in INSTANCES:
        url = INSTANCE_URL_TEMPLATE.format(name=name)
        dest = here / f"{name}.mps.gz"
        if fetch(url, dest, force=args.force):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. ok={ok} fail={fail} of {len(INSTANCES) + 1} files.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
