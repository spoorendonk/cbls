"""Pinning downloaded files by hash (lifted from the MIPfeas downloader, #196).

A benchmark's reference values are derived once from the library that publishes
them and pinned, so a library revision cannot silently redefine the yardstick
(docs/measuring.md, "A comparison table is an OUTPUT; the reference values
are the artifact"). A pin table is a CSV of `<key>,sha256,bytes`; these are the four
operations every downloader that pins needs: hash, read, write, compare.
"""

from __future__ import annotations

import csv
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pin_table(path: Path, key_column: str) -> dict[str, tuple[str, int]]:
    """`{key: (sha256, bytes)}` from a two-column pin table, empty when absent."""
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {row[key_column]: (row["sha256"], int(row["bytes"])) for row in csv.DictReader(fh)}


def write_pin_table(path: Path, key_column: str, rows: dict[str, tuple[str, int]]) -> None:
    # Temp-then-rename, like every other file a benchmark writes. A pin table is
    # committed data a run driver gates on, and a truncated table does not merely
    # lose rows: a downloader reads a key it cannot find as a first acquisition
    # and re-pins whatever is on disk with no flag, which is exactly the
    # overwrite the update flags exist to require.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([key_column, "sha256", "bytes"])
        writer.writerows((key, sha, size) for key, (sha, size) in sorted(rows.items()))
    tmp.replace(path)


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
