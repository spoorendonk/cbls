"""Tests for the MIPfeas roster derivation.

The roster is not published as a name list, so it is derived from two MIPLIB files
and the derivation is asserted. That assertion is what stops a MIPLIB revision
silently redefining the benchmark, and it is a guarantee the README makes in
writing — so it gets a test rather than being trusted.
"""

from __future__ import annotations

import pytest

from benchmarks.instances.mipfeas.download import (
    EXPECTED_ROSTER_SIZE,
    build_roster,
    parse_solu,
    parse_test_file,
)

#: One line per instance, exactly as MIPLIB's `.test` files are written.
TEST_FILE = "\n".join(f"inst{i}.mps.gz" for i in range(EXPECTED_ROSTER_SIZE + 2)) + "\n"


def _solu(overrides: dict[str, str] | None = None) -> str:
    """A solution file covering the test roster, `=opt=` unless overridden."""
    lines = []
    for i in range(EXPECTED_ROSTER_SIZE + 2):
        name = f"inst{i}"
        lines.append((overrides or {}).get(name, f"=opt=  {name}  {100 + i}"))
    return "\n".join(lines) + "\n"


def test_parse_test_file_strips_the_mps_gz_suffix() -> None:
    assert parse_test_file("a.mps.gz\nb.mps.gz\n") == ["a", "b"]


def test_parse_test_file_ignores_blanks_and_comments() -> None:
    assert parse_test_file("# header\n\na.mps.gz\n") == ["a"]


def test_parse_solu_reads_tags_and_values() -> None:
    entries = parse_solu("=opt=  a  1.5\n=best= b  2.0\n=inf=  c\n")
    assert entries["a"] == ("=opt=", 1.5)
    assert entries["b"] == ("=best=", 2.0)
    assert entries["c"] == ("=inf=", None)


def test_build_roster_excludes_infeasible_instances() -> None:
    # 235 names minus 2 infeasible = the expected 233.
    solu = _solu({"inst0": "=inf=  inst0", "inst1": "=inf=  inst1"})
    roster = build_roster(TEST_FILE, solu)
    assert len(roster) == EXPECTED_ROSTER_SIZE
    assert all(entry.instance not in ("inst0", "inst1") for entry in roster)


def test_build_roster_records_whether_a_reference_is_proven() -> None:
    solu = _solu({"inst0": "=inf= inst0", "inst1": "=inf= inst1", "inst2": "=best= inst2  42.0"})
    kinds = {entry.instance: entry.reference_kind for entry in build_roster(TEST_FILE, solu)}
    assert kinds["inst2"] == "best"
    assert kinds["inst3"] == "opt"


def test_build_roster_rejects_a_roster_of_the_wrong_size() -> None:
    # A MIPLIB revision that adds or drops an instance must fail the download
    # rather than silently redefine what "MIPfeas" means.
    solu = _solu({"inst0": "=inf=  inst0"})  # only one exclusion -> 234
    with pytest.raises(RuntimeError, match=f"expected {EXPECTED_ROSTER_SIZE}"):
        build_roster(TEST_FILE, solu)


def test_build_roster_rejects_an_instance_with_no_reference_value() -> None:
    # The Primal Integral needs a reference per instance; =unkn= cannot supply one.
    solu = _solu({"inst0": "=inf= inst0", "inst1": "=inf= inst1", "inst2": "=unkn= inst2"})
    with pytest.raises(RuntimeError, match="no usable reference value"):
        build_roster(TEST_FILE, solu)


def test_build_roster_rejects_an_instance_missing_from_the_solution_file() -> None:
    solu = "\n".join(_solu().splitlines()[:-1]) + "\n"
    with pytest.raises(RuntimeError, match="no solution-file entry"):
        build_roster(TEST_FILE, solu)
