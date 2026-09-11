"""Tests for the MIPfeas roster derivation.

The roster is not published as a name list, so it is derived from two MIPLIB files
and the derivation is asserted. That assertion is what stops a MIPLIB revision
silently redefining the benchmark, and it is a guarantee the README makes in
writing — so it gets a test rather than being trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.instances.mipfeas.download import (
    EXPECTED_ROSTER_SIZE,
    MANIFEST_FILENAME,
    PINNED_REFERENCE_FILES,
    REFERENCES_FILENAME,
    ROSTER_FILENAME,
    SMOKE_FILENAME,
    SMOKE_INSTANCES,
    SOLU_FILENAME,
    RosterEntry,
    build_roster,
    check_reference_pins,
    parse_solu,
    parse_test_file,
    pin_fetched_instances,
    read_pin_table,
    read_roster_csv,
    reference_changes,
    sha256_of,
    update_references,
    verify_instances,
    verify_mode,
    verify_references,
    write_manifest,
    write_pin_table,
    write_roster_csv,
)

#: The committed roster directory, whose pins a run is actually scored against.
INSTANCE_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "mipfeas"

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


# --- Pinned bytes -------------------------------------------------------------
#
# The roster's instances and its yardstick files are pinned by hash and the pins
# are *checked*, so a corrupted, truncated or upstream-revised file stops a run
# instead of silently changing what a published row measured (issue #137). None
# of these tests needs the roster on disk: they build tiny fixtures, because the
# real instances are 546 MiB and re-fetching them is the very substitution the
# pins exist to catch.


def _pin_instances(directory: Path, files: dict[str, bytes]) -> Path:
    """Write `files` as instances and pin exactly those bytes in a manifest."""
    for name, data in files.items():
        (directory / f"{name}.mps.gz").write_bytes(data)
    manifest = directory / MANIFEST_FILENAME
    write_pin_table(
        manifest,
        "instance",
        {name: (sha256_of(data), len(data)) for name, data in files.items()},
    )
    return manifest


def test_a_corrupted_instance_is_reported_against_its_pin(tmp_path: Path) -> None:
    manifest = _pin_instances(tmp_path, {"a": b"original bytes"})
    (tmp_path / "a.mps.gz").write_bytes(b"tampered")

    problems, absent = verify_instances(["a"], tmp_path, manifest)

    assert absent == []
    assert len(problems) == 1
    assert "a.mps.gz" in problems[0]
    assert sha256_of(b"original bytes")[:16] in problems[0]


def test_a_substitution_of_the_same_length_is_still_caught(tmp_path: Path) -> None:
    # A size-only check passes this, which is why the pin carries a hash.
    manifest = _pin_instances(tmp_path, {"a": b"AAAAAAAA"})
    (tmp_path / "a.mps.gz").write_bytes(b"BBBBBBBB")

    problems, _ = verify_instances(["a"], tmp_path, manifest)

    assert len(problems) == 1


def test_an_instance_the_manifest_never_pinned_is_a_problem(tmp_path: Path) -> None:
    # Indistinguishable from a substituted one, so it is not waved through.
    manifest = _pin_instances(tmp_path, {"a": b"x"})
    (tmp_path / "b.mps.gz").write_bytes(b"y")

    problems, _ = verify_instances(["a", "b"], tmp_path, manifest)

    assert [p for p in problems if p.startswith("b.mps.gz")]


def test_an_absent_instance_is_not_reported_as_corrupt(tmp_path: Path) -> None:
    # A checkout holding only the smoke subset is legitimate; the run driver is
    # what refuses a roster with missing instances.
    manifest = _pin_instances(tmp_path, {"a": b"x"})
    (tmp_path / "a.mps.gz").unlink()

    problems, absent = verify_instances(["a"], tmp_path, manifest)

    assert problems == []
    assert absent == ["a"]


def test_an_unpinned_roster_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    problems, absent = verify_instances(["a"], tmp_path, tmp_path / MANIFEST_FILENAME)

    assert len(problems) == 1
    assert MANIFEST_FILENAME in problems[0]
    assert absent == ["a"]


def _reference_dir(tmp_path: Path, roster: list[RosterEntry]) -> Path:
    """A directory holding the three pinned reference files, correctly pinned."""
    (tmp_path / SOLU_FILENAME).write_bytes(b"=opt= inst0 100\n")
    write_roster_csv(roster, tmp_path / ROSTER_FILENAME)
    by_name = {entry.instance: entry for entry in roster}
    smoke = [by_name[n] for n in SMOKE_INSTANCES if n in by_name] or roster[:1]
    write_roster_csv(smoke, tmp_path / SMOKE_FILENAME)
    write_pin_table(
        tmp_path / REFERENCES_FILENAME,
        "file",
        {
            name: (sha256_of((tmp_path / name).read_bytes()), (tmp_path / name).stat().st_size)
            for name in PINNED_REFERENCE_FILES
        },
    )
    return tmp_path


def test_an_edited_roster_table_fails_its_pin(tmp_path: Path) -> None:
    # The derivation is pinned as well as its input: a hand-edited reference value
    # moves every gap scored against it and is otherwise invisible.
    here = _reference_dir(tmp_path, [RosterEntry("inst0", 100.0, "opt")])
    write_roster_csv([RosterEntry("inst0", 99.0, "opt")], here / ROSTER_FILENAME)

    problems = verify_references(here)

    assert len(problems) == 1
    assert problems[0].startswith(f"{ROSTER_FILENAME}:")


def test_unpinned_reference_files_are_a_failure_not_a_pass(tmp_path: Path) -> None:
    problems = verify_references(tmp_path)

    assert len(problems) == 1
    assert REFERENCES_FILENAME in problems[0]


def test_verify_mode_exits_non_zero_when_a_pin_does_not_match(tmp_path: Path) -> None:
    here = _reference_dir(tmp_path, [RosterEntry("inst0", 100.0, "opt")])
    _pin_instances(here, {"inst0": b"bytes"})
    (here / "inst0.mps.gz").write_bytes(b"other")

    assert verify_mode(here, ["inst0"]) == 1


def test_verify_mode_exits_zero_on_an_intact_checkout(tmp_path: Path) -> None:
    here = _reference_dir(tmp_path, [RosterEntry("inst0", 100.0, "opt")])
    _pin_instances(here, {"inst0": b"bytes"})

    assert verify_mode(here, ["inst0"]) == 0


def test_the_committed_reference_files_match_their_pins() -> None:
    # The pins in the tree are the ones a run is scored against, so they are
    # checked here rather than only by whoever next runs acquisition.
    assert verify_references(INSTANCE_DIR) == []


# --- Overwriting a reference needs a flag -------------------------------------


def test_reference_changes_names_a_moved_value_and_a_moved_kind() -> None:
    before = [RosterEntry("a", 1.0, "opt"), RosterEntry("b", 2.0, "best")]
    after = [RosterEntry("a", 1.5, "opt"), RosterEntry("b", 2.0, "opt")]

    lines = reference_changes(before, after)

    assert len(lines) == 2
    assert any("a" in line and "1.0" in line and "1.5" in line for line in lines)
    assert any("b" in line and "best" in line and "opt" in line for line in lines)


def test_reference_changes_names_arrivals_and_departures() -> None:
    lines = reference_changes([RosterEntry("a", 1.0, "opt")], [RosterEntry("b", 2.0, "opt")])

    assert any(line.strip().startswith("- dropped") and " a " in line for line in lines)
    assert any(line.strip().startswith("+ added") and " b " in line for line in lines)


def test_reference_changes_is_empty_for_an_unchanged_roster() -> None:
    roster = [RosterEntry("a", 1.0, "opt")]
    assert reference_changes(roster, list(roster)) == []


def test_a_moved_solution_file_fails_the_download_rather_than_replacing_the_yardstick(
    tmp_path: Path,
) -> None:
    here = _reference_dir(tmp_path, [RosterEntry("inst0", 100.0, "opt")])

    problems = check_reference_pins(here, b"=opt= inst0 42\n", [RosterEntry("inst0", 42.0, "opt")])

    assert problems, "an upstream revision must not pass unnoticed"
    assert any(SOLU_FILENAME in p for p in problems)
    # And the diff a reviewer needs is in the complaint, not only the hashes.
    assert any("100.0" in p and "42.0" in p for p in problems)


def test_an_unchanged_solution_file_passes_the_pin_check(tmp_path: Path) -> None:
    here = _reference_dir(tmp_path, [RosterEntry("inst0", 100.0, "opt")])

    assert (
        check_reference_pins(here, b"=opt= inst0 100\n", [RosterEntry("inst0", 100.0, "opt")]) == []
    )


def test_update_references_rewrites_the_tables_and_repins_them(tmp_path: Path) -> None:
    # The smoke table is derived from the roster, so the fixture roster has to
    # carry the smoke names for the rewrite to be the real one.
    before = [RosterEntry(name, 100.0, "opt") for name in SMOKE_INSTANCES]
    here = _reference_dir(tmp_path, before)
    moved = [RosterEntry(name, 100.0, "opt") for name in SMOKE_INSTANCES[1:]]
    moved.insert(0, RosterEntry(SMOKE_INSTANCES[0], 42.0, "best"))

    update_references(here, b"=best= inst0 42\n", moved)

    assert read_roster_csv(here / ROSTER_FILENAME) == moved
    assert verify_references(here) == []
    # The kind survives the rewrite: a gap against a best-known value does not
    # mean what a gap against a proven optimum means.
    assert read_roster_csv(here / ROSTER_FILENAME)[0].reference_kind == "best"
    assert read_roster_csv(here / SMOKE_FILENAME)[0].reference_kind == "best"


def test_roster_csv_round_trips_the_reference_kind(tmp_path: Path) -> None:
    roster = [RosterEntry("a", 1.5, "opt"), RosterEntry("b", -2.0, "best")]
    write_roster_csv(roster, tmp_path / ROSTER_FILENAME)

    assert read_roster_csv(tmp_path / ROSTER_FILENAME) == roster


# --- Overwriting a pinned instance needs a flag -------------------------------


def test_write_manifest_refuses_to_repin_moved_bytes_and_writes_nothing(tmp_path: Path) -> None:
    manifest = _pin_instances(tmp_path, {"a": b"original"})
    before = manifest.read_text()
    (tmp_path / "a.mps.gz").write_bytes(b"substituted")

    changed = write_manifest(["a"], tmp_path, manifest, update=False)

    assert changed == ["a"]
    assert manifest.read_text() == before, "the pin must survive a refused update"


def test_write_manifest_repins_moved_bytes_when_told_to(tmp_path: Path) -> None:
    manifest = _pin_instances(tmp_path, {"a": b"original"})
    (tmp_path / "a.mps.gz").write_bytes(b"substituted")

    changed = write_manifest(["a"], tmp_path, manifest, update=True)

    assert changed == ["a"]
    assert read_pin_table(manifest, "instance")["a"] == (sha256_of(b"substituted"), 11)


def test_write_manifest_adds_an_instance_it_has_never_pinned(tmp_path: Path) -> None:
    # An acquisition is not an overwrite, so it needs no flag.
    manifest = _pin_instances(tmp_path, {"a": b"x"})
    (tmp_path / "b.mps.gz").write_bytes(b"yy")

    assert write_manifest(["a", "b"], tmp_path, manifest, update=False) == []
    assert read_pin_table(manifest, "instance")["b"] == (sha256_of(b"yy"), 2)


def test_an_absent_manifest_does_not_silently_re_pin_what_is_on_disk(tmp_path: Path) -> None:
    """The add path is flagless; a missing pin table must not borrow that.

    Delete `manifest.csv`, substitute an instance, and the unflagged add path
    pinned the substituted bytes and exited 0 -- after which every later
    `--verify` passed, because the manifest agreed with the file.
    """
    (tmp_path / "a.mps.gz").write_bytes(b"SUBSTITUTED")

    assert pin_fetched_instances(["a"], tmp_path, update=False) == 3
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_a_manifest_missing_one_row_does_not_silently_re_pin_it(tmp_path: Path) -> None:
    """The same hole through a short table rather than an absent one."""
    _pin_instances(tmp_path, {"a": b"x"})
    (tmp_path / "b.mps.gz").write_bytes(b"SUBSTITUTED")

    assert pin_fetched_instances(["a", "b"], tmp_path, update=False) == 3
    assert "b" not in read_pin_table(tmp_path / MANIFEST_FILENAME, "instance")


def test_the_flag_accepts_a_file_the_manifest_does_not_cover(tmp_path: Path) -> None:
    _pin_instances(tmp_path, {"a": b"x"})
    (tmp_path / "b.mps.gz").write_bytes(b"yy")

    assert pin_fetched_instances(["a", "b"], tmp_path, update=True) == 0
    assert read_pin_table(tmp_path / MANIFEST_FILENAME, "instance")["b"] == (sha256_of(b"yy"), 2)


def test_write_manifest_keeps_rows_outside_the_fetched_subset(tmp_path: Path) -> None:
    # The #103 failure mode: a subset run must not drop the other rows.
    manifest = _pin_instances(tmp_path, {"a": b"x", "b": b"yy"})
    (tmp_path / "b.mps.gz").unlink()

    write_manifest(["a"], tmp_path, manifest, update=False)

    assert set(read_pin_table(manifest, "instance")) == {"a", "b"}


def test_a_tampered_local_solution_file_is_reported_even_when_upstream_matches(
    tmp_path: Path,
) -> None:
    # The duplicate-suppression that keeps the solution file from being reported
    # twice must not swallow the case where upstream is fine and the committed copy
    # is not -- the script would otherwise print "match their pins" about it.
    here = _reference_dir(tmp_path, [RosterEntry("inst0", 100.0, "opt")])
    (here / SOLU_FILENAME).write_bytes(b"=opt= inst0 100\n# tampered\n")

    problems = check_reference_pins(
        here, b"=opt= inst0 100\n", [RosterEntry("inst0", 100.0, "opt")]
    )

    assert [p for p in problems if p.startswith(f"{SOLU_FILENAME}:")]


def test_a_refused_reference_update_leaves_the_committed_files_alone(tmp_path: Path) -> None:
    # write_smoke_csv is the only step that can refuse, so it runs first: a MIPLIB
    # revision that drops a smoke instance must not leave the solution file and the
    # roster replaced under the old pins.
    here = _reference_dir(tmp_path, [RosterEntry(name, 100.0, "opt") for name in SMOKE_INSTANCES])
    before = (here / SOLU_FILENAME).read_bytes(), (here / ROSTER_FILENAME).read_bytes()

    with pytest.raises(RuntimeError, match="Smoke instances not in the roster"):
        update_references(here, b"=opt= other 1\n", [RosterEntry("other", 1.0, "opt")])

    assert ((here / SOLU_FILENAME).read_bytes(), (here / ROSTER_FILENAME).read_bytes()) == before
    assert verify_references(here) == []
