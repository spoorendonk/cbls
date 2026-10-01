"""The MINLPLib held-out roster (#144) and the published roster it must not disturb.

Both rosters are functions of committed inputs: the catalogue snapshot
`heldout/pool.csv`, the published walk's binary-NL skips in
`heldout/unfetchable.csv`, and -- for the held-out set -- `HELDOUT_SEED`. These
tests re-derive both offline and compare them byte for byte against the
committed `bounds.csv` files, so a change to the selection code that would
silently redefine either roster fails here rather than in a published table.
HELDOUT.md beside the rosters carries the method and the composition tables
these counts pin.
"""

from __future__ import annotations

import shutil
import urllib.error
from pathlib import Path

import pytest

from benchmarks.instances.minlplib import download
from benchmarks.instances.minlplib.download import (
    DEFAULT_ROSTER,
    HELDOUT_DIRNAME,
    HELDOUT_SEED,
    POOL_FILENAME,
    UNFETCHABLE_FILENAME,
    Instance,
    bounds_text,
    build_heldout,
    class_counts,
    parse_csv,
    read_bounds_names,
    read_unfetchable,
    roster_walk_skips,
    select,
    select_heldout,
    survivors,
    walk,
)

INST_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "minlplib"
HELDOUT_DIR = INST_DIR / HELDOUT_DIRNAME

#: Recorded in HELDOUT.md. A catalogue refresh that changes these is a new pool,
#: and the held-out membership drawn from the old one stays as committed.
POOL_SIZE = 397
POOL_CLASSES = {
    "bilinear": 5,
    "mixed-integer": 53,
    "other": 263,
    "polynomial": 10,
    "transcendental": 66,
}
PUBLISHED_CLASSES = {
    "bilinear": 5,
    "mixed-integer": 15,
    "other": 14,
    "polynomial": 10,
    "transcendental": 6,
}
HELDOUT_CLASSES = {"mixed-integer": 17, "other": 17, "transcendental": 16}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return parse_csv((HELDOUT_DIR / POOL_FILENAME).read_bytes())


@pytest.fixture(scope="module")
def skipped() -> dict[str, set[str]]:
    return read_unfetchable(HELDOUT_DIR / UNFETCHABLE_FILENAME)


def published_walk(rows: list[dict[str, str]], skipped: dict[str, set[str]]) -> list[Instance]:
    return walk(select(rows), DEFAULT_ROSTER, skipped["published"])


def heldout_walk(rows: list[dict[str, str]], skipped: dict[str, set[str]]) -> list[Instance]:
    exclude = set(read_bounds_names(INST_DIR / "bounds.csv")) | skipped["published"]
    order = select_heldout(rows, exclude, HELDOUT_SEED)
    return walk(order, DEFAULT_ROSTER, skipped.get("heldout", set()))


def test_pool_snapshot_is_exactly_the_filter_survivors(rows: list[dict[str, str]]) -> None:
    # Every snapshot row passes the filter: the snapshot is the pool, not a
    # sample of the catalogue that happens to contain it.
    pool = survivors(rows)
    assert len(rows) == len(pool) == POOL_SIZE
    assert class_counts(pool) == POOL_CLASSES


def test_published_roster_is_reproduced_byte_for_byte(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    roster = published_walk(rows, skipped)
    assert bounds_text(roster) == (INST_DIR / "bounds.csv").read_bytes().decode()
    assert class_counts(roster) == PUBLISHED_CLASSES


def test_published_skips_are_exactly_the_walks_gaps(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    # No stale or missing entry: the recorded skips are precisely the candidates
    # the published walk passed over, so `walk` is replaying the real fetch.
    roster = read_bounds_names(INST_DIR / "bounds.csv")
    assert set(roster_walk_skips(select(rows), roster)) == skipped["published"]


def test_heldout_roster_is_reproduced_byte_for_byte(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    heldout = heldout_walk(rows, skipped)
    assert bounds_text(heldout) == (HELDOUT_DIR / "bounds.csv").read_bytes().decode()
    assert class_counts(heldout) == HELDOUT_CLASSES


def test_heldout_is_disjoint_from_the_published_roster() -> None:
    published = set(read_bounds_names(INST_DIR / "bounds.csv"))
    heldout = read_bounds_names(HELDOUT_DIR / "bounds.csv")
    assert len(heldout) == len(set(heldout)) == DEFAULT_ROSTER
    assert not published & set(heldout)


def test_heldout_draw_depends_on_the_seed(rows: list[dict[str, str]]) -> None:
    # Guards the seed actually reaching the order: a key that ignored it would
    # pass every byte-for-byte check above while making the seed decorative.
    exclude = set(read_bounds_names(INST_DIR / "bounds.csv"))
    first = [i.name for i in select_heldout(rows, exclude, HELDOUT_SEED)[:DEFAULT_ROSTER]]
    other = [i.name for i in select_heldout(rows, exclude, HELDOUT_SEED + 1)[:DEFAULT_ROSTER]]
    assert first != other


def test_heldout_is_not_the_size_ordered_next_fifty(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    # The rejected construction (HELDOUT.md): the published order's next fifty.
    unavailable = skipped["published"] | skipped.get("heldout", set())
    next_fifty = walk(select(rows), 2 * DEFAULT_ROSTER, unavailable)[DEFAULT_ROSTER:]
    heldout = heldout_walk(rows, skipped)
    assert {i.name for i in heldout} != {i.name for i in next_fifty}


def test_every_heldout_instance_has_a_text_nl_file() -> None:
    for name in read_bounds_names(HELDOUT_DIR / "bounds.csv"):
        path = HELDOUT_DIR / f"{name}.nl"
        assert path.is_file(), name
        assert path.read_bytes()[:1] == b"g", f"{name} is not a text NL file"
    on_disk = {p.stem for p in HELDOUT_DIR.glob("*.nl")}
    assert on_disk == set(read_bounds_names(HELDOUT_DIR / "bounds.csv"))


def test_roster_walk_skips_refuses_a_roster_the_catalogue_cannot_rebuild(
    rows: list[dict[str, str]],
) -> None:
    candidates = select(rows)
    roster = read_bounds_names(INST_DIR / "bounds.csv")
    with pytest.raises(ValueError, match="no longer pass the filter"):
        roster_walk_skips(candidates, [*roster, "not-an-instance"])
    with pytest.raises(ValueError, match="order"):
        roster_walk_skips(candidates, [roster[1], roster[0], *roster[2:]])


def _fake_minlplib(
    monkeypatch: pytest.MonkeyPatch, binary: set[str], down: set[str] | None = None
) -> list[str]:
    """Serve text NL for every instance except `binary` (binary NL) and `down` (outage)."""
    fetched: list[str] = []

    def fake_fetch(url: str, timeout: int = 120) -> bytes:
        name = url.rsplit("/", 1)[-1].removesuffix(".nl")
        fetched.append(name)
        if down and name in down:
            raise urllib.error.URLError("simulated outage")
        return b"b3 0 1 0\n" if name in binary else f"g3 1 1 0 # {name}\n".encode()

    monkeypatch.setattr(download, "fetch_bytes", fake_fetch)
    return fetched


def test_build_heldout_writes_the_committed_roster_and_then_refuses(
    rows: list[dict[str, str]],
    skipped: dict[str, set[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The live walk, against a fake server that serves binary NL exactly where the
    # committed record says, must write the committed files: this is what ties the
    # offline replay (`walk`) above to the code that actually fetched.
    shutil.copy(INST_DIR / "bounds.csv", tmp_path / "bounds.csv")
    _fake_minlplib(monkeypatch, set().union(*skipped.values()))
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 0
    for name in ("bounds.csv", POOL_FILENAME, UNFETCHABLE_FILENAME):
        assert (tmp_path / HELDOUT_DIRNAME / name).read_bytes() == (
            HELDOUT_DIR / name
        ).read_bytes(), name
    # A committed membership is not replaced without --force.
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 2


def test_build_heldout_force_keeps_existing_nl_files(
    rows: list[dict[str, str]],
    skipped: dict[str, set[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copy(INST_DIR / "bounds.csv", tmp_path / "bounds.csv")
    _fake_minlplib(monkeypatch, set().union(*skipped.values()))
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 0
    fetched = _fake_minlplib(monkeypatch, set().union(*skipped.values()))
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=True) == 0
    # Only the published walk's skips are re-checked; no held-out .nl is re-fetched.
    assert set(fetched) == skipped["published"]


def test_build_heldout_aborts_on_a_network_failure_without_writing(
    rows: list[dict[str, str]],
    skipped: dict[str, set[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An outage must not be recorded as "unfetchable": that would change the
    # membership. The first held-out candidate is down.
    shutil.copy(INST_DIR / "bounds.csv", tmp_path / "bounds.csv")
    first = read_bounds_names(HELDOUT_DIR / "bounds.csv")[0]
    _fake_minlplib(monkeypatch, set().union(*skipped.values()), down={first})
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 1
    assert not (tmp_path / HELDOUT_DIRNAME / "bounds.csv").exists()
    assert not (tmp_path / HELDOUT_DIRNAME / UNFETCHABLE_FILENAME).exists()
