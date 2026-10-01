"""The MINLPLib held-out roster (#144) and the published roster it must not disturb.

Both rosters are functions of committed inputs: the catalogue snapshot
`heldout/pool.csv`, the published walk's binary-NL skips in
`heldout/unfetchable.csv`, and -- for the held-out set -- `HELDOUT_SEED`. These
tests re-derive both offline and compare them byte for byte against the
committed `bounds.csv` files, so a change to the selection code that would
silently redefine either roster fails here rather than in a published table.
HELDOUT.md beside the rosters carries the method and the composition tables
these counts pin.

The byte-for-byte checks are circular by construction: the committed files were
written by the code they are compared against. They pin the rosters against
later *changes* to that code; they say nothing about whether the method is the
right one. That is argued in HELDOUT.md, and the class quotas and seed
dependence are checked below on their own terms.
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
    heldout_quotas,
    parse_csv,
    quota_walk,
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
#: Largest-remainder share of 50 by the published mix in the three shared
#: classes, 15 / 14 / 6: exactly 21.43 / 20 / 8.57, so 21 / 20 / 9.
HELDOUT_CLASSES = {"mixed-integer": 21, "other": 20, "transcendental": 9}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return parse_csv((HELDOUT_DIR / POOL_FILENAME).read_bytes())


@pytest.fixture(scope="module")
def skipped() -> dict[str, set[str]]:
    return read_unfetchable(HELDOUT_DIR / UNFETCHABLE_FILENAME)


def published_walk(rows: list[dict[str, str]], skipped: dict[str, set[str]]) -> list[Instance]:
    return walk(select(rows), DEFAULT_ROSTER, skipped["published"])


def heldout_walk(
    rows: list[dict[str, str]], skipped: dict[str, set[str]], seed: int = HELDOUT_SEED
) -> list[Instance]:
    """`build_heldout`'s draw, replayed offline with the real exclusion."""
    roster = set(read_bounds_names(INST_DIR / "bounds.csv"))
    by_class = select_heldout(rows, roster | skipped["published"], seed)
    published = [i for i in survivors(rows) if i.name in roster]
    quotas = heldout_quotas(published, by_class, DEFAULT_ROSTER)
    unavailable = skipped.get("heldout", set())
    return quota_walk(by_class, quotas, lambda i: i.name not in unavailable)


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


def test_heldout_draw_depends_on_the_seed(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    # Guards the seed actually reaching the order: a key that ignored it would
    # pass every byte-for-byte check above while making the seed decorative.
    first = {i.name for i in heldout_walk(rows, skipped)}
    other = {i.name for i in heldout_walk(rows, skipped, HELDOUT_SEED + 1)}
    assert first != other


def test_heldout_quotas_follow_the_published_mix_in_shared_classes(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    # Independent of the committed files: the quota rule itself, on the pool.
    roster = set(read_bounds_names(INST_DIR / "bounds.csv"))
    by_class = select_heldout(rows, roster | skipped["published"], HELDOUT_SEED)
    published = [i for i in survivors(rows) if i.name in roster]
    assert set(by_class) == {"mixed-integer", "other", "transcendental"}
    assert heldout_quotas(published, by_class, DEFAULT_ROSTER) == HELDOUT_CLASSES


def test_quota_walk_passes_over_rejects_without_losing_a_seat() -> None:
    def inst(name: str, cls: str) -> Instance:
        return Instance(name, 1, 1, 0.0, None, "min", cls, 0)

    by_class = {
        "a": [inst("a1", "a"), inst("a2", "a"), inst("a3", "a")],
        "b": [inst("b1", "b"), inst("b2", "b")],
    }
    chosen = quota_walk(by_class, {"a": 2, "b": 1}, lambda i: i.name != "a1")
    assert [i.name for i in chosen] == ["a2", "b1", "a3"]
    # A class that runs dry returns short rather than borrowing another's seats.
    short = quota_walk(by_class, {"a": 1, "b": 3}, lambda i: True)
    assert [i.name for i in short] == ["a1", "b1", "b2"]


def test_heldout_is_not_the_size_ordered_next_fifty(
    rows: list[dict[str, str]], skipped: dict[str, set[str]]
) -> None:
    # The rejected construction (HELDOUT.md): the published order's next fifty.
    unavailable = skipped["published"] | skipped.get("heldout", set())
    next_fifty = walk(select(rows), 2 * DEFAULT_ROSTER, unavailable)[DEFAULT_ROSTER:]
    heldout = heldout_walk(rows, skipped)
    assert {i.name for i in heldout} != {i.name for i in next_fifty}

    # And it has the property the doc chose it for: a size spread the narrow
    # next-fifty band lacks (216 against 92 on the committed pool).
    def spread(xs: list[Instance]) -> int:
        sizes = [i.nvars + i.ncons for i in xs]
        return max(sizes) - min(sizes)

    assert spread(heldout) > 2 * spread(next_fifty)


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
    monkeypatch: pytest.MonkeyPatch,
    binary: set[str],
    down: set[str] | None = None,
    html: set[str] | None = None,
) -> list[str]:
    """Text NL for every instance except `binary` (binary NL), `down` (outage), `html`."""
    fetched: list[str] = []

    def fake_fetch(url: str, timeout: int = 120) -> bytes:
        name = url.rsplit("/", 1)[-1].removesuffix(".nl")
        fetched.append(name)
        if down and name in down:
            raise urllib.error.URLError("simulated outage")
        if html and name in html:
            return b"<!doctype html><title>Service Unavailable</title>"
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


def test_build_heldout_treats_an_html_error_page_as_an_outage(
    rows: list[dict[str, str]],
    skipped: dict[str, set[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A soft-error page served with 200 must abort, not become a recorded skip.
    shutil.copy(INST_DIR / "bounds.csv", tmp_path / "bounds.csv")
    first = read_bounds_names(HELDOUT_DIR / "bounds.csv")[0]
    _fake_minlplib(monkeypatch, set().union(*skipped.values()), html={first})
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 1
    assert not (tmp_path / HELDOUT_DIRNAME / UNFETCHABLE_FILENAME).exists()


def test_build_heldout_refuses_a_short_published_roster(
    rows: list[dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `--limit 10` main-mode run leaves a 10-row bounds.csv; a draw against it
    # would exclude only those 10 and could admit the other 40 published instances.
    lines = (INST_DIR / "bounds.csv").read_bytes().splitlines(keepends=True)
    (tmp_path / "bounds.csv").write_bytes(b"".join(lines[:11]))
    fetched = _fake_minlplib(monkeypatch, set())
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=True) == 2
    assert not fetched


def test_build_heldout_refuses_when_a_published_skip_is_now_text_nl(
    rows: list[dict[str, str]],
    skipped: dict[str, set[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If an instance the published walk skipped as binary NL is served as text
    # today, a rebuild of the published roster would admit it: refuse, write nothing.
    shutil.copy(INST_DIR / "bounds.csv", tmp_path / "bounds.csv")
    now_text = sorted(skipped["published"])[0]
    fetched = _fake_minlplib(monkeypatch, skipped["published"] - {now_text})
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 1
    assert now_text in fetched
    assert not (tmp_path / HELDOUT_DIRNAME).exists()


def test_build_heldout_refuses_on_a_network_failure_in_the_published_recheck(
    rows: list[dict[str, str]],
    skipped: dict[str, set[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copy(INST_DIR / "bounds.csv", tmp_path / "bounds.csv")
    down = sorted(skipped["published"])[0]
    _fake_minlplib(monkeypatch, skipped["published"], down={down})
    assert build_heldout(tmp_path, rows, DEFAULT_ROSTER, force=False) == 1
    assert not (tmp_path / HELDOUT_DIRNAME).exists()
