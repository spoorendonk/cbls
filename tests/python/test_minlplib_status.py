"""The MINLPLib full-library selection and per-instance status (#283)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.instances.minlplib import download, status

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "build" / "cbls_minlplib_nl_check"


def _row(name: str, **cols: str) -> dict[str, str]:
    row = {
        "name": name,
        "probtype": "MBNLP",
        "convex": "True",
        "nvars": "500",
        "ncons": "400",
        "nbinvars": "3",
        "nintvars": "1",
        "objsense": "min",
        "primalbound": "1.5",
        "dualbound": "1.0",
    }
    row.update(cols)
    return row


def test_full_selection_takes_any_type_convex_and_size() -> None:
    # A convex MBNLP far over the roster's size budget, which the pinned
    # roster filter refuses, is in the full-library selection.
    row = _row("big")
    assert download.row_to_instance(row) is None
    inst = download.full_instance(row)
    assert inst is not None
    assert (inst.name, inst.nvars, inst.ncons, inst.primalbound) == ("big", 500, 400, 1.5)
    assert inst.ndiscvars == 4


@pytest.mark.parametrize("col", ["opvcpower", "opcvpower"])
def test_full_selection_takes_both_powers(col: str) -> None:
    row = _row("p", convex="False", nvars="5", ncons="5", probtype="NLP", **{col: "True"})
    assert download.row_to_instance(row) is None  # the pinned roster filter is unchanged
    assert download.full_instance(row) is not None


@pytest.mark.parametrize("col", sorted(download.FULL_EXCLUDED_OP_COLUMNS))
def test_full_selection_excludes_the_unmodelled_operators(col: str) -> None:
    row = _row("e", **{col: "True"})
    assert download.full_instance(row) is None
    assert download.full_excluded_ops(row) == [col]


@pytest.mark.parametrize("primal", ["", "inf", "-inf", "nan", "x"])
def test_full_selection_needs_a_finite_primal_bound(primal: str) -> None:
    assert download.full_instance(_row("n", primalbound=primal)) is None


def _fake_checker(tmp_path: Path) -> Path:
    """A checker that answers from each file's first word; "crash" kills it."""
    script = tmp_path / "checker"
    script.write_text(
        "#!/bin/sh\n"
        'for f in "$@"; do\n'
        '  w=$(head -n1 "$f")\n'
        '  case "$w" in\n'
        "    crash) exit 3;;\n"
        '    ok) printf "%s\\tmodelled\\n" "$f";;\n'
        '    bad) printf "%s\\treader-error\\tNL: broken\\n" "$f";;\n'
        '    *) printf "%s\\tadapter-unsupported\\topcode 49\\n" "$f";;\n'
        "  esac\n"
        "done\n"
    )
    script.chmod(0o755)
    return script


def _catalogue(tmp_path: Path) -> tuple[list[dict[str, str]], Path]:
    nl = tmp_path / "nl"
    nl.mkdir()
    rows = [
        _row("erf", operrorf="True"),
        _row("absent"),
        _row("noprimal", primalbound="inf"),
        _row("ok1"),
        _row("bad"),
        _row("atan"),
        _row("crash"),
        _row("ok2", opvcpower="True"),
    ]
    for name, word in [
        ("erf", "ok"),
        ("noprimal", "ok"),
        ("ok1", "ok"),
        ("bad", "bad"),
        ("atan", "atan"),
        ("crash", "crash"),
        ("ok2", "ok"),
    ]:
        (nl / f"{name}.nl").write_text(word + "\n")
    return rows, nl


def test_every_instance_gets_the_first_status_that_applies(tmp_path: Path) -> None:
    rows, nl = _catalogue(tmp_path)
    got = status.statuses(rows, nl, _fake_checker(tmp_path))
    assert [(s.name, s.status, s.detail) for s in got] == [
        ("erf", "unsupported-op", "operrorf"),
        ("absent", "missing-nl", ""),
        ("noprimal", "no-primal-bound", "inf"),
        ("ok1", "modelled", ""),
        ("bad", "reader-error", "NL: broken"),
        ("atan", "adapter-unsupported", "opcode 49"),
        # The crash takes its batch down; the batch is re-run a file at a
        # time, so only the crashing file is charged.
        ("crash", "reader-error", "checker crashed"),
        ("ok2", "modelled", ""),
    ]
    assert status.counts(got) == {
        "unsupported-op": 1,
        "missing-nl": 1,
        "no-primal-bound": 1,
        "reader-error": 2,
        "adapter-unsupported": 1,
        "modelled": 2,
    }


def test_the_cli_writes_the_status_table_and_the_modelled_bounds(tmp_path: Path) -> None:
    rows, nl = _catalogue(tmp_path)
    cat = tmp_path / "instancedata.csv"
    with open(cat, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "status.csv"
    bounds = tmp_path / "bounds.csv"
    argv = ["--catalogue", str(cat), "--nl-dir", str(nl), "--out", str(out)]
    argv += ["--checker", str(_fake_checker(tmp_path)), "--bounds", str(bounds)]
    assert status.main(argv) == 0
    with open(out, newline="") as fh:
        table = list(csv.DictReader(fh))
    assert [r["instance"] for r in table] == [r["name"] for r in rows]
    assert download.read_bounds_names(bounds) == ["ok1", "ok2"]


def test_a_missing_checker_is_refused(tmp_path: Path) -> None:
    rows, nl = _catalogue(tmp_path)
    argv = ["--catalogue", str(tmp_path / "x.csv"), "--nl-dir", str(nl)]
    argv += ["--out", str(tmp_path / "s.csv"), "--checker", str(tmp_path / "none")]
    assert status.main(argv) == 2


def test_the_real_checker_models_the_vendored_files(tmp_path: Path) -> None:
    if not CHECKER.is_file():
        pytest.skip("cbls_minlplib_nl_check not built")
    data = REPO / "tests" / "data"
    files = [data / "filter.nl", data / "ex8_4_4.nl", data / "kriging_peaks-red010.nl"]
    broken = tmp_path / "broken.nl"
    broken.write_text("g3 0 1 0\n")
    got = status.check_files(CHECKER, [*files, broken])
    assert [got[str(f)] for f in files] == [("modelled", "")] * 3
    assert got[str(broken)][0] == "reader-error"
