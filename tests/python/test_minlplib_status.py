"""The MINLPLib full-library selection and per-instance status (#283)."""

from __future__ import annotations

import csv
import subprocess
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
    """A checker that answers from each file's first word.

    "crash" exits 3, "hang" sleeps past any test timeout, "latin" answers with
    a byte that is not UTF-8 and "ctrl" with a \\x1c (a reader message quoting
    raw file bytes can carry either), "sos" with a reader coverage refusal.
    """
    script = tmp_path / "checker"
    script.write_text(
        "#!/bin/sh\n"
        'for f in "$@"; do\n'
        '  w=$(head -n1 "$f")\n'
        '  case "$w" in\n'
        "    crash) exit 3;;\n"
        "    hang) exec sleep 30;;\n"
        '    latin) printf "%s\\treader-error\\tNL: bad tag \\303\\n" "$f";;\n'
        '    ctrl) printf "%s\\treader-error\\tNL: a\\034b\\n" "$f";;\n'
        '    sos) printf "%s\\treader-unsupported\\tNL: suffix\\n" "$f";;\n'
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
        _row("sos"),
        _row("ok2", opvcpower="True"),
    ]
    for name, word in [
        ("erf", "ok"),
        ("noprimal", "ok"),
        ("ok1", "ok"),
        ("bad", "bad"),
        ("atan", "atan"),
        ("crash", "crash"),
        ("sos", "sos"),
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
        # Each file has its own checker process, so only the crashing file
        # is charged, and not as the reader's.
        ("crash", "checker-failed", "exit code 3"),
        ("sos", "reader-unsupported", "NL: suffix"),
        ("ok2", "modelled", ""),
    ]
    assert status.counts(got) == {
        "unsupported-op": 1,
        "missing-nl": 1,
        "no-primal-bound": 1,
        "reader-error": 1,
        "reader-unsupported": 1,
        "adapter-unsupported": 1,
        "checker-failed": 1,
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


def test_an_undecodable_checker_byte_does_not_stop_the_run(tmp_path: Path) -> None:
    (tmp_path / "x.nl").write_text("latin\n")
    got = status.check_files(_fake_checker(tmp_path), [tmp_path / "x.nl"])
    assert got[str(tmp_path / "x.nl")] == ("reader-error", "NL: bad tag \ufffd")


def test_a_control_byte_in_a_message_does_not_stop_the_run(tmp_path: Path) -> None:
    # str.splitlines() would split at \x1c and leave a line that parses as
    # nothing; only newlines delimit the checker's lines.
    (tmp_path / "x.nl").write_text("ctrl\n")
    got = status.check_files(_fake_checker(tmp_path), [tmp_path / "x.nl"])
    assert got[str(tmp_path / "x.nl")] == ("reader-error", "NL: a\x1cb")


def test_output_that_is_not_a_checker_line_is_a_checker_failure(tmp_path: Path) -> None:
    (tmp_path / "x.nl").write_text("ok\n")
    noisy = tmp_path / "noisy"
    noisy.write_text("#!/bin/sh\necho hello\n")
    noisy.chmod(0o755)
    got = status.check_files(noisy, [tmp_path / "x.nl"])
    assert got[str(tmp_path / "x.nl")][0] == "checker-failed"


def test_a_hung_file_is_charged_alone(tmp_path: Path) -> None:
    (tmp_path / "hang.nl").write_text("hang\n")
    (tmp_path / "ok.nl").write_text("ok\n")
    files = [tmp_path / "ok.nl", tmp_path / "hang.nl"]
    got = status.check_files(_fake_checker(tmp_path), files, timeout=0.5)
    assert got[str(files[0])] == ("modelled", "")
    assert got[str(files[1])][0] == "checker-failed"
    assert "timed out" in got[str(files[1])][1]


def test_a_checker_that_fails_on_everything_fails_the_run(tmp_path: Path) -> None:
    nl = tmp_path / "nl"
    nl.mkdir()
    rows = [_row("a"), _row("b")]
    for r in rows:
        (nl / f"{r['name']}.nl").write_text("ok\n")
    cat = tmp_path / "instancedata.csv"
    with open(cat, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    argv = ["--catalogue", str(cat), "--nl-dir", str(nl), "--out", str(tmp_path / "s.csv")]
    bounds = tmp_path / "bounds.csv"
    bounds.write_text("instance\nkept\n")
    argv += ["--bounds", str(bounds)]
    assert status.main([*argv, "--checker", str(broken)]) == 1
    # Nothing is written: an existing bounds.csv is not emptied.
    assert bounds.read_text() == "instance\nkept\n"
    assert not (tmp_path / "s.csv").exists()


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
    # filter.nl with an opcode no NL writer emits: a coverage refusal, as the
    # runner buckets it, not a malformed file.
    unknown = tmp_path / "unknown.nl"
    unknown.write_text((data / "filter.nl").read_text().replace("o42\n", "o999\n", 1))
    got = status.check_files(CHECKER, [*files, broken, unknown])
    assert [got[str(f)] for f in files] == [("modelled", "")] * 3
    assert got[str(broken)][0] == "reader-error"
    assert got[str(unknown)][0] == "reader-unsupported"


#: Two columns in [0,10] and the doubleton equality `x0 + x1 = 5`, minimise
#: `x0 + 2 x1`: presolve aggregates one column into the other (#328).
AGGREGABLE_NL = (
    "g3 0 1 0\n 2 1 1 0 0\n 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 0 0 0 0 0\n 0 0\n 0 0\n 0 0 0 0 0\n"
    "b\n0 0 10\n0 0 10\nr\n4 5\nO0 0\nn0\nG0 2\n0 1\n1 2\nC0\nn0\nJ0 2\n0 1\n1 1\n"
)


@pytest.mark.parametrize(
    ("flags", "aggregated"),
    [(["--presolve"], 1), (["--presolve", "--no-presolve-aggregation"], 0)],
    ids=["aggregation-on", "aggregation-off"],
)
def test_the_presolve_census_reports_aggregation(
    tmp_path: Path, flags: list[str], aggregated: int
) -> None:
    """`nl_check --presolve` reports `aggregated=`, and has the off switch (#347)."""
    if not CHECKER.is_file():
        pytest.skip("cbls_minlplib_nl_check not built")
    nl = tmp_path / "agg.nl"
    nl.write_text(AGGREGABLE_NL)
    out = subprocess.run(
        [str(CHECKER), *flags, str(nl)], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    fields = out.rstrip("\n").split("\t")
    assert fields[1] == "modelled", out
    assert f" aggregated={aggregated} " in fields[2], out
