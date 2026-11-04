"""`benchmarks/uc-chped/score.py`, the UC-CHPED campaign's table (#363).

The scorer sits in a directory no package can name, so it is run as the command
the campaign runs. The rows are generated in the runner's layout, and its columns
and roster are pinned against `uc_chped.cpp` itself. Gaps are hand-computed.
`fixtures/uc_chped_campaign_smoke` holds real runner output.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCORE = REPO / "benchmarks" / "uc-chped" / "score.py"
RUNNER_SOURCE = REPO / "benchmarks" / "uc-chped" / "uc_chped.cpp"
SMOKE = Path(__file__).parent / "fixtures" / "uc_chped_campaign_smoke"

COLUMNS = (
    "instance,periods,method,objective,lb,gap_pct,time_s,source,note,"
    "ub,time_limit_s,seed,feasible,verified,max_violation,feas_tol,commit_sha,search_config"
)
ROSTER = ["--roster", "ucp13:1,3", "ucp100-48p:48"]
CONFIG = "float_hook=on;lns=on;threads=4"


def cited(instance: str, periods: int, lb: float, ub: float) -> dict[str, object]:
    return {
        "instance": instance,
        "periods": periods,
        "method": "Pedroso MIP (1hr)",
        "objective": ub,
        "lb": lb,
        "gap_pct": f"{100 * (ub - lb) / lb:.2f}",
        "source": "Pedroso et al. 2014 Table 2",
        "note": "cited reference",
        "ub": ub,
    }


def measured(
    instance: str, periods: int, seed: int, objective: float | None, **cells: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "instance": instance,
        "periods": periods,
        "method": "CBLS ViolationLS",
        "objective": "" if objective is None else objective,
        "time_s": 3,
        "source": "this work",
        "time_limit_s": 3,
        "seed": seed,
        "feasible": "false" if objective is None else "true",
        "verified": "" if objective is None else "true",
        "feas_tol": "1e-06",
        "commit_sha": "abc1234",
        "search_config": CONFIG,
    }
    if objective is None:
        row["note"] = "infeasible at feas_tol"
    return {**row, **cells}


def seed_rows(seed: int, **cells: object) -> list[dict[str, object]]:
    """A whole table of the `ROSTER` campaign at one seed; `cells` land on every measured row."""
    return [
        cited("ucp13", 1, 100.0, 110.0),
        cited("ucp13", 3, 200.0, 200.0),
        measured("ucp13", 1, seed, 100.0 + 2.0 * seed, lb=100.0, ub=110.0, **cells),
        measured("ucp13", 3, seed, 200.0, lb=200.0, ub=200.0, **cells),
        measured("ucp100-48p", 48, seed, 5000.0 * seed, **cells),
    ]


def write_table(path: Path, rows: list[dict[str, object]], columns: str = COLUMNS) -> Path:
    out = io.StringIO()
    out.write("# UC-CHPED comparison table.\n# Run: commit abc1234.\n")
    writer = csv.DictWriter(out, fieldnames=columns.split(","), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.getvalue())
    return path


def run_score(*argv: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCORE), *map(str, argv)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def two_seeds(tmp_path: Path) -> list[Path]:
    return [write_table(tmp_path / f"seed{s}" / "c.csv", seed_rows(s)) for s in (1, 2)]


def refusal(tables: list[Path], *argv: str) -> str:
    result = run_score(*tables, *(argv or ROSTER))
    assert result.returncode == 2, result.stdout
    assert result.stdout == ""
    return result.stderr


def test_columns_are_the_runners() -> None:
    source = RUNNER_SOURCE.read_text()
    found = re.search(r"kCsvHeader =\s*((?:\"[^\"]*\"\s*)+);", source)
    assert found is not None
    assert "".join(re.findall(r"\"([^\"]*)\"", found.group(1))) == COLUMNS


def test_published_roster_is_the_runners(tmp_path: Path) -> None:
    source = RUNNER_SOURCE.read_text()
    pairs = re.findall(r"\{\"(ucp[^\"]*)\.jsonl\", \{([0-9, ]+)\}\}", source)
    assert len(pairs) == 8
    rows = [
        measured(name, int(t), 1, 100.0)
        for name, horizons in pairs
        for t in horizons.replace(" ", "").split(",")
    ]
    # Every row the runner's roster holds, and no --roster: only the published one passes.
    result = run_score(write_table(tmp_path / "c.csv", rows))
    assert result.returncode == 0, result.stderr
    assert "NOT THE PUBLISHED ROSTER" not in result.stdout


def test_the_real_smoke_scores_into_the_table() -> None:
    """The runner's own output, two seeds at `--threads 4`, in the full run's layout."""
    tables = [SMOKE / f"seed{s}" / "comparison.csv" for s in (1, 2)]
    for table in tables:
        lines = table.read_text().splitlines()
        assert next(ln for ln in lines if not ln.startswith("#")) == COLUMNS

    result = run_score(*tables, "--roster", "ucp13:1,3", "ucp100-48p:48")

    assert result.returncode == 0, result.stderr
    printed = result.stdout.splitlines()
    assert printed[0].startswith("NOT THE PUBLISHED ROSTER")
    for line in (
        "| ucp13 | 1 | 3 | 2/2 | 11701.3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 |",
        "| ucp13 | 3 | 3 | 2/2 | 39104.4 | 0.65 [0.51, 0.79] | 0.65 [0.51, 0.79] | 0.00 |",
        "| ucp100-48p | 48 | 3 | 2/2 | 11905181.3 | - | - | - |",
        "- engine commit: 9a62529 (commit_sha column)",
        "- seeds: 1, 2 (seed column)",
        "- cbls threads: 4 (search_config column)",
        "- no solution: 0",
    ):
        assert line in printed


def test_table_gives_the_median_and_spread_of_the_gaps_over_seeds(tmp_path: Path) -> None:
    result = run_score(*two_seeds(tmp_path), *ROSTER)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # ucp13 T=1: objectives 102 and 104 against lb 100, ub 110. Pedroso's gap is 10.
    assert "| ucp13 | 1 | 3 | 2/2 | 103.0 | 3.00 [2.00, 4.00] | -6.36 [-7.27, -5.45] | 10.00 |" in (
        lines
    )
    assert "| ucp13 | 3 | 3 | 2/2 | 200.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 |" in lines
    # No published bound: an objective and no gap.
    assert "| ucp100-48p | 48 | 3 | 2/2 | 7500.0 | - | - | - |" in lines
    assert "- engine commit: abc1234 (commit_sha column)" in lines
    assert "- seeds: 1, 2 (seed column)" in lines
    assert "- cbls threads: 4 (search_config column)" in lines
    assert "- no solution: 0" in lines


def test_a_subset_roster_says_so_first(tmp_path: Path) -> None:
    assert run_score(*two_seeds(tmp_path), *ROSTER).stdout.startswith("NOT THE PUBLISHED ROSTER")


def test_threads_read_as_one_when_the_cell_has_none(tmp_path: Path) -> None:
    table = write_table(tmp_path / "c.csv", seed_rows(1, search_config="lns=on"))
    lines = run_score(table, *ROSTER).stdout.splitlines()
    assert "- cbls threads: 1 (search_config column)" in lines


def test_out_writes_what_it_prints(tmp_path: Path) -> None:
    out = tmp_path / "table.md"
    result = run_score(*two_seeds(tmp_path), *ROSTER, "--out", out)
    assert out.read_text() == result.stdout


def test_an_unsolved_seed_is_counted_and_named_and_left_out_of_the_gap(tmp_path: Path) -> None:
    rows = seed_rows(2)
    rows[2] = measured("ucp13", 1, 2, None, lb=100.0, ub=110.0)
    tables = [write_table(tmp_path / "1.csv", seed_rows(1)), write_table(tmp_path / "2.csv", rows)]
    lines = run_score(*tables, *ROSTER).stdout.splitlines()
    assert "| ucp13 | 1 | 3 | 1/2 | 102.0 | 2.00 [2.00, 2.00] | -7.27 [-7.27, -7.27] | 10.00 |" in (
        lines
    )
    assert "- no solution: 1" in lines
    assert "  - ucp13 T=1 seed 2: infeasible at feas_tol" in lines


def test_no_seed_solving_a_pair_leaves_dashes(tmp_path: Path) -> None:
    rows = seed_rows(1)
    rows[2] = measured("ucp13", 1, 1, None, lb=100.0, ub=110.0)
    lines = run_score(write_table(tmp_path / "1.csv", rows), *ROSTER).stdout.splitlines()
    assert "| ucp13 | 1 | 3 | 0/1 | - | - | - | 10.00 |" in lines


def test_a_missing_pair_is_refused_by_name(tmp_path: Path) -> None:
    rows = [r for r in seed_rows(2) if r["instance"] != "ucp100-48p"]
    tables = [write_table(tmp_path / "1.csv", seed_rows(1)), write_table(tmp_path / "2.csv", rows)]
    assert "ucp100-48p T=48 seed 2" in refusal(tables)


def test_the_published_roster_is_required_without_the_flag(tmp_path: Path) -> None:
    table = write_table(tmp_path / "1.csv", seed_rows(1))
    assert "incomplete roster" in refusal([table], "--out=x")


def test_a_pair_outside_the_roster_is_refused(tmp_path: Path) -> None:
    rows = [*seed_rows(1), measured("ucp40", 1, 1, 9.0)]
    assert "outside the roster" in refusal([write_table(tmp_path / "1.csv", rows)])


def test_a_row_twice_is_refused(tmp_path: Path) -> None:
    rows = seed_rows(1)
    assert "twice" in refusal([write_table(tmp_path / "1.csv", [*rows, rows[-1]])])


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ({"verified": "false", "note": "verify-fail (2 errors)", "objective": ""}, "verification"),
        ({"verified": ""}, "never verified"),
        ({"note": "skipped: T > n_periods", "feasible": "", "seed": ""}, "measured nothing"),
    ],
)
def test_a_row_that_is_no_verified_measurement_is_refused(
    tmp_path: Path, cells: dict[str, object], expected: str
) -> None:
    rows = seed_rows(1)
    rows[-1].update(cells)
    assert expected in refusal([write_table(tmp_path / "1.csv", rows)])


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ({"commit_sha": "def5678"}, "mixed engine commits"),
        ({"search_config": "lns=on;threads=8"}, "mixed search configurations"),
        ({"feas_tol": "1e-09"}, "mixed feasibility tolerances"),
        ({"time_limit_s": 60}, "two budgets"),
    ],
)
def test_two_tables_that_differ_in_a_setting_are_refused(
    tmp_path: Path, cells: dict[str, object], expected: str
) -> None:
    tables = [
        write_table(tmp_path / "1.csv", seed_rows(1)),
        write_table(tmp_path / "2.csv", seed_rows(2, **cells)),
    ]
    assert expected in refusal(tables)


def test_two_horizons_may_have_two_budgets(tmp_path: Path) -> None:
    rows = seed_rows(1)
    rows[-1]["time_limit_s"] = 600
    result = run_score(write_table(tmp_path / "1.csv", rows), *ROSTER)
    assert result.returncode == 0, result.stderr
    assert "| ucp100-48p | 48 | 600 | 1/1 |" in result.stdout


def test_an_empty_commit_is_refused(tmp_path: Path) -> None:
    table = write_table(tmp_path / "1.csv", seed_rows(1, commit_sha=""))
    assert "run with --commit" in refusal([table])


def test_two_tables_at_one_seed_are_refused(tmp_path: Path) -> None:
    tables = [write_table(tmp_path / f"{n}.csv", seed_rows(1)) for n in "ab"]
    # Their rows collide first, or the seed check names them: either refusal names the seed.
    assert "seed 1" in refusal(tables)


def test_one_table_at_two_seeds_is_refused(tmp_path: Path) -> None:
    rows = seed_rows(1)
    rows[-1]["seed"] = 2
    assert "mixed seeds" in refusal([write_table(tmp_path / "1.csv", rows)], *ROSTER)


def test_cited_numbers_that_differ_between_tables_are_refused(tmp_path: Path) -> None:
    rows = seed_rows(2)
    rows[0] = cited("ucp13", 1, 100.0, 111.0)
    rows[2]["ub"] = 111.0
    tables = [write_table(tmp_path / "1.csv", seed_rows(1)), write_table(tmp_path / "2.csv", rows)]
    assert "Pedroso's numbers" in refusal(tables)


def test_a_measured_row_with_other_bounds_than_the_cited_ones_is_refused(tmp_path: Path) -> None:
    rows = seed_rows(1)
    rows[2]["lb"] = 90.0
    assert "are not Pedroso's" in refusal([write_table(tmp_path / "1.csv", rows)])


def test_a_table_in_the_older_published_layout_is_refused(tmp_path: Path) -> None:
    columns = "instance,periods,method,objective,lb,gap_pct,time_s,source,note"
    old = [{k: v for k, v in cited("ucp13", 1, 1.0, 1.0).items() if k in columns.split(",")}]
    assert "columns" in refusal([write_table(tmp_path / "1.csv", old, columns)])


def test_a_torn_row_is_refused(tmp_path: Path) -> None:
    table = write_table(tmp_path / "1.csv", seed_rows(1))
    table.write_text(table.read_text() + "ucp13,1\n")
    assert "torn row" in refusal([table])


def test_a_missing_table_is_refused(tmp_path: Path) -> None:
    assert "refused" in refusal([tmp_path / "absent.csv"])
