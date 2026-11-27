"""`benchmarks/uc-chped/score.py`, the UC-CHPED campaign's table (#363).

The scorer sits in a directory no package can name, so it is run as the command
the campaign runs. The rows are generated in the runner's layout, and its columns
and roster are pinned against `uc_chped.cpp` itself. Gaps are hand-computed.
`fixtures/uc_chped_campaign_smoke` holds real runner output.
"""

from __future__ import annotations

import csv
import io
import json
import math
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
        "| ucp13 | 3 | 3 | 2/2 | 39025.1 | 0.45 [0.01, 0.89] | 0.45 [0.01, 0.89] | 0.00 |",
        "| ucp100-48p | 48 | 3 | 2/2 | 11841399.1 | - | - | - |",
        "- engine commit: 7c398e9 (commit_sha column)",
        "- seeds: 1, 2 (seed column)",
        "- cbls threads: 4 (search_config column)",
        "- no solution: 0",
        "- below Pedroso's LB, to check before reading the gap: 0",
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
        ({"note": "lost-workers"}, "lost portfolio workers"),
        ({"objective": "garbled"}, "objective is 'garbled'"),
        ({"lb": "garbled"}, "lb is 'garbled'"),
        ({"time_limit_s": ""}, "no budget"),
        ({"time_limit_s": 0}, "no budget"),
        ({"periods": "many"}, "periods is 'many'"),
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


@pytest.mark.parametrize("commit", ["", "unknown"])
def test_a_commit_the_runner_was_not_told_is_refused(tmp_path: Path, commit: str) -> None:
    table = write_table(tmp_path / "1.csv", seed_rows(1, commit_sha=commit))
    assert "run with --commit" in refusal([table])


def test_two_tables_at_one_seed_are_refused(tmp_path: Path) -> None:
    tables = [write_table(tmp_path / f"{n}.csv", seed_rows(1)) for n in "ab"]
    # Their rows collide first, or the seed check names them: either refusal names the seed.
    assert "seed 1" in refusal(tables)


def test_two_tables_at_one_seed_with_no_row_in_common_are_refused(tmp_path: Path) -> None:
    rows = seed_rows(1)
    tables = [
        write_table(tmp_path / "a.csv", rows[:4]),
        write_table(tmp_path / "b.csv", rows[4:]),
    ]
    assert "is in both" in refusal(tables)


@pytest.mark.parametrize("seed", ["", "one"])
def test_a_seed_that_is_no_number_is_refused(tmp_path: Path, seed: str) -> None:
    rows = seed_rows(1)
    for row in rows[2:]:
        row["seed"] = seed
    assert "the seed cell" in refusal([write_table(tmp_path / "1.csv", rows)])


def test_an_objective_below_pedrosos_lb_is_named_under_the_table(tmp_path: Path) -> None:
    rows = seed_rows(1)
    rows[2]["objective"] = 90.0
    result = run_score(write_table(tmp_path / "1.csv", rows), *ROSTER)
    assert result.returncode == 0, result.stderr
    assert "- below Pedroso's LB, to check before reading the gap: 1" in result.stdout
    assert "  - ucp13 T=1 seed 1: 90.00 against 100.00" in result.stdout


def test_a_slip_below_a_large_lb_stays_visible(tmp_path: Path) -> None:
    rows = seed_rows(1)
    rows[0] = cited("ucp13", 1, 1112371.0, 1112371.0)
    rows[2].update(objective=1112370.5, lb=1112371.0, ub=1112371.0)
    result = run_score(write_table(tmp_path / "1.csv", rows), *ROSTER)
    assert result.returncode == 0, result.stderr
    assert "  - ucp13 T=1 seed 1: 1112370.50 against 1112371.00" in result.stdout


def test_the_table_does_not_depend_on_the_order_of_the_tables(tmp_path: Path) -> None:
    tables = []
    for seed in (1, 2, 10):
        rows = seed_rows(seed)
        rows[-1].update(objective="", feasible="false", verified="", note="infeasible at feas_tol")
        tables.append(write_table(tmp_path / f"{seed}.csv", rows))
    forward = run_score(*tables, *ROSTER)
    backward = run_score(*reversed(tables), *ROSTER)
    assert forward.returncode == 0, forward.stderr
    assert forward.stdout == backward.stdout
    assert forward.stdout.index("seed 2:") < forward.stdout.index("seed 10:")


def test_a_repeated_horizon_is_refused(tmp_path: Path) -> None:
    table = write_table(tmp_path / "1.csv", seed_rows(1))
    assert "repeats a horizon" in refusal([table], "--roster", "ucp13:1,1")


def test_an_undecodable_table_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "1.csv"
    table.write_bytes(b"\xff\xfe\x00garbage")
    assert "refused" in refusal([table])


def test_an_out_over_an_input_table_is_refused_and_leaves_it(tmp_path: Path) -> None:
    tables = two_seeds(tmp_path)
    before = tables[0].read_text()
    assert "would overwrite" in refusal(tables, *ROSTER, "--out", str(tables[0]))
    assert tables[0].read_text() == before


def test_an_out_over_the_published_table_is_refused(tmp_path: Path) -> None:
    published = REPO / "benchmarks" / "instances" / "uc-chped" / "comparison.csv"
    assert "would overwrite" in refusal(two_seeds(tmp_path), *ROSTER, "--out", str(published))


def test_an_unwritable_out_still_prints_the_table(tmp_path: Path) -> None:
    out = tmp_path / "missing" / "dir" / "table.md"
    result = run_score(*two_seeds(tmp_path), *ROSTER, "--out", out)
    assert result.returncode == 2
    assert "| ucp13 | 1 |" in result.stdout
    assert "cannot write" in result.stderr


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


def scip_record(
    instance: str, periods: int, method: str, lb: float | None, ub: float | None, **cells: object
) -> dict[str, object]:
    """One `ucp_valve_scip.py --out` line, with the fields the scorer reads."""
    return {
        "instance": instance,
        "periods": periods,
        "method": method,
        "cpu_limit_s": 3600.0,
        "scip_clock": "wall",
        "lb": lb,
        "ub": ub,
        "status": "converged" if method == "pwl" else "optimal",
        "concavity_violated_units": [],
        "scip": "10.0.0",
        "check": [],
        **cells,
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def scip_lines(tmp_path: Path, records: list[dict[str, object]]) -> list[str]:
    result = run_score(
        *two_seeds(tmp_path), *ROSTER, "--scip", write_jsonl(tmp_path / "s", records)
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_scip_table_gives_each_methods_bounds_and_cbls_gaps_to_the_best(tmp_path: Path) -> None:
    lines = scip_lines(
        tmp_path,
        [
            scip_record("ucp13", 1, "pwl", 100.0, 101.0),
            scip_record("ucp13", 1, "exact", 99.0, 100.5),
            # The port names the base instance; the roster names the extension.
            scip_record("ucp100", 48, "pwl", 4000.0, 4500.0),
        ],
    )
    # cbls 102 and 104: against the best LB 100 (pwl) and the best UB 100.5 (exact).
    assert (
        "| ucp13 | 1 | 103.0 | 100.00 | 101.00 | 1.00 | 99.00 | 100.50 | 1.52 "
        "| 3.00 [2.00, 4.00] | 2.49 [1.49, 3.48] |"
    ) in lines
    assert (
        "| ucp100-48p | 48 | 7500.0 | 4000.00 | 4500.00 | 12.50 | - | - | - "
        "| 87.50 [25.00, 150.00] | 66.67 [11.11, 122.22] |"
    ) in lines
    assert "| ucp13 | 3 | 200.0 | - | - | - | - | - | - | - | - |" in lines
    assert any(
        line.startswith(
            "- SCIP 10.0.0; budget (process CPU s): pwl 3600; exact 3600 (ucp_valve_scip.py "
            "records). Each SCIP solve is limited in wall time"
        )
        for line in lines
    )
    assert "- no SCIP record: 3" in lines
    assert "  - ucp100-48p T=48 exact" in lines
    assert "- cbls below the best SCIP LB, to check before reading the gap: 0" in lines


@pytest.mark.parametrize(
    ("config", "sentence"),
    [
        (CONFIG, "cbls's has demand >= (FIDELITY.md section 1.4)."),
        (CONFIG + ";demand=eq", "so does cbls (--demand-equality, search config demand=eq)."),
    ],
)
def test_scip_table_names_the_demand_model_cbls_solved(
    config: str, sentence: str, tmp_path: Path
) -> None:
    """A `--demand-equality` campaign (#442) is on SCIP's `=` model, not `>=`."""
    tables = [
        write_table(tmp_path / f"seed{s}" / "c.csv", seed_rows(s, search_config=config))
        for s in (1, 2)
    ]
    records = write_jsonl(tmp_path / "s", [scip_record("ucp13", 1, "pwl", 100.0, 101.0)])
    result = run_score(*tables, *ROSTER, "--scip", records)
    assert result.returncode == 0, result.stderr
    assert sentence in result.stdout


def test_without_scip_there_is_no_scip_table(tmp_path: Path) -> None:
    result = run_score(*two_seeds(tmp_path), *ROSTER)
    assert "SCIP references" not in result.stdout


def test_a_pwl_lb_that_is_no_bound_is_starred_and_not_used(tmp_path: Path) -> None:
    record = scip_record("ucp13", 1, "pwl", 103.0, 105.0, concavity_violated_units=[2])
    lines = scip_lines(tmp_path, [record])
    starred = "| ucp13 | 1 | 103.0 | 103.00* | 105.00 | - | - | - | - | - | -1.90 [-2.86, -0.95] |"
    assert starred in lines
    assert "- pwl LB not a bound (some unit breaks c <= e*f^2/2), starred: 1" in lines


def test_cbls_below_the_best_scip_lb_is_named(tmp_path: Path) -> None:
    lines = scip_lines(tmp_path, [scip_record("ucp13", 1, "exact", 103.0, 103.0)])
    assert "- cbls below the best SCIP LB, to check before reading the gap: 1" in lines
    assert "  - ucp13 T=1 seed 1: 102.00 against 103.00" in lines


def test_a_scip_run_with_no_solution_is_named(tmp_path: Path) -> None:
    record = scip_record("ucp13", 3, "pwl", None, None, status="infeasible")
    lines = scip_lines(tmp_path, [record])
    assert "- no SCIP solution: 1" in lines
    assert "  - ucp13 T=3 pwl: infeasible" in lines


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        ([scip_record("ucp13", 1, "pwl", 1.0, 2.0, check=["demand t=1"])], "re-check"),
        ([scip_record("ucp13", 1, "milp", 1.0, 2.0)], "unknown method"),
        ([scip_record("ucp40", 1, "pwl", 1.0, 2.0)], "not in the roster"),
        ([scip_record("ucp13", 1, "pwl", 1.0, 2.0)] * 2, "twice"),
        ([scip_record("ucp13", 1, "pwl", math.nan, 2.0)], "finite"),
        ([{"instance": "ucp13", "periods": 1}], "not a ucp_valve_scip.py record"),
        (
            [
                scip_record("ucp13", 1, "pwl", 1.0, 2.0),
                scip_record("ucp13", 1, "exact", 1.0, 2.0, scip="9.2.0"),
            ],
            "mixed SCIP versions",
        ),
        (
            [
                scip_record("ucp13", 1, "pwl", 1.0, 2.0),
                scip_record("ucp13", 3, "pwl", 1.0, 2.0, cpu_limit_s=60.0),
            ],
            "mixed SCIP budgets",
        ),
        ([scip_record("ucp13", 1, "pwl", 1.0, 2.0, cpu_limit_s=-1.0)], "positive number"),
        ([scip_record("ucp13", 1, "pwl", 1.0, 2.0, scip_clock="cpu")], "not 'wall'"),
        ([scip_record("ucp13", 1, "pwl", 10**400, 2.0)], "finite"),
        ([], "no records"),
    ],
)
def test_a_scip_record_that_cannot_be_read_honestly_is_refused(
    tmp_path: Path, records: list[dict[str, object]], reason: str
) -> None:
    scip = write_jsonl(tmp_path / "s.jsonl", records)
    assert reason in refusal(two_seeds(tmp_path), *ROSTER, "--scip", str(scip))


def test_a_scip_line_that_is_no_json_is_refused(tmp_path: Path) -> None:
    scip = tmp_path / "s.jsonl"
    scip.write_text("{torn\n")
    assert "not JSON" in refusal(two_seeds(tmp_path), *ROSTER, "--scip", str(scip))


def test_an_out_over_a_scip_file_is_refused(tmp_path: Path) -> None:
    scip = write_jsonl(tmp_path / "s.jsonl", [scip_record("ucp13", 1, "pwl", 1.0, 2.0)])
    refused = refusal(two_seeds(tmp_path), *ROSTER, "--scip", str(scip), "--out", str(scip))
    assert "overwrite" in refused


def test_an_lb_that_is_not_positive_gives_no_gap(tmp_path: Path) -> None:
    lines = scip_lines(tmp_path, [scip_record("ucp13", 1, "exact", -50.0, 100.0)])
    assert (
        "| ucp13 | 1 | 103.0 | - | - | - | -50.00 | 100.00 | - | - | 3.00 [2.00, 4.00] |" in lines
    )


def test_scip_files_come_one_per_flag_and_leave_the_tables_alone(tmp_path: Path) -> None:
    pwl = write_jsonl(tmp_path / "pwl.jsonl", [scip_record("ucp13", 1, "pwl", 100.0, 101.0)])
    exact = write_jsonl(tmp_path / "exact.jsonl", [scip_record("ucp13", 1, "exact", 99.0, 100.5)])
    result = run_score("--scip", pwl, "--scip", exact, *two_seeds(tmp_path), *ROSTER)
    assert result.returncode == 0, result.stderr
    assert (
        "| ucp13 | 1 | 103.0 | 100.00 | 101.00 | 1.00 | 99.00 | 100.50 | 1.52 "
        "| 3.00 [2.00, 4.00] | 2.49 [1.49, 3.48] |"
    ) in result.stdout.splitlines()


def test_a_best_scip_lb_above_the_best_ub_is_named_not_refused(tmp_path: Path) -> None:
    records = [
        scip_record("ucp13", 1, "pwl", 110.0, 120.0),
        scip_record("ucp13", 1, "exact", 90.0, 100.0),
    ]
    lines = scip_lines(tmp_path, records)
    assert "- best SCIP LB above the best SCIP UB, to check: 1" in lines
    assert "  - ucp13 T=1: LB 110.00 above UB 100.00" in lines


TRACE_RUNNER_SOURCE = REPO / "benchmarks" / "uc-chped" / "trace_recorder.h"
TRACE_HEADER = "instance,periods,time_limit_s,time_seconds,batches,objective,new_best,commit_sha"

#: Per seed of `two_seeds`: (instance, T, incumbents before the end, final objective).
#: The closing row puts the final objective at the budget, as `record_final` does.
TRACE: dict[int, list[tuple[str, int, list[tuple[float, float]], float]]] = {
    1: [
        ("ucp13", 1, [(1.5, 125.0)], 102.0),
        ("ucp13", 3, [(0.0, 200.0)], 200.0),
        ("ucp100-48p", 48, [(2.0, 5000.0)], 5000.0),
    ],
    2: [
        ("ucp13", 1, [(0.0, 104.0)], 104.0),
        ("ucp13", 3, [(0.0, 200.0)], 200.0),
        ("ucp100-48p", 48, [(2.0, 10000.0)], 10000.0),
    ],
}


def write_trace(
    path: Path, seed: int, *, budget: float = 3.0, commit: str = "abc1234", drop: str = ""
) -> Path:
    """`uc_chped --trace` for one seed of `two_seeds`; `drop` leaves that instance out."""
    lines = [TRACE_HEADER]
    for instance, periods, points, final in TRACE[seed]:
        if instance == drop:
            continue
        rows = [*points, (budget, final)]
        lines += [
            f"{instance},{periods},{budget:g},{t:g},7,{z:g},{int(k < len(points))},{commit}"
            for k, (t, z) in enumerate(rows)
        ]
    path.write_text("\n".join(lines) + "\n")
    return path


def trace_args(
    tmp_path: Path, *, budget: float = 3.0, commit: str = "abc1234", drop: str = ""
) -> list[str]:
    """`--trace SEED:CSV` for both seeds of `two_seeds`."""
    return [
        f"--trace={s}:"
        + str(write_trace(tmp_path / f"trace{s}.csv", s, budget=budget, commit=commit, drop=drop))
        for s in (1, 2)
    ]


def driver_record(
    instance: str, periods: int, method: str, lb: float | None, ub: float | None, **cells: object
) -> dict[str, object]:
    """A `ucp_valve_scip.py --backend driver` line: wall budget, threads, trajectory."""
    record = scip_record(instance, periods, method, lb, ub, backend="driver", threads=4)
    del record["cpu_limit_s"]
    return {**record, "time_limit_s": 3.0, "trajectory": [], **cells}


def test_trace_columns_are_the_recorders() -> None:
    assert f'"{TRACE_HEADER}"' in TRACE_RUNNER_SOURCE.read_text()


def test_equal_time_table_gives_cbls_primal_integral_and_final_gap(tmp_path: Path) -> None:
    result = run_score(*two_seeds(tmp_path), *ROSTER, *trace_args(tmp_path))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # ref 102 (seed 1's final). Seed 1: gap 2 on [0, 1.5), 23/125 on [1.5, 3): PI 1.092;
    # seed 2: 2/104 throughout. Final gaps 0 and 1.92 %.
    no_scip = " | - | - | - | - |"
    assert (
        "| ucp13 | 1 | 3 | 102.00 | 0.5556 [0.0192, 1.0920] | 0.96 [0.00, 1.92]" + no_scip in lines
    )
    assert (
        "| ucp13 | 3 | 3 | 200.00 | 0.0000 [0.0000, 0.0000] | 0.00 [0.00, 0.00]" + no_scip in lines
    )
    # ref 5000. Seed 1: 2 on [0, 2) then 0; seed 2: 2 then 1/2.
    assert (
        "| ucp100-48p | 48 | 3 | 5000.00 | 1.4167 [1.3333, 1.5000] | 25.00 [0.00, 50.00]" + no_scip
    ) in lines
    assert "- no incumbent from any method, so no reference: 0" in lines


def test_equal_time_table_scores_the_driver_trajectories(tmp_path: Path) -> None:
    records = write_jsonl(
        tmp_path / "s",
        [
            driver_record("ucp13", 1, "pwl", 100.0, 101.0, trajectory=[[0.0, 103.0], [2.0, 101.0]]),
            # SCIP's objective (101) bounds the true cost (100.5) from above.
            driver_record("ucp13", 1, "exact", 99.0, 100.5, trajectory=[[1.0, 101.0]]),
        ],
    )
    result = run_score(*two_seeds(tmp_path), *ROSTER, "--scip", records, *trace_args(tmp_path))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # ref 100.5. exact: 2 on [0, 1), 0.5/101 on [1, 3); pwl: 2.5/103 on [0, 2), 0.5/101 after.
    assert (
        "| ucp13 | 1 | 3 | 100.50 | 0.5658 [0.0337, 1.0980] | 2.42 [1.47, 3.37] "
        "| 0.0178 [0.0178, 0.0178] | 0.50 [0.50, 0.50] "
        "| 0.6700 [0.6700, 0.6700] | 0.00 [0.00, 0.00] |"
    ) in lines
    assert any(
        line.startswith("- SCIP 10.0.0 through cbls_scip_traj at 4 threads; budget (wall s")
        and "pwl T=1 3; exact T=1 3" in line
        for line in lines
    )


def test_a_row_with_no_incumbent_anywhere_is_named(tmp_path: Path) -> None:
    tables = []
    for s in (1, 2):
        rows = [r for r in seed_rows(s) if r["instance"] != "ucp100-48p"]
        rows.append(measured("ucp100-48p", 48, s, None))
        tables.append(write_table(tmp_path / f"seed{s}" / "c.csv", rows))
    traces = []
    for s in (1, 2):
        path = write_trace(tmp_path / f"trace{s}.csv", s, drop="ucp100-48p")
        with path.open("a") as fh:
            fh.write("ucp100-48p,48,3,3,7,,0,abc1234\n")  # the closing row, no incumbent
        traces.append(f"--trace={s}:{path}")
    result = run_score(*tables, *ROSTER, *traces)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "| ucp100-48p | 48 | 3 | - | - | - | - | - | - | - |" in lines
    assert "  - ucp100-48p T=48" in lines


@pytest.mark.parametrize(
    ("budget", "commit", "drop", "argv", "reason"),
    [
        (3.0, "def5678", "", [], "commit 'def5678'"),
        (4.0, "abc1234", "", [], "budget 4 s"),
        (3.0, "abc1234", "ucp100-48p", [], "no trace for 2 rows"),
        (3.0, "abc1234", "", ["--trace=3:x.csv"], "not a table's"),
        (3.0, "abc1234", "", ["--trace=x.csv"], "SEED:CSV"),
    ],
)
def test_a_trace_that_does_not_match_the_tables_is_refused(
    tmp_path: Path, budget: float, commit: str, drop: str, argv: list[str], reason: str
) -> None:
    traces = trace_args(tmp_path, budget=budget, commit=commit, drop=drop)
    assert reason in refusal(two_seeds(tmp_path), *ROSTER, *traces, *argv)


def test_a_seed_traced_twice_and_a_torn_trace_are_refused(tmp_path: Path) -> None:
    traces = trace_args(tmp_path)
    assert "twice" in refusal(two_seeds(tmp_path), *ROSTER, *traces, traces[0])
    torn = tmp_path / "trace1.csv"
    torn.write_text("instance,periods\nucp13,1\n")
    assert "columns are not the trace's" in refusal(two_seeds(tmp_path), *ROSTER, *traces)


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        ([scip_record("ucp13", 1, "pwl", 1.0, 2.0)], "use ucp_valve_scip.py --backend driver"),
        ([driver_record("ucp13", 1, "pwl", 1.0, 2.0, threads=8)], "SCIP ran 8 threads, cbls 4"),
        ([driver_record("ucp13", 1, "pwl", 1.0, 2.0, time_limit_s=60.0)], "ran 60 s, cbls 3"),
        (
            [driver_record("ucp13", 1, "pwl", 1.0, 2.0, trajectory=[[2.0, 3.0], [1.0, 2.0]])],
            "out of time order",
        ),
        ([driver_record("ucp13", 1, "pwl", 1.0, 2.0, trajectory=None)], "no trajectory list"),
        (
            [
                driver_record("ucp13", 1, "pwl", 1.0, 2.0),
                scip_record("ucp13", 1, "exact", 1.0, 2.0),
            ],
            "mixed SCIP backends",
        ),
    ],
)
def test_scip_records_not_at_cbls_time_and_threads_are_refused(
    tmp_path: Path, records: list[dict[str, object]], reason: str
) -> None:
    scip = write_jsonl(tmp_path / "s.jsonl", records)
    traces = trace_args(tmp_path)
    assert reason in refusal(two_seeds(tmp_path), *ROSTER, "--scip", str(scip), *traces)


def test_an_out_over_a_trace_is_refused(tmp_path: Path) -> None:
    traces = trace_args(tmp_path)
    out = traces[0].partition(":")[2]
    assert "would overwrite" in refusal(two_seeds(tmp_path), *ROSTER, *traces, "--out", out)
