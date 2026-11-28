"""File I/O from Python (#464): `.cbls` round trips, MPS and AMPL `.nl` adapters.

Each format is read, adapted and solved through `cbls.solve`; the solves are
capped by iterations, not the clock, so the asserted point is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import cbls
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# min x + 2y  s.t.  x + y >= 3,  x <= 2,  x, y integer in [0, 10]: optimum (2, 1), 4.
TINY_MPS = """\
NAME          TINY
ROWS
 N  COST
 G  C1
 L  C2
COLUMNS
    MARKER    'MARKER'    'INTORG'
    X         COST        1.0   C1   1.0
    X         C2          1.0
    Y         COST        2.0   C1   1.0
    MARKER    'MARKER'    'INTEND'
RHS
    RHS       C1          3.0   C2   2.0
BOUNDS
 UP BND       X           10.0
 UP BND       Y           10.0
ENDATA
"""


def _capped(iterations: int = 20_000) -> cbls.SearchConfig:
    config = cbls.SearchConfig()
    config.max_iterations = iterations
    return config


def _tiny_mps(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mps"
    path.write_text(TINY_MPS)
    return path


def _built_model() -> tuple[cbls.Model, int, int]:
    m = cbls.Model()
    x = m.int_var(0, 10, "x")
    y = m.float_var(-1.5, 2.5, "y")
    m.add_constraint(m.geq(m.sum([x, y]), m.constant(3.0)))
    m.minimize(m.sum([m.prod(x, x), m.prod(y, y)]))
    m.close()
    return m, x, y


def test_save_then_load_round_trips_variables_and_nodes(tmp_path: Path) -> None:
    m, _, _ = _built_model()
    path = tmp_path / "model.cbls"
    cbls.save_model(m, str(path))
    loaded = cbls.load_model(str(path))
    assert loaded.num_vars() == m.num_vars()
    assert loaded.num_nodes() == m.num_nodes()
    for i in range(m.num_vars()):
        a, b = m.var(i), loaded.var(i)
        assert (a.name, a.type, a.lb, a.ub) == (b.name, b.type, b.lb, b.ub)
    # Node names are renumbered on the first load, so a loaded model's file is
    # the fixed point: saving and loading it again changes nothing.
    again = tmp_path / "again.cbls"
    cbls.save_model(loaded, str(again))
    third = tmp_path / "third.cbls"
    cbls.save_model(cbls.load_model(str(again)), str(third))
    assert third.read_text() == again.read_text()


def test_a_loaded_model_solves_to_the_same_point_as_the_built_one(tmp_path: Path) -> None:
    m, _, _ = _built_model()
    path = tmp_path / "model.cbls"
    cbls.save_model(m, str(path))
    loaded = cbls.load_model(str(path))
    built = cbls.solve(m, time_limit=60.0, threads=1, config=_capped())
    read = cbls.solve(loaded, time_limit=60.0, threads=1, config=_capped())
    assert built.feasible and read.feasible
    assert read.values == built.values
    assert read.objective == built.objective


def test_the_example_cbls_file_loads_and_solves() -> None:
    m = cbls.load_model(str(EXAMPLES / "simple.cbls"))
    result = cbls.solve(m, time_limit=60.0, threads=1, config=_capped())
    assert result.feasible
    assert sum(result.values) >= 5.0 - 1e-6


def test_load_model_of_a_missing_file_raises() -> None:
    with pytest.raises(ValueError, match="cannot open"):
        cbls.load_model("/nonexistent/model.cbls")


def test_read_mps_reports_the_problem_header(tmp_path: Path) -> None:
    problem = cbls.read_mps(str(_tiny_mps(tmp_path)))
    assert problem.name == "TINY"
    assert problem.objective_row_name == "COST"
    assert (problem.num_vars, problem.num_rows, problem.num_nonzeros) == (2, 2, 5)
    assert not problem.maximize


def test_mps_to_model_solves_to_the_known_optimum(tmp_path: Path) -> None:
    result = cbls.mps_to_model(cbls.read_mps(str(_tiny_mps(tmp_path))))
    assert len(result.var_handles) == 2
    assert len(result.constraint_node_ids) == 2
    assert result.objective_node_id >= 0
    assert result.n_clamped_columns == 0
    solved = cbls.solve(result.model, time_limit=60.0, threads=1, config=_capped())
    assert solved.feasible
    assert solved.objective == 4.0
    assert result.original_values() == [2.0, 1.0]
    assert [solved.value(h) for h in result.var_handles] == [2.0, 1.0]


def test_mps_to_model_options_keep_the_cpp_defaults_and_apply() -> None:
    options = cbls.MpsToModelOptions()
    assert (options.inf_clamp, options.propagate_bounds, options.max_propagation_passes) == (
        1.0e9,
        True,
        10,
    )


def test_mps_without_propagation_runs_no_pass(tmp_path: Path) -> None:
    problem = cbls.read_mps(str(_tiny_mps(tmp_path)))
    options = cbls.MpsToModelOptions()
    options.propagate_bounds = False
    assert cbls.mps_to_model(problem, options).bound_stats.passes == 0
    assert cbls.mps_to_model(problem).bound_stats.passes > 0


def test_read_mps_of_a_missing_file_raises() -> None:
    with pytest.raises(RuntimeError, match="Cannot open"):
        cbls.read_mps("/nonexistent/problem.mps")


def test_read_nl_reports_the_problem_header() -> None:
    problem = cbls.read_nl(str(DATA / "filter.nl"))
    assert (problem.n_vars, problem.n_cons, problem.n_objs) == (2, 1, 1)
    assert (problem.n_defined_vars, problem.n_discrete_vars) == (0, 0)


def test_nl_to_model_solves_filter_to_its_best_known_point() -> None:
    result = cbls.nl_to_model(cbls.read_nl(str(DATA / "filter.nl")))
    assert result.supported
    assert result.skipped_reasons == []
    assert len(result.var_handles) == 2
    solved = cbls.solve(result.model, time_limit=60.0, threads=1, config=_capped())
    assert solved.feasible
    # The best-known point tests/test_minlplib.cpp checks the C++ path against.
    assert result.original_values() == pytest.approx([10.0, 1290.226825443540065], rel=1e-6)


def test_nl_to_model_options_keep_the_cpp_defaults() -> None:
    options = cbls.NlToModelOptions()
    assert (
        options.propagate_bounds,
        options.max_propagation_passes,
        options.inf_clamp,
        options.int_inf_clamp,
    ) == (True, 10, 1.0e9, 1.0e6)


def test_an_unsupported_nl_operator_is_reported_not_raised(tmp_path: Path) -> None:
    # filter.nl's unary log10 (o42) as atan (o49): parseable, outside the adapter's set.
    atan = tmp_path / "atan.nl"
    atan.write_text((DATA / "filter.nl").read_text().replace("o42\n", "o49\n", 1))
    result = cbls.nl_to_model(cbls.read_nl(str(atan)))
    assert not result.supported
    assert any("49" in reason for reason in result.skipped_reasons), result.skipped_reasons


def test_an_unknown_nl_opcode_is_refused_by_the_reader(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.nl"
    unknown.write_text((DATA / "filter.nl").read_text().replace("o42\n", "o999\n", 1))
    with pytest.raises(RuntimeError, match="NL_UNKNOWN_OPCODE"):
        cbls.read_nl(str(unknown))


def test_read_nl_of_a_missing_file_raises() -> None:
    with pytest.raises(RuntimeError, match="cannot open"):
        cbls.read_nl("/nonexistent/problem.nl")
