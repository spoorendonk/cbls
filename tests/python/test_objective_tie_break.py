"""`SearchConfig.fj_objective_tie_break` / `fj_objective_start` from Python (#218).

The mechanisms are pinned in C++ (`tests/test_fj_objective_tie.cpp`). What is
left for here is what only the binding can get wrong: a field missing,
misnamed, defaulting on, or not reaching the solve it is handed to.
"""

import _cbls_core as cbls


def _scan_model() -> tuple["cbls.Model", int]:
    """x >= 5 and x <= 4 keep the run infeasible; z is read only by the
    objective (maximise z), so only the objective flags move it."""
    m = cbls.Model()
    x = m.int_var(0, 10)
    z = m.int_var(0, 10)
    m.add_constraint(m.geq(x, m.constant(5.0)))
    m.add_constraint(m.leq(x, m.constant(4.0)))
    m.minimize(m.neg(z))
    m.close()
    return m, z


def _final_z(**flags: bool) -> float:
    m, z = _scan_model()
    config = cbls.SearchConfig()
    config.max_iterations = 2_000
    config.use_compound_moves = False
    for name, value in flags.items():
        setattr(config, name, value)
    r = cbls.solve(m, 0.0, 1, config=config)
    assert not r.feasible
    return float(m.var(-(z + 1)).value)  # a variable handle is -(id + 1)


def test_objective_flags_default_off() -> None:
    config = cbls.SearchConfig()
    assert config.fj_objective_tie_break is False
    assert config.fj_objective_start is False


def test_objective_tie_break_reaches_the_solve() -> None:
    assert _final_z() == 0.0
    assert _final_z(fj_objective_tie_break=True) == 10.0


def test_objective_start_reaches_the_solve() -> None:
    assert _final_z(fj_objective_start=True) == 10.0
