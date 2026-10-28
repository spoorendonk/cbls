"""`SearchConfig.restart_randomization` and `.luby_restart_unit` from Python (#239).

The schedule and the kicks are pinned in C++ (`tests/test_restart_options.cpp`).
What is left for here is what only the binding can get wrong: a field missing,
misnamed, defaulting on, or not reaching the solve it is handed to.
"""

import _cbls_core as cbls
import pytest


def _infeasible() -> "cbls.Model":
    """Integer columns summing to 0.5: kicks every batch until the cap."""
    m = cbls.Model()
    xs = [m.int_var(0, 10_000_000) for _ in range(50)]
    m.add_constraint(m.eq_expr(m.sum(xs), m.constant(0.5)))
    m.close()
    return m


def test_restart_options_default_off() -> None:
    config = cbls.SearchConfig()
    assert config.restart_randomization is False
    assert config.luby_restart_unit == 0


def _iterations(seed: int, field: str, value: object) -> int:
    config = cbls.SearchConfig()
    config.max_iterations = 20_000
    config.use_compound_moves = False
    setattr(config, field, value)
    r = cbls.solve(_infeasible(), 0.0, seed, config=config)
    assert not r.feasible
    return int(r.iterations)


@pytest.mark.parametrize(
    ("field", "off", "on"),
    [("restart_randomization", False, True), ("luby_restart_unit", 0, 4)],
)
def test_restart_option_reaches_the_solve(field: str, off: object, on: object) -> None:
    """Clock-free, so each arm is deterministic; where the last batch ends,
    and so the count past the cap, follows the trajectory and would match if
    the field did not reach the search."""
    for seed in (1, 2, 3):
        assert _iterations(seed, field, off) != _iterations(seed, field, on), seed


def test_negative_luby_restart_unit_is_refused() -> None:
    config = cbls.SearchConfig()
    config.luby_restart_unit = -1
    with pytest.raises(ValueError, match="luby_restart_unit"):
        cbls.solve(_infeasible(), 0.0, 1, config=config)
