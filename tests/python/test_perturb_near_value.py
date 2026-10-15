"""`SearchConfig.perturb_near_value` from Python (#217).

The sampler and the kick are pinned in C++ (`tests/test_near_value_perturb.cpp`).
What is left for here is what only the binding can get wrong: the field missing,
misnamed, defaulting on, or not reaching the solve it is handed to.
"""

import _cbls_core as cbls


def _infeasible() -> "cbls.Model":
    """Integer columns summing to 0.5: kicks every batch until the cap."""
    m = cbls.Model()
    xs = [m.int_var(0, 10_000_000) for _ in range(50)]
    m.add_constraint(m.eq_expr(m.sum(xs), m.constant(0.5)))
    m.close()
    return m


def test_perturb_near_value_defaults_off() -> None:
    assert cbls.SearchConfig().perturb_near_value is False


def test_perturb_near_value_reaches_the_solve() -> None:
    """Clock-free, so each arm is deterministic and the two differ only in the
    kick: where a batch ends, and so the count past the cap, follows the
    trajectory, and would match if the flag did not reach the kick."""
    for seed in (1, 2, 3):
        iterations = {}
        for near in (False, True):
            config = cbls.SearchConfig()
            config.max_iterations = 20_000
            config.use_compound_moves = False
            config.perturb_near_value = near
            r = cbls.solve(_infeasible(), 0.0, seed, config=config)
            assert not r.feasible
            iterations[near] = r.iterations
        assert iterations[False] != iterations[True], seed
