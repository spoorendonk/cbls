"""The structural-batch configuration surface exposed to Python (#165).

What is bound is configuration, not extension: `StructuralSelection`,
`SearchConfig.structural_sample_size` and a `NeighbourList` built either from
rows or from a cost callback. `MoveGenerator` itself is deliberately NOT
subclassable from Python -- it needs the trampoline/GIL machinery of #132 and
would sit on the move-generation hot path -- so these tests pin the config
surface and the validation that keeps an ill-formed neighbour list from reaching
an unchecked index (#156).
"""

from typing import Any

import _cbls_core as cbls
import pytest


def vid(handle: int) -> int:
    return -(handle + 1)


def set_cover_model(columns: int = 40, rows: int = 12) -> tuple[Any, int]:
    """One Set over `columns`, `rows` coverage rows, weighted objective.

    Big enough that the search does not converge inside the small iteration
    budgets below, which is what makes two policies distinguishable by their
    result rather than by both reaching the same trivial optimum.
    """
    m = cbls.Model()
    chosen = m.set_var(columns, 1, rows, "chosen")
    for r in range(rows):
        covered = m.lambda_sum(
            chosen, lambda e, base=r: 1.0 if (e * 7 + base) % 5 == 0 or e % rows == base else 0.0
        )
        m.add_constraint(m.geq(covered, m.constant(1.0)))
    m.minimize(m.lambda_sum(chosen, lambda e: 1.0 + float((e * 13) % 19)))
    m.close()
    return m, vid(chosen)


class TestStructuralSelection:
    def test_default_is_first_improving(self) -> None:
        cfg = cbls.SearchConfig()
        assert cfg.structural_selection == cbls.StructuralSelection.FirstImprovingSample
        assert cfg.structural_sample_size == 8
        assert cfg.structural_neighbours is None

    @pytest.mark.parametrize(
        "selection",
        [
            cbls.StructuralSelection.FirstImprovingSample,
            cbls.StructuralSelection.BestOfSample,
            cbls.StructuralSelection.ViolationGuided,
        ],
    )
    def test_every_policy_solves_a_set_model(self, selection: cbls.StructuralSelection) -> None:
        m, _ = set_cover_model()
        cfg = cbls.SearchConfig()
        cfg.max_iterations = 2000
        cfg.structural_batch_probability = 1.0
        cfg.structural_selection = selection
        cfg.structural_sample_size = 4
        result = cbls.solve(m, 0.0, 42, True, None, None, 3, None, cfg)
        assert result.feasible

    def test_selection_reaches_the_engine(self) -> None:
        """A non-default policy must change the search, not just the config.

        The check that matters is that the field is APPLIED. Were the apply step
        neutered, all three policies would run the identical trajectory from the
        identical seed, and this assertion is the one that notices.
        """
        results = {}
        for selection in (
            cbls.StructuralSelection.FirstImprovingSample,
            cbls.StructuralSelection.BestOfSample,
            cbls.StructuralSelection.ViolationGuided,
        ):
            m, chosen = set_cover_model()
            cfg = cbls.SearchConfig()
            cfg.max_iterations = 400
            cfg.structural_batch_probability = 1.0
            cfg.structural_selection = selection
            cfg.structural_sample_size = 6
            cbls.solve(m, 0.0, 7, True, None, None, 3, None, cfg)
            results[selection] = sorted(m.var(chosen).elements)
        assert len({tuple(v) for v in results.values()}) > 1


class TestNeighbourList:
    def test_nearest_neighbours_is_k_nearest_ties_by_id(self) -> None:
        nl = cbls.nearest_neighbours(4, 2, lambda a, b: abs(a - b))
        assert nl.universe() == 4
        assert len(nl) == 4
        # Element 1 is equidistant from 0 and 2; ascending id breaks the tie.
        assert nl.neighbours_of(1) == [0, 2]
        assert nl.neighbours_of(0) == [1, 2]

    def test_out_of_range_element_is_empty_not_a_crash(self) -> None:
        nl = cbls.nearest_neighbours(4, 2, lambda a, b: abs(a - b))
        assert nl.neighbours_of(-1) == []
        assert nl.neighbours_of(4) == []
        assert nl.neighbours_of(10_000_000) == []

    def test_rows_constructor_validates(self) -> None:
        assert cbls.NeighbourList([[1], [0]]).neighbours_of(0) == [1]
        with pytest.raises(ValueError):
            cbls.NeighbourList([[0, 9], [0]])

    def test_arrays_are_read_only(self) -> None:
        """The engine indexes this list unchecked, so Python cannot rewrite it."""
        nl = cbls.nearest_neighbours(4, 2, lambda a, b: abs(a - b))
        assert nl.offsets[0] == 0
        assert len(nl.ids) == nl.offsets[-1]
        with pytest.raises(AttributeError):
            nl.ids = [99]
        with pytest.raises(AttributeError):
            nl.offsets = [0, 1]

    def test_config_round_trips_a_neighbour_list(self) -> None:
        nl = cbls.nearest_neighbours(6, 2, lambda a, b: abs(a - b))
        cfg = cbls.SearchConfig()
        cfg.structural_neighbours = nl
        assert cfg.structural_neighbours is not None
        assert cfg.structural_neighbours.neighbours_of(1) == nl.neighbours_of(1)
        cfg.structural_neighbours = None
        assert cfg.structural_neighbours is None

    def test_neighbour_list_reaches_the_engine(self) -> None:
        """Supplying one must change where the built-in generators look.

        Same seed, same budget, same model: the only difference is the list. A
        binding that stored it and never passed it on would give two identical
        runs.
        """
        results = []
        for neighbours in (
            None,
            cbls.nearest_neighbours(40, 3, lambda a, b: float(abs(a - b))),
        ):
            m, chosen = set_cover_model()
            cfg = cbls.SearchConfig()
            cfg.max_iterations = 400
            cfg.structural_batch_probability = 1.0
            cfg.structural_neighbours = neighbours
            result = cbls.solve(m, 0.0, 42, True, None, None, 3, None, cfg)
            assert result.feasible
            results.append(tuple(sorted(m.var(chosen).elements)))
        assert results[0] != results[1]
