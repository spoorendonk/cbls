"""Python can tell the modeller's rows from close()'s hidden domain rows (#248).

`close()` appends a row `arg >= floor` per log, sqrt and fractional pow whose
argument can leave the op's domain. They sit in `constraint_ids()` after the
modeller's rows, so a caller mapping row indices back to its own rows needs
`num_user_constraints()` to know where its rows end.
"""

from __future__ import annotations

import _cbls_core as cbls


def test_a_log_of_a_free_variable_adds_one_domain_row_after_the_users() -> None:
    m = cbls.Model()
    x = m.float_var(-10.0, 10.0)
    m.add_constraint(m.leq(m.log_expr(x), m.constant(1.0)))
    assert m.num_user_constraints() == 1
    assert m.num_domain_rows() == 0

    m.close()
    rows = m.constraint_ids()
    assert len(rows) == 2
    assert m.num_user_constraints() == 1
    assert m.num_domain_rows() == 1
    assert not m.is_domain_node(rows[0])
    assert m.is_domain_node(rows[1])
