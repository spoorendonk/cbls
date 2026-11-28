"""Solve a MINLPLib instance (.nl), print the point and verify it (docs/examples/minlplib.md).

Usage: python examples/minlplib.py benchmarks/instances/minlplib/st_e38.nl
"""

import sys

import cbls

INTEGER_TYPES = (cbls.VarType.Bool, cbls.VarType.Int)


def verify(model: cbls.Model, tol: float = 1e-6) -> list[str]:
    """What cbls::verify_model checks of the point: bounds, integrality, constraints."""
    errors: list[str] = []
    for i in range(model.num_vars()):
        v = model.var(i)
        if not v.lb - tol <= v.value <= v.ub + tol:
            errors.append(f"variable {i} = {v.value} outside [{v.lb}, {v.ub}]")
        if v.type in INTEGER_TYPES and abs(v.value - round(v.value)) > tol:
            errors.append(f"variable {i} = {v.value} is not integral")
    for c in model.constraint_ids():
        # A constraint node's value is its violation; a domain row allows none.
        if model.node_value(c) > (0.0 if model.is_domain_node(c) else tol):
            errors.append(f"constraint node {c} violated by {model.node_value(c)}")
    return errors


def main(path: str) -> int:
    converted = cbls.nl_to_model(cbls.read_nl(path))
    if not converted.supported:  # an operator cbls does not model
        print("unsupported:", "; ".join(converted.skipped_reasons))
        return 1
    model = converted.model

    result = cbls.solve(model, time_limit=2.0, seed=42, threads=1)

    print(f"feasible  = {result.feasible}")
    print(f"objective = {result.objective:.6f}")  # in the model's own sense
    for column, value in enumerate(converted.original_values()):
        print(f"  x{column} = {value:.6f}")
    errors = verify(model)
    print("verify: PASS" if not errors else f"verify: FAIL ({len(errors)} errors)")
    for error in errors:
        print("  " + error)
    return 0 if result.feasible and not errors else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MODEL.nl", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
