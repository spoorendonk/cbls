"""Find a feasible point of a MIPLIB instance (.mps.gz) and verify it (docs/examples/mip.md).

Usage: python examples/mip.py benchmarks/instances/miplib-fj/neos5.mps.gz
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
            errors.append(f"{v.name} = {v.value} outside [{v.lb}, {v.ub}]")
        if v.type in INTEGER_TYPES and abs(v.value - round(v.value)) > tol:
            errors.append(f"{v.name} = {v.value} is not integral")
    for c in model.constraint_ids():
        # A constraint node's value is its violation; a domain row allows none.
        if model.node_value(c) > (0.0 if model.is_domain_node(c) else tol):
            errors.append(f"constraint node {c} violated by {model.node_value(c)}")
    return errors


def main(path: str) -> int:
    problem = cbls.read_mps(path)
    converted = cbls.mps_to_model(problem)
    model = converted.model
    print(f"{problem.name}: {problem.num_vars} columns, {problem.num_rows} rows")

    result = cbls.solve(model, time_limit=2.0, seed=42, threads=1)

    print(f"feasible  = {result.feasible}")
    print(f"objective = {result.objective:.6f}")  # in the minimise sense (#472)
    # One value per MPS column; a handle encodes variable id i as -(i + 1).
    values = converted.original_values()
    nonzero = [
        f"{model.var(-(h + 1)).name}={v:g}"
        for h, v in zip(converted.var_handles, values, strict=True)
        if v != 0.0
    ]
    shown = " ".join(nonzero[:8]) + (" ..." if len(nonzero) > 8 else "")
    print(f"nonzero   = {len(nonzero)} of {len(values)}: {shown}")
    errors = verify(model)
    print("verify: PASS" if not errors else f"verify: FAIL ({len(errors)} errors)")
    for error in errors:
        print("  " + error)
    return 0 if result.feasible and not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
