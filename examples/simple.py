"""The Python twin of examples/simple.cpp: minimise x^2 + y^2 over [0, 10]^2."""

import cbls


def main() -> None:
    m = cbls.Model()
    x = m.Float(0, 10, "x")
    y = m.Float(0, 10, "y")
    m.minimize(x**2 + y**2)
    m.close()

    result = cbls.solve(m, time_limit=5.0, threads=1)  # cbls::solve, as simple.cpp
    print(f"objective = {result.objective:f}")
    print(f"feasible = {'true' if result.feasible else 'false'}")
    print(f"iterations = {result.iterations}")
    print(f"time = {result.time_seconds:.3f}s")
    print(f"x = {result.value(x):g}, y = {result.value(y):g}")


if __name__ == "__main__":
    main()
