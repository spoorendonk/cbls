#include <cbls/cbls.h>
#include <cstdio>

int main() {
    cbls::Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto two = m.constant(2);
    auto obj = m.sum({m.pow_expr(x, two), m.pow_expr(y, two)});
    m.minimize(obj);
    m.close();

    auto result = cbls::solve(m, 5.0);
    printf("objective = %f\n", result.objective);
    printf("feasible = %s\n", result.feasible ? "true" : "false");
    printf("iterations = %ld\n", result.iterations);
    printf("time = %.3fs\n", result.time_seconds);
    return 0;
}
