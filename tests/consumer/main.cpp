// A downstream program, built against cbls the two ways README's "Use from
// C++" documents (tests/consumer/run.cmake). Exits nonzero on a wrong result.
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <cstdio>
#include <exception>

namespace {

int run(const char* mps_path) {
    // examples/simple.cpp, on a short budget.
    cbls::Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
    auto result = cbls::solve(m, 0.2);
    std::printf("objective = %f\n", result.objective);
    if (!result.feasible || result.objective > 1e-3) {
        return 1;
    }

    // A gzipped MPS read pulls in the reader and with it zlib, the static
    // library's link-time dependency.
    auto loaded = cbls::mps_to_model(cbls::read_mps(mps_path));
    std::printf("mps columns = %zu\n", loaded.var_handles.size());
    return loaded.var_handles.empty() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cbls_consumer <file.mps.gz>\n");
        return 2;
    }
    try {
        return run(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cbls_consumer: %s\n", e.what());
        return 1;
    }
}
