// Find a feasible point of a MIPLIB instance (.mps.gz) and verify it
// (docs/examples/mip.md).
//
// Usage: build/cbls_example_mip benchmarks/instances/miplib-fj/neos5.mps.gz
#include <cbls/cbls.h>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

int run(const char* path) {
    const cbls::MpsProblem problem = cbls::read_mps(path);
    cbls::MpsToModelResult converted = cbls::mps_to_model(problem);
    cbls::Model& model = converted.model;
    std::printf("%s: %zu columns, %zu rows\n", problem.name.c_str(), problem.vars.size(),
                problem.rows.size());

    cbls::SearchResult result = cbls::solve(model, /*time_limit=*/2.0, /*seed=*/42);

    std::printf("feasible  = %s\n", result.feasible ? "true" : "false");
    std::printf("objective = %.6f\n", result.objective);  // in the minimise sense
    const std::vector<double> values = converted.original_values(model);
    int nonzero = 0;
    for (double value : values) {
        nonzero += value != 0.0 ? 1 : 0;
    }
    std::printf("nonzero   = %d of %zu\n", nonzero, values.size());
    const cbls::VerifyResult verified = cbls::verify_model(model);  // tolerance 1e-6
    verified.print_diagnostics(stdout);
    return result.feasible && verified.ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.mps[.gz]\n", argv[0]);
        return 2;
    }
    try {  // a reader error (missing file, bad syntax) throws
        return run(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
