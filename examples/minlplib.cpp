// Solve a MINLPLib instance (.nl), print the point and verify it
// (docs/examples/minlplib.md).
//
// Usage: build/cbls_example_minlplib benchmarks/instances/minlplib/st_e38.nl
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

int run(const char* path) {
    cbls::NlToModelResult converted = cbls::nl_to_model(cbls::read_nl(path));
    if (!converted.supported) {  // an operator cbls does not model
        for (const std::string& reason : converted.skipped_reasons) {
            std::printf("unsupported: %s\n", reason.c_str());
        }
        return 1;
    }
    cbls::Model& model = converted.model;

    cbls::SearchResult result = cbls::solve(model, /*time_limit=*/2.0, /*seed=*/42);

    std::printf("feasible  = %s\n", result.feasible ? "true" : "false");
    // The engine minimises: a maximisation model's objective comes back negated
    // (+inf, no point recorded, stays as it is).
    const bool flip = model.is_maximizing() && std::isfinite(result.objective);
    const double objective = flip ? -result.objective : result.objective;
    std::printf("objective = %.6f\n", objective);
    const std::vector<double> values = converted.original_values(model);
    for (size_t column = 0; column < values.size(); ++column) {
        std::printf("  x%zu = %.6f\n", column, values[column]);
    }
    const cbls::VerifyResult verified = cbls::verify_model(model);  // tolerance 1e-6
    verified.print_diagnostics(stdout);
    return result.feasible && verified.ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.nl\n", argv[0]);
        return 2;
    }
    try {  // a reader error (missing file, bad syntax) throws
        return run(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
