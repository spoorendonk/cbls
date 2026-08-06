// Tests for the engine contract the MIPfeas incumbent trace depends on.
//
// benchmarks/mipfeas/mipfeas.cpp records the Primal Integral's step function from
// SolveCallback, filtering on `std::isfinite(p.objective)` rather than on
// `p.feasible`. That choice is only correct if the objective field means "an
// incumbent exists" — these tests pin that meaning down, because a change to it
// would silently mis-time every published Primal Integral rather than fail a build.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

// Collects every progress event so the sequence can be asserted on as a whole.
class ProgressCollector : public cbls::SolveCallback {
public:
    void on_progress(const cbls::SolveProgress& p) override { events.push_back(p); }

    std::vector<cbls::SolveProgress> events;
};

// A model whose feasible region needs searching for, so that progress events are
// emitted both before and after the first incumbent:
//   min  3 a + 2 b + 4 c   s.t.  a + b + c >= 2,  a,b,c in {0,1}
// Optimum: a=1, b=1, c=0 -> 5.
void build_binary_model(cbls::Model& m) {
    cbls::Expr a(&m, m.bool_var("a"));
    cbls::Expr b(&m, m.bool_var("b"));
    cbls::Expr c(&m, m.bool_var("c"));
    cbls::Expr two(&m, m.constant(2.0));
    m.add_constraint(a + b + c >= two);
    m.minimize(3.0 * a + 2.0 * b + 4.0 * c);
    m.close();
}

}  // namespace

TEST_CASE("SolveProgress objective is infinite until an incumbent exists", "[mipfeas][trace]") {
    cbls::Model model;
    build_binary_model(model);
    ProgressCollector collector;
    cbls::SearchConfig cfg;
    cfg.max_iterations = 2000;
    cbls::SearchResult result =
        cbls::solve(model, /*time_limit=*/0.0, /*seed=*/7, /*use_fj=*/true, /*hook=*/nullptr,
                    /*lns=*/nullptr, /*lns_interval=*/3, &collector, cfg);

    REQUIRE(result.feasible);
    REQUIRE_FALSE(collector.events.empty());

    // Once finite, the objective never reverts to infinite: it carries the best
    // feasible objective so far, which cannot be un-found.
    bool seen_finite = false;
    for (const auto& e : collector.events) {
        if (std::isfinite(e.objective)) {
            seen_finite = true;
        } else {
            REQUIRE_FALSE(seen_finite);
        }
    }
    REQUIRE(seen_finite);
}

TEST_CASE("a trace filtered on a finite objective is monotone and ends at the result",
          "[mipfeas][trace]") {
    cbls::Model model;
    build_binary_model(model);
    ProgressCollector collector;
    cbls::SearchConfig cfg;
    cfg.max_iterations = 2000;
    cbls::SearchResult result =
        cbls::solve(model, /*time_limit=*/0.0, /*seed=*/11, /*use_fj=*/true, /*hook=*/nullptr,
                    /*lns=*/nullptr, /*lns_interval=*/3, &collector, cfg);
    REQUIRE(result.feasible);

    // Mirrors the runner's TraceRecorder: strict improvements on a finite objective.
    std::vector<std::pair<double, double>> trace;
    double last = std::numeric_limits<double>::infinity();
    for (const auto& e : collector.events) {
        if (std::isfinite(e.objective) && e.objective < last) {
            last = e.objective;
            trace.emplace_back(e.time_seconds, e.objective);
        }
    }

    REQUIRE_FALSE(trace.empty());
    for (size_t i = 1; i < trace.size(); ++i) {
        REQUIRE(trace[i].second < trace[i - 1].second);
        REQUIRE(trace[i].first >= trace[i - 1].first);
    }
    // The profile must end where the run ended, or the tail of every Primal
    // Integral is scored against a solution the run did not return.
    REQUIRE(trace.back().second == result.objective);
}

TEST_CASE("an incumbent can be reported while the current point is infeasible",
          "[mipfeas][trace]") {
    // The reason the runner filters on isfinite(objective) rather than p.feasible:
    // the two fields describe different things, so a run that has an incumbent can
    // still report feasible == false for the point it is currently sitting on. If
    // this ever stops happening the filter choice is merely redundant, not wrong,
    // so the test asserts the weaker invariant that always holds.
    cbls::Model model;
    build_binary_model(model);
    ProgressCollector collector;
    cbls::SearchConfig cfg;
    cfg.max_iterations = 2000;
    cbls::SearchResult result =
        cbls::solve(model, /*time_limit=*/0.0, /*seed=*/3, /*use_fj=*/true, /*hook=*/nullptr,
                    /*lns=*/nullptr, /*lns_interval=*/3, &collector, cfg);
    REQUIRE(result.feasible);

    // Every event that reports the current point feasible must also carry an
    // incumbent: reaching a feasible point is what records one.
    for (const auto& e : collector.events) {
        if (e.feasible) {
            REQUIRE(std::isfinite(e.objective));
        }
    }
}
