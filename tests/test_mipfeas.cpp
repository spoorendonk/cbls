// Tests for the MIPfeas class: the engine contract its incumbent trace depends
// on, and one end-to-end solve of a vendored MIPLIB instance.
//
// benchmarks/mipfeas/mipfeas.cpp records the Primal Integral's step function from
// SolveCallback, filtering on `std::isfinite(p.objective)` rather than on
// `p.feasible`. That choice is only correct if the objective field means "an
// incumbent exists" — the trace tests pin that meaning down, because a change to it
// would silently mis-time every published Primal Integral rather than fail a build.
//
// The solve test at the bottom covers the other half: that the engine still
// reaches a genuine, integral, no-better-than-optimal MILP solution at all.

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/io_mps.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
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

// ---------------------------------------------------------------------------
// End-to-end solve (issue #124).

namespace {

const char* kPk1Path = "benchmarks/instances/miplib-fj/pk1.mps.gz";
const char* kMiplibSoluPath = "benchmarks/instances/miplib-fj/miplib2017-v22.solu";

// The proven optimum recorded for `name` in a MIPLIB .solu file. Read rather
// than hard-coded so the reference the test scores against is the vendored
// file's, not a copy of it that can drift.
double proven_optimum(const std::string& name) {
    for (const cbls::SoluEntry& e : cbls::read_solu(kMiplibSoluPath)) {
        if (e.name == name) {
            REQUIRE(e.is_optimal);
            return e.value;
        }
    }
    FAIL("no .solu entry for " << name);
    return 0.0;
}

}  // namespace

// The MIPfeas roster's own instances are deliberately NOT vendored (~546 MiB;
// benchmarks/instances/mipfeas/.gitignore says so), so the in-suite solve runs
// on the ~180 KB vendored MIPLIB subset in benchmarks/instances/miplib-fj/ —
// same MIPLIB 2017 instances, same proven optima, and present in a fresh clone
// with no network. That directory's README is headed "Retired"; the data is
// explicitly kept, and this test is now one of its dependents.
//
// `pk1` is the instance picked: 86 columns over 45 rows, 55 of them integer
// (x2-x56) and 31 continuous, with the objective a single continuous column
// (x1) — so the run exercises the float path as well as the integer jumps, and
// the reference guard below needs a tolerance band rather than exact integer
// arithmetic. The engine reaches a feasible point on every one of seeds 1-10 at
// both 1000 and 2000 GLS iterations (~0.12-0.24s), and on seed 42 at the 2000
// the test uses, so the budget here is not near an edge. pk1's optimum, 11, is
// proven — `=opt= pk1 11` in the vendored miplib2017-v22.solu, and the same
// 11.0 in the MIPfeas roster.csv, which is derived from v36.
//
// The quality assertion is deliberately one-sided. Objective quality on pk1
// varies by two orders of magnitude across seeds (87 at seed 1, ~1e7 at seed 7
// at this budget), so an upper bound would be either vacuous or flaky; a
// solution BELOW the proven optimum, on the other hand, is unambiguously a bug
// in the model, the reader or the objective bookkeeping rather than a quality
// regression. This is the in-suite version of the run scorer's below_reference
// check.
TEST_CASE("MIPfeas pk1 solves to a feasible point never better than its optimum",
          "[mipfeas][solve]") {
    const double reference = proven_optimum("pk1");
    const cbls::MpsProblem prob = cbls::read_mps(kPk1Path);
    REQUIRE(prob.vars.size() == 86);

    // The scorer's own rule is the relative term alone — below_reference in
    // benchmarks/mipfeas/primal_integral.py, 1e-6 * (|reference| + 1). The
    // absolute floor of 10x the feasibility tolerance added here only widens it
    // (1.2e-5 against 1.0e-5 on pk1), because a point may sit feas_tol outside a
    // row and buy a little objective with that slack.
    const double band =
        std::max(1e-6 * (std::abs(reference) + 1.0), 10.0 * cbls::kDefaultFeasibilityTolerance);

    for (uint64_t seed : {1ULL, 7ULL, 42ULL}) {
        INFO("seed " << seed);
        cbls::MpsToModelResult built = cbls::mps_to_model(prob);
        REQUIRE(built.objective_node_id >= 0);
        cbls::FloatIntensifyHook hook;
        // LNS cannot fire at this budget — diversify() needs 100 stagnant
        // batches of 1000 iterations — so it is passed for parity with
        // benchmarks/mipfeas/mipfeas.cpp, not because it is under test.
        cbls::LNS lns(0.3);
        cbls::SearchResult result = solve_deterministic(built.model, 2000, seed, &hook, &lns);

        CAPTURE(result.best_violation, result.objective, result.iterations);
        REQUIRE(result.feasible);
        REQUIRE(result.best_violation <= cbls::kDefaultFeasibilityTolerance);
        REQUIRE(std::isfinite(result.objective));
        // The iteration budget is what stopped the run: solve_deterministic
        // disables the wall clock, and a regression that re-armed it would make
        // this test machine-dependent while still passing (#104).
        REQUIRE(result.termination == cbls::TerminationReason::IterationLimit);

        // An Int column left fractional means the returned point is not a
        // solution of the MIP at all, whatever the residual says.
        int n_fractional = 0;
        for (const auto& v : built.model.variables()) {
            if (v.type == cbls::VarType::Int && std::abs(v.value - std::round(v.value)) > 1e-9) {
                ++n_fractional;
            }
        }
        REQUIRE(n_fractional == 0);

        // solve() restores best_state and re-evaluates before returning, so the
        // objective it reports must be the one the model holds. Without this the
        // reference guard below only constrains a number, not the solution.
        const double model_objective = built.model.node(built.objective_node_id).value;
        REQUIRE(std::abs(model_objective - result.objective) <=
                1e-6 * (std::abs(result.objective) + 1.0));

        REQUIRE(result.objective >= reference - band);
    }
}
