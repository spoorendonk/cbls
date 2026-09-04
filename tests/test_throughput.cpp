// Throughput floors, one per benchmarked problem class (issue #125).
//
// WHAT THESE ASSERT, AND WHAT THEY DO NOT. Every published result in this repo
// is a time-limited solve, so the engine's value is the work it completes inside
// a fixed wall clock. Nothing else in the suite would notice that halving: the
// correctness tests bound the search by GLS *iterations* (solve_deterministic,
// time_limit = 0) precisely so their outcome does not depend on machine speed,
// which makes every one of them blind to the rate at which those iterations
// arrive. These three tests are the inverse: a fixed wall-clock budget, and an
// assertion on the iterations completed within it.
//
// They are FLOORS ON WORK, not ceilings on time. `timing_structural_batch_deadline`
// in test_search.cpp remains the suite's only assertion on wall-clock *duration*;
// these assert on a count, with the clock as the budget rather than the subject.
//
// The floors sit 5-10x below the measured rate on purpose. The target is an
// order-of-magnitude collapse — an accidental algorithmic regression, an O(n)
// scan that became O(n^2), a cache that stopped caching. A floor set close to the
// observed rate catches machine variance instead and gets disabled within a
// month, which is worse than no test at all. If one of these fires, the rate did
// not drift; something structural broke. docs/profiling.md is the tool for
// attributing where the time went.
//
// WHAT THEY DO NOT COVER: the FJ inner loop is what they measure, as the trap
// note below makes concrete. A 10x regression confined to batch-level code —
// LNS, diversification, resync, the inner-solver hook — costs too little of a
// 1s budget to move these numbers, and would pass green. Covering that wants a
// different instrument, not a tighter floor here.
//
// Each run also asserts `termination == TimeLimit`, which pins the iteration
// count to meaning "work per second" rather than "work until it stopped".
//
// Be honest about what that buys TODAY: nothing. The other reasons are all
// unreachable here — IterationLimit needs max_iterations > 0 (the config default
// is 0), NoBudget needs no deadline, Feasible needs a model with no objective,
// and all three of these carry one. A loop that ignored the deadline would hang
// into the ctest TIMEOUT rather than report a different reason. So it is a
// tautology that guards a future early-exit path, not a live check. Asserting
// that `time_seconds` lands near kBudgetSeconds would buy the wall-clock half
// for real; it is left out because the TIMEOUT already bounds that direction.
//
// The three models are the three benchmarked classes of CLAUDE.md's benchmark
// table, and each is the instance an existing end-to-end solve test already uses
// (or, for MINLP, a mid-size sibling of it — see below), so a failure here is
// isolated to throughput rather than confounded with a model or reader change.
//
// SHOWN TO FAIL (CLAUDE.md, "Measuring and testing engine changes"). Verified at
// 1bc4bb8 on the machine tabulated below: `git archive HEAD | tar -x` into a
// scratch tree, then a 5ms `sleep_for` at the `++iterations_` site in
// src/feasibility_jump.cpp. All three floors went red together --
//
//     milp/pk1        8498 it/s -> 191    (floor 500)
//     minlp/chain50   6001 it/s -> 190    (floor 300)
//     uc/ucp13-1p    39840 it/s -> 193    (floor 2500)
//
// -- while timing_structural_batch_deadline still passed, which is correct: it
// asserts on duration, not on work done.
//
// THE TRAP, if you re-verify this: stalling the `while (!past_deadline())` loop
// in src/search.cpp instead changes these rates by NOTHING. That loop iterates
// over *batches*; a batch runs many GLS iterations, and `SearchResult::iterations`
// counts the inner ones. A 200us stall there was invisible at all three classes
// (8498/6001/39840 it/s, i.e. the healthy rate). Cripple the inner site or you
// will conclude these tests cannot fail when they can.
//
// Registration: hand-registered in tests/CMakeLists.txt with LABELS "timing;slow"
// and an explicit TIMEOUT. The `timing` label makes them greppable as a class
// (`ctest -L timing`); the `slow` label keeps their three wall-clock seconds out
// of the `ctest -LE slow` pre-commit set. pre-push and CI run everything.

#include "benchmarks/uc-chped/data.h"
#include "benchmarks/uc-chped/uc_model.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/io_mps.h>
#include <cbls/io_nl.h>
#include <cstdio>

namespace {

// ---------------------------------------------------------------------------
// MEASUREMENT PROVENANCE for every floor in this file.
//
//   Machine:  AMD Ryzen 5 5600H (12 logical cores), 12 GiB RAM
//   Build:    Release (-O3) — the type CMakeLists.txt defaults to, so the
//             ```build fence, CI and a plain `cmake -B build` all measure this.
//   Commit:   floors derived at 6a8f46a (integration-wave1), 2026-09-02.
//
// Three regimes were measured, five and three repeats respectively:
//
//   solo      the test alone on a quiet box (load average < 1)
//   -j12      the same test inside a full `ctest -j$(nproc)` run of the suite,
//             which is how pre-push and CI actually run it
//   stress    24 spinning CPU hogs alongside it — 25x oversubscription, well
//             past anything the suite itself produces. Recorded only to show
//             the floors have headroom left even there; NOT what they are set
//             from.
//
//   class          solo (it/s)      -j12 (worst)   stress (worst)   FLOOR
//   ---------------------------------------------------------------------
//   milp/pk1       8500 -  8700          3617            1056         500
//   minlp/chain50  5950 -  6250          2111             913         300
//   uc/ucp13-1p   37800 - 41100         14412            3718        2500
//
// The solo ranges are deliberately wider than any single sitting produces: three
// independent measurement passes on this same box disagreed by more than one
// pass's own spread (an early pass read milp 8584-8754 and uc 40969-41353, both
// of which later passes missed low every time). A range a re-measurement cannot
// land in is worse than useless for telling a real regression from different
// hardware, so these span every observation taken. Re-measure with
// `build/tests/cbls_tests "[throughput]"`, which prints all three rates.
//
// Every floor is set from the WORST -j12 observation, divided by 5-10 as the
// issue requires: 7.2x for pk1, 7.0x for chain50, 5.8x for ucp13. Contention on
// this box costs 1.6-2.9x, so a floor derived from the solo column would be
// flaky by construction; measure both ways before changing one. ucp13 has the
// least headroom of the three and is the first that will fire.
//
// If you re-derive these on other hardware, replace the whole table rather than
// scaling it — the three classes do not degrade at the same rate under load
// (chain50 lost 2.9x where pk1 lost 2.0x), so a single fudge factor is wrong.
// ---------------------------------------------------------------------------

// One second is long enough that model build, the initial full_evaluate and the
// first batch's warm-up are a small share of it, and short enough that three of
// these cost the full suite three seconds. Shorter budgets were rejected: under
// `-j12` a 0.1s budget has enough scheduling variance to swamp the signal.
constexpr double kBudgetSeconds = 1.0;

// THE FLOORS ARE RELEASE-ONLY. Every number in the table above was measured at
// -O3, and an unoptimized build runs this engine 5-8x slower — a Debug `ctest
// -j12` puts the UC class at ~2000 it/s against a floor of 2500, i.e. red, with
// nothing at the failure site to say why. Debug is a documented supported
// override (README's build section), so these skip rather than fail there:
// ctest reports "Skipped" via the SKIP_RETURN_CODE 4 already set on all three in
// tests/CMakeLists.txt. The gates are unaffected either way — pre-commit runs
// `-LE slow`, which excludes these, and pre-push builds Release.
//
// A regression that only shows up unoptimized is not something these can catch,
// and that is deliberate: the published configuration is the one worth a floor.
#ifdef NDEBUG
constexpr bool kOptimizedBuild = true;
#else
constexpr bool kOptimizedBuild = false;
#endif

// Run `model` for exactly `kBudgetSeconds` of wall clock in the configuration
// the benchmark runners use — FJ on, FloatIntensifyHook, LNS(0.3) — and report
// what came back. Everything about this differs from `solve_deterministic` by
// design: the clock binds and the iteration count is the measurement.
//
// The hook and LNS are included rather than stripped out because the rate that
// matters is the rate of the published configuration; a floor measured without
// them would not notice a regression inside them. Both are cheap here (LNS
// cannot even fire at this budget — diversify() needs 100 stagnant batches).
//
// This matches benchmarks/mipfeas/mipfeas.cpp and benchmarks/minlplib/minlplib.cpp
// exactly. It does NOT match the UC runner: benchmarks/uc-chped/uc_chped.cpp
// warm-starts with greedy_uc_initialize + fj_nl_initialize and sets
// cfg.skip_init, so the UC floor below measures a COLD start where its runner
// measures a warm one. That is fine for a throughput floor — the inner loop is
// the same either way — but the UC number is not comparable to that runner's.
cbls::SearchResult run_for_budget(cbls::Model& model, uint64_t seed) {
    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    cbls::SearchConfig cfg;  // max_iterations = 0: the wall clock is the only budget
    return cbls::solve(model, kBudgetSeconds, seed, /*use_fj=*/true, &hook, &lns,
                       /*lns_interval=*/3, /*callback=*/nullptr, cfg);
}

// Report the observed rate on stdout so a re-measurement needs no instrumentation:
// `build/tests/cbls_tests "[throughput]"` prints all three.
void report(const char* label, const cbls::SearchResult& r) {
    printf("\nthroughput %s: %lld iterations in %.3fs = %.0f it/s\n", label,
           static_cast<long long>(r.iterations), r.time_seconds,
           static_cast<double>(r.iterations) / r.time_seconds);
}

}  // namespace

// ---------------------------------------------------------------------------
// Class 1: MILP feasibility (benchmark `mipfeas`, epic #87).
//
// `pk1` is the instance the mipfeas end-to-end solve test already uses: 86
// columns over 45 rows, 55 integer and 31 continuous, from the vendored ~180 KB
// MIPLIB subset in benchmarks/instances/miplib-fj/ (present in a fresh clone,
// no network). It exercises the integer jump path and the float path together,
// which is what a MILP throughput number should cover.

TEST_CASE("MILP feasibility throughput floor", "[throughput][timing][milp]") {
    if (!kOptimizedBuild) {
        SKIP("throughput floors are derived at -O3; see kOptimizedBuild above");
    }
    const cbls::MpsProblem prob = cbls::read_mps("benchmarks/instances/miplib-fj/pk1.mps.gz");
    cbls::MpsToModelResult built = cbls::mps_to_model(prob);
    REQUIRE(built.objective_node_id >= 0);

    cbls::SearchResult result = run_for_budget(built.model, /*seed=*/42);
    report("milp/pk1", result);

    // The clock is what stopped it, so `iterations` is a rate and not a total.
    REQUIRE(result.termination == cbls::TerminationReason::TimeLimit);
    // 8584-8754 it/s solo, 4337 worst under -j12; floor is 8.7x below that.
    // See the provenance table above for the machine and build type.
    REQUIRE(result.iterations >= 500);
}

// ---------------------------------------------------------------------------
// Class 2: non-convex MINLP (benchmark `minlplib`, epic #87).
//
// `chain50` rather than `ex4_1_8` (the instance the minlplib solve test uses):
// ex4_1_8 has two variables and one row, so its iteration rate measures loop
// overhead — a regression in delta_evaluate or in the jump-table scan over a
// real DAG would be invisible there. chain50 is 102 continuous columns over 51
// rows, classified `polynomial` in bounds.csv, and the runner reaches a feasible
// point on it (comparison.csv: feasible, 14.4% to BKS), so the search spends its
// budget on real work rather than thrashing.
//
// Deliberately NOT elec25/elec50, the other mid-size non-convex instances
// vendored here: their Coulomb objective is +inf at coincident points, the
// obj<=bound row clamps to ~1e30 and swamps the real rows in floating point
// (#100). A model whose violation signal is degenerate is a poor throughput
// probe — the search's behaviour there is a known bug, not a workload.

TEST_CASE("non-convex MINLP throughput floor", "[throughput][timing][minlp]") {
    if (!kOptimizedBuild) {
        SKIP("throughput floors are derived at -O3; see kOptimizedBuild above");
    }
    cbls::NlProblem prob = cbls::read_nl("benchmarks/instances/minlplib/chain50.nl");
    cbls::NlToModelResult built = cbls::nl_to_model(prob);
    REQUIRE(built.supported);
    REQUIRE(built.objective_node_id >= 0);

    cbls::SearchResult result = run_for_budget(built.model, /*seed=*/42);
    report("minlp/chain50", result);

    REQUIRE(result.termination == cbls::TerminationReason::TimeLimit);
    // 6185-6252 it/s solo, 2111 worst under -j12; floor is 7.0x below that.
    // This class degrades hardest under contention (2.9x), which is why the
    // floor is not simply a fixed fraction of the solo rate.
    REQUIRE(result.iterations >= 300);
}

// ---------------------------------------------------------------------------
// Class 3: the mixed bool+float unit-commitment model (benchmark `uc-chped`,
// epics #25/#91/#92).
//
// The 13-unit 1-period sub-instance: 13 commitment Bools and 13 dispatch Floats
// over the same builder the benchmark runner uses. This is the only one of the
// three classes where a Bool flip and a Newton float jump compete inside the same
// jump table, so it is the one that would notice a regression confined to that
// interaction. 13 units keeps the model small enough that the rate is dominated
// by iteration cost rather than by the one-off full_evaluate.

TEST_CASE("unit-commitment throughput floor", "[throughput][timing][uc]") {
    if (!kOptimizedBuild) {
        SKIP("throughput floors are derived at -O3; see kOptimizedBuild above");
    }
    auto ucp13 = cbls::uc_chped::load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
    auto inst = cbls::uc_chped::make_subinstance(ucp13, 1);
    auto ucm = cbls::uc_chped::build_uc_model(inst);
    REQUIRE(ucm.model.num_vars() == 26);

    cbls::SearchResult result = run_for_budget(ucm.model, /*seed=*/42);
    report("uc/ucp13-1p", result);

    REQUIRE(result.termination == cbls::TerminationReason::TimeLimit);
    // 40969-41353 it/s solo, 14412 worst under -j12; floor is 5.8x below that.
    REQUIRE(result.iterations >= 2500);
}
