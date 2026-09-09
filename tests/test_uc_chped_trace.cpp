// The UC-CHPED anytime trace (#147).
//
// The defect this covers is not "the trace file is malformed" -- it is that the
// runner passed `nullptr` where solve() takes a SolveCallback, so no trace
// existed at all. A test that only checked a header would pass with the
// callback still `nullptr`, so the central case here drives a REAL solve() with
// the recorder attached and asserts that rows came out of it.
//
// Wall-clock-free, as tests/test_bench_flags.cpp is and for the same reason:
// `time_limit = 0` disables solve()'s clock entirely, so the run is bounded by
// `max_iterations` alone and the incumbents it records are reproducible for a
// given seed. The one thing that is NOT asserted is the row count, because
// solve() also emits a periodic progress report roughly once a second and that
// is a clock decision -- so the assertions are on what every row must say, and
// on the improvement sequence, neither of which the periodic reports disturb.

#include "benchmarks/uc-chped/data.h"
#include "benchmarks/uc-chped/greedy_init.h"
#include "benchmarks/uc-chped/trace_recorder.h"
#include "benchmarks/uc-chped/uc_model.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using cbls::uc_chped::kTraceHeader;
using cbls::uc_chped::TraceRecorder;

namespace {

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream is(line);
    while (std::getline(is, cell, sep)) {
        cells.push_back(cell);
    }
    return cells;
}

std::vector<std::string> rows_of(const std::string& csv) {
    std::vector<std::string> out;
    std::istringstream is(csv);
    std::string line;
    while (std::getline(is, line)) {
        if (!line.empty()) {
            out.push_back(line);
        }
    }
    return out;
}

/// A progress report with an incumbent, which is the only kind the trace
/// records.
cbls::SolveProgress incumbent(double time_seconds, double objective, bool new_best) {
    cbls::SolveProgress p;
    p.time_seconds = time_seconds;
    p.objective = objective;
    p.feasible = true;
    p.new_best = new_best;
    return p;
}

/// The runner's own start: greedy commitment plus a short FJ polish, with the
/// polish's wall clock disabled so nothing in the run depends on the machine.
void warm_start(cbls::uc_chped::UCModel& ucm, const cbls::uc_chped::UCInstance& inst,
                uint64_t seed) {
    cbls::uc_chped::greedy_uc_initialize(ucm.model, inst, ucm);
    cbls::RNG rng(seed);
    cbls::ViolationManager vm(ucm.model);
    cbls::fj_nl_initialize(ucm.model, vm, 200, &rng, /*time_limit=*/0.0);
}

}  // namespace

TEST_CASE("the uc-chped trace header names exactly the cells a row carries", "[uc-chped][trace]") {
    // Pinned literally. The column set is the interface the analysis reads, and
    // a header that drifts from the rows underneath it is worse than no trace:
    // every column after the drift is silently misread.
    REQUIRE(std::string(kTraceHeader) ==
            "instance,periods,time_seconds,objective,new_best,commit_sha");

    std::ostringstream out;
    TraceRecorder recorder(out, "ucp13", 24, "abc1234");
    recorder.on_progress(incumbent(1.5, 496191.8, /*new_best=*/true));

    const auto header = split(kTraceHeader, ',');
    const auto rows = rows_of(out.str());
    REQUIRE(rows.size() == 1);
    const auto cells = split(rows[0], ',');
    REQUIRE(cells.size() == header.size());
    CHECK(cells[0] == "ucp13");
    CHECK(cells[1] == "24");
    CHECK(cells[4] == "1");
    CHECK(cells[5] == "abc1234");
}

TEST_CASE("the trace identifies the horizon, not just the instance", "[uc-chped][trace]") {
    // This runner solves one row per (instance, horizon) pair. Without the
    // horizon cell the two blocks below would be indistinguishable, and a trace
    // that cannot be read back to the row it describes defends no budget.
    std::ostringstream out;
    TraceRecorder one(out, "ucp13", 1, "abc1234");
    TraceRecorder twentyfour(out, "ucp13", 24, "abc1234");
    one.on_progress(incumbent(0.1, 12782.0, true));
    twentyfour.on_progress(incumbent(0.2, 496191.0, true));

    const auto rows = rows_of(out.str());
    REQUIRE(rows.size() == 2);
    CHECK(split(rows[0], ',')[1] == "1");
    CHECK(split(rows[1], ',')[1] == "24");
}

TEST_CASE("the trace records nothing before there is an incumbent", "[uc-chped][trace]") {
    std::ostringstream out;
    TraceRecorder recorder(out, "ucp13", 24, "abc1234");

    cbls::SolveProgress searching;  // feasible defaults to false
    searching.time_seconds = 0.5;
    recorder.on_progress(searching);

    // `feasible` with a non-finite objective is a documented state (#100): a
    // feasibility witness whose objective overflowed. There is no incumbent
    // VALUE to plot, and an "inf" in a numeric column would read as a result.
    cbls::SolveProgress overflowed = incumbent(0.6, std::numeric_limits<double>::infinity(), true);
    recorder.on_progress(overflowed);

    CHECK(out.str().empty());
}

TEST_CASE("a comma in an identifier cannot shift the trace's columns", "[uc-chped][trace]") {
    std::ostringstream out;
    TraceRecorder recorder(out, "ucp13", 24, "abc1234,dirty");
    recorder.on_progress(incumbent(1.0, 5.0, true));

    const auto rows = rows_of(out.str());
    REQUIRE(rows.size() == 1);
    CHECK(split(rows[0], ',').size() == split(kTraceHeader, ',').size());
    CHECK(split(rows[0], ',')[5] == "abc1234;dirty");
}

TEST_CASE("a recorder handed to solve() records the run's incumbents", "[uc-chped][trace]") {
    // The wiring test. Hand solve() the recorder the way benchmarks/uc-chped/
    // uc_chped.cpp does and require that rows come back: with the callback
    // argument back at `nullptr` -- which is what this issue found -- solve()
    // reports nothing and this file stays empty.
    auto ucp13 = cbls::uc_chped::load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
    auto inst = cbls::uc_chped::make_subinstance(ucp13, 3);
    auto ucm = cbls::uc_chped::build_uc_model(inst);
    warm_start(ucm, inst, /*seed=*/42);

    std::ostringstream out;
    TraceRecorder recorder(out, "ucp13", inst.n_periods, "abc1234");

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    cbls::SearchConfig cfg;
    cfg.skip_init = true;
    cfg.max_iterations = 300;
    const cbls::SearchResult result =
        cbls::solve(ucm.model, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/false, &hook, &lns,
                    /*lns_interval=*/3, &recorder, cfg);
    REQUIRE(result.feasible);

    const auto rows = rows_of(out.str());
    REQUIRE(!rows.empty());

    const size_t n_columns = split(kTraceHeader, ',').size();
    double previous_best = std::numeric_limits<double>::infinity();
    int improvements = 0;
    for (const auto& row : rows) {
        const auto cells = split(row, ',');
        REQUIRE(cells.size() == n_columns);
        CHECK(cells[0] == "ucp13");
        CHECK(cells[1] == "3");
        CHECK(cells[5] == "abc1234");
        CHECK(std::stod(cells[2]) >= 0.0);  // wall time, the one machine-dependent cell
        if (cells[4] == "1") {
            const double objective = std::stod(cells[3]);
            // The incumbent only ever improves, so a trace whose new_best rows
            // are not decreasing is not an anytime profile.
            CHECK(objective < previous_best);
            previous_best = objective;
            ++improvements;
        }
    }
    CHECK(improvements > 0);
    // The last recorded incumbent is the one solve() returns: a trace that
    // stopped short of the answer would understate what the budget bought.
    // Compared on a relative band, not bit-for-bit: the cell was formatted to
    // ten significant digits and parsed back.
    CHECK(std::abs(previous_best - result.objective) <= 1e-6 * std::abs(result.objective));
}
