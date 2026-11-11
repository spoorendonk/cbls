#pragma once

// The anytime trace for the UC-CHPED runner (#147): what the incumbent was
// doing when the wall-clock budget stopped, so a bad published gap can be told
// apart as an engine limitation or a too-short budget.
//
// Shape follows benchmarks/minlplib/minlplib.cpp's TraceRecorder (`new_best`,
// one row per report), plus four columns this question needs:
//
//   * `periods` -- one solve per (instance, horizon), so `instance` alone does
//     not identify a row's solve.
//   * `time_limit_s` -- "flattened before the budget?" needs the budget, and
//     the budget map is what these traces exist to change: without the
//     column an archived trace stops stating its budget once the map moves.
//   * `batches` -- `SolveProgress::iteration`, the ViolationLS batch count (NOT
//     comparison.csv's GLS `iterations`); separates "converged" from "barely
//     started" in a flat tail.
//   * `commit_sha` -- engine commit on every row (#147), as comparison.csv: a
//     search-trajectory change invalidates a trace.
//
// Unlike minlplib's: takes an `ostream&` (testable into a stringstream),
// flushes each row, runs identifiers through csv_text, prints ten significant
// digits. Each is argued at its definition.
//
// A header so tests/test_uc_chped_trace.cpp can pin the recorder's contract
// through a real solve(). That test cannot see the runner's main(); the
// runner passing a real callback, not `nullptr`, is pinned by
// tests/python/test_cli.py::test_uc_chped_records_an_anytime_trace.

#include <algorithm>
#include <cbls/search.h>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace cbls::uc_chped {

/// The CSV cell rule both output files share: substitute, never quote, since
/// the readers (grep, awk, spreadsheet import) do not all implement quoting.
/// So every field or record terminator is replaced: ',' would shift columns,
/// '"' opens a quote on a quoting reader, '\r' splits a row on a CRLF-aware one.
inline std::string csv_text(std::string s) {
    std::replace(s.begin(), s.end(), ',', ';');
    std::replace(s.begin(), s.end(), '"', '\'');
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
    return s;
}

/// A trace number at comparison.csv's ten significant digits: the stream
/// default of six rounds a six-figure UC objective to the unit, and whether the
/// incumbent flattened is read in the last digits. Local, not a sticky stream
/// precision that would reformat the caller's other output.
inline std::string trace_num(double v) {
    std::ostringstream os;
    os.precision(10);
    os << v;
    return os.str();
}

/// The trace's column set, pinned by tests/test_uc_chped_trace.cpp against
/// what TraceRecorder writes: a header that misnames the rows is worse than none.
inline constexpr const char* kTraceHeader =
    "instance,periods,time_limit_s,time_seconds,batches,objective,new_best,commit_sha";

/// Writes one row per progress report that has an incumbent, plus one closing
/// row per solve (record_final), each keyed by its (instance, periods) solve.
/// With record_final called, every solve appears and its last row is at the
/// run's final time.
class TraceRecorder : public cbls::SolveCallback {
public:
    TraceRecorder(std::ostream& out, std::string instance, int periods, double time_limit_s,
                  std::string commit_sha)
        : out_(out),
          instance_(csv_text(std::move(instance))),
          periods_(periods),
          time_limit_s_(time_limit_s),
          commit_sha_(csv_text(std::move(commit_sha))) {}

    void on_progress(const cbls::SolveProgress& p) override {
        // The WHOLE predicate is "incumbent with a value": `p.objective` is
        // `best_feasible_obj_`, +inf until the first finite-objective incumbent
        // (+inf with a witness is an overflowed objective, #100; "inf" in a
        // numeric column would read as a result). Must NOT also test
        // `p.feasible` as minlplib's recorder does: that is real_feasible() on
        // the CURRENT assignment, which leaves the incumbent once the bound
        // tightens, so it would drop the ~1s samples of the flat tail #147
        // collects.
        last_batches_ = p.iteration;
        if (!std::isfinite(p.objective)) {
            return;
        }
        // Flushed per row (write_row): a full run is over an hour, and an
        // interrupted one must keep the profile that judges its budget.
        write_row(p.time_seconds, p.iteration, trace_num(p.objective), p.new_best);
    }

    /// One closing row per solve; the runner calls it after `solve()` returns.
    ///
    /// Periodic reports come at most once a second and `record_best` resets
    /// that timer, so without this row every horizon's trace stopped 0.65-1.45s
    /// short of its budget (measured, #147), and one that improved inside that
    /// window ends on a rising incumbent that reads as "still improving". A solve with no valued
    /// incumbent would otherwise write no rows, indistinguishable from a filtered roster or a
    /// `nullptr` callback; this row states it with an EMPTY objective cell, which no periodic row
    /// has. Hence `max(time_seconds)` per (instance, periods) is the run's end, and `new_best` is
    /// 0: the row reports where the budget left the search.
    ///
    /// `batches` is the last batch count `on_progress` saw, NOT
    /// `SearchResult::iterations` (the GLS count), so the column means one
    /// thing on every row; empty when the solve never reported.
    void record_final(const cbls::SearchResult& result) {
        const std::string objective =
            std::isfinite(result.objective) ? trace_num(result.objective) : std::string();
        write_row(result.time_seconds, last_batches_, objective, /*new_best=*/false);
    }

private:
    void write_row(double time_seconds, int64_t batches, const std::string& objective,
                   bool new_best) {
        out_ << instance_ << "," << periods_ << "," << trace_num(time_limit_s_) << ","
             << trace_num(time_seconds) << ",";
        if (batches >= 0) {
            out_ << batches;
        }
        out_ << "," << objective << "," << (new_best ? 1 : 0) << "," << commit_sha_ << '\n';
        out_.flush();
    }

    std::ostream& out_;
    std::string instance_;
    int periods_;
    double time_limit_s_;
    std::string commit_sha_;
    /// Last batch count seen, or -1 when `solve()` never reported progress.
    int64_t last_batches_ = -1;
};

}  // namespace cbls::uc_chped
