#pragma once

// The anytime trace for the UC-CHPED runner (#147).
//
// The published output of this benchmark is a gap percentage measured at a
// wall-clock budget. Without a record of what the incumbent was doing when the
// clock stopped, a bad number is uninterpretable in either direction: an engine
// limitation and a budget that was simply too short look identical. This is the
// record.
//
// Shape follows benchmarks/minlplib/minlplib.cpp's TraceRecorder -- same
// feasible-incumbent filter, same `new_best` column -- and deliberately differs
// in exactly two places, both forced:
//
//   * `periods`. This runner solves one row per (instance, horizon) pair, not
//     one per instance, so an `instance` column alone cannot say which solve a
//     trace row belongs to. Two rows for `ucp13` would otherwise be
//     indistinguishable.
//   * `commit_sha`. Issue #147's Rules require the engine commit on every row,
//     for the same reason comparison.csv carries it: a search-trajectory change
//     silently invalidates a trace, and without the commit the next reader
//     cannot tell drift from a bug.
//
// The header lives here rather than inside uc_chped.cpp so a Catch2 test can
// drive the recorder through a real solve() without a runner main() --
// tests/test_uc_chped_trace.cpp is what stops the callback quietly going back
// to the `nullptr` this issue exists to fix.

#include <algorithm>
#include <cbls/search.h>
#include <cmath>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace cbls::uc_chped {

/// The CSV cell rule this benchmark's two output files share. A comma inside a
/// free-text cell would shift every column after it, so the writer substitutes
/// rather than quoting: these cells are short diagnostics and identifiers, and a
/// quoted field would need escaping rules the readers of these tables (grep,
/// awk, a spreadsheet import) do not all implement.
///
/// Nothing here quotes, so every character that would end a field or a record
/// has to be substituted rather than escaped -- including '"', which a reader
/// that does honour quoting would otherwise treat as opening one, and '\r',
/// which turns a row into two on a CRLF-aware reader.
inline std::string csv_text(std::string s) {
    std::replace(s.begin(), s.end(), ',', ';');
    std::replace(s.begin(), s.end(), '"', '\'');
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
    return s;
}

/// A trace number at the same precision comparison.csv publishes its objective
/// with. The stream default is six significant digits, which rounds a
/// six-figure UC objective to the nearest unit -- and "did the incumbent stop
/// improving before the budget expired?" is exactly the question that reads the
/// last few digits. Formatted locally rather than by setting precision on the
/// stream, which would be sticky and would silently reformat whatever else the
/// caller writes to it.
inline std::string trace_num(double v) {
    std::ostringstream os;
    os.precision(10);
    os << v;
    return os.str();
}

/// The trace's column set. Pinned by tests/test_uc_chped_trace.cpp against what
/// TraceRecorder actually writes, because a header that names one set of columns
/// while the rows carry another is worse than no trace at all.
inline constexpr const char* kTraceHeader =
    "instance,periods,time_seconds,objective,new_best,commit_sha";

/// Writes one row per progress report that has an incumbent, identifying the
/// (instance, horizon) solve it belongs to.
class TraceRecorder : public cbls::SolveCallback {
public:
    TraceRecorder(std::ostream& out, std::string instance, int periods, std::string commit_sha)
        : out_(out),
          instance_(csv_text(std::move(instance))),
          periods_(periods),
          commit_sha_(csv_text(std::move(commit_sha))) {}

    void on_progress(const cbls::SolveProgress& p) override {
        // No incumbent yet, or a feasibility witness whose objective overflowed
        // (SearchResult documents `feasible` with a non-finite objective, #100).
        // Either way there is no incumbent VALUE to plot against time, and an
        // "inf" in a numeric column would read as a solve result.
        if (!p.feasible || !std::isfinite(p.objective)) {
            return;
        }
        // Flushed per row, as the results CSV is: a full run is over an hour and
        // an interrupted one must not lose its buffered profile -- which is the
        // half of the record that says whether a budget was long enough.
        out_ << instance_ << "," << periods_ << "," << trace_num(p.time_seconds) << ","
             << trace_num(p.objective) << "," << (p.new_best ? 1 : 0) << "," << commit_sha_ << '\n';
        out_.flush();
    }

private:
    std::ostream& out_;
    std::string instance_;
    int periods_;
    std::string commit_sha_;
};

}  // namespace cbls::uc_chped
