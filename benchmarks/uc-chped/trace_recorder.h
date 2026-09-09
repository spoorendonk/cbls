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
// `new_best` column, same one-row-per-report writer. Four columns are added,
// each because this benchmark cannot answer its question without them:
//
//   * `periods`. This runner solves one row per (instance, horizon) pair, not
//     one per instance, so an `instance` column alone cannot say which solve a
//     trace row belongs to. Two rows for `ucp13` would otherwise be
//     indistinguishable.
//   * `time_limit_s`. The question is whether the incumbent flattened BEFORE
//     THE BUDGET, so a trace that does not carry the budget is not self-
//     contained -- and the budget map is exactly what these traces exist to
//     change, so an archived trace would stop stating what it was measured
//     under the moment the map is edited.
//   * `batches`. `SolveProgress::iteration`, the ViolationLS batch count (NOT
//     comparison.csv's `iterations`, which is the GLS count). It separates a
//     flat tail that means "converged" from one that means "barely got going",
//     which are opposite arguments about the budget.
//   * `commit_sha`. Issue #147's Rules require the engine commit on every row,
//     for the same reason comparison.csv carries it: a search-trajectory change
//     silently invalidates a trace, and without the commit the next reader
//     cannot tell drift from a bug.
//
// It also differs from minlplib's in ways that are not the column set: it takes
// an `ostream&` rather than an `ofstream&` (which is what lets a test drive it
// into a stringstream), it flushes each row, it runs the identifier cells
// through csv_text, and it formats to ten significant digits rather than the
// stream default of six. Each is argued at its own definition below.
//
// The header lives here rather than inside uc_chped.cpp so a Catch2 test can
// drive the recorder through a real solve() without a runner main()
// (tests/test_uc_chped_trace.cpp). That test pins the recorder's CONTRACT, but
// it cannot see the runner -- uc_chped.cpp is a main() with no header, so the
// test builds its own solve() call and stays green if the runner's argument
// reverts. What stops the RUNNER going back to the `nullptr` this issue exists
// to fix is tests/python/test_cli.py's test_uc_chped_records_an_anytime_trace,
// which runs the binary and requires rows rather than a bare header.

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
    "instance,periods,time_limit_s,time_seconds,batches,objective,new_best,commit_sha";

/// Writes one row per progress report that has an incumbent, identifying the
/// (instance, horizon) solve it belongs to.
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
        // "Is there an incumbent with a value yet?", and that is the WHOLE
        // predicate: `p.objective` is the search's `best_feasible_obj_`, which
        // is +inf until the first finite-objective incumbent is recorded and
        // finite from then on. The +inf case is the feasibility witness whose
        // objective overflowed (#100) -- feasible, but with no value to plot,
        // and an "inf" in a numeric column would read as a solve result.
        //
        // Deliberately NOT also `!p.feasible`, which minlplib's recorder tests:
        // that field is real_feasible() on the CURRENT assignment, not on the
        // incumbent, and after every improvement the objective bound is
        // tightened below the incumbent and the search moves off the feasible
        // point it just recorded. Testing it would drop the ~1s periodic
        // samples whenever the search happened to be mid-flight -- deleting the
        // flat tail, which is the one thing #147 exists to collect. It would
        // also make the last row "the last instant the search was incidentally
        // feasible" rather than "where the budget left it".
        if (!std::isfinite(p.objective)) {
            return;
        }
        // Flushed per row, as the results CSV is: a full run is over an hour and
        // an interrupted one must not lose its buffered profile -- which is the
        // half of the record that says whether a budget was long enough.
        out_ << instance_ << "," << periods_ << "," << trace_num(time_limit_s_) << ","
             << trace_num(p.time_seconds) << "," << p.iteration << "," << trace_num(p.objective)
             << "," << (p.new_best ? 1 : 0) << "," << commit_sha_ << '\n';
        out_.flush();
    }

private:
    std::ostream& out_;
    std::string instance_;
    int periods_;
    double time_limit_s_;
    std::string commit_sha_;
};

}  // namespace cbls::uc_chped
