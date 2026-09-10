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
#include <cstdint>
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

/// Writes one row per progress report that has an incumbent, plus one closing
/// row per solve, identifying the (instance, horizon) solve each belongs to.
///
/// Every solve the runner starts therefore appears in the trace, and the last
/// row for a (instance, periods) pair is always the one at the run's final
/// time -- see record_final for why the file is unreadable without both
/// guarantees.
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
        last_batches_ = p.iteration;
        if (!std::isfinite(p.objective)) {
            return;
        }
        // Flushed per row, as the results CSV is: a full run is over an hour and
        // an interrupted one must not lose its buffered profile -- which is the
        // half of the record that says whether a budget was long enough.
        write_row(p.time_seconds, p.iteration, trace_num(p.objective), p.new_best);
    }

    /// One closing row per solve, written after `solve()` returns.
    ///
    /// Without it the trace answers the wrong question twice over.
    ///
    ///   1. The engine emits a periodic report at most once a second and
    ///      `record_best` resets that timer, so an improvement inside the final
    ///      second is the last thing in the file and the record then ends on a
    ///      rising incumbent with budget apparently left -- which reads as
    ///      "still improving when the clock stopped" when what actually
    ///      followed was a flat second nobody sampled. Measured before this
    ///      existed: every horizon's last row fell 0.65-1.45s short of its
    ///      budget, and on one the last row WAS an improvement.
    ///   2. A solve that never finds a valued incumbent writes no rows at all,
    ///      and an absent (instance, horizon) is indistinguishable from a
    ///      filtered roster, an interrupted campaign, or the callback
    ///      regressing to the `nullptr` this issue exists to remove. "There was
    ///      never anything to flatten" is a real answer to #147's question and
    ///      the trace has to be able to state it: the row is written with an
    ///      EMPTY objective cell, which no periodic row can produce.
    ///
    /// So every solve appears in the trace exactly once at its final time,
    /// `max(time_seconds)` per (instance, periods) is the end of the run rather
    /// than the last thing that happened to be sampled, and `new_best` is 0
    /// because this row reports where the budget left the search, not an
    /// improvement.
    ///
    /// `batches` carries the last figure `on_progress` saw, NOT
    /// `SearchResult::iterations` -- that is the GLS iteration count, a
    /// different quantity from this column's batch count, and putting it here
    /// would silently change what the column means on exactly one row per
    /// solve. A solve that never reported leaves the cell empty.
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
