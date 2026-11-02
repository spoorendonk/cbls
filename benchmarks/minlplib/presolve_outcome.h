#pragma once

// What the linear-part presolve (#342) did to one `.nl` build, as one line of
// text: `presolve=<status> rows=<removed> cols=<removed> aggregated=<cols>
// bounds=<tightened> seconds=<presolve time>`. Shared by cbls_minlplib (printed
// per instance under --presolve, #343) and cbls_minlplib_nl_check --presolve
// (the census), so both name an outcome alike. A doubleton-equality
// aggregation (#328) removes one column and its row, so it counts in `rows=`
// and `cols=`; `aggregated=` says how many of those it was (#347).

#include <array>
#include <cbls/io_nl.h>
#include <cstdio>
#include <string>

namespace cbls::bench {

/// `built` must come from `nl_to_model` with presolve enabled. Statuses:
/// `reduced` and `unchanged` (presolve ran), `infeasible` (presolve proved the
/// linear part infeasible; the plain model was built), `rejected` (presolve
/// refused the input; the plain model was built), and `not-run` (the linear
/// part could not be handed to presolve, e.g. the objective's linear part names
/// a non-column; the plain model was built). The last two carry zero counts;
/// `infeasible` carries what presolve did before it found the conflict.
inline std::string presolve_outcome(const NlToModelResult& built) {
    const PresolveStats& s = built.presolve_stats;
    const char* status = "not-run";
    if (!built.presolve_rejected.empty()) {
        status = "rejected";
    } else if (s.status == PresolveStatus::Infeasible) {
        status = "infeasible";
    } else if (built.postsolve.n_original() != 0) {
        status = s.status == PresolveStatus::Reduced ? "reduced" : "unchanged";
    }
    const int rows =
        s.rows_empty + s.rows_free + s.rows_singleton + s.rows_redundant + s.cols_aggregated;
    const int cols = s.cols_fixed + s.cols_empty + s.cols_aggregated;
    std::array<char, 32> secs{};
    std::snprintf(secs.data(), secs.size(), "%.3g", s.seconds);
    return std::string("presolve=") + status + " rows=" + std::to_string(rows) +
           " cols=" + std::to_string(cols) + " aggregated=" + std::to_string(s.cols_aggregated) +
           " bounds=" + std::to_string(s.bounds_tightened) + " seconds=" + secs.data();
}

}  // namespace cbls::bench
