#pragma once

#include <benchmarks/common/runner_args.h>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

namespace cbls::bench {

/// The published artifact a run's `--out` or `--trace` resolves to.
struct PublishedTarget {
    bool table = true;  ///< the comparison table; otherwise the anytime trace
    std::string path;

    [[nodiscard]] const char* artifact() const { return table ? "table" : "anytime trace"; }
    /// The flag that redirects it: a trace-only run told to "pass --out
    /// elsewhere" would be sent to change a path it never set.
    [[nodiscard]] const char* redirect() const { return table ? "--out" : "--trace"; }
};

/// The guards every runner that publishes `<inst_dir>/comparison.csv` and
/// `<inst_dir>/anytime_trace.csv` shares. `out` is `--out` already
/// defaulted, `trace` is `--trace` or empty, `why` is the first flag that
/// makes the run other than the published protocol (nullptr when none).
/// Reports and exits 2 on a refusal; otherwise returns the published artifact
/// the run writes, or nullopt when it writes none. A runner adds its own
/// further conditions on the returned target.
///
/// The guards compare the FILES (`same_file`), not which flags were typed:
/// the documented command passes `--out <the published table>` explicitly,
/// and an emptiness test let it rewrite the table from a one-instance run.
/// Both names are reserved whether or not a file exists there yet.
inline std::optional<PublishedTarget> check_published_write(const std::string& out,
                                                            const std::string& trace,
                                                            const std::string& inst_dir,
                                                            const char* why, bool commit_set) {
    const std::string published = inst_dir + "/comparison.csv";
    const std::string published_trace = inst_dir + "/anytime_trace.csv";
    const bool trace_set = !trace.empty();
    // One path for both flags: the table's closing rename lands on the trace.
    if (trace_set && same_file(out, trace)) {
        std::fprintf(stderr,
                     "--out and --trace name the same file %s; the table would replace it\n",
                     out.c_str());
        std::exit(2);
    }
    // The table is written to `<out>.tmp` and renamed into place, so a trace
    // there puts two truncating writers on one inode and publishes the
    // interleave as the table.
    if (trace_set && same_file(trace, out + ".tmp")) {
        std::fprintf(stderr,
                     "--trace names the table's temp file %s; the two writers would interleave "
                     "and the result would be published as the table\n",
                     trace.c_str());
        std::exit(2);
    }
    // A CROSSED artifact is refused for every run, ahead of any question of
    // protocol: only `--out` writes the table and only `--trace` the trace.
    // `--trace` truncates on open, so `--trace <the table>` on a legitimate
    // published run would empty the table before any solve.
    if (trace_set && same_file(trace, published)) {
        std::fprintf(stderr, "--trace cannot write the published table %s (only --out writes it)\n",
                     published.c_str());
        std::exit(2);
    }
    if (same_file(out, published_trace)) {
        std::fprintf(stderr,
                     "--out cannot write the published anytime trace %s (only --trace writes it)\n",
                     published_trace.c_str());
        std::exit(2);
    }
    const bool writes_table = same_file(out, published);
    const bool writes_trace = trace_set && same_file(trace, published_trace);
    if (!writes_table && !writes_trace) {
        return std::nullopt;
    }
    PublishedTarget target{writes_table, writes_table ? published : published_trace};
    if (why != nullptr) {
        std::fprintf(stderr,
                     "%s cannot write the published %s %s "
                     "(pass %s elsewhere for an unpublished run)\n",
                     why, target.artifact(), target.path.c_str(), target.redirect());
        std::exit(2);
    }
    // "unknown" provenance is one a later reader cannot tell engine drift
    // from a bug with; every trace row carries the commit too.
    if (!commit_set) {
        std::fprintf(stderr,
                     "writing %s requires an explicit --commit SHA "
                     "(pass %s elsewhere for an unpublished run)\n",
                     target.path.c_str(), target.redirect());
        std::exit(2);
    }
    return target;
}

}  // namespace cbls::bench
