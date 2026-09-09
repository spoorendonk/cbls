#pragma once

#include <array>
#include <benchmarks/common/runner_args.h>
#include <cbls/inner_solver.h>
#include <cbls/lns.h>
#include <cbls/search.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace cbls::bench {

// One ablation arm of the search configuration, as typed on the command line.
//
// Why this is shared rather than written twice: an arm is only an experiment if
// the row it produced says which arm it was, and the row and `--help` cannot be
// allowed to drift from what the solve actually received. So the flag table,
// the parse, the validation, the application onto `SearchConfig`/`solve()`'s
// arguments and the canonical `search_config` cell all live here, and the
// runners hold none of it (#136).
//
// WHAT IS DELIBERATELY ABSENT. A flag that is accepted, recorded on the row and
// then ignored by the engine is worse than no flag: it produces an ablation that
// measures nothing while looking rigorous. Two such knobs were considered and
// left out, each verified against the engine rather than against its
// documentation:
//
//   * the GLS weight-decay parameter rho -- `solve()` resamples it at loop
//     entry, at every stagnation and on every new best, and `SearchConfig` has
//     no field for it at all (the case #136 is named after).
//   * `use_fj` -- both `solve()`'s argument and `SearchConfig::use_fj`. The
//     first line of `solve()` in src/search.cpp is `(void)use_fj;` with the
//     comment "GFJ is always the engine now; the flag is vestigial", and
//     `config.use_fj` is read only by `ParallelSearch`, which no benchmark
//     runner uses. `--fj`/`--no-fj` would therefore have been exactly the rho
//     trap: the two runners already pass opposite values (minlplib true,
//     uc-chped false) and get the same engine.
//
// Adding a flag here means checking the same thing: that src/search.cpp or
// src/feasibility_jump.cpp actually reads the field, and that a test can show
// two values of it producing two different runs (tests/test_bench_flags.cpp).
struct SearchFlags {
    // Defaults are taken from the engine rather than restated, so that a change
    // to `SearchConfig`'s defaults cannot silently make "default" on the row
    // mean something the engine no longer does. The two booleans that are not
    // `SearchConfig` fields (the hook and LNS are `solve()` pointer arguments)
    // default to what both runners pass today: on.
    bool float_hook = true;
    bool lns = true;
    // Held as int64_t even where the engine field is `int`: parse_int64 is the
    // shared integer policy, and range-checking once in validate_search_flags()
    // is what keeps a narrowing conversion out of the apply step.
    int64_t lns_interval = SearchConfig{}.lns_interval;
    bool compound_moves = SearchConfig{}.use_compound_moves;
    double novelty_prob = SearchConfig{}.novelty_jump_probability;
    int64_t unproductive_iters = SearchConfig{}.unproductive_iterations;
    int64_t perturbation_period = SearchConfig{}.perturbation_period;
    int64_t max_iterations = SearchConfig{}.max_iterations;
    // Not a `SearchConfig` field: `solve()` disables its wall clock when handed
    // `time_limit <= 0`, and both runners guard their `--time-limit` as
    // strictly positive. This is the way to ask for the iteration-budgeted,
    // clock-free arm without loosening that guard.
    bool no_time_limit = false;
};

/// One flag's contribution to the two strings that must not drift apart: the
/// key it takes in the row's `search_config` cell, and its `--help` fragment.
struct SearchFlagSpec {
    const char* key;
    const char* usage;
};

/// The flag table. Order is the canonical order of the `search_config` cell and
/// of the usage line; both are generated from this array, so a flag cannot be
/// documented and unrecorded (or the reverse), and the cell's key order is
/// deterministic by construction rather than by convention.
inline constexpr std::array<SearchFlagSpec, 9> kSearchFlagSpecs = {{
    {"float_hook", "[--no-float-hook]"},
    {"lns", "[--no-lns]"},
    {"lns_interval", "[--lns-interval N]"},
    {"compound_moves", "[--compound-moves|--no-compound-moves]"},
    {"novelty_prob", "[--novelty-prob P]"},
    {"unproductive_iters", "[--unproductive-iters N]"},
    {"perturbation_period", "[--perturbation-period N]"},
    {"max_iterations", "[--max-iterations N]"},
    {"time_limit", "[--no-time-limit]"},
}};

inline constexpr std::size_t kSearchFlagCount = kSearchFlagSpecs.size();

/// Matches and consumes one search-configuration flag under the cursor.
///
/// Returns false when the argument is not one of ours, leaving the cursor where
/// it was, so a runner chains this into its own `else if` ladder ahead of the
/// unknown-option branch. Value flags go through `ArgCursor::value_flag`, which
/// means a trailing `--lns-interval` with no value does NOT match here and falls
/// through to that branch -- the rule runner_args.h exists to state.
///
/// Bad values follow the runners' shared policy unchanged: a non-integer reports
/// and exits 2 inside parse_int64, a non-number double becomes NaN and is caught
/// by validate_search_flags() below.
inline bool match_search_flag(ArgCursor& c, const std::string& s, SearchFlags& f) {
    const char* v = nullptr;
    if (s == "--no-float-hook") {
        f.float_hook = false;
    } else if (s == "--no-lns") {
        f.lns = false;
    } else if (c.value_flag("--lns-interval", v)) {
        f.lns_interval = parse_int64("--lns-interval", v);
    } else if (s == "--compound-moves") {
        f.compound_moves = true;
    } else if (s == "--no-compound-moves") {
        f.compound_moves = false;
    } else if (c.value_flag("--novelty-prob", v)) {
        f.novelty_prob = parse_double("--novelty-prob", v);
    } else if (c.value_flag("--unproductive-iters", v)) {
        f.unproductive_iters = parse_int64("--unproductive-iters", v);
    } else if (c.value_flag("--perturbation-period", v)) {
        f.perturbation_period = parse_int64("--perturbation-period", v);
    } else if (c.value_flag("--max-iterations", v)) {
        f.max_iterations = parse_int64("--max-iterations", v);
    } else if (s == "--no-time-limit") {
        f.no_time_limit = true;
    } else {
        return false;
    }
    return true;
}

/// Range and coherence checks over the parsed flags.
///
/// Returns false with a one-line reason for the runner to report; the runner
/// owns the exit, as it does for every other flag value (runner_args.h). Kept
/// out of the parse so that it is callable from a test without ending the
/// process.
///
/// `time_limit_set` says whether the runner also saw an explicit `--time-limit`,
/// which only this layer can turn into the conflict it is.
inline bool validate_search_flags(const SearchFlags& f, bool time_limit_set, std::string& error) {
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    if (f.lns_interval < 1 || f.lns_interval > kIntMax) {
        // 0 would disable LNS through a second, undocumented route (solve()
        // skips the repair when lns_interval <= 0), which would make two
        // different `search_config` cells mean the same run. `--no-lns` is the
        // one spelling for that arm.
        error = "--lns-interval must be >= 1 (use --no-lns to switch LNS off)";
        return false;
    }
    // NaN is tested first and by name. parse_double returns NaN for a value
    // that is not a number, and NaN compares false against every bound, so a
    // range test alone would ACCEPT the typo it is meant to catch. (Written as
    // `!(p >= 0 && p <= 1)` instead, the NaN case is covered but
    // readability-simplify-boolean-expr asks for a DeMorgan rewrite that
    // silently loses it -- so the check is right about the form and wrong about
    // the meaning, and this spelling satisfies both.)
    if (std::isnan(f.novelty_prob) || f.novelty_prob < 0.0 || f.novelty_prob > 1.0) {
        error = "--novelty-prob must be in [0, 1]";
        return false;
    }
    if (f.perturbation_period < 1 || f.perturbation_period > kIntMax) {
        error = "--perturbation-period must be >= 1";
        return false;
    }
    if (f.max_iterations < 0) {
        error = "--max-iterations must be >= 0 (0 = unlimited)";
        return false;
    }
    // No positivity guard on --unproductive-iters, deliberately: SearchConfig
    // documents `<= 0` as "restore the old fixed cadence", which is a
    // legitimate arm and the control case for the #102 exit. Only a
    // non-integer is rejected, and parse_int64 has already done that.
    if (f.no_time_limit && time_limit_set) {
        error = "--no-time-limit and --time-limit contradict each other";
        return false;
    }
    if (f.no_time_limit && f.max_iterations <= 0) {
        // solve() with neither budget returns immediately (TerminationReason::
        // NoBudget), so this combination would publish a full-looking table of
        // empty results -- the same hazard the runners' `--time-limit > 0`
        // guards exist for.
        error = "--no-time-limit requires --max-iterations N with N > 0";
        return false;
    }
    return true;
}

/// The first flag whose value is not the engine default, or nullptr when the
/// arm is the default configuration.
///
/// This is what lets a runner's published-table guard treat an ablation arm the
/// way it treats a shortened budget or a partial roster: a table generated under
/// a non-default search configuration is not the published measurement, whatever
/// else was passed.
inline const char* first_non_default_search_flag(const SearchFlags& f) {
    const SearchFlags d;
    if (f.float_hook != d.float_hook) {
        return "--no-float-hook";
    }
    if (f.lns != d.lns) {
        return "--no-lns";
    }
    if (f.lns_interval != d.lns_interval) {
        return "--lns-interval";
    }
    if (f.compound_moves != d.compound_moves) {
        return "--compound-moves";
    }
    if (f.novelty_prob != d.novelty_prob) {
        return "--novelty-prob";
    }
    if (f.unproductive_iters != d.unproductive_iters) {
        return "--unproductive-iters";
    }
    if (f.perturbation_period != d.perturbation_period) {
        return "--perturbation-period";
    }
    if (f.max_iterations != d.max_iterations) {
        return "--max-iterations";
    }
    if (f.no_time_limit != d.no_time_limit) {
        return "--no-time-limit";
    }
    return nullptr;
}

/// A double as the `search_config` cell spells it: `%g`, which is stable across
/// runs (nothing here calls setlocale) and short enough to read in a table.
inline std::string search_flag_number(double v) {
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%g", v);
    return {buf.data()};
}

/// The flags' values in `kSearchFlagSpecs` order. Held apart from the string so
/// that the keys and the values are zipped from one array rather than written
/// out twice in the same order and hoped over.
inline std::array<std::string, kSearchFlagCount> search_config_values(const SearchFlags& f) {
    const char* const on = "on";
    const char* const off = "off";
    return {{
        f.float_hook ? on : off,
        f.lns ? on : off,
        std::to_string(f.lns_interval),
        f.compound_moves ? on : off,
        search_flag_number(f.novelty_prob),
        std::to_string(f.unproductive_iters),
        std::to_string(f.perturbation_period),
        std::to_string(f.max_iterations),
        f.no_time_limit ? off : on,
    }};
}

/// The canonical `key=value;key=value` cell recorded on every row.
///
/// Deterministic, comma-free (so it needs no CSV quoting) and emitted in one
/// fixed order for every runner, so two rows produced by the same arm compare
/// equal as strings and `sort | uniq -c` over a results file is a census of the
/// arms it holds.
inline std::string search_config_string(const SearchFlags& f) {
    const auto values = search_config_values(f);
    std::string out;
    for (std::size_t i = 0; i < kSearchFlagCount; ++i) {
        if (i > 0) {
            out += ';';
        }
        out += kSearchFlagSpecs[i].key;
        out += '=';
        out += values[i];
    }
    return out;
}

/// The `--help` fragment for these flags, generated from the same table as the
/// cell above.
inline std::string search_flags_usage() {
    std::string out;
    for (const auto& spec : kSearchFlagSpecs) {
        if (!out.empty()) {
            out += ' ';
        }
        out += spec.usage;
    }
    return out;
}

/// Writes the flags onto the search configuration. Call validate_search_flags()
/// first: the narrowing casts here are safe only because it has range-checked
/// the two fields the engine holds as `int`.
inline void apply_search_flags(const SearchFlags& f, SearchConfig& cfg) {
    cfg.use_compound_moves = f.compound_moves;
    cfg.novelty_jump_probability = f.novelty_prob;
    cfg.unproductive_iterations = f.unproductive_iters;
    cfg.perturbation_period = static_cast<int>(f.perturbation_period);
    cfg.max_iterations = f.max_iterations;
}

/// The remaining four are `solve()` arguments rather than `SearchConfig` fields,
/// so they are applied at the call rather than onto the config.
inline InnerSolverHook* hook_argument(const SearchFlags& f, InnerSolverHook& hook) {
    return f.float_hook ? &hook : nullptr;
}

inline LNS* lns_argument(const SearchFlags& f, LNS& lns) {
    return f.lns ? &lns : nullptr;
}

inline int lns_interval_argument(const SearchFlags& f) {
    return static_cast<int>(f.lns_interval);
}

/// The wall-clock budget `solve()` should be handed: 0 disables its clock, which
/// is what `--no-time-limit` asks for.
inline double time_limit_argument(const SearchFlags& f, double time_limit) {
    return f.no_time_limit ? 0.0 : time_limit;
}

}  // namespace cbls::bench
