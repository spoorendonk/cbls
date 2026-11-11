#pragma once

#include <array>
#include <benchmarks/common/runner_args.h>
#include <cbls/inner_solver.h>
#include <cbls/lns.h>
#include <cbls/move_generator.h>
#include <cbls/search.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>

namespace cbls::bench {

// One ablation arm of the search configuration, as typed on the command line.
// Flag table, parse, validation, application onto `SearchConfig`/`solve()` and
// the `search_config` cell all live here, so the row and `--help` cannot drift
// from what the solve received (#136).
//
// A flag must be read by the engine, else its arm measures nothing while
// looking rigorous. Hence no flag for GLS rho (`solve()` resamples it; no
// `SearchConfig` field, #136) nor for `use_fj` (`solve()` ignores it,
// src/search.cpp; `ParallelSearch` only forwards it, src/pool.cpp). A new flag
// needs src/search.cpp or src/feasibility_jump.cpp to read its field and a
// test showing two values give two runs (tests/test_bench_flags.cpp).
//
// Whether a flag is read depends on the model too (#198): structural flags are
// dead on a scalar model (`effective_structural_probability` is 0.0 without a
// List/Set generator), FJ-side flags on a pure-List one. So each flag has a
// family, each runner declares its families, and validate_search_flags()
// refuses the rest. The cell and usage line carry only declared families,
// which keeps a scalar runner's cell byte-identical to its committed tables.
//
// The families, by what reads them:
//   * common     -- the diversification loop every model runs: LNS, the kick
//                   cadence, the budgets.
//   * scalar     -- FJ and its satellites: the float hook (continuous scalars),
//                   compound/Novelty Jump batches, and FJ's unproductive-batch
//                   exit (`gfj.unproductive_iterations`, read nowhere else).
//   * structural -- the structural batch's probability, selection policy and
//                   sample size.
//   * neighbours -- `--neighbours K`. Its own family because the flag carries
//                   only K: the cost the K nearest are ranked by is the
//                   runner's, so a runner that supplies no cost must not
//                   declare it, and the flag is then refused.
inline constexpr unsigned kCommonSearchFlags = 1U << 0U;
inline constexpr unsigned kScalarSearchFlags = 1U << 1U;
inline constexpr unsigned kStructuralSearchFlags = 1U << 2U;
inline constexpr unsigned kNeighbourSearchFlags = 1U << 3U;
inline constexpr std::size_t kSearchFlagFamilyCount = 4;
/// What a runner over scalar models (minlplib, uc-chped, and mipfeas for the
/// five shared flags it takes, #267) declares.
inline constexpr unsigned kScalarRunnerSearchFlags = kCommonSearchFlags | kScalarSearchFlags;

/// The family's spelling in a refusal.
inline const char* search_flag_family_name(std::size_t family_index) {
    switch (family_index) {
        case 0:
            return "common";
        case 1:
            return "scalar";
        case 2:
            return "structural";
        default:
            return "neighbour";
    }
}

struct SearchFlags {
    // Defaults come from `SearchConfig{}`, so "default" on the row tracks the
    // engine. The hook and LNS are `solve()` pointer arguments, not fields:
    // they default to on, as every runner passes.
    bool float_hook = true;
    bool lns = true;
    // int64_t even where the engine field is `int`: validate_search_flags()
    // range-checks once, so the apply step never narrows unchecked.
    int64_t lns_interval = SearchConfig{}.lns_interval;
    bool compound_moves = SearchConfig{}.use_compound_moves;
    double novelty_prob = SearchConfig{}.novelty_jump_probability;
    int64_t unproductive_iters = SearchConfig{}.unproductive_iterations;
    int64_t perturbation_period = SearchConfig{}.perturbation_period;
    int64_t max_iterations = SearchConfig{}.max_iterations;
    // Not a `SearchConfig` field: passes `time_limit <= 0` (no wall clock) to
    // `solve()` for the iteration-budgeted arm, without loosening the runners'
    // strictly positive `--time-limit` guard.
    bool no_time_limit = false;
    // The structural batch (#198). Negative is the engine's "automatic": the
    // default, not typeable, written `auto` in the cell.
    double structural_prob = SearchConfig{}.structural_batch_probability;
    StructuralSelection structural_selection = SearchConfig{}.structural_selection;
    int64_t structural_sample_size = SearchConfig{}.structural_sample_size;
    // 0 = no neighbour list, the engine's default (a null
    // `structural_neighbours`). Typed values must be >= 1.
    int64_t neighbours = 0;
    // Objective tie-break (#218), on by default since #226; the cell records it
    // only when off, so default rows before and after #226 differ only by commit.
    bool objective_tie_break = SearchConfig{}.fj_objective_tie_break;
    // First flag typed per family (indexed by family bit), or nullptr. Set
    // whatever the value, so a flag restated at its default is still refused
    // on a runner that cannot read it.
    std::array<const char*, kSearchFlagFamilyCount> typed{};
};

/// One flag's contribution to the two strings that must not drift apart: the
/// key it takes in the row's `search_config` cell, and its `--help` fragment.
struct SearchFlagSpec {
    const char* key;
    const char* usage;
    unsigned family;
    // Opt-in: written to the cell only when non-default, so a flag can join a
    // family with committed tables and keep their default cell byte-identical
    // (#226).
    bool opt_in = false;
};

/// The flag table, in the canonical order of the cell and the usage line; both
/// are generated from it, so no flag is documented but unrecorded or reverse.
/// New flags go at the END: a cell is this table filtered by family, so
/// appending keeps every existing runner's cell byte-identical.
inline constexpr std::array<SearchFlagSpec, 14> kSearchFlagSpecs = {{
    {"float_hook", "[--no-float-hook]", kScalarSearchFlags},
    {"lns", "[--no-lns]", kCommonSearchFlags},
    {"lns_interval", "[--lns-interval N]", kCommonSearchFlags},
    {"compound_moves", "[--compound-moves|--no-compound-moves]", kScalarSearchFlags},
    {"novelty_prob", "[--novelty-prob P]", kScalarSearchFlags},
    {"unproductive_iters", "[--unproductive-iters N]", kScalarSearchFlags},
    {"perturbation_period", "[--perturbation-period N]", kCommonSearchFlags},
    {"max_iterations", "[--max-iterations N]", kCommonSearchFlags},
    {"time_limit", "[--no-time-limit]", kCommonSearchFlags},
    {"structural_prob", "[--structural-prob P]", kStructuralSearchFlags},
    {"structural_selection",
     "[--structural-selection auto|first_improving|best_of_sample|violation_guided|granular_best]",
     kStructuralSearchFlags},
    {"structural_sample_size", "[--structural-sample-size N]", kStructuralSearchFlags},
    {"neighbours", "[--neighbours K]", kNeighbourSearchFlags},
    {"objective_tie_break", "[--no-objective-tie-break]", kScalarSearchFlags, true},
}};

inline constexpr std::size_t kSearchFlagCount = kSearchFlagSpecs.size();

/// The index of a family bit, for `SearchFlags::typed`.
inline std::size_t search_flag_family_index(unsigned family) {
    std::size_t i = 0;
    while ((family >> i) != 1U) {
        ++i;
    }
    return i;
}

/// The structural and neighbour families' half of match_search_flag(): the
/// flags only List/Set runners read (#198). Returns the flag matched and
/// consumed, setting `family` to its family, or nullptr with the cursor and
/// `family` left as they were.
inline const char* match_structural_search_flag(ArgCursor& c, SearchFlags& f, unsigned& family) {
    const char* v = nullptr;
    if (c.value_flag("--structural-prob", v)) {
        // Outside [0, 1] (negative too: "automatic" is the omitted flag) becomes
        // NaN, which validation refuses. `+ 0.0` folds a typed -0 onto 0.
        const double p = parse_double("--structural-prob", v) + 0.0;
        f.structural_prob = p >= 0.0 && p <= 1.0 ? p : std::numeric_limits<double>::quiet_NaN();
        family = kStructuralSearchFlags;
        return "--structural-prob";
    }
    if (c.value_flag("--structural-selection", v)) {
        if (!try_parse_structural_selection(v, f.structural_selection)) {
            std::fprintf(stderr,
                         "--structural-selection: unknown policy '%s' (expected first_improving, "
                         "best_of_sample, violation_guided or granular_best)\n",
                         v);
            std::exit(2);
        }
        family = kStructuralSearchFlags;
        return "--structural-selection";
    }
    if (c.value_flag("--structural-sample-size", v)) {
        f.structural_sample_size = parse_int64("--structural-sample-size", v);
        family = kStructuralSearchFlags;
        return "--structural-sample-size";
    }
    if (c.value_flag("--neighbours", v)) {
        f.neighbours = parse_int64("--neighbours", v);
        family = kNeighbourSearchFlags;
        return "--neighbours";
    }
    return nullptr;
}

/// Matches and consumes one search-configuration flag under the cursor.
///
/// Returns false, cursor unmoved, when the argument is not ours, so a runner
/// chains it ahead of its unknown-option branch. A value flag with no value
/// (trailing `--lns-interval`) does not match and falls through there
/// (runner_args.h). A bad integer or `--structural-selection` word exits 2
/// here; a bad double becomes NaN for validate_search_flags(). Every flag
/// matches whatever the runner declares; validation refuses on `typed`.
inline bool match_search_flag(ArgCursor& c, const std::string& s, SearchFlags& f) {
    const char* v = nullptr;
    const char* flag = nullptr;
    unsigned family = kCommonSearchFlags;
    if (s == "--no-float-hook") {
        f.float_hook = false;
        flag = "--no-float-hook";
        family = kScalarSearchFlags;
    } else if (s == "--no-lns") {
        f.lns = false;
        flag = "--no-lns";
    } else if (c.value_flag("--lns-interval", v)) {
        f.lns_interval = parse_int64("--lns-interval", v);
        flag = "--lns-interval";
    } else if (s == "--compound-moves" || s == "--no-compound-moves") {
        f.compound_moves = s == "--compound-moves";
        flag = f.compound_moves ? "--compound-moves" : "--no-compound-moves";
        family = kScalarSearchFlags;
    } else if (c.value_flag("--novelty-prob", v)) {
        // `+ 0.0` folds -0 onto 0, else one run has two cells.
        f.novelty_prob = parse_double("--novelty-prob", v) + 0.0;
        flag = "--novelty-prob";
        family = kScalarSearchFlags;
    } else if (c.value_flag("--unproductive-iters", v)) {
        f.unproductive_iters = parse_int64("--unproductive-iters", v);
        flag = "--unproductive-iters";
        family = kScalarSearchFlags;
    } else if (c.value_flag("--perturbation-period", v)) {
        f.perturbation_period = parse_int64("--perturbation-period", v);
        flag = "--perturbation-period";
    } else if (c.value_flag("--max-iterations", v)) {
        f.max_iterations = parse_int64("--max-iterations", v);
        flag = "--max-iterations";
    } else if (s == "--no-time-limit") {
        f.no_time_limit = true;
        flag = "--no-time-limit";
    } else if (const char* structural = match_structural_search_flag(c, f, family);
               structural != nullptr) {
        flag = structural;
    } else if (s == "--no-objective-tie-break") {
        f.objective_tie_break = false;
        flag = "--no-objective-tie-break";
        family = kScalarSearchFlags;
    } else {
        return false;
    }
    const std::size_t idx = search_flag_family_index(family);
    if (f.typed[idx] == nullptr) {
        f.typed[idx] = flag;
    }
    return true;
}

/// The policy the run executes (#203): `auto` resolved against whether
/// `--neighbours` supplies a list. The cell, the published-table guard and the
/// no-effect rules compare this, so `auto` and its resolution are one arm.
inline StructuralSelection effective_structural_selection(const SearchFlags& f) {
    return resolve_structural_selection(f.structural_selection, f.neighbours != 0);
}

/// What the default selection runs under `f`'s neighbour list.
inline StructuralSelection default_structural_selection(const SearchFlags& f) {
    return resolve_structural_selection(SearchConfig{}.structural_selection, f.neighbours != 0);
}

/// The structural half of validate_search_flags(): its own family and
/// no-effect rules. Every default passes.
inline bool validate_structural_flags(const SearchFlags& f, std::string& error) {
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    const SearchFlags defaults;
    // match_search_flag() maps any typed value outside [0, 1] to NaN, so
    // `auto` has one spelling: the omitted flag.
    if (std::isnan(f.structural_prob)) {
        error = "--structural-prob must be in [0, 1] (omit it for the engine's automatic value)";
        return false;
    }
    if (f.structural_sample_size < 1 || f.structural_sample_size > kIntMax) {
        error = "--structural-sample-size must be >= 1";
        return false;
    }
    if (f.neighbours < 0 || f.neighbours > kIntMax ||
        (f.neighbours == 0 &&
         f.typed[search_flag_family_index(kNeighbourSearchFlags)] != nullptr)) {
        // A typed 0 would be a second spelling of the default "no list".
        error = "--neighbours must be >= 1 (omit it for no neighbour list)";
        return false;
    }
    // No-effect combinations, refused as the scalar pairs are: the engine
    // reads the sample size only under BestOfSample and ViolationGuided, and
    // at probability 0 reads no structural flag. GranularBest without a list
    // is ViolationGuided by another name.
    if (f.structural_selection == StructuralSelection::GranularBest && f.neighbours == 0) {
        error = "--structural-selection granular_best needs --neighbours K";
        return false;
    }
    // `auto` without a list runs first_improving, so the same rule holds there.
    if (effective_structural_selection(f) == StructuralSelection::FirstImprovingSample &&
        f.structural_sample_size != defaults.structural_sample_size) {
        error =
            "--structural-sample-size has no effect under --structural-selection "
            "first_improving";
        return false;
    }
    if (f.structural_prob == 0.0 &&
        (effective_structural_selection(f) != default_structural_selection(f) ||
         f.structural_sample_size != defaults.structural_sample_size || f.neighbours != 0)) {
        error =
            "--structural-prob 0 runs no structural batch, so the structural selection, "
            "sample size and neighbours are read by nothing";
        return false;
    }
    return true;
}

/// Range and coherence checks. Returns false with a one-line reason; the
/// runner owns the exit (runner_args.h), so a test can call this.
/// `applicable` is the runner's families (#198): a flag from any other is
/// refused by name. `time_limit_set`: the runner saw an explicit
/// `--time-limit`, which conflicts with `--no-time-limit`.
inline bool validate_search_flags(const SearchFlags& f, unsigned applicable, bool time_limit_set,
                                  std::string& error) {
    for (std::size_t i = 0; i < kSearchFlagFamilyCount; ++i) {
        if (f.typed[i] != nullptr && (applicable & (1U << i)) == 0) {
            error = std::string(f.typed[i]) + " is a " + search_flag_family_name(i) +
                    " search flag, which this runner's models cannot use: it would be "
                    "recorded and read by nothing";
            return false;
        }
    }
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    if (f.lns_interval < 1 || f.lns_interval > kIntMax) {
        // solve() skips LNS when lns_interval <= 0: refused so `--no-lns` is
        // that arm's one spelling (one run, one cell).
        error = "--lns-interval must be >= 1 (use --no-lns to switch LNS off)";
        return false;
    }
    // NaN (parse_double's non-number) must be tested by name: it passes every
    // range test. `!(p >= 0 && p <= 1)` would catch it, but
    // readability-simplify-boolean-expr's DeMorgan rewrite would lose it.
    if (std::isnan(f.novelty_prob) || f.novelty_prob < 0.0 || f.novelty_prob > 1.0) {
        error = "--novelty-prob must be in [0, 1]";
        return false;
    }
    // The engine arms the unproductive-batch exit only on `> 0`, so negative
    // and 0 are one run; 0 is that fixed-cadence arm's (#102 control) one
    // spelling, keeping one run, one cell.
    if (f.unproductive_iters < 0) {
        error = "--unproductive-iters must be >= 0 (0 restores the fixed kick cadence)";
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
    // A typed flag the engine would not read is refused. Engine short-circuits
    // (src/search.cpp): the kick tests `lns_ != nullptr` before
    // `lns_interval_`; the batch picker tests `use_compound_moves` before
    // `novelty_jump_probability`. This also keeps the cell a census of runs.
    const SearchFlags defaults;
    if (!f.lns && f.lns_interval != defaults.lns_interval) {
        error = "--lns-interval has no effect with --no-lns";
        return false;
    }
    if (!f.compound_moves && f.novelty_prob != defaults.novelty_prob) {
        error = "--novelty-prob has no effect without --compound-moves";
        return false;
    }
    if (f.no_time_limit && time_limit_set) {
        error = "--no-time-limit and --time-limit contradict each other";
        return false;
    }
    if (f.no_time_limit && f.max_iterations <= 0) {
        // With neither budget solve() returns at once (NoBudget): a
        // full-looking table of empty results, as `--time-limit > 0` guards.
        error = "--no-time-limit requires --max-iterations N with N > 0";
        return false;
    }
    return validate_structural_flags(f, error);
}

/// The first flag whose value differs from `d`, or nullptr for the default arm.
/// `d` is the runner's published protocol (engine defaults unless pinned, as
/// uc-chped pins compound moves). Lets the published-table guard refuse a
/// non-default arm as it refuses a short budget or partial roster.
inline const char* first_non_default_search_flag(const SearchFlags& f,
                                                 const SearchFlags& d = SearchFlags{}) {
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
        return f.compound_moves ? "--compound-moves" : "--no-compound-moves";
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
    if (f.structural_prob != d.structural_prob) {
        return "--structural-prob";
    }
    if (effective_structural_selection(f) !=
        resolve_structural_selection(d.structural_selection, f.neighbours != 0)) {
        return "--structural-selection";
    }
    if (f.structural_sample_size != d.structural_sample_size) {
        return "--structural-sample-size";
    }
    if (f.neighbours != d.neighbours) {
        return "--neighbours";
    }
    if (f.objective_tie_break != d.objective_tie_break) {
        return "--no-objective-tie-break";
    }
    return nullptr;
}

/// A double as the `search_config` cell spells it.
///
/// `%.17g` round-trips a double, so the cell names the value the search
/// received; `%g`'s six digits would merge 0.1234567 and 0.1234568. Stable:
/// nothing here calls setlocale.
inline std::string search_flag_number(double v) {
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%.17g", v);
    return {buf.data()};
}

/// The flags' values in `kSearchFlagSpecs` order, so keys and values zip from
/// one array.
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
        f.structural_prob < 0.0 ? std::string("auto") : search_flag_number(f.structural_prob),
        structural_selection_name(effective_structural_selection(f)),
        std::to_string(f.structural_sample_size),
        f.neighbours == 0 ? std::string(off) : std::to_string(f.neighbours),
        f.objective_tie_break ? on : off,
    }};
}

/// The canonical `key=value;key=value` cell recorded on every row, over the
/// runner's declared families only (`applicable`).
///
/// Deterministic, comma-free (no CSV quoting), fixed order: rows of one arm
/// compare equal, so `sort | uniq -c` is a census of arms. validate_search_flags'
/// no-effect rules keep it one cell per run.
inline std::string search_config_string(const SearchFlags& f, unsigned applicable) {
    const auto values = search_config_values(f);
    const auto defaults = search_config_values(SearchFlags{});
    std::string out;
    for (std::size_t i = 0; i < kSearchFlagCount; ++i) {
        if ((kSearchFlagSpecs[i].family & applicable) == 0) {
            continue;
        }
        if (kSearchFlagSpecs[i].opt_in && values[i] == defaults[i]) {
            continue;
        }
        if (!out.empty()) {
            out += ';';
        }
        out += kSearchFlagSpecs[i].key;
        out += '=';
        out += values[i];
    }
    return out;
}

/// The `--help` fragment for the runner's declared flags, generated from the
/// same table as the cell above.
inline std::string search_flags_usage(unsigned applicable) {
    std::string out;
    for (const auto& spec : kSearchFlagSpecs) {
        if ((spec.family & applicable) == 0) {
            continue;
        }
        if (!out.empty()) {
            out += ' ';
        }
        out += spec.usage;
    }
    return out;
}

/// Writes the flags onto the config. Call validate_search_flags() first: the
/// `int` casts are safe only after its range checks. `--neighbours` needs the
/// runner's cost, so it goes through neighbours_argument().
inline void apply_search_flags(const SearchFlags& f, SearchConfig& cfg) {
    cfg.use_compound_moves = f.compound_moves;
    cfg.novelty_jump_probability = f.novelty_prob;
    cfg.unproductive_iterations = f.unproductive_iters;
    cfg.perturbation_period = static_cast<int>(f.perturbation_period);
    cfg.max_iterations = f.max_iterations;
    cfg.structural_batch_probability = f.structural_prob;
    cfg.structural_selection = f.structural_selection;
    cfg.structural_sample_size = static_cast<int>(f.structural_sample_size);
    cfg.fj_objective_tie_break = f.objective_tie_break;
}

/// `--neighbours` against the instance: `nearest_neighbours` clamps K to
/// `universe - 1`, so a larger K is one run under another cell. Call before
/// neighbours_argument().
inline bool validate_neighbours(const SearchFlags& f, int universe, std::string& error) {
    if (f.neighbours > 0 && f.neighbours > static_cast<int64_t>(universe) - 1) {
        error = "--neighbours " + std::to_string(f.neighbours) + " exceeds the " +
                std::to_string(universe > 0 ? universe - 1 : 0) +
                " other elements this instance has (a larger K is the same run)";
        return false;
    }
    return true;
}

/// The K-nearest list for `--neighbours K`, ranked by the runner's `cost` over
/// `universe` elements, or null without the flag (engine default). Only a
/// runner declaring `kNeighbourSearchFlags` calls this.
inline std::shared_ptr<const NeighbourList> neighbours_argument(
    const SearchFlags& f, int universe, const std::function<double(int, int)>& cost) {
    if (f.neighbours == 0) {
        return nullptr;
    }
    return std::make_shared<const NeighbourList>(
        nearest_neighbours(universe, static_cast<int>(f.neighbours), cost));
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
