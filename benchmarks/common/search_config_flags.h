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
//
// PER-RUNNER APPLICABILITY (#198). "Read by the engine" is a property of the
// flag AND the model. The structural flags are read by nothing on a scalar
// model (`effective_structural_probability` resolves the batch to 0.0 when no
// List/Set generator exists), and the FJ-side flags are read by nothing on a
// pure-List model, which runs no FJ batch at all. So every flag belongs to a
// family below, every runner declares the families its models can use, and a
// flag outside that set is REFUSED by validate_search_flags() -- not accepted,
// recorded and ignored. The cell and the usage line carry only the declared
// families, which is also what keeps a scalar runner's cell byte-identical to
// the one its committed tables already hold.
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
/// What a runner over scalar models (minlplib, uc-chped) declares.
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
    // The structural batch (#198). A negative probability is the engine's
    // "automatic" and has no spelling on the command line: it is the default,
    // and the cell writes it as `auto`.
    double structural_prob = SearchConfig{}.structural_batch_probability;
    StructuralSelection structural_selection = SearchConfig{}.structural_selection;
    int64_t structural_sample_size = SearchConfig{}.structural_sample_size;
    // 0 = no neighbour list, the engine's default (a null
    // `structural_neighbours`). Typed values must be >= 1.
    int64_t neighbours = 0;
    // OR-Tools' near-value kick (#217), whose default #226 is deciding.
    // Recorded in the cell only when on (see SearchFlagSpec::opt_in).
    bool perturb_near_value = SearchConfig{}.perturb_near_value;
    // The first flag typed from each family, indexed by the family's bit, or
    // nullptr. Set by match_search_flag() whatever the value, so that a flag
    // restated at its default is refused on a runner that cannot read it too:
    // that is still a flag the runner would accept and ignore.
    std::array<const char*, kSearchFlagFamilyCount> typed{};
};

/// One flag's contribution to the two strings that must not drift apart: the
/// key it takes in the row's `search_config` cell, and its `--help` fragment.
struct SearchFlagSpec {
    const char* key;
    const char* usage;
    unsigned family;
    // An opt-in flag is written into the cell only when its value differs from
    // the default. That is what lets a flag join a family whose runners already
    // have committed tables: their default cell stays byte-identical, and an
    // arm that turns the flag on is still named by it (#226).
    bool opt_in = false;
};

/// The flag table. Order is the canonical order of the `search_config` cell and
/// of the usage line; both are generated from this array, so a flag cannot be
/// documented and unrecorded (or the reverse), and the cell's key order is
/// deterministic by construction rather than by convention.
///
/// New flags go at the END: a runner's cell is the table filtered by its
/// families, so appending keeps every existing runner's cell byte-identical.
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
     "[--structural-selection first_improving|best_of_sample|violation_guided]",
     kStructuralSearchFlags},
    {"structural_sample_size", "[--structural-sample-size N]", kStructuralSearchFlags},
    {"neighbours", "[--neighbours K]", kNeighbourSearchFlags},
    {"perturb_near_value", "[--perturb-near-value]", kScalarSearchFlags, true},
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
/// by validate_search_flags() below. An unknown `--structural-selection` word
/// has no NaN to become, so it reports and exits 2 here, as a bad integer does.
///
/// Every flag is matched whatever the runner declares; whether the runner may
/// take it is validate_search_flags()' call, made on `typed`.
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
        // `+ 0.0` folds a typed -0 onto 0: the same run must not be recorded
        // under two cells (`novelty_prob=-0` and `novelty_prob=0`).
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
    } else if (c.value_flag("--structural-prob", v)) {
        // Out of [0, 1] -- a negative included, since "automatic" is spelled by
        // omitting the flag -- becomes NaN, the value validation refuses. A
        // NaN from a non-number already is one.
        // `+ 0.0` folds a typed -0 onto 0, as for --novelty-prob.
        const double p = parse_double("--structural-prob", v) + 0.0;
        f.structural_prob = p >= 0.0 && p <= 1.0 ? p : std::numeric_limits<double>::quiet_NaN();
        flag = "--structural-prob";
        family = kStructuralSearchFlags;
    } else if (c.value_flag("--structural-selection", v)) {
        if (!try_parse_structural_selection(v, f.structural_selection)) {
            std::fprintf(stderr,
                         "--structural-selection: unknown policy '%s' (expected first_improving, "
                         "best_of_sample or violation_guided)\n",
                         v);
            std::exit(2);
        }
        flag = "--structural-selection";
        family = kStructuralSearchFlags;
    } else if (c.value_flag("--structural-sample-size", v)) {
        f.structural_sample_size = parse_int64("--structural-sample-size", v);
        flag = "--structural-sample-size";
        family = kStructuralSearchFlags;
    } else if (c.value_flag("--neighbours", v)) {
        f.neighbours = parse_int64("--neighbours", v);
        flag = "--neighbours";
        family = kNeighbourSearchFlags;
    } else if (s == "--perturb-near-value") {
        f.perturb_near_value = true;
        flag = "--perturb-near-value";
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

/// The structural half of validate_search_flags(), split out because it is a
/// separate family with its own no-effect rules. Every default passes.
inline bool validate_structural_flags(const SearchFlags& f, std::string& error) {
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    const SearchFlags defaults;
    // match_search_flag() turns every typed value outside [0, 1] into NaN,
    // a negative included: the engine's "automatic" is the default and is
    // spelled by leaving the flag out, so `auto` has one spelling.
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
        // 0 is the default "no list"; typing it would be a second spelling of
        // the default arm, so the typed flag must ask for an actual list.
        error = "--neighbours must be >= 1 (omit it for no neighbour list)";
        return false;
    }
    // No-effect combinations, refused for the census reason the scalar pairs
    // are. The engine reads the sample size only under BestOfSample and
    // ViolationGuided (FirstImprovingSample "takes exactly one generate()
    // call"), and at an explicit probability of 0 it never runs a structural
    // batch, so neither the policy, the sample size nor the neighbour list is
    // read.
    if (f.structural_selection == StructuralSelection::FirstImprovingSample &&
        f.structural_sample_size != defaults.structural_sample_size) {
        error =
            "--structural-sample-size has no effect under --structural-selection "
            "first_improving";
        return false;
    }
    if (f.structural_prob == 0.0 &&
        (f.structural_selection != defaults.structural_selection ||
         f.structural_sample_size != defaults.structural_sample_size || f.neighbours != 0)) {
        error =
            "--structural-prob 0 runs no structural batch, so the structural selection, "
            "sample size and neighbours are read by nothing";
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
/// `applicable` is the runner's declared families (#198); a flag typed from
/// any other family is refused, naming the flag. `time_limit_set` says whether
/// the runner also saw an explicit `--time-limit`, which only this layer can
/// turn into the conflict it is.
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
    // A negative and a zero are THE SAME RUN -- the engine arms the
    // unproductive-batch exit only on `> 0` -- so accepting both would put two
    // different `search_config` cells on one configuration, breaking the
    // invariant search_config_string() states and that the two no-effect
    // refusals below are justified by. 0 is the documented spelling for the
    // fixed-cadence arm; a negative is a typo.
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
    // Two combinations in which a flag the caller typed would be READ BY
    // NOTHING, and so must be refused rather than accepted, recorded and
    // silently ignored -- the failure this whole header exists to prevent, and
    // the same standard the rho and use_fj rejections are held to.
    //
    // They are short-circuits in the engine, not opinions:
    // `src/search.cpp`'s kick site tests `lns_ != nullptr` before
    // `lns_interval_`, and its batch picker tests `config_.use_compound_moves`
    // before drawing against `novelty_jump_probability`. Refusing them also
    // keeps the recorded cell a census of RUNS rather than of typing: with
    // these out, no two accepted command lines that produce the same run can
    // produce two different `search_config` strings.
    const SearchFlags defaults;
    if (!f.lns && f.lns_interval != defaults.lns_interval) {
        error = "--lns-interval has no effect with --no-lns";
        return false;
    }
    if (!f.compound_moves && f.novelty_prob != defaults.novelty_prob) {
        error = "--novelty-prob has no effect without --compound-moves";
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
    return validate_structural_flags(f, error);
}

/// The first flag whose value is not the runner's default, or nullptr when the
/// arm is the default configuration. `d` is the runner's published protocol:
/// the engine's defaults unless the runner pins a field away from them, as
/// uc-chped does for compound moves.
///
/// This is what lets a runner's published-table guard treat an ablation arm the
/// way it treats a shortened budget or a partial roster: a table generated under
/// a non-default search configuration is not the published measurement, whatever
/// else was passed.
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
    if (f.structural_selection != d.structural_selection) {
        return "--structural-selection";
    }
    if (f.structural_sample_size != d.structural_sample_size) {
        return "--structural-sample-size";
    }
    if (f.neighbours != d.neighbours) {
        return "--neighbours";
    }
    if (f.perturb_near_value != d.perturb_near_value) {
        return "--perturb-near-value";
    }
    return nullptr;
}

/// A double as the `search_config` cell spells it.
///
/// `%.17g`, not `%g`. The cell exists so a results file states the
/// configuration it was produced under, and `%g`'s six significant digits
/// cannot: `--novelty-prob 0.1234567` and `--novelty-prob 0.1234568` are two
/// different runs that both recorded `novelty_prob=0.123457`. Seventeen digits
/// is what round-trips an IEEE double exactly, so the cell names the value the
/// search actually received. Stable across runs -- nothing here calls
/// setlocale -- and the trailing digits only appear for a value that has them.
inline std::string search_flag_number(double v) {
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%.17g", v);
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
        f.structural_prob < 0.0 ? std::string("auto") : search_flag_number(f.structural_prob),
        structural_selection_name(f.structural_selection),
        std::to_string(f.structural_sample_size),
        f.neighbours == 0 ? std::string(off) : std::to_string(f.neighbours),
        f.perturb_near_value ? on : off,
    }};
}

/// The canonical `key=value;key=value` cell recorded on every row, over the
/// runner's declared families only (`applicable`).
///
/// Deterministic, comma-free (so it needs no CSV quoting) and emitted in one
/// fixed order for every runner, so two rows produced by the same arm compare
/// equal as strings and `sort | uniq -c` over a results file is a census of the
/// arms it holds. That last claim is what validate_search_flags' two
/// no-effect-in-this-combination rules protect: without them `--no-lns` and
/// `--no-lns --lns-interval 5` would be one run under two different cells.
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

/// Writes the flags onto the search configuration. Call validate_search_flags()
/// first: the narrowing casts here are safe only because it has range-checked
/// the fields the engine holds as `int`.
///
/// `--neighbours` is not applied here: it needs the runner's cost, so it goes
/// through neighbours_argument() below.
inline void apply_search_flags(const SearchFlags& f, SearchConfig& cfg) {
    cfg.use_compound_moves = f.compound_moves;
    cfg.novelty_jump_probability = f.novelty_prob;
    cfg.unproductive_iterations = f.unproductive_iters;
    cfg.perturbation_period = static_cast<int>(f.perturbation_period);
    cfg.max_iterations = f.max_iterations;
    cfg.structural_batch_probability = f.structural_prob;
    cfg.structural_selection = f.structural_selection;
    cfg.structural_sample_size = static_cast<int>(f.structural_sample_size);
    cfg.perturb_near_value = f.perturb_near_value;
}

/// The neighbour flag's check against the instance, which only the runner can
/// make: `nearest_neighbours` clamps K to `universe - 1`, so every K from there
/// up is one run under several cells. Call it before neighbours_argument().
inline bool validate_neighbours(const SearchFlags& f, int universe, std::string& error) {
    if (f.neighbours > 0 && f.neighbours > static_cast<int64_t>(universe) - 1) {
        error = "--neighbours " + std::to_string(f.neighbours) + " exceeds the " +
                std::to_string(universe > 0 ? universe - 1 : 0) +
                " other elements this instance has (a larger K is the same run)";
        return false;
    }
    return true;
}

/// The K-nearest neighbour list `--neighbours K` asks for, ranked by the
/// runner's `cost` over its universe of `universe` elements, or null when the
/// flag was not given (the engine's default). Only a runner that declares
/// `kNeighbourSearchFlags` calls this; it is what supplies the cost.
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
