// Tests for the benchmark runners' search-configuration flags (#136).
//
// Two halves, and the second is the point. A flag that parses is worth nothing
// on its own: the failure this issue exists to prevent is a flag that is
// accepted, recorded on the results row and then ignored by the engine, which
// produces an ablation that measures nothing while looking rigorous. So every
// flag below is also given a PROBE -- two runs that differ only in that flag,
// asserted to produce different observables from the same seed.
//
// Every probe is wall-clock-free: `time_limit = 0` disables solve()'s clock
// entirely, so each run is bounded by `max_iterations` alone and is
// bit-reproducible for a given seed. A probe that depended on elapsed time
// would be flaky, and a flaky probe is not evidence.

#include <benchmarks/common/search_config_flags.h>
#include <catch2/catch_test_macros.hpp>
#include <cbls/inner_solver.h>
#include <cbls/lns.h>
#include <cbls/model.h>
#include <cbls/move_generator.h>
#include <cbls/search.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using cbls::LNS;
using cbls::Model;
using cbls::SearchConfig;
using cbls::SearchResult;
using cbls::TerminationReason;
using cbls::bench::SearchFlags;

namespace {

// What minlplib and uc-chped declare, which is what every test of the original
// nine flags runs under: their cell is pinned byte-for-byte by committed tables.
constexpr unsigned kScalar = cbls::bench::kScalarRunnerSearchFlags;
// What a List runner that supplies a cost declares (cbls_cvrp, #197).
constexpr unsigned kStructural = cbls::bench::kCommonSearchFlags |
                                 cbls::bench::kStructuralSearchFlags |
                                 cbls::bench::kNeighbourSearchFlags;

struct ParseOutcome {
    SearchFlags flags;
    std::vector<std::string> unmatched;
};

/// Drives `match_search_flag` over a command line exactly as a runner does:
/// through an ArgCursor, with anything it does not match falling through to
/// what would be the runner's unknown-option branch.
ParseOutcome parse(const std::vector<std::string>& args) {
    std::vector<std::string> storage;
    storage.emplace_back("cbls_probe");  // argv[0]; the cursor advances past it
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& a : storage) {
        argv.push_back(a.data());
    }

    ParseOutcome out;
    cbls::bench::ArgCursor cursor(static_cast<int>(argv.size()), argv.data());
    while (cursor.advance()) {
        const std::string s = cursor.arg();
        if (!cbls::bench::match_search_flag(cursor, s, out.flags)) {
            out.unmatched.push_back(s);
        }
    }
    return out;
}

/// Unsatisfiable and objective-bearing: 8 variables capped at 10 cannot sum to
/// 2500, and the objective keeps the pure-feasibility break from ending the run.
/// So no batch ever improves, every batch is stagnant, and the diversification
/// machinery the cadence flags govern is exercised on a fixed schedule rather
/// than on whatever the search happens to find.
///
/// Small on purpose. An LNS repair is a whole fj_nl_initialize pass and its cost
/// tracks the variable count hard -- measured here at ~24ms per repair on 8
/// variables against ~630ms on 50 -- and the cadence probes need tens of kicks.
void build_stagnant_model(Model& m) {
    std::vector<int32_t> vars;
    vars.reserve(8);
    for (int i = 0; i < 8; ++i) {
        vars.push_back(m.int_var(0, 10));
    }
    std::vector<int32_t> args(vars.begin(), vars.end());
    args.push_back(m.constant(-2500.0));
    m.add_constraint(m.abs_expr(m.sum(args)));
    m.minimize(m.sum(vars));
    m.close();
}

/// A continuous model with a real optimum the inner solver can descend to:
/// minimise x^2 + y^2 subject to x + y >= 1, whose optimum is (0.5, 0.5).
/// FeasibilityJump alone lands somewhere on the constraint; the
/// FloatIntensifyHook is what walks the pair down it.
void build_continuous_model(Model& m) {
    const int32_t x = m.float_var(0, 10);
    const int32_t y = m.float_var(0, 10);
    const int32_t one = m.constant(1.0);
    const int32_t neg1 = m.constant(-1.0);
    const int32_t two = m.constant(2.0);
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));  // 1 - x - y <= 0
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
}

/// Runs one arm exactly as a runner runs it: the flags are applied onto the
/// config and onto solve()'s arguments through the shared header, and nothing
/// else differs between two calls. `batch_iterations` is a test knob rather than
/// a flag -- it sets how often the loop reaches a batch boundary, which is where
/// every cadence decision is taken.
SearchResult run_arm(Model& model, const SearchFlags& flags, int64_t batch_iterations) {
    cbls::FloatIntensifyHook hook;
    LNS lns(0.3);
    SearchConfig cfg;
    cfg.batch_iterations = batch_iterations;
    cbls::bench::apply_search_flags(flags, cfg);
    return cbls::solve(model, cbls::bench::time_limit_argument(flags, 0.0), /*seed=*/42,
                       /*use_fj=*/true, cbls::bench::hook_argument(flags, hook),
                       cbls::bench::lns_argument(flags, lns),
                       cbls::bench::lns_interval_argument(flags), nullptr, cfg);
}

/// Builds a fresh model per arm: solve() mutates the model (assignment, and the
/// objective bound it folds in), so two arms must never share one.
SearchResult probe(const std::vector<std::string>& args, void (*build)(Model&),
                   int64_t batch_iterations) {
    const ParseOutcome parsed = parse(args);
    REQUIRE(parsed.unmatched.empty());
    std::string error;
    REQUIRE(
        cbls::bench::validate_search_flags(parsed.flags, kScalar, /*time_limit_set=*/false, error));
    Model m;
    build(m);
    return run_arm(m, parsed.flags, batch_iterations);
}

/// A tiny capacitated routing model on Lists only -- three routes partitioning
/// ten customers, at most four per route, minimising Euclidean route length
/// from a depot at the origin. FJ has nothing to jump on it, so it is what the
/// structural flags are read on (#198), and what cbls_cvrp's models look like.
struct RoutingFixture {
    static constexpr int kCustomers = 10;
    std::vector<double> x;
    std::vector<double> y;

    RoutingFixture() {
        for (int i = 0; i < kCustomers; ++i) {
            // Two rings, so nearest neighbours are not simply index order.
            const double r = (i % 2 == 0) ? 10.0 : 25.0;
            const double a = 0.7 * i;
            x.push_back(r * std::cos(a));
            y.push_back(r * std::sin(a));
        }
    }
    [[nodiscard]] double cost(int a, int b) const { return std::hypot(x[a] - x[b], y[a] - y[b]); }
    [[nodiscard]] double depot(int e) const { return std::hypot(x[e], y[e]); }

    void build(Model& m) const {
        std::vector<int32_t> routes;
        std::vector<int32_t> lengths;
        for (int r = 0; r < 3; ++r) {
            const int32_t route = m.list_var(kCustomers, 0, kCustomers, cbls::ListInit::Empty);
            routes.push_back(route);
            m.add_constraint(
                m.sum({m.lambda_sum(route, [](int) { return 1.0; }), m.constant(-4.0)}));
            lengths.push_back(m.pair_lambda_sum(
                route, [this](int a, int b) { return cost(a, b); },
                [this](int e) { return depot(e); }, [this](int e) { return depot(e); }));
        }
        m.add_list_partition(routes, cbls::Cover::Exact);
        m.minimize(m.sum(lengths));
        m.close();
    }
};

/// The structural counterpart of probe(): validated under the List runner's
/// families, with `--neighbours` resolved against the fixture's cost exactly as
/// a runner would resolve it against its instance's.
SearchResult structural_probe(const std::vector<std::string>& args) {
    const ParseOutcome parsed = parse(args);
    REQUIRE(parsed.unmatched.empty());
    std::string error;
    REQUIRE(cbls::bench::validate_search_flags(parsed.flags, kStructural, false, error));
    const RoutingFixture fx;
    Model m;
    fx.build(m);
    LNS lns(0.3);
    SearchConfig cfg;
    cfg.batch_iterations = 20;
    cbls::bench::apply_search_flags(parsed.flags, cfg);
    REQUIRE(cbls::bench::validate_neighbours(parsed.flags, RoutingFixture::kCustomers, error));
    cfg.structural_neighbours = cbls::bench::neighbours_argument(
        parsed.flags, RoutingFixture::kCustomers, [&fx](int a, int b) { return fx.cost(a, b); });
    return cbls::solve(m, cbls::bench::time_limit_argument(parsed.flags, 0.0), /*seed=*/42,
                       /*use_fj=*/true, /*hook=*/nullptr,
                       cbls::bench::lns_argument(parsed.flags, lns),
                       cbls::bench::lns_interval_argument(parsed.flags), nullptr, cfg);
}

}  // namespace

// ---------------------------------------------------------------------------
// Parse, validate, record
// ---------------------------------------------------------------------------

TEST_CASE("search flags default to the engine's own configuration", "[bench][flags]") {
    const SearchFlags f;
    const SearchConfig engine;
    REQUIRE(f.lns_interval == engine.lns_interval);
    REQUIRE(f.compound_moves == engine.use_compound_moves);
    REQUIRE(f.novelty_prob == engine.novelty_jump_probability);
    REQUIRE(f.unproductive_iters == engine.unproductive_iterations);
    REQUIRE(f.perturbation_period == engine.perturbation_period);
    REQUIRE(f.max_iterations == engine.max_iterations);
    REQUIRE(f.float_hook);
    REQUIRE(f.lns);
    REQUIRE_FALSE(f.no_time_limit);
    REQUIRE(cbls::bench::first_non_default_search_flag(f) == nullptr);
}

TEST_CASE("every search flag sets the field it claims", "[bench][flags]") {
    // Every flag at once, and a combination that is actually runnable: the two
    // no-effect pairs (--no-lns with --lns-interval, --novelty-prob without
    // --compound-moves) are refused by validate_search_flags, so --no-lns is
    // checked on its own below rather than folded in here.
    const ParseOutcome parsed =
        parse({"--no-float-hook", "--lns-interval", "7", "--compound-moves", "--novelty-prob",
               "0.25", "--unproductive-iters", "0", "--perturbation-period", "9",
               "--max-iterations", "1234", "--no-time-limit"});
    REQUIRE(parsed.unmatched.empty());
    std::string error;
    REQUIRE(
        cbls::bench::validate_search_flags(parsed.flags, kScalar, /*time_limit_set=*/false, error));
    const SearchFlags& f = parsed.flags;
    REQUIRE_FALSE(f.float_hook);
    REQUIRE(f.lns);
    REQUIRE(f.lns_interval == 7);
    REQUIRE(f.compound_moves);
    REQUIRE(f.novelty_prob == 0.25);
    REQUIRE(f.unproductive_iters == 0);
    REQUIRE(f.perturbation_period == 9);
    REQUIRE(f.max_iterations == 1234);
    REQUIRE(f.no_time_limit);

    // ...and that those land on the objects solve() is actually handed.
    SearchConfig cfg;
    cbls::bench::apply_search_flags(f, cfg);
    REQUIRE(cfg.use_compound_moves);
    REQUIRE(cfg.novelty_jump_probability == 0.25);
    REQUIRE(cfg.unproductive_iterations == 0);
    REQUIRE(cfg.perturbation_period == 9);
    REQUIRE(cfg.max_iterations == 1234);

    cbls::FloatIntensifyHook hook;
    LNS lns(0.3);
    REQUIRE(cbls::bench::hook_argument(f, hook) == nullptr);
    REQUIRE(cbls::bench::lns_argument(f, lns) == &lns);
    REQUIRE(cbls::bench::lns_interval_argument(f) == 7);
    REQUIRE(cbls::bench::time_limit_argument(f, 60.0) == 0.0);

    // The one flag the line above cannot carry.
    const SearchFlags no_lns = parse({"--no-lns"}).flags;
    REQUIRE_FALSE(no_lns.lns);
    REQUIRE(cbls::bench::lns_argument(no_lns, lns) == nullptr);
}

TEST_CASE("the on/off pairs are order-independent", "[bench][flags]") {
    REQUIRE(parse({"--compound-moves"}).flags.compound_moves);
    REQUIRE_FALSE(parse({"--no-compound-moves"}).flags.compound_moves);
    // Last one wins, which is what lets a wrapper script append an override.
    REQUIRE_FALSE(parse({"--compound-moves", "--no-compound-moves"}).flags.compound_moves);
    REQUIRE(parse({"--no-compound-moves", "--compound-moves"}).flags.compound_moves);
}

TEST_CASE("a value flag with no value falls through to the unknown-option path", "[bench][flags]") {
    // The ArgCursor rule (benchmarks/common/runner_args.h): a trailing
    // `--lns-interval` must not read past argv's end, and must not silently keep
    // the default either -- a published results table is what a silently-kept
    // default costs. It has to reach the runner's unknown-option branch, which
    // reports and exits 2.
    for (const char* flag : {"--lns-interval", "--novelty-prob", "--unproductive-iters",
                             "--perturbation-period", "--max-iterations"}) {
        const ParseOutcome parsed = parse({flag});
        REQUIRE(parsed.unmatched == std::vector<std::string>{flag});
        REQUIRE(cbls::bench::first_non_default_search_flag(parsed.flags) == nullptr);
    }
}

TEST_CASE("an unrelated argument is left for the runner", "[bench][flags]") {
    const ParseOutcome parsed = parse({"--seed", "3", "--no-lns", "instances/dir"});
    REQUIRE_FALSE(parsed.flags.lns);
    REQUIRE(parsed.unmatched == std::vector<std::string>{"--seed", "3", "instances/dir"});
}

TEST_CASE("the search config cell is canonical and deterministic", "[bench][flags]") {
    REQUIRE(cbls::bench::search_config_string(SearchFlags{}, kScalar) ==
            "float_hook=on;lns=on;lns_interval=3;compound_moves=on;novelty_prob=0.5;"
            "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on");

    const SearchFlags f =
        parse({"--no-float-hook", "--lns-interval", "7", "--compound-moves", "--novelty-prob",
               "0.25", "--unproductive-iters", "0", "--perturbation-period", "9",
               "--max-iterations", "1234", "--no-time-limit"})
            .flags;
    const std::string cell = cbls::bench::search_config_string(f, kScalar);
    REQUIRE(cell ==
            "float_hook=off;lns=on;lns_interval=7;compound_moves=on;novelty_prob=0.25;"
            "unproductive_iters=0;perturbation_period=9;max_iterations=1234;time_limit=off");
    // No comma, or the cell would shift every column after it in a CSV that
    // does not quote (both runners' writers do not).
    REQUIRE(cell.find(',') == std::string::npos);
    // Same flags in a different order are the same arm, so the cell must be the
    // same string: a results file is meant to be grouped by this column.
    REQUIRE(cbls::bench::search_config_string(
                parse({"--max-iterations", "1234", "--no-time-limit", "--perturbation-period", "9",
                       "--unproductive-iters", "0", "--novelty-prob", "0.25", "--compound-moves",
                       "--lns-interval", "7", "--no-float-hook"})
                    .flags,
                kScalar) == cell);
}

TEST_CASE("one run cannot be recorded under two different cells", "[bench][flags]") {
    // The cell is meant to be a census of RUNS, and the engine short-circuits
    // twice: it tests the LNS pointer before the interval, and the compound
    // switch before the probability. So `--no-lns --lns-interval 5` IS
    // `--no-lns`, and would otherwise be stamped as a different arm. Refusing
    // the pair keeps the claim true -- and is the same no-effect rule the rho
    // and use_fj rejections rest on.
    std::string error;
    REQUIRE_FALSE(cbls::bench::validate_search_flags(
        parse({"--no-lns", "--lns-interval", "5"}).flags, kScalar, false, error));
    REQUIRE(error.find("--lns-interval") != std::string::npos);
    REQUIRE_FALSE(cbls::bench::validate_search_flags(
        parse({"--no-compound-moves", "--novelty-prob", "0.25"}).flags, kScalar, false, error));
    REQUIRE(error.find("--novelty-prob") != std::string::npos);

    // Restating the default alongside the switch is not a no-effect flag; it is
    // a no-op, and nothing was asked for that the run does not do.
    REQUIRE(cbls::bench::validate_search_flags(parse({"--no-lns", "--lns-interval", "3"}).flags,
                                               kScalar, false, error));
    REQUIRE(cbls::bench::validate_search_flags(parse({"--novelty-prob", "0.5"}).flags, kScalar,
                                               false, error));
    // ...and each is fine in the combination the engine actually reads.
    REQUIRE(cbls::bench::validate_search_flags(parse({"--lns-interval", "5"}).flags, kScalar, false,
                                               error));
    REQUIRE(cbls::bench::validate_search_flags(
        parse({"--compound-moves", "--novelty-prob", "0.25"}).flags, kScalar, false, error));
}

TEST_CASE("the usage line documents exactly the recorded flags", "[bench][flags]") {
    // Both strings come from kSearchFlagSpecs, so this pins that they stay
    // generated from it rather than hand-maintained in parallel -- and that a
    // runner's line and cell carry exactly its declared families (#198).
    for (const unsigned families : {kScalar, kStructural}) {
        const std::string usage = cbls::bench::search_flags_usage(families);
        const std::string cell = cbls::bench::search_config_string(SearchFlags{}, families);
        for (const auto& spec : cbls::bench::kSearchFlagSpecs) {
            const bool declared = (spec.family & families) != 0;
            REQUIRE((usage.find(spec.usage) != std::string::npos) == declared);
            REQUIRE((cell.find(std::string(spec.key) + "=") != std::string::npos) == declared);
        }
    }
}

TEST_CASE("a non-default arm is named for the published-table guard", "[bench][flags]") {
    // What keeps an ablation arm from regenerating a published comparison table
    // as a side effect. Every flag has to be visible to it, or one arm gets a
    // free pass.
    const std::vector<std::vector<std::string>> arms = {
        {"--no-float-hook"},
        {"--no-lns"},
        {"--lns-interval", "5"},
        {"--no-compound-moves"},
        {"--novelty-prob", "0.1"},
        {"--unproductive-iters", "0"},
        {"--perturbation-period", "7"},
        {"--max-iterations", "10"},
        {"--no-time-limit"},
        {"--structural-prob", "1"},
        {"--structural-selection", "violation_guided"},
        {"--structural-sample-size", "4"},
        {"--neighbours", "5"},
    };
    REQUIRE(arms.size() == cbls::bench::kSearchFlagCount);
    for (const auto& arm : arms) {
        const ParseOutcome parsed = parse(arm);
        REQUIRE(parsed.unmatched.empty());
        REQUIRE(cbls::bench::first_non_default_search_flag(parsed.flags) != nullptr);
    }
    // A flag restated at its default is not an arm.
    REQUIRE(cbls::bench::first_non_default_search_flag(parse({"--compound-moves"}).flags) ==
            nullptr);
    // A runner whose protocol pins a field away from the engine (uc-chped runs
    // compound moves off) judges arms against its own defaults: the engine's
    // default is then the arm, named in the direction it was asked for, and
    // restating the runner's value is not.
    SearchFlags off;
    off.compound_moves = false;
    REQUIRE(std::string(cbls::bench::first_non_default_search_flag(parse({}).flags, off)) ==
            "--compound-moves");
    REQUIRE(cbls::bench::first_non_default_search_flag(parse({"--no-compound-moves"}).flags, off) ==
            nullptr);
    REQUIRE(std::string(cbls::bench::first_non_default_search_flag(
                parse({"--no-compound-moves"}).flags)) == "--no-compound-moves");
    REQUIRE(cbls::bench::first_non_default_search_flag(parse({"--lns-interval", "3"}).flags) ==
            nullptr);
}

TEST_CASE("validation rejects values the engine cannot use", "[bench][flags]") {
    std::string error;
    auto rejected = [&error](const std::vector<std::string>& args, bool time_limit_set = false) {
        return !cbls::bench::validate_search_flags(parse(args).flags, kScalar, time_limit_set,
                                                   error);
    };

    REQUIRE(rejected({"--lns-interval", "0"}));  // --no-lns is the one spelling for that arm
    REQUIRE(error.find("--lns-interval") != std::string::npos);
    REQUIRE(rejected({"--lns-interval", "-1"}));
    REQUIRE(rejected({"--lns-interval", "3000000000"}));  // wider than the engine's int field
    REQUIRE(rejected({"--novelty-prob", "1.5"}));
    REQUIRE(rejected({"--novelty-prob", "-0.1"}));
    REQUIRE(rejected({"--perturbation-period", "0"}));
    REQUIRE(rejected({"--max-iterations", "-1"}));
    // --no-time-limit without an iteration budget is a run with no budget at
    // all, which returns instantly and would publish a table of empty results.
    REQUIRE(rejected({"--no-time-limit"}));
    REQUIRE(error.find("--max-iterations") != std::string::npos);
    REQUIRE(rejected({"--no-time-limit", "--max-iterations", "0"}));
    // ...and it contradicts an explicit wall clock.
    REQUIRE(rejected({"--no-time-limit", "--max-iterations", "10"}, /*time_limit_set=*/true));

    REQUIRE(cbls::bench::validate_search_flags(SearchFlags{}, kScalar, false, error));
    REQUIRE(cbls::bench::validate_search_flags(
        parse({"--no-time-limit", "--max-iterations", "10"}).flags, kScalar, false, error));
}

TEST_CASE("a non-number double is rejected rather than defaulted", "[bench][flags]") {
    // The shared reporting policy: parse_double reports the bad token and
    // returns NaN, and the guard is what turns it into an exit. NaN must fail
    // the range check rather than pass it, or a typo would silently keep the
    // default. (The integer half of the policy calls std::exit inside
    // parse_int64 and so is pinned by the runners' driver tests instead.)
    const ParseOutcome parsed = parse({"--novelty-prob", "not-a-number"});
    REQUIRE(parsed.unmatched.empty());
    std::string error;
    REQUIRE_FALSE(cbls::bench::validate_search_flags(parsed.flags, kScalar, false, error));
    REQUIRE(error.find("--novelty-prob") != std::string::npos);
}

TEST_CASE("the --unproductive-iters flag takes zero on purpose", "[bench][flags]") {
    // SearchConfig documents `<= 0` as "restore the old fixed cadence", which is
    // the control arm for the #102 early exit -- so this flag must NOT get the
    // positivity guard every other integer flag has. Zero is its spelling.
    std::string error;
    const ParseOutcome parsed = parse({"--unproductive-iters", "0"});
    REQUIRE(cbls::bench::validate_search_flags(parsed.flags, kScalar, false, error));
    SearchConfig cfg;
    cbls::bench::apply_search_flags(parsed.flags, cfg);
    REQUIRE(cfg.unproductive_iterations == 0);
}

TEST_CASE("a negative --unproductive-iters is the same run as zero, so it is refused",
          "[bench][flags]") {
    // The engine arms the early exit only on `> 0`, so -1 and 0 produce the
    // IDENTICAL run -- but they record two different `search_config` cells.
    // That breaks the invariant search_config_string() states outright, and
    // that invariant is the justification for refusing the two no-effect flag
    // combinations elsewhere in this file. An earlier cut accepted both.
    std::string error;
    for (const char* value : {"-1", "-300"}) {
        const ParseOutcome parsed = parse({"--unproductive-iters", value});
        REQUIRE_FALSE(cbls::bench::validate_search_flags(parsed.flags, kScalar, false, error));
        REQUIRE(error.find("--unproductive-iters") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Probes: each flag changes the run, not merely the config object
// ---------------------------------------------------------------------------

TEST_CASE("the --no-lns flag stops the repairs the default arm makes", "[bench][flags][probe]") {
    // batch_iterations = 1 puts a batch boundary after every GLS iteration and
    // --perturbation-period 1 makes every one of them a kick, so the default
    // interval of 3 turns roughly a third of them into repairs. The two arms
    // differ in exactly one flag -- `--lns-interval` cannot appear here, since
    // it has no effect alongside `--no-lns` and is refused with it. Nothing
    // depends on machine speed: the budget is 25 GLS iterations, clock off.
    const std::vector<std::string> base = {"--perturbation-period", "1", "--max-iterations", "25"};
    std::vector<std::string> without = base;
    without.emplace_back("--no-lns");

    const SearchResult with_lns = probe(base, build_stagnant_model, /*batch_iterations=*/1);
    const SearchResult no_lns = probe(without, build_stagnant_model, /*batch_iterations=*/1);

    REQUIRE(with_lns.termination == TerminationReason::IterationLimit);
    REQUIRE(with_lns.lns_repairs > 0);  // not vacuous: the default arm really repairs
    REQUIRE(no_lns.lns_repairs == 0);
    // The kicks themselves survive -- --no-lns removes the expensive half of
    // diversification, not diversification.
    REQUIRE(no_lns.perturbations > 0);
}

TEST_CASE("the --lns-interval flag changes how often a kick is a repair", "[bench][flags][probe]") {
    // FJ batches only: a Novelty batch's iteration charge depends on the moves
    // it found, so with compound moves on the two arms' budgets end after
    // different numbers of batches and the kick counts stop being comparable.
    const SearchResult every = probe({"--perturbation-period", "1", "--lns-interval", "1",
                                      "--max-iterations", "25", "--no-compound-moves"},
                                     build_stagnant_model, /*batch_iterations=*/1);
    const SearchResult fifth = probe({"--perturbation-period", "1", "--lns-interval", "5",
                                      "--max-iterations", "25", "--no-compound-moves"},
                                     build_stagnant_model, /*batch_iterations=*/1);
    REQUIRE(fifth.lns_repairs > 0);
    REQUIRE(every.lns_repairs > fifth.lns_repairs);
    // Same number of kicks either way; only their composition changed.
    REQUIRE(every.perturbations == fifth.perturbations);
}

TEST_CASE("the --perturbation-period flag changes the diversification cadence",
          "[bench][flags][probe]") {
    // --no-lns on both arms: the cadence is what is under test, and letting the
    // kicks draw the expensive half of diversification would make the probe
    // slower without making it sharper.
    const SearchResult often =
        probe({"--no-lns", "--perturbation-period", "1", "--max-iterations", "200"},
              build_stagnant_model, /*batch_iterations=*/1);
    const SearchResult rarely =
        probe({"--no-lns", "--perturbation-period", "50", "--max-iterations", "200"},
              build_stagnant_model, /*batch_iterations=*/1);
    REQUIRE(rarely.perturbations > 0);
    REQUIRE(often.perturbations > rarely.perturbations);
}

TEST_CASE("the --unproductive-iters flag changes when a batch gives up", "[bench][flags][probe]") {
    // The #102 route, isolated from the perturbation_period route: 12 batches of
    // 50 iterations never reach a period of 100, so with the exit switched off
    // (`0`, the documented old-cadence arm) the run cannot kick at all. Switched
    // on at 5, a batch that has stopped reducing the real rows ends early and
    // the kick is taken as due.
    const SearchResult early = probe({"--no-lns", "--unproductive-iters", "5",
                                      "--perturbation-period", "100", "--max-iterations", "600"},
                                     build_stagnant_model, /*batch_iterations=*/50);
    const SearchResult never = probe({"--no-lns", "--unproductive-iters", "0",
                                      "--perturbation-period", "100", "--max-iterations", "600"},
                                     build_stagnant_model, /*batch_iterations=*/50);
    REQUIRE(early.perturbations > 0);
    REQUIRE(never.perturbations == 0);
}

TEST_CASE("the --max-iterations flag bounds the run", "[bench][flags][probe]") {
    const SearchResult small =
        probe({"--no-lns", "--max-iterations", "100"}, build_stagnant_model, 1);
    const SearchResult large =
        probe({"--no-lns", "--max-iterations", "500"}, build_stagnant_model, 1);
    REQUIRE(small.termination == TerminationReason::IterationLimit);
    REQUIRE(large.termination == TerminationReason::IterationLimit);
    REQUIRE(small.iterations < large.iterations);
    REQUIRE(small.iterations >= 100);
    REQUIRE(large.iterations >= 500);
}

TEST_CASE("the --no-time-limit flag hands solve() an iteration budget alone",
          "[bench][flags][probe]") {
    // The flag's whole job: solve() disables its wall clock at `time_limit <= 0`,
    // which both runners' `--time-limit > 0` guards otherwise make unreachable.
    const SearchFlags f = parse({"--no-time-limit", "--max-iterations", "200"}).flags;
    REQUIRE(cbls::bench::time_limit_argument(f, 600.0) == 0.0);

    Model m;
    build_stagnant_model(m);
    const SearchResult result = run_arm(m, f, /*batch_iterations=*/1);
    REQUIRE(result.termination == TerminationReason::IterationLimit);

    // And why validate_search_flags insists on the iteration budget: with
    // neither budget the same call does no work at all.
    SearchFlags no_budget = f;
    no_budget.max_iterations = 0;
    Model m2;
    build_stagnant_model(m2);
    const SearchResult empty = run_arm(m2, no_budget, /*batch_iterations=*/1);
    REQUIRE(empty.termination == TerminationReason::NoBudget);
    REQUIRE(empty.iterations == 0);
}

TEST_CASE("the --compound-moves flag changes the batches the search runs",
          "[bench][flags][probe]") {
    // With Novelty Jump on and its probability at 1 every batch is a compound
    // batch, which is a different search from the Feasibility-Jump-only arm.
    // Read off the batch counters, not the iteration count: since #201 that
    // reports the larger of GLS iterations and batches, so an all-Novelty run
    // reports its 300 batches against the FJ arm's 300 GLS iterations.
    const std::vector<std::string> budget = {"--no-lns", "--max-iterations", "300",
                                             "--perturbation-period", "5"};
    std::vector<std::string> on = budget;
    on.insert(on.end(), {"--compound-moves", "--novelty-prob", "1.0"});
    std::vector<std::string> off = budget;
    off.emplace_back("--no-compound-moves");

    const SearchResult off_arm = probe(off, build_stagnant_model, /*batch_iterations=*/10);
    const SearchResult on_arm = probe(on, build_stagnant_model, /*batch_iterations=*/10);
    REQUIRE(off_arm.counters.novelty_batches == 0);
    REQUIRE(on_arm.counters.novelty_batches > 0);
    REQUIRE(on_arm.counters.fj_batches == 0);
}

TEST_CASE("the --novelty-prob flag changes how many batches are compound",
          "[bench][flags][probe]") {
    // Isolates the probability from the switch: both arms have compound moves
    // enabled, so any difference is the probability's doing.
    const std::vector<std::string> budget = {
        "--no-lns", "--compound-moves", "--max-iterations", "300", "--perturbation-period",
        "5",        "--novelty-prob"};
    std::vector<std::string> always = budget;
    always.emplace_back("1.0");
    std::vector<std::string> never = budget;
    never.emplace_back("0.0");

    const SearchResult always_arm = probe(always, build_stagnant_model, /*batch_iterations=*/10);
    const SearchResult never_arm = probe(never, build_stagnant_model, /*batch_iterations=*/10);
    REQUIRE(never_arm.counters.novelty_batches == 0);
    REQUIRE(always_arm.counters.novelty_batches > 0);
    REQUIRE(always_arm.counters.fj_batches == 0);
}

TEST_CASE("the --no-float-hook flag changes where a continuous run lands",
          "[bench][flags][probe]") {
    // The hook is the only thing that descends the objective over Float
    // variables once a feasible point exists, so removing it must show up in the
    // objective -- on a model whose optimum (0.5, 0.5) is strictly inside the
    // region FeasibilityJump's own start lands in.
    const SearchResult with_hook =
        probe({"--max-iterations", "2000"}, build_continuous_model, /*batch_iterations=*/50);
    const SearchResult without_hook = probe({"--max-iterations", "2000", "--no-float-hook"},
                                            build_continuous_model, /*batch_iterations=*/50);
    REQUIRE(with_hook.feasible);
    REQUIRE(without_hook.feasible);
    REQUIRE(with_hook.objective != without_hook.objective);
    // Not merely different: the hook is what gets this model near its optimum.
    REQUIRE(with_hook.objective < without_hook.objective);
}

// ---------------------------------------------------------------------------
// The structural flags and the per-runner mask (#198)
// ---------------------------------------------------------------------------

TEST_CASE("every structural flag sets the field it claims", "[bench][flags]") {
    const ParseOutcome parsed =
        parse({"--structural-prob", "0.75", "--structural-selection", "best_of_sample",
               "--structural-sample-size", "3", "--neighbours", "4"});
    REQUIRE(parsed.unmatched.empty());
    std::string error;
    REQUIRE(cbls::bench::validate_search_flags(parsed.flags, kStructural, false, error));
    SearchConfig cfg;
    cbls::bench::apply_search_flags(parsed.flags, cfg);
    REQUIRE(cfg.structural_batch_probability == 0.75);
    REQUIRE(cfg.structural_selection == cbls::StructuralSelection::BestOfSample);
    REQUIRE(cfg.structural_sample_size == 3);
    const RoutingFixture fx;
    const auto list = cbls::bench::neighbours_argument(
        parsed.flags, RoutingFixture::kCustomers, [&fx](int a, int b) { return fx.cost(a, b); });
    REQUIRE(list != nullptr);
    REQUIRE(list->of(0).size() == 4);
    // Unflagged, the engine's defaults are untouched: automatic probability and
    // no neighbour list.
    const SearchFlags defaults;
    SearchConfig untouched;
    cbls::bench::apply_search_flags(defaults, untouched);
    REQUIRE(untouched.structural_batch_probability == SearchConfig{}.structural_batch_probability);
    REQUIRE(cbls::bench::neighbours_argument(defaults, 10, [](int, int) { return 0.0; }) ==
            nullptr);
}

TEST_CASE("a List runner's cell records the structural arm and omits the scalar flags",
          "[bench][flags]") {
    REQUIRE(cbls::bench::search_config_string(SearchFlags{}, kStructural) ==
            "lns=on;lns_interval=3;perturbation_period=100;max_iterations=0;time_limit=on;"
            "structural_prob=auto;structural_selection=first_improving;"
            "structural_sample_size=8;neighbours=off");
    const SearchFlags f = parse({"--structural-prob", "1", "--structural-selection",
                                 "violation_guided", "--neighbours", "20"})
                              .flags;
    REQUIRE(cbls::bench::search_config_string(f, kStructural) ==
            "lns=on;lns_interval=3;perturbation_period=100;max_iterations=0;time_limit=on;"
            "structural_prob=1;structural_selection=violation_guided;"
            "structural_sample_size=8;neighbours=20");
}

TEST_CASE("a flag outside the runner's families is refused, not recorded", "[bench][flags]") {
    std::string error;
    // Structural flags on a scalar runner -- even restated at the default,
    // which is still a flag that would be accepted and read by nothing.
    for (const std::vector<std::string>& args :
         std::vector<std::vector<std::string>>{{"--structural-prob", "1"},
                                               {"--structural-selection", "first_improving"},
                                               {"--structural-sample-size", "8"},
                                               {"--neighbours", "5"}}) {
        REQUIRE_FALSE(cbls::bench::validate_search_flags(parse(args).flags, kScalar, false, error));
        REQUIRE(error.find(args.front()) != std::string::npos);
    }
    // Scalar-only flags on a List runner.
    for (const std::vector<std::string>& args :
         std::vector<std::vector<std::string>>{{"--no-float-hook"},
                                               {"--compound-moves"},
                                               {"--no-compound-moves"},
                                               {"--novelty-prob", "0.5"},
                                               {"--unproductive-iters", "300"}}) {
        REQUIRE_FALSE(
            cbls::bench::validate_search_flags(parse(args).flags, kStructural, false, error));
        REQUIRE(error.find(args.front()) != std::string::npos);
    }
    // A List runner that supplies no cost does not declare the neighbour
    // family, and the flag is refused rather than ignored.
    const unsigned no_cost = kStructural & ~cbls::bench::kNeighbourSearchFlags;
    REQUIRE_FALSE(cbls::bench::validate_search_flags(parse({"--neighbours", "5"}).flags, no_cost,
                                                     false, error));
    REQUIRE(error.find("--neighbours") != std::string::npos);
    // The common flags are taken everywhere.
    for (const unsigned families : {kScalar, kStructural, no_cost}) {
        REQUIRE(cbls::bench::validate_search_flags(
            parse({"--no-lns", "--perturbation-period", "7", "--max-iterations", "9"}).flags,
            families, false, error));
    }
}

TEST_CASE("structural flag values the engine cannot use are refused", "[bench][flags]") {
    std::string error;
    auto rejected = [&error](const std::vector<std::string>& args) {
        return !cbls::bench::validate_search_flags(parse(args).flags, kStructural, false, error);
    };
    REQUIRE(rejected({"--structural-prob", "1.5"}));
    REQUIRE(error.find("--structural-prob") != std::string::npos);
    // "Automatic" is the default and is spelled by omitting the flag.
    REQUIRE(rejected({"--structural-prob", "-1"}));
    REQUIRE(rejected({"--structural-prob", "nope"}));
    REQUIRE(
        rejected({"--structural-selection", "best_of_sample", "--structural-sample-size", "0"}));
    REQUIRE(rejected(
        {"--structural-selection", "best_of_sample", "--structural-sample-size", "3000000000"}));
    REQUIRE(rejected({"--neighbours", "0"}));
    REQUIRE(error.find("--neighbours") != std::string::npos);
    REQUIRE(rejected({"--neighbours", "-2"}));
    // The no-effect pairs: the sample size is inert under first_improving, and
    // at probability 0 no structural batch runs to read any of them.
    REQUIRE(rejected({"--structural-sample-size", "4"}));
    REQUIRE(error.find("--structural-sample-size") != std::string::npos);
    REQUIRE(rejected({"--structural-prob", "0", "--neighbours", "5"}));
    REQUIRE(rejected({"--structural-prob", "0", "--structural-selection", "violation_guided"}));
    REQUIRE_FALSE(rejected({"--structural-prob", "0"}));
    // -0 is the 0 arm, and must be recorded as it: one run, one cell.
    REQUIRE(
        cbls::bench::search_config_string(parse({"--structural-prob", "-0"}).flags, kStructural) ==
        cbls::bench::search_config_string(parse({"--structural-prob", "0"}).flags, kStructural));
    REQUIRE(cbls::bench::search_config_string(parse({"--novelty-prob", "-0"}).flags, kScalar) ==
            cbls::bench::search_config_string(parse({"--novelty-prob", "0"}).flags, kScalar));
    // K past the instance's other elements is clamped by nearest_neighbours,
    // so the runner refuses it against its universe.
    REQUIRE(cbls::bench::validate_neighbours(parse({"--neighbours", "9"}).flags, 10, error));
    REQUIRE_FALSE(cbls::bench::validate_neighbours(parse({"--neighbours", "10"}).flags, 10, error));
    REQUIRE(error.find("--neighbours") != std::string::npos);
    REQUIRE(cbls::bench::validate_neighbours(SearchFlags{}, 1, error));
    REQUIRE_FALSE(rejected({"--structural-prob", "0", "--structural-selection", "first_improving",
                            "--structural-sample-size", "8"}));
}

TEST_CASE("the --structural-prob flag changes the batch mix", "[bench][flags][probe]") {
    // On a List-only model the automatic setting is 1.0 (#201); an explicit 0.5
    // is honoured, so half the batches become (empty) FJ batches.
    const std::vector<std::string> budget = {"--no-lns", "--max-iterations", "200",
                                             "--no-time-limit"};
    std::vector<std::string> half = budget;
    half.insert(half.end(), {"--structural-prob", "0.5"});
    const SearchResult auto_arm = structural_probe(budget);
    const SearchResult half_arm = structural_probe(half);
    REQUIRE(auto_arm.counters.fj_batches == 0);
    REQUIRE(half_arm.counters.fj_batches > 0);
    REQUIRE(half_arm.counters.structural_batches > 0);
}

TEST_CASE("the --structural-selection flag changes how many candidates are scored",
          "[bench][flags][probe]") {
    const std::vector<std::string> budget = {"--no-lns", "--max-iterations", "50",
                                             "--no-time-limit", "--structural-selection"};
    std::vector<std::string> first = budget;
    first.emplace_back("first_improving");
    std::vector<std::string> best = budget;
    best.emplace_back("best_of_sample");
    const SearchResult first_arm = structural_probe(first);
    const SearchResult best_arm = structural_probe(best);
    REQUIRE(first_arm.counters.structural_moves_tried != best_arm.counters.structural_moves_tried);
}

TEST_CASE("the --structural-sample-size flag changes how many candidates are scored",
          "[bench][flags][probe]") {
    const std::vector<std::string> budget = {"--no-lns",
                                             "--max-iterations",
                                             "50",
                                             "--no-time-limit",
                                             "--structural-selection",
                                             "best_of_sample",
                                             "--structural-sample-size"};
    std::vector<std::string> small = budget;
    small.emplace_back("1");
    std::vector<std::string> large = budget;
    large.emplace_back("40");
    const SearchResult small_arm = structural_probe(small);
    const SearchResult large_arm = structural_probe(large);
    REQUIRE(large_arm.counters.structural_moves_tried > small_arm.counters.structural_moves_tried);
}

TEST_CASE("the --neighbours flag changes the structural trajectory", "[bench][flags][probe]") {
    // A neighbour list redirects the built-in generators' targets, so the same
    // seed walks a different path. Read off the full set of observables a run
    // reports; any one of them differing is a different run.
    const std::vector<std::string> budget = {"--no-lns", "--max-iterations", "100",
                                             "--no-time-limit"};
    std::vector<std::string> near = budget;
    near.insert(near.end(), {"--neighbours", "2"});
    const SearchResult uniform = structural_probe(budget);
    const SearchResult granular = structural_probe(near);
    const bool differs =
        uniform.objective != granular.objective ||
        uniform.counters.structural_moves_tried != granular.counters.structural_moves_tried ||
        uniform.counters.structural_moves_accepted != granular.counters.structural_moves_accepted;
    REQUIRE(differs);
}
