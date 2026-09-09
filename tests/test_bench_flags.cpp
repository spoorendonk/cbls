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
#include <cbls/search.h>
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
    REQUIRE(cbls::bench::validate_search_flags(parsed.flags, /*time_limit_set=*/false, error));
    Model m;
    build(m);
    return run_arm(m, parsed.flags, batch_iterations);
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
    const ParseOutcome parsed =
        parse({"--no-float-hook", "--no-lns", "--lns-interval", "7", "--compound-moves",
               "--novelty-prob", "0.25", "--unproductive-iters", "0", "--perturbation-period", "9",
               "--max-iterations", "1234", "--no-time-limit"});
    REQUIRE(parsed.unmatched.empty());
    const SearchFlags& f = parsed.flags;
    REQUIRE_FALSE(f.float_hook);
    REQUIRE_FALSE(f.lns);
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
    REQUIRE(cbls::bench::lns_argument(f, lns) == nullptr);
    REQUIRE(cbls::bench::lns_interval_argument(f) == 7);
    REQUIRE(cbls::bench::time_limit_argument(f, 60.0) == 0.0);
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
    for (const std::string& flag : {"--lns-interval", "--novelty-prob", "--unproductive-iters",
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
    REQUIRE(cbls::bench::search_config_string(SearchFlags{}) ==
            "float_hook=on;lns=on;lns_interval=3;compound_moves=off;novelty_prob=0.5;"
            "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on");

    const SearchFlags f =
        parse({"--no-float-hook", "--no-lns", "--lns-interval", "7", "--compound-moves",
               "--novelty-prob", "0.25", "--unproductive-iters", "0", "--perturbation-period", "9",
               "--max-iterations", "1234", "--no-time-limit"})
            .flags;
    const std::string cell = cbls::bench::search_config_string(f);
    REQUIRE(cell ==
            "float_hook=off;lns=off;lns_interval=7;compound_moves=on;novelty_prob=0.25;"
            "unproductive_iters=0;perturbation_period=9;max_iterations=1234;time_limit=off");
    // No comma, or the cell would shift every column after it in a CSV that
    // does not quote (both runners' writers do not).
    REQUIRE(cell.find(',') == std::string::npos);
    // Same flags in a different order are the same arm, so the cell must be the
    // same string: a results file is meant to be grouped by this column.
    REQUIRE(cbls::bench::search_config_string(
                parse({"--max-iterations", "1234", "--no-time-limit", "--perturbation-period", "9",
                       "--unproductive-iters", "0", "--novelty-prob", "0.25", "--compound-moves",
                       "--lns-interval", "7", "--no-lns", "--no-float-hook"})
                    .flags) == cell);
}

TEST_CASE("the usage line documents exactly the recorded flags", "[bench][flags]") {
    // Both strings come from kSearchFlagSpecs, so this pins that they stay
    // generated from it rather than hand-maintained in parallel.
    const std::string usage = cbls::bench::search_flags_usage();
    const std::string cell = cbls::bench::search_config_string(SearchFlags{});
    for (const auto& spec : cbls::bench::kSearchFlagSpecs) {
        REQUIRE(usage.find(spec.usage) != std::string::npos);
        REQUIRE(cell.find(std::string(spec.key) + "=") != std::string::npos);
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
        {"--compound-moves"},
        {"--novelty-prob", "0.1"},
        {"--unproductive-iters", "0"},
        {"--perturbation-period", "7"},
        {"--max-iterations", "10"},
        {"--no-time-limit"},
    };
    REQUIRE(arms.size() == cbls::bench::kSearchFlagCount);
    for (const auto& arm : arms) {
        const ParseOutcome parsed = parse(arm);
        REQUIRE(parsed.unmatched.empty());
        REQUIRE(cbls::bench::first_non_default_search_flag(parsed.flags) != nullptr);
    }
    // A flag restated at its default is not an arm.
    REQUIRE(cbls::bench::first_non_default_search_flag(parse({"--no-compound-moves"}).flags) ==
            nullptr);
    REQUIRE(cbls::bench::first_non_default_search_flag(parse({"--lns-interval", "3"}).flags) ==
            nullptr);
}

TEST_CASE("validation rejects values the engine cannot use", "[bench][flags]") {
    std::string error;
    auto rejected = [&error](const std::vector<std::string>& args, bool time_limit_set = false) {
        return !cbls::bench::validate_search_flags(parse(args).flags, time_limit_set, error);
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

    REQUIRE(cbls::bench::validate_search_flags(SearchFlags{}, false, error));
    REQUIRE(cbls::bench::validate_search_flags(
        parse({"--no-time-limit", "--max-iterations", "10"}).flags, false, error));
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
    REQUIRE_FALSE(cbls::bench::validate_search_flags(parsed.flags, false, error));
    REQUIRE(error.find("--novelty-prob") != std::string::npos);
}

TEST_CASE("the --unproductive-iters flag takes a non-positive value on purpose", "[bench][flags]") {
    // SearchConfig documents `<= 0` as "restore the old fixed cadence", which is
    // the control arm for the #102 early exit -- so this flag must NOT get the
    // positivity guard every other integer flag has.
    std::string error;
    for (const std::string& value : {"0", "-1"}) {
        const ParseOutcome parsed = parse({"--unproductive-iters", value});
        REQUIRE(cbls::bench::validate_search_flags(parsed.flags, false, error));
        SearchConfig cfg;
        cbls::bench::apply_search_flags(parsed.flags, cfg);
        REQUIRE(cfg.unproductive_iterations <= 0);
    }
}

// ---------------------------------------------------------------------------
// Probes: each flag changes the run, not merely the config object
// ---------------------------------------------------------------------------

TEST_CASE("the --no-lns flag stops the repairs the default arm makes", "[bench][flags][probe]") {
    // batch_iterations = 1 puts a batch boundary after every GLS iteration, and
    // --perturbation-period 1 makes every one of them a kick; --lns-interval 1
    // makes every kick an LNS repair. Nothing here depends on machine speed: the
    // budget is 25 GLS iterations and the clock is off.
    const std::vector<std::string> base = {"--perturbation-period", "1", "--lns-interval", "1",
                                           "--max-iterations",      "25"};
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
    const SearchResult every =
        probe({"--perturbation-period", "1", "--lns-interval", "1", "--max-iterations", "25"},
              build_stagnant_model, /*batch_iterations=*/1);
    const SearchResult fifth =
        probe({"--perturbation-period", "1", "--lns-interval", "5", "--max-iterations", "25"},
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
    // batch, which is a different search from the Feasibility-Jump-only default
    // -- and Novelty batches do not charge fj.iterations(), so the difference
    // shows in the iteration count as well as in the trajectory.
    const std::vector<std::string> budget = {"--no-lns", "--max-iterations", "300",
                                             "--perturbation-period", "5"};
    std::vector<std::string> on = budget;
    on.insert(on.end(), {"--compound-moves", "--novelty-prob", "1.0"});

    const SearchResult off_arm = probe(budget, build_stagnant_model, /*batch_iterations=*/10);
    const SearchResult on_arm = probe(on, build_stagnant_model, /*batch_iterations=*/10);
    REQUIRE(off_arm.iterations != on_arm.iterations);
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
    REQUIRE(always_arm.iterations != never_arm.iterations);
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
