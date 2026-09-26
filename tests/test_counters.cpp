// `SearchResult::counters` (#169): where a run spent its work.
//
// The counters are observational, so the tests that matter are the IDENTITIES
// rather than the magnitudes: a bucket sum that does not reconstruct the total,
// or a portfolio total that does not reconstruct its workers', is a counter
// nobody can read. Magnitudes are asserted only where a zero would mean the
// mechanism never ran at all.
//
// Every run here is iteration-budgeted with `time_limit = 0.0`, both so the
// numbers are reproducible and because that is the regime in which
// `inner_solver_seconds` is deliberately blind -- which this file also pins.

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

Model quadratic_model() {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));  // x + y >= 1
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
    return m;
}

// A List and a Set, both consumed by the objective and a constraint, so the
// structural batch has generators to sweep and moves to score. An unconsumed
// structured variable would give a sweep with nothing to improve.
Model structured_model() {
    Model m;
    auto route = m.list_var(8, "route");
    auto chosen = m.set_var(10, 3, 6, "chosen");
    std::vector<std::vector<double>> cost(8, std::vector<double>(8, 0.0));
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            cost[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                static_cast<double>(((i * 7) + (j * 3)) % 11);
        }
    }
    const int32_t tour = m.pair_lambda_sum(
        route,
        [cost](int a, int b) { return cost[static_cast<size_t>(a)][static_cast<size_t>(b)]; },
        PairMode::Cyclic);
    const int32_t load =
        m.lambda_sum(chosen, [](int e) { return static_cast<double>((e * 5) % 7) + 1.0; });
    m.add_constraint(m.sum({m.constant(-12.0), load}));  // load >= 12
    m.minimize(m.sum({tour, load}));
    m.close();
    return m;
}

// Counts this worker's batch_end events into a slot the TEST keeps, so the count
// outlives the worker. A tracer dies with its worker, so a plain member would be
// unreadable by the time the portfolio returns.
struct CountingTracer : Tracer {
    std::shared_ptr<std::atomic<int64_t>> batches;
    explicit CountingTracer(std::shared_ptr<std::atomic<int64_t>> slot)
        : batches(std::move(slot)) {}
    void batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool /*improved*/) override {
        batches->fetch_add(1, std::memory_order_relaxed);
    }
};

SearchResult run(Model& model, const SearchConfig& config, uint64_t seed,
                 InnerSolverHook* hook = nullptr) {
    return solve(model, /*time_limit=*/0.0, seed, true, hook, nullptr, 3, nullptr, config);
}

}  // namespace

TEST_CASE("batches by kind sum to the batch count", "[counters]") {
    // The identity the breakdown rests on: `pick_batch_kind` returns exactly one
    // of three, and every batch is counted once. A kind that stopped being
    // counted -- a fourth added without a bucket, say -- shows up here and
    // nowhere else.
    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    // Both non-default batch kinds armed, so all three buckets can be nonzero.
    config.use_compound_moves = true;
    config.novelty_jump_probability = 0.3;

    Model m = structured_model();
    const SearchResult r = run(m, config, /*seed=*/4);
    const SearchCounters& c = r.counters;

    REQUIRE(c.batches > 0);
    REQUIRE(c.fj_batches + c.novelty_batches + c.structural_batches == c.batches);
    // Not a tautology of the line above: a model with List/Set variables and the
    // automatic structural probability must actually run structural batches, or
    // the identity would hold trivially with everything in one bucket.
    REQUIRE(c.structural_batches > 0);
    REQUIRE(c.fj_batches > 0);
    // The third bucket, and the only place in the tree it is shown nonzero. The
    // identity above holds just as well with novelty batches counted into
    // fj_batches, so without this the whole file passes on that bug.
    REQUIRE(c.novelty_batches > 0);
}

TEST_CASE("a scalar model runs feasibility-jump batches only", "[counters]") {
    // The control for the test above, and the statement that the buckets track
    // the batch kind rather than just the count: with no structured variable the
    // structural probability resolves to 0.0, and compound moves are off by
    // default.
    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;

    Model m = quadratic_model();
    const SearchCounters& c = run(m, config, /*seed=*/4).counters;

    REQUIRE(c.batches > 0);
    REQUIRE(c.fj_batches == c.batches);
    REQUIRE(c.novelty_batches == 0);
    REQUIRE(c.structural_batches == 0);
    REQUIRE(c.by_generator.empty());
    REQUIRE(c.structural_moves_tried == 0);
}

TEST_CASE("structural counters total their per-generator rows", "[counters][structural]") {
    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;

    Model m = structured_model();
    const SearchCounters& c = run(m, config, /*seed=*/6).counters;

    // One row per built-in generator: one per structured variable, so two here.
    REQUIRE(c.by_generator.size() == 2);
    int64_t tried = 0;
    int64_t accepted = 0;
    for (const GeneratorCounters& g : c.by_generator) {
        REQUIRE_FALSE(g.name.empty());
        REQUIRE(g.moves_accepted <= g.moves_tried);
        tried += g.moves_tried;
        accepted += g.moves_accepted;
    }
    REQUIRE(c.structural_moves_tried == tried);
    REQUIRE(c.structural_moves_accepted == accepted);
    // A sweep that scored nothing would satisfy every equality above.
    REQUIRE(c.structural_moves_tried > 0);
}

TEST_CASE("BestOfSample commits at most one candidate per sample", "[counters][structural]") {
    // The counters make the policy difference visible, which is the point of
    // counting tried and accepted separately rather than only "moves". Under
    // BestOfSample each generator call scores its whole sample and commits one
    // winner at most, so the acceptance ratio is bounded by the number of
    // sweeps -- where FirstImprovingSample may commit several from one sample.
    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;
    config.structural_selection = StructuralSelection::BestOfSample;

    Model m = structured_model();
    const SearchCounters& c = run(m, config, /*seed=*/6).counters;

    REQUIRE(c.structural_batches > 0);
    REQUIRE(c.structural_moves_tried > 0);
    // Not vacuous: the bound below is satisfied by zero acceptances, so the sweep
    // has to be shown to commit something before it says anything.
    REQUIRE(c.structural_moves_accepted > 0);
    // At most one commit per generator per structural batch.
    REQUIRE(c.structural_moves_accepted <=
            c.structural_batches * static_cast<int64_t>(c.by_generator.size()));
}

TEST_CASE("the inner solver is counted but not timed without a clock", "[counters]") {
    // Both halves matter. The CALL COUNT has to be right on an iteration-budgeted
    // run, because that is the regime the tests and the determinism claim live
    // in. The SECONDS have to stay zero there, because filling them would mean
    // the loop reads a clock on a run documented to read none -- see
    // SearchCounters::inner_solver_seconds and docs/architecture.md.
    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;
    FloatIntensifyHook hook;

    Model m = quadratic_model();
    const SearchCounters& c = run(m, config, /*seed=*/8, &hook).counters;

    REQUIRE(c.inner_solver_calls > 0);
    REQUIRE(c.inner_solver_seconds == 0.0);
}

TEST_CASE("the inner solver is timed when the run has a clock", "[counters]") {
    // The other side of the gate: with a wall-clock budget the seconds are
    // filled, so a host asking "what did the polish cost" gets an answer on the
    // runs where the question is meaningful.
    SearchConfig config;
    config.batch_iterations = 100;
    FloatIntensifyHook hook;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.3, /*seed=*/8, true, &hook, nullptr, 3, nullptr, config);

    REQUIRE(r.counters.inner_solver_calls > 0);
    REQUIRE(r.counters.inner_solver_seconds > 0.0);
    // Sanity, not a timing assertion: the polish cannot have taken longer than
    // the run that contained it.
    REQUIRE(r.counters.inner_solver_seconds <= r.time_seconds);
}

TEST_CASE("no inner solver means no inner-solver calls", "[counters]") {
    SearchConfig config;
    config.max_iterations = 2000;

    Model m = quadratic_model();
    const SearchCounters& c = run(m, config, /*seed=*/8).counters;

    REQUIRE(c.inner_solver_calls == 0);
    REQUIRE(c.inner_solver_seconds == 0.0);
    REQUIRE(c.portfolio_restarts == 0);  // a single solve cannot restart itself
}

TEST_CASE("merge sums scalars and merges generator rows by name", "[counters]") {
    // Unit-level, because both portfolio aggregation sites call this one function
    // and the name-keyed half is what keeps a worker that built no batch from
    // shifting the rows of the ones that did.
    SearchCounters a;
    a.structural_moves_tried = 15;
    a.structural_moves_accepted = 3;
    a.batches = 3;
    a.fj_batches = 2;
    a.structural_batches = 1;
    a.inner_solver_calls = 1;
    a.inner_solver_seconds = 0.25;
    a.portfolio_restarts = 1;
    a.by_generator.push_back({"list:0", 10, 2});
    a.by_generator.push_back({"set:1", 5, 1});

    SearchCounters b;
    b.structural_moves_tried = 8;
    b.structural_moves_accepted = 4;
    b.batches = 4;
    b.novelty_batches = 4;
    b.inner_solver_calls = 2;
    b.inner_solver_seconds = 0.5;
    b.portfolio_restarts = 2;
    // Deliberately in the other order, and with one name `a` does not have.
    b.by_generator.push_back({"set:1", 7, 3});
    b.by_generator.push_back({"list:9", 1, 1});

    a.merge(b);

    REQUIRE(a.batches == 7);
    REQUIRE(a.fj_batches == 2);
    REQUIRE(a.novelty_batches == 4);
    REQUIRE(a.structural_batches == 1);
    REQUIRE(a.inner_solver_calls == 3);
    REQUIRE(a.inner_solver_seconds == 0.75);
    REQUIRE(a.portfolio_restarts == 3);
    // The totals are summed as scalars rather than re-derived from the rows below,
    // which is what keeps them consistent with a `by_generator` merge that appends
    // a name the other side did not have.
    REQUIRE(a.structural_moves_tried == 23);
    REQUIRE(a.structural_moves_accepted == 7);

    REQUIRE(a.by_generator.size() == 3);
    REQUIRE(a.by_generator[0].name == "list:0");
    REQUIRE(a.by_generator[0].moves_tried == 10);
    // Matched by name, not by position: `set:1` is `b`'s first row and `a`'s
    // second.
    REQUIRE(a.by_generator[1].name == "set:1");
    REQUIRE(a.by_generator[1].moves_tried == 12);
    REQUIRE(a.by_generator[1].moves_accepted == 4);
    REQUIRE(a.by_generator[2].name == "list:9");
    REQUIRE(a.by_generator[2].moves_tried == 1);
}

TEST_CASE("merge walks positionally when the two row sets agree", "[counters]") {
    // The shape every portfolio worker actually produces -- same generators, same
    // order, suffixed names -- which the test above does not reach: it differs in
    // one name and so falls through to the by-name scan. This does not prove WHICH
    // path ran, and is not meant to: the two must agree, and that agreement is the
    // property. What it pins is the walk's own arithmetic, which nothing else
    // touches.
    SearchCounters a;
    a.by_generator.push_back({"builtin_list#0", 10, 2});
    a.by_generator.push_back({"builtin_list#1", 5, 1});

    SearchCounters b;
    b.by_generator.push_back({"builtin_list#0", 3, 1});
    b.by_generator.push_back({"builtin_list#1", 7, 4});

    a.merge(b);

    REQUIRE(a.by_generator.size() == 2);
    REQUIRE(a.by_generator[0].name == "builtin_list#0");
    REQUIRE(a.by_generator[0].moves_tried == 13);
    REQUIRE(a.by_generator[0].moves_accepted == 3);
    REQUIRE(a.by_generator[1].name == "builtin_list#1");
    REQUIRE(a.by_generator[1].moves_tried == 12);
    REQUIRE(a.by_generator[1].moves_accepted == 5);
}

TEST_CASE("an empty target takes every row unchanged", "[counters]") {
    // The FIRST merge at both aggregation sites, and the one the size compare
    // would otherwise send down the by-name path once per worker.
    SearchCounters empty;
    SearchCounters other;
    other.batches = 5;
    other.by_generator.push_back({"builtin_set#0", 4, 2});
    other.by_generator.push_back({"builtin_set#1", 6, 0});

    empty.merge(other);

    REQUIRE(empty.batches == 5);
    REQUIRE(empty.by_generator.size() == 2);
    REQUIRE(empty.by_generator[0].name == "builtin_set#0");
    REQUIRE(empty.by_generator[0].moves_tried == 4);
    REQUIRE(empty.by_generator[1].moves_accepted == 0);
}

TEST_CASE("BatchKind has a distinct stable token", "[counters]") {
    const std::vector<BatchKind> all = {BatchKind::FeasibilityJump, BatchKind::NoveltyJump,
                                        BatchKind::Structural};
    std::vector<std::string> tokens;
    for (BatchKind k : all) {
        REQUIRE(batch_kind_name(k) != nullptr);
        tokens.emplace_back(batch_kind_name(k));
    }
    REQUIRE(tokens[0] == "feasibility_jump");
    REQUIRE(tokens[1] == "novelty_jump");
    REQUIRE(tokens[2] == "structural");
}

TEST_CASE("two generators of one kind keep separate rows", "[counters][structural]") {
    // The built-ins name themselves by TYPE -- `StandardStructuralGenerator::name()`
    // returns "builtin_list" or "builtin_set" -- so a model with two List variables
    // gives two generators with the SAME name. `SearchCounters::merge` keys
    // `by_generator` on that name, and a portfolio merges through it even at one
    // worker with no restart, so a collision makes the portfolio report a different
    // row shape for the same model than `solve()` does, with one row absorbing the
    // other's counts.
    auto build = [] {
        Model m;
        auto a = m.list_var(6, "a");
        auto b = m.list_var(6, "b");
        auto cost = [](int i, int j) { return static_cast<double>(((i * 7) + (j * 3)) % 11); };
        m.minimize(m.sum({m.pair_lambda_sum(a, cost, PairMode::Cyclic),
                          m.pair_lambda_sum(b, cost, PairMode::Cyclic)}));
        m.close();
        return m;
    };

    SearchConfig config;
    config.max_iterations = 300;
    config.structural_batch_probability = 1.0;

    Model single = build();
    const SearchCounters& direct = run(single, config, /*seed=*/5).counters;
    REQUIRE(direct.by_generator.size() == 2);
    REQUIRE(direct.by_generator[0].name != direct.by_generator[1].name);

    // One worker, no restart: the only merge is absorb's, into an empty vector.
    ParallelConfig par_config;
    par_config.n_threads = 1;
    ParallelSearch ps(1);
    const SearchResult r = ps.solve(build, /*time_limit=*/0.3, /*seed=*/5, config,
                                    /*hook_factory=*/nullptr, /*lns_factory=*/nullptr,
                                    /*callback=*/nullptr, par_config);
    REQUIRE(r.counters.by_generator.size() == 2);
    int64_t tried = 0;
    for (const GeneratorCounters& g : r.counters.by_generator) {
        tried += g.moves_tried;
    }
    REQUIRE(r.counters.structural_moves_tried == tried);
}

TEST_CASE("a portfolio's counters are the sum of its workers'", "[counters][parallel]") {
    // The half-aggregation bug this guards against: the portfolio sums some
    // counters in `WorkerAccumulator::absorb` and some in `solve_portfolio`'s own
    // loop, and a field added to one and not the other reads as a mechanism that
    // never fired. Both sites go through `SearchCounters::merge`.
    //
    // A wall clock, not an iteration budget: with no shared clock a worker runs
    // exactly one solve and never restarts, which would leave
    // `portfolio_restarts` untestable here.
    SearchConfig config;
    config.batch_iterations = 100;

    // Each worker counts its OWN batch_end events into a slot this test keeps, so
    // the aggregate can be compared against its parts from OUTSIDE the
    // aggregation. Without that the test could only re-check an identity that
    // holds within the aggregate -- which a half-aggregation bug preserves.
    std::mutex mutex;
    std::vector<std::shared_ptr<std::atomic<int64_t>>> per_worker;

    ParallelConfig par_config;
    par_config.n_threads = 3;
    par_config.tracer_factory = [&mutex, &per_worker](int /*worker*/) -> std::unique_ptr<Tracer> {
        auto slot = std::make_shared<std::atomic<int64_t>>(0);
        {
            const std::scoped_lock lock(mutex);
            per_worker.push_back(slot);
        }
        return std::make_unique<CountingTracer>(std::move(slot));
    };

    ParallelSearch ps(3);
    const SearchResult r = ps.solve(
        [] { return structured_model(); }, /*time_limit=*/0.4, /*seed=*/21, config,
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);
    const SearchCounters& c = r.counters;

    // THE aggregation check: each worker's OWN tracer count, summed outside the
    // portfolio, equals the batches it reports. Be precise about what that does
    // and does not cover: `count_batch` and `trace_batch_end` are adjacent
    // statements in one loop body, so a bug in COUNTING a batch moves both numbers
    // together and this cannot see it. What it does see is the AGGREGATION -- the
    // tracer side never touches `SearchCounters::merge` -- across a worker's
    // restarts and across the workers, which is the half-aggregation bug named
    // above.
    REQUIRE(per_worker.size() == 3);
    int64_t observed = 0;
    for (const auto& slot : per_worker) {
        observed += slot->load(std::memory_order_relaxed);
    }
    REQUIRE(observed == c.batches);

    // The same identity as the single-threaded case: summing three workers'
    // buckets preserves it.
    REQUIRE(c.batches > 0);
    REQUIRE(c.fj_batches + c.novelty_batches + c.structural_batches == c.batches);
    REQUIRE(c.structural_batches > 0);
    // Merged by name across workers, so three workers on one model still give one
    // row per generator rather than three.
    REQUIRE(c.by_generator.size() == 2);
    int64_t tried = 0;
    for (const GeneratorCounters& g : c.by_generator) {
        tried += g.moves_tried;
    }
    REQUIRE(c.structural_moves_tried == tried);
}

TEST_CASE("a restarted worker's restarts are counted", "[counters][parallel]") {
    // A tight iteration budget with a live wall clock is what makes a worker
    // return early and be restarted on the time its predecessor left -- the
    // property `pool.h` states and `tests/test_parallel.cpp` pins from the
    // iteration side. Here it is the counter that has to see it.
    SearchConfig config;
    config.max_iterations = 200;
    config.batch_iterations = 50;

    ParallelConfig par_config;
    par_config.n_threads = 2;

    ParallelSearch ps(2);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/23, config,
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(r.counters.portfolio_restarts > 0);
    // Not load-bearing -- two workers at 200 iterations each already exceed this
    // without any restart. The counter above is the assertion; this only says the
    // run did real work.
    REQUIRE(r.iterations > config.max_iterations);
}
