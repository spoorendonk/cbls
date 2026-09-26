// `Tracer` (#169): the host's event stream, separate from the throttled
// `SolveCallback` progress stream.
//
// Two properties are worth testing and the rest is noise:
//
//  1. a recording tracer sees ORDERED, WELL-FORMED events for a short
//     deterministic run -- and the counts it derives agree with the counters the
//     same run reports, which is the only independent check available on an
//     event stream;
//  2. a null tracer changes NOTHING -- same objective, same iteration count,
//     same final assignment, event for event with the run that had one.
//
// What is deliberately NOT tested is a per-iteration event, because there is
// none: every method fires at batch, kick or inner-solver granularity. A test
// asserting a per-iteration cadence would be asserting a contract the class is
// written to refuse.

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
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

// A 20-column integer model: the target is easy to reach and the optimum is the
// balanced assignment, so the search plateaus and keeps stagnating -- which is
// what makes a diversification kick actually fire. The quadratic model above
// improves on nearly every batch and takes almost no kicks at all.
Model integer_model() {
    Model m;
    std::vector<int32_t> xs;
    std::vector<int32_t> squares;
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    for (int i = 0; i < 20; ++i) {
        xs.push_back(m.int_var(0, 10));
        squares.push_back(m.pow_expr(xs.back(), two));
    }
    std::vector<int32_t> row;
    row.push_back(m.constant(60.0));
    for (int32_t x : xs) {
        row.push_back(m.prod(neg1, x));
    }
    m.add_constraint(m.sum(row));
    m.minimize(m.sum(squares));
    m.close();
    return m;
}

// Records the stream in order, plus just enough per-kind state to assert the
// ordering rules without the test having to parse a log.
class RecordingTracer : public Tracer {
public:
    struct Event {
        enum class Kind : std::uint8_t { BatchEnd, NewBest, Kick, Lns, Hook };
        Kind kind;
        BatchKind batch = BatchKind::FeasibilityJump;
        KickKind kick_kind = KickKind::Perturb;
        int64_t iterations = 0;
        double objective = 0.0;
        double seconds = 0.0;
        bool flag = false;  // `improved` for BatchEnd, `accepted` for Lns
    };

    void batch_end(BatchKind kind, int64_t iterations, bool improved) override {
        events_.push_back(
            {Event::Kind::BatchEnd, kind, KickKind::Perturb, iterations, 0.0, 0.0, improved});
    }
    void new_best(double objective, double seconds) override {
        events_.push_back({Event::Kind::NewBest, BatchKind::FeasibilityJump, KickKind::Perturb, 0,
                           objective, seconds, false});
    }
    void kick(KickKind kind) override {
        events_.push_back(
            {Event::Kind::Kick, BatchKind::FeasibilityJump, kind, 0, 0.0, 0.0, false});
    }
    void lns(bool accepted) override {
        events_.push_back({Event::Kind::Lns, BatchKind::FeasibilityJump, KickKind::Perturb, 0, 0.0,
                           0.0, accepted});
    }
    void hook(double seconds) override {
        events_.push_back({Event::Kind::Hook, BatchKind::FeasibilityJump, KickKind::Perturb, 0, 0.0,
                           seconds, false});
    }

    [[nodiscard]] const std::vector<Event>& events() const { return events_; }
    [[nodiscard]] int64_t count(Event::Kind kind) const {
        int64_t n = 0;
        for (const Event& e : events_) {
            n += (e.kind == kind) ? 1 : 0;
        }
        return n;
    }

private:
    std::vector<Event> events_;
};

using Kind = RecordingTracer::Event::Kind;

}  // namespace

TEST_CASE("a recording tracer sees ordered batch events", "[tracer]") {
    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;
    config.tracer = &tracer;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/12, true, nullptr, nullptr, 3, nullptr, config);

    REQUIRE_FALSE(tracer.events().empty());
    // One batch_end per batch, and the run's own counter agrees. This is the
    // check that makes the stream trustworthy: the tracer and the counters are
    // filled at different sites and neither reads the other.
    REQUIRE(tracer.count(Kind::BatchEnd) == r.counters.batches);

    // `iterations` is the run's cumulative GLS count, so it is non-decreasing
    // across the stream and ends at the value the result reports.
    int64_t previous = -1;
    int64_t last_batch_iterations = -1;
    for (const auto& e : tracer.events()) {
        if (e.kind != Kind::BatchEnd) {
            continue;
        }
        REQUIRE(e.iterations >= previous);
        previous = e.iterations;
        last_batch_iterations = e.iterations;
        REQUIRE(e.batch == BatchKind::FeasibilityJump);  // a scalar model runs FJ only
    }
    REQUIRE(last_batch_iterations == r.iterations);

    // A new best arrives before the batch_end of the batch that produced it,
    // and the first event of a run that improves on its first batch is
    // therefore a new_best rather than a batch_end.
    REQUIRE(tracer.count(Kind::NewBest) > 0);
    REQUIRE(tracer.events().front().kind == Kind::NewBest);
}

TEST_CASE("a batch that improves is reported as improving", "[tracer]") {
    // The `improved` flag has to track the batch that produced the new best, or
    // a host correlating the two reads the improvement against the wrong batch.
    // Every new_best must be followed by a batch_end with improved = true
    // before the next batch_end that is not.
    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 3000;
    config.batch_iterations = 100;
    config.tracer = &tracer;

    Model m = quadratic_model();
    solve(m, /*time_limit=*/0.0, /*seed=*/12, true, nullptr, nullptr, 3, nullptr, config);

    bool pending_best = false;
    int64_t checked = 0;
    for (const auto& e : tracer.events()) {
        if (e.kind == Kind::NewBest) {
            pending_best = true;
            continue;
        }
        if (e.kind != Kind::BatchEnd) {
            continue;
        }
        if (pending_best) {
            REQUIRE(e.flag);
            ++checked;
            pending_best = false;
        }
    }
    REQUIRE(checked > 0);
}

TEST_CASE("a perturb kick is reported once per kick", "[tracer]") {
    // The kick count has to match `SearchResult::perturbations`, which counts
    // the same thing from the other side. `perturbation_period = 1` makes every
    // non-improving batch a kick, so the count is large enough to be meaningful.
    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 3000;
    config.batch_iterations = 50;
    config.perturbation_period = 1;
    config.tracer = &tracer;

    Model m = integer_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/14, true, nullptr, nullptr, 3, nullptr, config);

    REQUIRE(r.perturbations > 0);
    REQUIRE(tracer.count(Kind::Kick) == r.perturbations);
    // No LNS was supplied, so no kick can have drawn its LNS half.
    REQUIRE(tracer.count(Kind::Lns) == 0);
    for (const auto& e : tracer.events()) {
        if (e.kind == Kind::Kick) {
            REQUIRE(e.kick_kind == KickKind::Perturb);
        }
    }
}

TEST_CASE("an LNS kick reports its kind and its outcome", "[tracer]") {
    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 3000;
    config.batch_iterations = 50;
    config.perturbation_period = 1;
    config.lns_interval = 2;
    config.tracer = &tracer;
    LNS lns(0.3);

    Model m = integer_model();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/14, true, nullptr, &lns,
                                 config.lns_interval, nullptr, config);

    REQUIRE(r.lns_repairs > 0);
    // One LNS kick event and one outcome event per repair, and they agree with
    // the counters the result reports.
    REQUIRE(tracer.count(Kind::Lns) == r.lns_repairs);
    int64_t lns_kicks = 0;
    int64_t accepted = 0;
    for (const auto& e : tracer.events()) {
        lns_kicks += (e.kind == Kind::Kick && e.kick_kind == KickKind::LNS) ? 1 : 0;
        accepted += (e.kind == Kind::Lns && e.flag) ? 1 : 0;
    }
    REQUIRE(lns_kicks == r.lns_repairs);
    REQUIRE(accepted == r.lns_repairs_accepted);
    // Every kick is still counted once in total, LNS ones included.
    REQUIRE(tracer.count(Kind::Kick) == r.perturbations);

    // A `KickKind::LNS` is immediately followed by its `lns()` outcome, which is
    // what lets a host pair them without a correlation id.
    const auto& events = tracer.events();
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == Kind::Kick && events[i].kick_kind == KickKind::LNS) {
            REQUIRE(i + 1 < events.size());
            REQUIRE(events[i + 1].kind == Kind::Lns);
        }
    }
}

TEST_CASE("an inner-solver call is reported with its duration", "[tracer]") {
    // The one event that costs a clock read on a run with no wall clock, which
    // is why the counter beside it reads 0.0 there and this does not. See
    // SearchCounters::inner_solver_seconds and Tracer's own note.
    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 3000;
    config.batch_iterations = 100;
    config.tracer = &tracer;
    FloatIntensifyHook hook;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/16, true, &hook, nullptr, 3, nullptr, config);

    REQUIRE(tracer.count(Kind::Hook) == r.counters.inner_solver_calls);
    REQUIRE(tracer.count(Kind::Hook) > 0);
    double traced_seconds = 0.0;
    for (const auto& e : tracer.events()) {
        if (e.kind == Kind::Hook) {
            REQUIRE(e.seconds >= 0.0);
            traced_seconds += e.seconds;
        }
    }
    // Summed and required POSITIVE, which `>= 0.0` per event is not. The subject of
    // this test is that a tracer WIDENS polish_and_record's timing gate, and a gate
    // that still only timed the hook under has_deadline_ would reach here handing
    // every event a 0.0.
    REQUIRE(traced_seconds > 0.0);
    // The counter is blind on a clockless run; the event is not. Both halves of
    // that asymmetry are deliberate.
    REQUIRE(r.counters.inner_solver_seconds == 0.0);
}

TEST_CASE("structural batches are reported as structural", "[tracer][structural]") {
    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 300;
    config.batch_iterations = 50;
    // Every batch structural, so the kind is asserted rather than sampled.
    config.structural_batch_probability = 1.0;
    config.tracer = &tracer;

    Model m;
    auto route = m.list_var(8, "route");
    auto cost = [](int a, int b) { return static_cast<double>(((a * 7) + (b * 3)) % 11); };
    m.minimize(m.pair_lambda_sum(route, cost, PairMode::Cyclic));
    m.close();

    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/18, true, nullptr, nullptr, 3, nullptr, config);

    REQUIRE(tracer.count(Kind::BatchEnd) == r.counters.batches);
    REQUIRE(r.counters.structural_batches == r.counters.batches);
    for (const auto& e : tracer.events()) {
        if (e.kind == Kind::BatchEnd) {
            REQUIRE(e.batch == BatchKind::Structural);
        }
    }
}

TEST_CASE("an adoption is reported as an Adopt kick", "[tracer][parallel]") {
    // `KickKind::Adopt` is the one kick a single-threaded run can never take, and
    // the arm that emits it is one `else` branch in `maybe_diversify` -- so
    // without this, deleting that branch fails nothing.
    //
    // The harness is `tests/test_parallel.cpp`'s: a capacity-one pool holding a
    // gift the search cannot reach on its own, so the draw is deterministic and
    // the adoption actually happens. `SearchCoordination` is the portfolio's own
    // channel, driven here directly because that is far cheaper and far more
    // deterministic than arranging a real portfolio to stall.
    constexpr int kVars = 80;
    constexpr double kTarget = 240.0;
    auto build = []() {
        Model m;
        std::vector<int32_t> xs;
        std::vector<int32_t> squares;
        auto two = m.constant(2);
        auto neg1 = m.constant(-1.0);
        xs.reserve(kVars);
        for (int i = 0; i < kVars; ++i) {
            xs.push_back(m.int_var(0, 10));
            squares.push_back(m.pow_expr(xs.back(), two));
        }
        std::vector<int32_t> row;
        row.push_back(m.constant(kTarget));
        for (int32_t x : xs) {
            row.push_back(m.prod(neg1, x));
        }
        m.add_constraint(m.sum(row));
        m.minimize(m.sum(squares));
        m.close();
        return m;
    };

    Model donor = build();
    Model::State balanced = donor.copy_state();
    for (int i = 0; i < kVars; ++i) {
        balanced.values[static_cast<size_t>(i)] = 3.0;
    }
    Solution gift;
    gift.state = balanced;
    gift.objective = 720.0;  // the balanced optimum: 80 * 3^2
    gift.feasible = true;
    SolutionPool pool(1);
    pool.submit(gift);
    SearchCoordination coord;
    coord.pool = &pool;

    RecordingTracer tracer;
    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;  // reach a full-period kick inside the budget
    config.tracer = &tracer;

    Model m = build();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/3, true, nullptr, nullptr, 3,
                                 nullptr, config, &coord);

    REQUIRE(r.feasible);
    int64_t adoptions = 0;
    for (const auto& e : tracer.events()) {
        adoptions += (e.kind == Kind::Kick && e.kick_kind == KickKind::Adopt) ? 1 : 0;
    }
    REQUIRE(adoptions > 0);
    // An adoption REPLACES the perturb half of the kick but is still a kick, and
    // `adopt_from_pool` bumps `perturbations_` for it -- so the three kinds
    // together still reconstruct `SearchResult::perturbations`, which is the same
    // identity the perturb-only test above asserts. What the event stream adds is
    // the BREAKDOWN: the result records only the total, so an adoption is
    // indistinguishable from a perturb there.
    REQUIRE(tracer.count(Kind::Kick) == r.perturbations);
    REQUIRE(adoptions < tracer.count(Kind::Kick));
}

TEST_CASE("a null tracer changes nothing", "[tracer]") {
    // The load-bearing test, and the reason the default is a null pointer: every
    // benchmark in the tree and every published figure comes from a run without
    // one, so attaching a tracer must be the only difference a tracer makes.
    //
    // Compared on the full final assignment, not just the objective: an event
    // site that drew from the RNG, or that moved the assignment, would show up
    // here and nowhere cheaper.
    auto run = [](Tracer* tracer) {
        SearchConfig config;
        config.max_iterations = 5000;
        config.batch_iterations = 100;
        config.perturbation_period = 2;
        config.tracer = tracer;
        Model m = integer_model();
        FloatIntensifyHook hook;
        LNS lns(0.3);
        const SearchResult r =
            solve(m, /*time_limit=*/0.0, /*seed=*/20, true, &hook, &lns, 2, nullptr, config);
        return std::pair<SearchResult, Model::State>{r, m.copy_state()};
    };

    RecordingTracer tracer;
    const auto with = run(&tracer);
    const auto without = run(nullptr);

    REQUIRE_FALSE(tracer.events().empty());
    REQUIRE(with.first.objective == without.first.objective);
    REQUIRE(with.first.iterations == without.first.iterations);
    REQUIRE(with.first.perturbations == without.first.perturbations);
    REQUIRE(with.first.lns_repairs == without.first.lns_repairs);
    REQUIRE(with.first.lns_repairs_accepted == without.first.lns_repairs_accepted);
    REQUIRE(with.first.termination == without.first.termination);
    REQUIRE(with.first.counters.batches == without.first.counters.batches);
    REQUIRE(with.second.values == without.second.values);
    REQUIRE(with.second.elements == without.second.elements);
}

TEST_CASE("the base Tracer is a no-op on every event", "[tracer]") {
    // A subclass overrides only what it wants, so every method has to be
    // callable and do nothing. Calling them directly, since a search that
    // exercised all five would prove less about the defaults.
    Tracer base;
    base.batch_end(BatchKind::Structural, 7, true);
    base.new_best(1.0, 2.0);
    base.kick(KickKind::Adopt);
    base.lns(true);
    base.hook(0.5);
    SUCCEED("the default implementations are no-ops");
}

TEST_CASE("a throwing tracer propagates, and a portfolio absorbs it", "[tracer][parallel]") {
    // Both halves of the contract `Tracer` states. It matters because a host's
    // tracer writes to a log or a socket and CAN fail, and the two entry points
    // answer differently: a single solve hands the exception to its caller, a
    // portfolio treats one worker's failure as that worker's.
    struct Thrower : Tracer {
        std::atomic<int>* throws;
        explicit Thrower(std::atomic<int>* counter) : throws(counter) {}
        void batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool /*improved*/) override {
            throws->fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error("trace failed");
        }
    };

    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;
    // Counted rather than assumed. Every other assertion below is also satisfied by
    // a tracer that was never ATTACHED -- a restart_config that dropped
    // cfg.tracer, a factory never called -- which is exactly the defect the
    // per-worker tracer contract exists to prevent.
    std::atomic<int> throws{0};

    SECTION("out of a single solve, unchanged") {
        Thrower tracer(&throws);
        config.tracer = &tracer;
        Model m = quadratic_model();
        REQUIRE_THROWS_AS(
            solve(m, /*time_limit=*/0.0, /*seed=*/24, true, nullptr, nullptr, 3, nullptr, config),
            std::runtime_error);
        // The first batch_end threw and nothing ran after it.
        REQUIRE(throws.load(std::memory_order_relaxed) == 1);
    }

    SECTION("absorbed per worker, so a surviving peer still answers") {
        // Only worker 0 throws. Its peer has no tracer and runs normally, so the
        // portfolio returns that worker's result rather than propagating -- which
        // is the same rule a raising SolveCallback gets.
        ParallelConfig par_config;
        par_config.n_threads = 2;
        par_config.tracer_factory = [&throws](int worker) -> std::unique_ptr<Tracer> {
            return worker == 0 ? std::make_unique<Thrower>(&throws) : nullptr;
        };
        ParallelSearch ps(2);
        const SearchResult r = ps.solve(
            [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/24, config,
            /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);
        REQUIRE(r.feasible);
        REQUIRE(r.iterations > 0);
        // Worker 0's tracer actually fired, so this section shows a throw ABSORBED
        // rather than one that never happened. kMaxWorkerRetries bounds it at three
        // attempts, each raising on its first batch.
        REQUIRE(throws.load(std::memory_order_relaxed) > 0);
    }
}

TEST_CASE("KickKind has a distinct stable token", "[tracer]") {
    const std::vector<KickKind> all = {KickKind::Perturb, KickKind::LNS, KickKind::Adopt};
    std::set<std::string> tokens;
    for (KickKind k : all) {
        REQUIRE(kick_kind_name(k) != nullptr);
        tokens.insert(kick_kind_name(k));
    }
    REQUIRE(tokens.size() == all.size());
    REQUIRE(std::string(kick_kind_name(KickKind::Perturb)) == "perturb");
    REQUIRE(std::string(kick_kind_name(KickKind::LNS)) == "lns");
    REQUIRE(std::string(kick_kind_name(KickKind::Adopt)) == "adopt");
}

TEST_CASE("the portfolio builds one tracer per worker", "[tracer][parallel]") {
    // Per worker, not one shared instance: the events arrive on the reporting
    // worker's own thread with no mutex between them, so a shared tracer would
    // be a data race. The factory is what makes that the default shape.
    //
    // The factory is called from N threads at once, so this one locks -- the
    // same reason tests/test_parallel.cpp's CountingGenerator does.
    struct WorkerTracer : Tracer {
        int index;
        int64_t batches = 0;
        explicit WorkerTracer(int i) : index(i) {}
        void batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool /*improved*/) override {
            ++batches;
        }
    };

    std::mutex mutex;
    // Addresses as integers, recorded at construction: each tracer dies with its
    // worker, so keeping the pointers would leave this test holding dangling
    // ones to compare.
    std::vector<std::uintptr_t> addresses;
    std::vector<int> indices;

    ParallelConfig par_config;
    par_config.n_threads = 3;
    par_config.tracer_factory = [&mutex, &addresses,
                                 &indices](int worker) -> std::unique_ptr<Tracer> {
        auto tracer = std::make_unique<WorkerTracer>(worker);
        {
            const std::scoped_lock lock(mutex);
            addresses.push_back(reinterpret_cast<std::uintptr_t>(tracer.get()));
            indices.push_back(worker);
        }
        return tracer;
    };

    ParallelSearch ps(3);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/22, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    // One tracer per worker, each with its own index, and three distinct objects.
    REQUIRE(addresses.size() == 3);
    const std::set<int> seen(indices.begin(), indices.end());
    const std::set<int> expected{0, 1, 2};
    REQUIRE(seen == expected);
    const std::set<std::uintptr_t> distinct(addresses.begin(), addresses.end());
    REQUIRE(distinct.size() == 3);
    // The tracers are dead by now -- each dies with its worker -- so the only
    // safe cross-check is the portfolio's own aggregate, which must have seen at
    // least as many batches as there were workers.
    REQUIRE(r.counters.batches >= 3);
}
