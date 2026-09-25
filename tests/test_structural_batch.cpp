// The registered-move-generator pipeline behind the STRUCTURAL batch (#165):
// the generator API, G_v-restricted candidate scoring, the selection policies
// and granular neighbour lists.
//
// The batch is driven DIRECTLY here rather than through `solve()`, because what
// these tests are about is which candidates get proposed, scored and committed
// on a known assignment -- a question a full search answers only statistically.
// The end-to-end guarantee (an unconfigured run keeps its pre-#165 trajectory)
// is pinned separately, in tests/test_structural_equivalence.cpp.

#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/dag_ops.h>
#include <cbls/move_generator.h>
#include <cbls/search.h>
#include <cbls/structural_batch.h>
#include <cbls/violation.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace cbls;

namespace {

const std::chrono::steady_clock::time_point kNoDeadline{};

// ---------------------------------------------------------------------------
// A two-variable custom generator: move one element from set A to set B.
//
// Its counters live behind a shared_ptr ON PURPOSE, so the test can observe a
// clone's behaviour. That is exactly what a real generator must NOT do with
// mutable state -- see "each portfolio worker gets its own generator" in
// tests/test_parallel.cpp, which pins the other half.
// ---------------------------------------------------------------------------
struct TransferStats {
    int generated = 0;
    int commits = 0;
    int clones = 0;
};

class TransferGenerator final : public MoveGenerator {
public:
    TransferGenerator(int32_t from, int32_t to, std::shared_ptr<TransferStats> stats)
        : scope_{from, to}, stats_(std::move(stats)) {}

    [[nodiscard]] std::string_view name() const override { return "transfer"; }
    [[nodiscard]] ConstSpan<int32_t> scope() const override { return {scope_.data(), 2}; }

    void generate(MoveContext& ctx, std::vector<Move>& out) override {
        ++stats_->generated;
        const Variable& from = ctx.model.var(scope_[0]);
        const Variable& to = ctx.model.var(scope_[1]);
        if (from.elements.empty()) {
            return;
        }
        const auto pos =
            static_cast<size_t>(ctx.rng.integers(0, static_cast<int64_t>(from.elements.size())));
        const int32_t element = from.elements[pos];
        if (std::find(to.elements.begin(), to.elements.end(), element) != to.elements.end()) {
            return;
        }
        Move move;
        move.move_type = "transfer";
        std::vector<int32_t> shrunk = from.elements;
        shrunk.erase(shrunk.begin() + static_cast<std::ptrdiff_t>(pos));
        std::vector<int32_t> grown = to.elements;
        grown.push_back(element);
        move.changes.push_back({scope_[0], 0.0, shrunk});
        move.changes.push_back({scope_[1], 0.0, grown});
        out.push_back(std::move(move));
    }

    void on_commit(const Move& move) override {
        REQUIRE(move.changes.size() == 2);
        ++stats_->commits;
    }

    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        ++stats_->clones;
        return std::make_unique<TransferGenerator>(scope_[0], scope_[1], stats_);
    }

private:
    std::array<int32_t, 2> scope_;
    std::shared_ptr<TransferStats> stats_;
};

// Counts calls and proposes nothing, so a test can ask whether a generator was
// reached at all.
class SilentGenerator final : public MoveGenerator {
public:
    SilentGenerator(int32_t var_id, std::shared_ptr<int> calls)
        : var_id_(var_id), calls_(std::move(calls)) {}
    [[nodiscard]] std::string_view name() const override { return "silent"; }
    [[nodiscard]] ConstSpan<int32_t> scope() const override { return {&var_id_, 1}; }
    void generate(MoveContext& /*ctx*/, std::vector<Move>& /*out*/) override { ++*calls_; }
    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        return std::make_unique<SilentGenerator>(var_id_, calls_);
    }

private:
    int32_t var_id_;
    std::shared_ptr<int> calls_;
};

// Burns a fixed slice of wall clock per `generate` call and proposes nothing,
// so a sweep over many of them takes a predictable time and the only thing that
// can cut it short is the deadline check between generators.
class SlowGenerator final : public MoveGenerator {
public:
    SlowGenerator(int32_t var_id, std::shared_ptr<int> calls)
        : var_id_(var_id), calls_(std::move(calls)) {}
    [[nodiscard]] std::string_view name() const override { return "slow"; }
    [[nodiscard]] ConstSpan<int32_t> scope() const override { return {&var_id_, 1}; }
    void generate(MoveContext& /*ctx*/, std::vector<Move>& /*out*/) override {
        ++*calls_;
        // A spin rather than a sleep: the batch's bound is about CPU work
        // overrunning a deadline, and a sleeping thread would be descheduled
        // where the real sweep is not.
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        while (std::chrono::steady_clock::now() < until) {
        }
    }
    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        return std::make_unique<SlowGenerator>(var_id_, calls_);
    }

private:
    int32_t var_id_;
    std::shared_ptr<int> calls_;
};

// Two Sets over the same six-element universe. The rows want at least three
// elements in A and at most one in B, so moving an element from B to A is the
// improving direction and no built-in single-variable move can make it in one
// step while both bounds bind.
struct TwoSetModel {
    Model model;
    int32_t a = -1;
    int32_t b = -1;
};

TwoSetModel two_set_model() {
    TwoSetModel ts;
    Model& m = ts.model;
    Expr a = m.Set(6, 0, 6, "a");
    Expr b = m.Set(6, 0, 6, "b");
    ts.a = vid(a.handle);
    ts.b = vid(b.handle);
    Expr size_a = Expr(&m, m.lambda_sum(a.handle, [](int) { return 1.0; }));
    Expr size_b = Expr(&m, m.lambda_sum(b.handle, [](int) { return 1.0; }));
    m.add_constraint(size_a >= m.Constant(3.0));
    m.add_constraint(size_b <= m.Constant(1.0));
    m.close();
    return ts;
}

void set_elements(Model& m, int32_t var_id, const std::vector<int32_t>& elements) {
    m.var_mut(var_id).elements = elements;
}

}  // namespace

TEST_CASE("a registered generator's multi-variable moves are scored and committed",
          "[structural][moves]") {
    TwoSetModel ts = two_set_model();
    set_elements(ts.model, ts.a, {0});
    set_elements(ts.model, ts.b, {1, 2, 3});
    full_evaluate(ts.model);

    ViolationManager vm(ts.model);
    auto stats = std::make_shared<TransferStats>();
    SearchConfig config;
    config.default_structural_generators = false;  // the custom generator alone
    config.move_generators.push_back(std::make_shared<const TransferGenerator>(ts.b, ts.a, stats));

    StructuralBatch batch(ts.model, config, /*enabled=*/true);
    REQUIRE(batch.generator_count() == 1);
    // The batch runs its OWN clone, never the registered instance.
    REQUIRE(stats->clones == 1);

    RNG rng(42);
    bool moved = false;
    for (int pass = 0; pass < 8 && !moved; ++pass) {
        moved = batch.run(ts.model, vm, rng, /*has_deadline=*/false, kNoDeadline);
    }
    REQUIRE(stats->generated > 0);
    REQUIRE(moved);
    REQUIRE(stats->commits > 0);
    // A committed transfer is a real two-variable change: B shrank, A grew.
    REQUIRE(ts.model.var(ts.a).elements.size() > 1);
    REQUIRE(ts.model.var(ts.b).elements.size() < 3);
}

TEST_CASE("the built-ins are registered per structured variable", "[structural][moves]") {
    TwoSetModel ts = two_set_model();
    std::vector<std::shared_ptr<const MoveGenerator>> defaults =
        default_move_generators(ts.model, nullptr);
    REQUIRE(defaults.size() == 2);
    REQUIRE(defaults[0]->name() == "builtin_set");
    REQUIRE(defaults[0]->scope().size() == 1);
    REQUIRE(defaults[0]->scope()[0] == ts.a);
    REQUIRE(defaults[1]->scope()[0] == ts.b);

    // A scalar-only model has none, which is what lets solve() skip the batch
    // without a per-variable scan.
    Model scalar;
    Expr x = scalar.Bool();
    scalar.minimize(x + scalar.Constant(0.0));
    scalar.close();
    REQUIRE(default_move_generators(scalar, nullptr).empty());
}

TEST_CASE("G_v-restricted scoring is the same double as the full rescan",
          "[structural][moves][violation]") {
    // The exactness argument: a row outside the moved variables' G_v cannot have
    // changed, so `now == snapshot[i]` holds bitwise and BOTH versions skip it;
    // ascending order then keeps the surviving terms in one sequence. So the
    // comparison below is `==` on doubles, deliberately, not an approximation.
    TwoSetModel ts = two_set_model();
    set_elements(ts.model, ts.a, {0, 4});
    set_elements(ts.model, ts.b, {1, 2, 3});
    full_evaluate(ts.model);
    ViolationManager vm(ts.model);
    // Skewed weights, so a mis-selected row would change the sum rather than
    // cancel against a uniform one.
    for (size_t i = 0; i < vm.weights.size(); ++i) {
        vm.weights[i] = 1.0 + (7.5 * static_cast<double>(i));
    }

    std::vector<double> baseline;
    vm.snapshot_violations(baseline);

    RNG rng(7);
    int scored = 0;
    int accepted_full = 0;
    for (int trial = 0; trial < 200; ++trial) {
        for (int32_t var_id : {ts.a, ts.b}) {
            std::vector<Move> moves;
            generate_standard_moves(ts.model.var(var_id), rng, moves, nullptr);
            for (const Move& move : moves) {
                SavedValues saved = save_move_values(ts.model, move);
                std::vector<int32_t> touched = apply_move(ts.model, move);
                delta_evaluate(ts.model, touched);
                const double full = vm.weighted_delta_from(baseline);
                const double restricted =
                    vm.weighted_delta_from(baseline, ts.model.constraints_of_var(var_id));
                INFO("full " << full << " restricted " << restricted);
                REQUIRE(full == restricted);
                REQUIRE((full < -1e-12) == (restricted < -1e-12));
                ++scored;
                accepted_full += (full < -1e-12) ? 1 : 0;
                undo_move(ts.model, move, saved);
                delta_evaluate(ts.model, touched);
            }
        }
    }
    REQUIRE(scored > 100);
    // The comparison would be vacuous if no candidate ever improved anything.
    REQUIRE(accepted_full > 0);
}

TEST_CASE("the restricted delta rejects a malformed row list", "[structural][violation]") {
    TwoSetModel ts = two_set_model();
    full_evaluate(ts.model);
    ViolationManager vm(ts.model);
    std::vector<double> baseline;
    vm.snapshot_violations(baseline);
    const std::vector<int32_t> out_of_range{99};
    REQUIRE_THROWS_AS(vm.weighted_delta_from(baseline, {out_of_range.data(), 1}),
                      std::out_of_range);
    const std::vector<double> short_snapshot(baseline.size() - 1, 0.0);
    const std::vector<int32_t> row{0};
    REQUIRE_THROWS_AS(vm.weighted_delta_from(short_snapshot, {row.data(), 1}),
                      std::invalid_argument);
}

TEST_CASE("ViolationGuided skips a scope whose rows are all satisfied", "[structural][moves]") {
    // Both rows hold at this assignment, so no move over either Set can lower
    // the weighted violation -- which is why skipping is exact rather than a
    // heuristic. The guided policy must not spend a single candidate here, and
    // the default policy must still spend them (otherwise the test would pass on
    // a batch that never ran at all).
    TwoSetModel ts = two_set_model();
    set_elements(ts.model, ts.a, {0, 1, 2});
    set_elements(ts.model, ts.b, {});
    full_evaluate(ts.model);
    ViolationManager vm(ts.model);
    REQUIRE(vm.total_violation() == 0.0);

    auto guided_calls = std::make_shared<int>(0);
    SearchConfig guided;
    guided.default_structural_generators = false;
    guided.structural_selection = StructuralSelection::ViolationGuided;
    guided.move_generators.push_back(std::make_shared<const SilentGenerator>(ts.a, guided_calls));
    StructuralBatch guided_batch(ts.model, guided, true);
    RNG rng(42);
    REQUIRE_FALSE(guided_batch.run(ts.model, vm, rng, false, kNoDeadline));
    REQUIRE(*guided_calls == 0);

    auto plain_calls = std::make_shared<int>(0);
    SearchConfig plain;
    plain.default_structural_generators = false;
    plain.move_generators.push_back(std::make_shared<const SilentGenerator>(ts.a, plain_calls));
    StructuralBatch plain_batch(ts.model, plain, true);
    REQUIRE_FALSE(plain_batch.run(ts.model, vm, rng, false, kNoDeadline));
    REQUIRE(*plain_calls == 1);
}

TEST_CASE("a sampling policy draws more candidates than the default", "[structural][moves]") {
    TwoSetModel ts = two_set_model();
    set_elements(ts.model, ts.a, {0});
    set_elements(ts.model, ts.b, {1, 2, 3});
    full_evaluate(ts.model);
    ViolationManager vm(ts.model);
    RNG rng(42);

    auto first_stats = std::make_shared<TransferStats>();
    SearchConfig first;
    first.default_structural_generators = false;
    first.move_generators.push_back(
        std::make_shared<const TransferGenerator>(ts.b, ts.a, first_stats));
    StructuralBatch(ts.model, first, true).run(ts.model, vm, rng, false, kNoDeadline);
    REQUIRE(first_stats->generated == 1);

    TwoSetModel again = two_set_model();
    set_elements(again.model, again.a, {0});
    set_elements(again.model, again.b, {1, 2, 3});
    full_evaluate(again.model);
    ViolationManager vm2(again.model);
    auto best_stats = std::make_shared<TransferStats>();
    SearchConfig best;
    best.default_structural_generators = false;
    best.structural_selection = StructuralSelection::BestOfSample;
    best.structural_sample_size = 5;
    best.move_generators.push_back(
        std::make_shared<const TransferGenerator>(again.b, again.a, best_stats));
    RNG rng2(42);
    StructuralBatch(again.model, best, true).run(again.model, vm2, rng2, false, kNoDeadline);
    REQUIRE(best_stats->generated > 1);
    REQUIRE(best_stats->generated <= 6);  // bounded: sample_size + the first call
}

TEST_CASE("an expired deadline stops the sweep before any generator runs", "[structural][moves]") {
    TwoSetModel ts = two_set_model();
    set_elements(ts.model, ts.a, {0});
    set_elements(ts.model, ts.b, {1, 2, 3});
    full_evaluate(ts.model);
    ViolationManager vm(ts.model);
    auto calls = std::make_shared<int>(0);
    SearchConfig config;
    config.default_structural_generators = false;
    config.move_generators.push_back(std::make_shared<const SilentGenerator>(ts.a, calls));
    StructuralBatch batch(ts.model, config, true);
    RNG rng(42);
    const auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    REQUIRE_FALSE(batch.run(ts.model, vm, rng, /*has_deadline=*/true, past));
    REQUIRE(*calls == 0);
}

TEST_CASE("nearest_neighbours is k-nearest, nearest first, ties by id",
          "[structural][neighbours]") {
    // Points on a line at 0, 1, 2, 3: element 1's nearest are 0 and 2 (both at
    // distance 1, so id order decides), then 3.
    const auto cost = [](int a, int b) { return std::abs(static_cast<double>(a - b)); };
    NeighbourList nl = nearest_neighbours(4, 2, cost);
    REQUIRE(nl.universe() == 4);
    const ConstSpan<int32_t> of_one = nl.of(1);
    REQUIRE(of_one.size() == 2);
    REQUIRE(of_one[0] == 0);
    REQUIRE(of_one[1] == 2);
    REQUIRE(nl.of(0).size() == 2);
    REQUIRE(nl.of(0)[0] == 1);
    REQUIRE(nl.of(0)[1] == 2);

    // Total on an out-of-range element rather than a heap read (#156's class).
    REQUIRE(nl.of(-1).empty());
    REQUIRE(nl.of(4).empty());
    REQUIRE(nl.of(4000).empty());

    // k above the universe is clamped, and a degenerate universe is empty.
    REQUIRE(nearest_neighbours(4, 99, cost).of(0).size() == 3);
    REQUIRE(nearest_neighbours(1, 3, cost).empty());
    REQUIRE(nearest_neighbours(4, 0, cost).empty());
}

TEST_CASE("NeighbourList validates its arrays at construction", "[structural][neighbours]") {
    REQUIRE_NOTHROW(NeighbourList(std::vector<int32_t>{0, 1, 2}, std::vector<int32_t>{1, 0}));
    // id outside the universe
    REQUIRE_THROWS_AS(NeighbourList(std::vector<int32_t>{0, 1, 2}, std::vector<int32_t>{5, 0}),
                      std::invalid_argument);
    // offsets not starting at 0
    REQUIRE_THROWS_AS(NeighbourList(std::vector<int32_t>{1, 2}, std::vector<int32_t>{0, 0}),
                      std::invalid_argument);
    // offsets not ending at ids.size()
    REQUIRE_THROWS_AS(NeighbourList(std::vector<int32_t>{0, 1}, std::vector<int32_t>{0, 0}),
                      std::invalid_argument);
    // offsets not monotone
    REQUIRE_THROWS_AS(NeighbourList(std::vector<int32_t>{0, 2, 1}, std::vector<int32_t>{0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(NeighbourList(std::vector<std::vector<int32_t>>{{0, 7}, {0}}),
                      std::invalid_argument);
}

TEST_CASE("a neighbour list restricts where a built-in move looks", "[structural][neighbours]") {
    // A List of four elements whose neighbour list names only element 3 as a
    // neighbour of anything: every guided move pair must therefore involve the
    // position holding element 3, where the uniform draw would not.
    Model m;
    Expr route = m.List(4, "route");
    m.minimize(Expr(&m, m.pair_lambda_sum(route.handle, [](int a, int b) {
        return static_cast<double>(std::abs(a - b));
    })));
    m.close();
    const int32_t rid = vid(route.handle);
    m.var_mut(rid).elements = {0, 1, 2, 3};
    full_evaluate(m);

    auto rows = std::vector<std::vector<int32_t>>{{3}, {3}, {3}, {0}};
    auto nl = std::make_shared<const NeighbourList>(rows);

    RNG rng(42);
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<Move> moves;
        generate_standard_moves(m.var(rid), rng, moves, nl.get());
        REQUIRE_FALSE(moves.empty());
        // A swap of positions i and j leaves every other position alone, so the
        // two positions the pair used are exactly the two that differ.
        const std::vector<int32_t>& swapped = moves[0].changes.front().new_elements;
        const std::vector<int32_t>& before = m.var(rid).elements;
        std::vector<int32_t> differing;
        for (size_t p = 0; p < before.size(); ++p) {
            if (before[p] != swapped[p]) {
                differing.push_back(before[p]);
            }
        }
        REQUIRE(differing.size() == 2);
        INFO("pair " << differing[0] << "," << differing[1]);
        REQUIRE((differing[0] == 3 || differing[1] == 3 || differing[0] == 0 || differing[1] == 0));
    }
}

TEST_CASE("structural selection names round-trip", "[structural]") {
    for (StructuralSelection s :
         {StructuralSelection::FirstImprovingSample, StructuralSelection::BestOfSample,
          StructuralSelection::ViolationGuided}) {
        StructuralSelection back = StructuralSelection::BestOfSample;
        REQUIRE(try_parse_structural_selection(structural_selection_name(s), back));
        REQUIRE(back == s);
    }
    StructuralSelection unused = StructuralSelection::FirstImprovingSample;
    REQUIRE_FALSE(try_parse_structural_selection("nope", unused));
}

TEST_CASE("a deadline that passes mid-sweep stops the sweep between generators",
          "[structural][moves]") {
    // The discriminating test for #105's bound, and the reason it asserts a
    // COUNT rather than a duration. Its wall-clock predecessor
    // (`structural batch respects the wall-clock deadline`, tests/test_search.cpp)
    // stopped discriminating when #165 restricted the candidate scan to G_v: the
    // 40 000 filler rows that made an unbounded sweep cost 1.134s are no longer
    // read, and that model's unbounded sweep re-measured at 0.024s against a
    // 0.10s budget -- it passed with the check deleted. This one does not: with
    // the check removed every one of the 200 generators runs, every time.
    //
    // Robust in the right direction under load. Each generate burns 200us of
    // CPU, so a loaded machine gets through FEWER generators before the 10ms
    // deadline, never more, and `visited < kGenerators` holds harder.
    constexpr int kGenerators = 200;
    TwoSetModel ts = two_set_model();
    set_elements(ts.model, ts.a, {0});
    set_elements(ts.model, ts.b, {1, 2, 3});
    full_evaluate(ts.model);
    ViolationManager vm(ts.model);

    auto calls = std::make_shared<int>(0);
    SearchConfig config;
    config.default_structural_generators = false;
    for (int i = 0; i < kGenerators; ++i) {
        config.move_generators.push_back(std::make_shared<const SlowGenerator>(ts.a, calls));
    }
    StructuralBatch batch(ts.model, config, /*enabled=*/true);
    REQUIRE(batch.generator_count() == kGenerators);

    RNG rng(42);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    batch.run(ts.model, vm, rng, /*has_deadline=*/true, deadline);

    INFO("generators visited: " << *calls << " of " << kGenerators);
    REQUIRE(*calls > 0);            // the sweep did start
    REQUIRE(*calls < kGenerators);  // and the deadline cut it short
}

TEST_CASE("a NaN cost does not break the neighbour-list sort", "[structural][neighbours]") {
    // Two or more NaN costs is the case that makes a naive comparator
    // non-strict, which is undefined behaviour inside partial_sort rather than
    // merely a wrong order. A cost callback returning NaN for every unreachable
    // pair produces exactly that.
    const auto all_nan = [](int, int) { return std::numeric_limits<double>::quiet_NaN(); };
    NeighbourList nl = nearest_neighbours(8, 3, all_nan);
    REQUIRE(nl.universe() == 8);
    // Every cost ties, so the id tiebreak alone decides: element 4's nearest are
    // the three lowest OTHER ids.
    const ConstSpan<int32_t> of_four = nl.of(4);
    REQUIRE(of_four.size() == 3);
    REQUIRE(of_four[0] == 0);
    REQUIRE(of_four[1] == 1);
    REQUIRE(of_four[2] == 2);

    // A single NaN among finite costs sorts last.
    const auto one_nan = [](int a, int b) {
        if (a == 0 && b == 1) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::abs(static_cast<double>(a - b));
    };
    NeighbourList mixed = nearest_neighbours(4, 3, one_nan);
    const ConstSpan<int32_t> of_zero = mixed.of(0);
    REQUIRE(of_zero.size() == 3);
    REQUIRE(of_zero[2] == 1);  // the NaN pair is last
}

TEST_CASE("NeighbourList::of is total at the integer boundary", "[structural][neighbours]") {
    // `e + 1` on an int32_t overflows at INT32_MAX, which is undefined behaviour
    // in the one accessor written to be total on every input -- and this type is
    // reachable from Python, where the index is whatever the caller passed.
    NeighbourList nl =
        nearest_neighbours(4, 2, [](int a, int b) { return std::abs(static_cast<double>(a - b)); });
    REQUIRE(nl.of(std::numeric_limits<int32_t>::max()).empty());
    REQUIRE(nl.of(std::numeric_limits<int32_t>::min()).empty());
    REQUIRE(NeighbourList().of(0).empty());
    REQUIRE(NeighbourList().of(std::numeric_limits<int32_t>::max()).empty());
}

TEST_CASE("a generator whose scope names an unknown variable is refused at registration",
          "[structural][moves]") {
    // Reported where the generator is handed over rather than as a
    // std::out_of_range thrown out of solve() some way into a run, where the
    // caller can no longer tell which generator did it.
    TwoSetModel ts = two_set_model();
    auto calls = std::make_shared<int>(0);
    SearchConfig config;
    config.default_structural_generators = false;
    config.move_generators.push_back(std::make_shared<const SilentGenerator>(99, calls));
    REQUIRE_THROWS_AS(StructuralBatch(ts.model, config, /*enabled=*/true), std::out_of_range);
    REQUIRE(*calls == 0);
}
