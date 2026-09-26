// Column generation (#168): a ColumnGenerator priced from the GLS weights
// during search, applied through Model::extend (#167).
//
// The end-to-end model is one-dimensional cutting stock on the Falkenauer
// uniform instance u120_00, started from the trivial pattern set (one item per
// roll), with a bounded-knapsack pricer over the GLS row weights.

#include "cbls/column_generator.h"
#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"
#include "cbls/model.h"
#include "cbls/model_extension.h"
#include "cbls/pool.h"
#include "cbls/search.h"
#include "cbls/tracer.h"
#include "cbls/violation.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

// ---------------------------------------------------------------------------
// The instance.
//
// Falkenauer (1996) class "u" instance u120_00: 120 items, bin capacity 150,
// item sizes uniform in [20, 100], optimum 48 bins. Copied from the OR-Library
// distribution, J.E. Beasley, file `binpack1.txt`
// (https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack1.txt, the first
// of its 20 instances; sha256 of the file as fetched on 2026-09-26:
// 891fe4b3be86371b120ccaf37ce4525b5b0a2fc249d38725cd3216fbaa42b172). The same
// instances are distributed by BPPLIB (Delorme, Iori & Martello 2018,
// "BPPLIB: a library for bin packing and cutting stock problems", Optimization
// Letters 12) as Falkenauer_U. Listed in the file's order.
// ---------------------------------------------------------------------------
constexpr int kU120Capacity = 150;
constexpr int kU120Optimum = 48;
constexpr int kU120Items[120] = {
    42, 69, 67, 57, 93, 90, 38, 36, 45, 42, 33, 79, 27, 57, 44, 84, 86, 92, 46, 38, 85, 33, 82, 73,
    49, 70, 59, 23, 57, 72, 74, 69, 33, 42, 28, 46, 30, 64, 29, 74, 41, 49, 55, 98, 80, 32, 25, 38,
    82, 30, 35, 39, 57, 84, 62, 50, 55, 27, 30, 36, 20, 78, 47, 26, 45, 41, 58, 98, 91, 96, 73, 84,
    37, 93, 91, 43, 73, 85, 81, 79, 71, 80, 76, 83, 41, 78, 70, 23, 42, 87, 43, 84, 60, 55, 49, 78,
    73, 62, 36, 44, 94, 69, 32, 96, 70, 84, 58, 78, 25, 80, 58, 66, 83, 24, 98, 60, 42, 43, 43, 39};

// Cutting stock: one demand row per DISTINCT item size.
struct CuttingStock {
    int capacity = 0;
    std::vector<int> sizes;   // distinct, descending
    std::vector<int> demand;  // parallel to sizes
    [[nodiscard]] int total_items() const {
        int n = 0;
        for (const int d : demand) {
            n += d;
        }
        return n;
    }
};

CuttingStock u120_00() {
    std::map<int, int, std::greater<>> counts;
    for (const int w : kU120Items) {
        ++counts[w];
    }
    CuttingStock cs;
    cs.capacity = kU120Capacity;
    for (const auto& [w, d] : counts) {
        cs.sizes.push_back(w);
        cs.demand.push_back(d);
    }
    return cs;
}

// The model: Int x_p per pattern p (rolls cut with it), a row
// `sum_p a_ip x_p >= d_i` per size, and `minimize sum_p x_p`. Row i is
// constraint index i. The trivial pattern set is one item of one size per roll.
struct CuttingModel {
    Model model;
    std::vector<int32_t> row_sums;  // the Sum node on each row's left-hand side
    int32_t objective_sum = -1;
};

CuttingModel build_trivial(const CuttingStock& cs) {
    CuttingModel cm;
    Model& m = cm.model;
    std::vector<int32_t> objective_terms;
    for (size_t i = 0; i < cs.sizes.size(); ++i) {
        const int32_t x = m.int_var(0, cs.demand[i]);
        cm.row_sums.push_back(m.sum({m.prod(m.constant(1.0), x)}));
        objective_terms.push_back(x);
    }
    for (size_t i = 0; i < cs.sizes.size(); ++i) {
        m.add_constraint(m.geq(cm.row_sums[i], m.constant(cs.demand[i])));
    }
    cm.objective_sum = m.sum(objective_terms);
    m.minimize(cm.objective_sum);
    m.close();
    return cm;
}

// The bounded knapsack that prices a pattern: maximise sum_i v_i a_i subject to
// sum_i w_i a_i <= capacity and 0 <= a_i <= min(d_i, capacity / w_i). Each unit
// of each size is a 0/1 item, which at capacity 150 is a few hundred items by
// 151 capacities -- trivially small.
std::vector<int> best_pattern(const CuttingStock& cs, const std::vector<double>& value) {
    struct Unit {
        size_t size_idx;
        int w;
        double v;
    };
    std::vector<Unit> units;
    for (size_t i = 0; i < cs.sizes.size(); ++i) {
        const int copies = std::min(cs.demand[i], cs.capacity / cs.sizes[i]);
        for (int k = 0; k < copies; ++k) {
            units.push_back({i, cs.sizes[i], value[i]});
        }
    }
    const int cap = cs.capacity;
    std::vector<double> dp(static_cast<size_t>(cap) + 1, 0.0);
    std::vector<std::vector<uint8_t>> take(units.size(),
                                           std::vector<uint8_t>(static_cast<size_t>(cap) + 1, 0));
    for (size_t u = 0; u < units.size(); ++u) {
        for (int c = cap; c >= units[u].w; --c) {
            const double with = dp[static_cast<size_t>(c - units[u].w)] + units[u].v;
            if (with > dp[static_cast<size_t>(c)]) {
                dp[static_cast<size_t>(c)] = with;
                take[u][static_cast<size_t>(c)] = 1;
            }
        }
    }
    std::vector<int> a(cs.sizes.size(), 0);
    int c = cap;
    for (size_t u = units.size(); u-- > 0;) {
        if (take[u][static_cast<size_t>(c)] != 0) {
            ++a[units[u].size_idx];
            c -= units[u].w;
        }
    }
    return a;
}

std::vector<std::pair<int32_t, double>> signature_of(const std::vector<int>& a) {
    std::vector<std::pair<int32_t, double>> sig;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != 0) {
            sig.emplace_back(static_cast<int32_t>(i), static_cast<double>(a[i]));
        }
    }
    return sig;
}

// The pricer. The reduced-cost analogue of the issue: a pattern a with cost 1
// changes the weighted violation by about W_obj - sum_i W_i a_i, so the column
// worth adding is the knapsack optimum under values W_i, whenever that beats
// W_obj.
class KnapsackPricer : public ColumnGenerator {
public:
    KnapsackPricer(CuttingStock cs, std::vector<int32_t> row_sums, int32_t objective_sum)
        : cs_(std::move(cs)), row_sums_(std::move(row_sums)), objective_sum_(objective_sum) {}

    void price(const PricingContext& ctx, PricingEvent /*why*/, ModelExtension& ext) override {
        if (!seeded_) {
            // The base model's columns count as duplicates too.
            for (size_t i = 0; i < cs_.sizes.size(); ++i) {
                ctx.signatures.insert({{static_cast<int32_t>(i), 1.0}}, 1.0);
            }
            seeded_ = true;
        }
        std::vector<double> value(cs_.sizes.size());
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] = ctx.weights[i];
        }
        const double w_obj = ctx.weights[static_cast<size_t>(ctx.objective_constraint_idx)];
        // Up to kPerCall columns a call. After each, the values of the sizes it
        // used are halved, so the next knapsack looks for a pattern over the
        // rows the previous ones left cheap -- a cheap way to get a few
        // DIFFERENT columns out of one weight vector instead of the same one.
        int64_t room = std::min<int64_t>(kPerCall, ctx.columns_remaining);
        for (int k = 0; k < kPerCall && room > 0; ++k) {
            const std::vector<int> a = best_pattern(cs_, value);
            double gain = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                gain += value[i] * a[i];
            }
            if (gain <= w_obj) {
                break;
            }
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i] > 0) {
                    value[i] *= 0.5;
                }
            }
            if (!ctx.signatures.insert(signature_of(a), 1.0)) {
                continue;  // already a column
            }
            stage(a, ext);
            --room;
        }
    }

    void stage(const std::vector<int>& a, ModelExtension& ext) const {
        int ub = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] > 0) {
                ub = std::max(ub, (cs_.demand[i] + a[i] - 1) / a[i]);
            }
        }
        const int32_t x = ext.int_var(0, ub);
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] > 0) {
                ext.append_to_sum(row_sums_[i], ext.prod(ext.constant(a[i]), x));
            }
        }
        ext.append_to_sum(objective_sum_, x);
    }

    [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
        return std::make_unique<KnapsackPricer>(*this);
    }

private:
    static constexpr int kPerCall = 4;
    CuttingStock cs_;
    std::vector<int32_t> row_sums_;
    int32_t objective_sum_;
    bool seeded_ = false;
};

// Every row of `m` satisfied at `state`, re-evaluated from scratch.
bool feasible_at(Model m, const Model::State& state, double tol = 1e-6) {
    m.set_objective_bound(std::numeric_limits<double>::infinity());
    m.restore_state(state);
    full_evaluate(m);
    for (size_t i = 0; i < m.constraint_ids().size(); ++i) {
        if (static_cast<int32_t>(i) == m.objective_constraint_idx()) {
            continue;
        }
        if (!(m.node_value(m.constraint_ids()[i]) <= tol)) {
            return false;
        }
    }
    return true;
}

SearchConfig iteration_budget(int64_t iterations) {
    SearchConfig cfg;
    cfg.max_iterations = iterations;
    return cfg;
}

// Stages nothing, and records what it saw.
class RecordingGenerator : public ColumnGenerator {
public:
    struct Call {
        PricingEvent why;
        int64_t batches;
        double remaining;
        std::chrono::steady_clock::time_point at;
    };
    explicit RecordingGenerator(std::shared_ptr<std::vector<Call>> log) : log_(std::move(log)) {}
    void price(const PricingContext& ctx, PricingEvent why, ModelExtension& /*ext*/) override {
        log_->push_back(
            {why, ctx.batches, ctx.remaining_seconds, std::chrono::steady_clock::now()});
    }
    [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
        return std::make_unique<RecordingGenerator>(*this);
    }

private:
    std::shared_ptr<std::vector<Call>> log_;
};

// A tracer that keeps the order of batch ends, pricing calls and kicks.
class OrderTracer : public Tracer {
public:
    enum class Kind : std::uint8_t { BatchEnd, Pricing, Kick };
    struct Event {
        Kind kind;
        bool improved = false;
        PricingEvent why = PricingEvent::Periodic;
    };
    void batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool improved) override {
        events.push_back({Kind::BatchEnd, improved, PricingEvent::Periodic});
    }
    void kick(KickKind /*kind*/) override {
        events.push_back({Kind::Kick, false, PricingEvent::Periodic});
    }
    void pricing(PricingEvent why, int64_t /*columns*/, int64_t /*rows*/,
                 double /*seconds*/) override {
        events.push_back({Kind::Pricing, false, why});
    }
    std::vector<Event> events;
};

// A small model that never becomes feasible: a + b <= 0 and a + b >= 2 over two
// Bools. The GLS weights grow for as long as it runs, and it kicks.
Model infeasible_pair() {
    Model m;
    const int32_t a = m.bool_var();
    const int32_t b = m.bool_var();
    const int32_t lhs = m.sum({m.prod(m.constant(1.0), a), m.prod(m.constant(1.0), b)});
    m.add_constraint(m.leq(lhs, m.constant(0.0)));
    m.add_constraint(m.geq(lhs, m.constant(2.0)));
    m.close();
    return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// The context: the live weights, at a safe point.
// ---------------------------------------------------------------------------

TEST_CASE("the pricer receives exactly the live GLS weights at a safe point", "[column]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);

    struct Seen {
        int calls = 0;
        bool aliased = true;        // weights IS vm.weights, not a copy
        bool sized = true;          // one weight per constraint of the model as it is now
        bool total_matches = true;  // those weights are the ones the manager totals with
        bool consistent = true;     // node values match a fresh evaluation
        bool diverged = false;      // GLS dynamics visible: some weight != 1
    };
    class Checker : public ColumnGenerator {
    public:
        explicit Checker(std::shared_ptr<Seen> seen) : seen_(std::move(seen)) {}
        void price(const PricingContext& ctx, PricingEvent /*why*/,
                   ModelExtension& /*ext*/) override {
            Seen& s = *seen_;
            ++s.calls;
            const std::vector<double>& vw = ctx.violations.weights;
            s.aliased =
                s.aliased && ctx.weights.begin() == vw.data() && ctx.weights.size() == vw.size();
            s.sized = s.sized && ctx.weights.size() == ctx.model.constraint_ids().size();
            double total = 0.0;
            for (size_t i = 0; i < ctx.weights.size(); ++i) {
                total += ctx.weights[i] *
                         std::max(0.0, ctx.model.node_value(ctx.model.constraint_ids()[i]));
                s.diverged = s.diverged || ctx.weights[i] != 1.0;
            }
            // Integer data, so the incremental and the from-scratch sums agree
            // exactly; a tolerance would hide a stale cone.
            const double managed = ctx.violations.total_violation();
            s.total_matches = s.total_matches && std::abs(total - managed) <= 1e-9 * (1.0 + total);
            Model copy(ctx.model);
            full_evaluate(copy);
            s.consistent = s.consistent && copy.node_values() == ctx.model.node_values();
        }
        [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
            return std::make_unique<Checker>(*this);
        }

    private:
        std::shared_ptr<Seen> seen_;
    };

    auto seen = std::make_shared<Seen>();
    SearchConfig cfg = iteration_budget(60'000);
    cfg.column_generator = std::make_shared<Checker>(seen);
    cfg.pricing_period = 1;  // every batch
    cfg.structural_batch_probability = 0.0;
    const SearchResult r = solve(cm.model, 0.0, 7, true, nullptr, nullptr, 3, nullptr, cfg);

    REQUIRE(seen->calls > 10);
    REQUIRE(r.counters.pricing_calls == seen->calls);
    REQUIRE(seen->aliased);
    REQUIRE(seen->sized);
    REQUIRE(seen->total_matches);
    REQUIRE(seen->consistent);
    REQUIRE(seen->diverged);
}

// ---------------------------------------------------------------------------
// The event schedule.
// ---------------------------------------------------------------------------

TEST_CASE("Periodic pricing fires every pricing_period batches", "[column]") {
    Model m = infeasible_pair();
    auto log = std::make_shared<std::vector<RecordingGenerator::Call>>();
    SearchConfig cfg = iteration_budget(40'000);
    cfg.batch_iterations = 100;
    cfg.column_generator = std::make_shared<RecordingGenerator>(log);
    cfg.pricing_period = 3;
    cfg.price_on_stagnation = false;
    const SearchResult r = solve(m, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg);

    REQUIRE(r.counters.batches >= 30);
    REQUIRE(static_cast<int64_t>(log->size()) == r.counters.batches / 3);
    for (size_t k = 0; k < log->size(); ++k) {
        REQUIRE((*log)[k].why == PricingEvent::Periodic);
        REQUIRE((*log)[k].batches == 3 * static_cast<int64_t>(k + 1));
        REQUIRE(std::isinf((*log)[k].remaining));  // no wall clock
    }
}

TEST_CASE("Stagnation pricing fires immediately before every kick", "[column]") {
    Model m = infeasible_pair();
    auto log = std::make_shared<std::vector<RecordingGenerator::Call>>();
    OrderTracer tracer;
    SearchConfig cfg = iteration_budget(60'000);
    cfg.batch_iterations = 100;
    cfg.perturbation_period = 10;
    cfg.column_generator = std::make_shared<RecordingGenerator>(log);
    cfg.tracer = &tracer;  // pricing_period = 0, price_on_stagnation defaults on
    const SearchResult r = solve(m, 0.0, 3, true, nullptr, nullptr, 3, nullptr, cfg);

    REQUIRE(r.perturbations > 5);
    REQUIRE(static_cast<int>(log->size()) == r.perturbations);
    int kicks = 0;
    for (size_t k = 0; k < tracer.events.size(); ++k) {
        const OrderTracer::Event& e = tracer.events[k];
        if (e.kind == OrderTracer::Kind::Kick) {
            ++kicks;
            // The event right before every kick is its pricing call.
            REQUIRE(k > 0);
            REQUIRE(tracer.events[k - 1].kind == OrderTracer::Kind::Pricing);
            REQUIRE(tracer.events[k - 1].why == PricingEvent::Stagnation);
        }
        if (e.kind == OrderTracer::Kind::Pricing) {
            // ...and every pricing call is followed by a kick.
            REQUIRE(k + 1 < tracer.events.size());
            REQUIRE(tracer.events[k + 1].kind == OrderTracer::Kind::Kick);
        }
    }
    REQUIRE(kicks == r.perturbations);
    for (const auto& c : *log) {
        REQUIRE(c.why == PricingEvent::Stagnation);
    }
}

TEST_CASE("NewBest pricing fires on every improving batch, before its weight reset", "[column]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    auto log = std::make_shared<std::vector<RecordingGenerator::Call>>();
    OrderTracer tracer;
    SearchConfig cfg = iteration_budget(60'000);
    cfg.column_generator = std::make_shared<RecordingGenerator>(log);
    cfg.price_on_stagnation = false;
    cfg.price_on_new_best = true;
    cfg.tracer = &tracer;
    (void)solve(cm.model, 0.0, 5, true, nullptr, nullptr, 3, nullptr, cfg);

    int improving = 0;
    int priced = 0;
    for (size_t k = 0; k < tracer.events.size(); ++k) {
        const OrderTracer::Event& e = tracer.events[k];
        if (e.kind == OrderTracer::Kind::BatchEnd && e.improved) {
            ++improving;
            REQUIRE(k + 1 < tracer.events.size());
            REQUIRE(tracer.events[k + 1].kind == OrderTracer::Kind::Pricing);
            REQUIRE(tracer.events[k + 1].why == PricingEvent::NewBest);
        }
        if (e.kind == OrderTracer::Kind::Pricing) {
            ++priced;
            REQUIRE(k > 0);
            REQUIRE(tracer.events[k - 1].kind == OrderTracer::Kind::BatchEnd);
            REQUIRE(tracer.events[k - 1].improved);
        }
    }
    REQUIRE(improving > 0);
    REQUIRE(priced == improving);
    REQUIRE(static_cast<int>(log->size()) == priced);
}

TEST_CASE("NewBest pricing sees the weights the improving batch left", "[column]") {
    // The weight reset for a new best happens AFTER the call, so the call sees
    // what the batch did to the weights. The model makes that observable: from
    // x = y = 0 the only way to x = y = 1 is through a GLS local minimum --
    // flipping either variable alone trades one unit of `x + y >= 2` for one of
    // `x == y` -- so the batch that first reaches feasibility has bumped a weight
    // on the way. Priced after the reset instead, every call would see all ones.
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    const int32_t both = m.sum({x, y});
    m.add_constraint(m.eq_expr(m.sum({m.prod(m.constant(1.0), x), m.prod(m.constant(-1.0), y)}),
                               m.constant(0.0)));
    m.add_constraint(m.geq(both, m.constant(2.0)));
    m.minimize(both);
    m.close();
    auto non_flat = std::make_shared<int>(0);
    class FlatCheck : public ColumnGenerator {
    public:
        explicit FlatCheck(std::shared_ptr<int> n) : n_(std::move(n)) {}
        void price(const PricingContext& ctx, PricingEvent /*why*/,
                   ModelExtension& /*ext*/) override {
            for (size_t i = 0; i < ctx.weights.size(); ++i) {
                if (ctx.weights[i] != 1.0) {
                    ++*n_;
                    return;
                }
            }
        }
        [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
            return std::make_unique<FlatCheck>(*this);
        }

    private:
        std::shared_ptr<int> n_;
    };
    SearchConfig cfg = iteration_budget(2'000);
    cfg.column_generator = std::make_shared<FlatCheck>(non_flat);
    cfg.price_on_stagnation = false;
    cfg.price_on_new_best = true;
    const SearchResult r = solve(m, 0.0, 5, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(r.feasible);
    REQUIRE(r.counters.pricing_calls > 0);
    REQUIRE(*non_flat > 0);
}

// ---------------------------------------------------------------------------
// End to end: cutting stock from the trivial pattern set.
// ---------------------------------------------------------------------------

TEST_CASE("a knapsack pricer beats the trivial pattern set on u120_00", "[column]") {
    const CuttingStock cs = u120_00();
    REQUIRE(cs.total_items() == 120);
    constexpr int64_t kCap = 400;
    for (const uint64_t seed : {1U, 2U, 3U, 4U, 5U}) {
        CAPTURE(seed);
        SearchConfig base = iteration_budget(30'000);

        CuttingModel plain = build_trivial(cs);
        const SearchResult without =
            solve(plain.model, 0.0, seed, true, nullptr, nullptr, 3, nullptr, base);

        CuttingModel grown = build_trivial(cs);
        const size_t base_vars = grown.model.num_vars();
        SearchConfig cfg = base;
        cfg.column_generator =
            std::make_shared<KnapsackPricer>(cs, grown.row_sums, grown.objective_sum);
        cfg.pricing_period = 5;
        cfg.max_generated_columns = kCap;
        const SearchResult with =
            solve(grown.model, 0.0, seed, true, nullptr, nullptr, 3, nullptr, cfg);

        REQUIRE(without.feasible);
        REQUIRE(with.feasible);
        UNSCOPED_INFO("seed " << seed << ": without " << without.objective << ", with "
                              << with.objective << " (optimum " << kU120Optimum << "), columns "
                              << with.counters.columns_added);
        REQUIRE(with.objective < without.objective);
        REQUIRE(with.objective >= kU120Optimum);
        // Under the cap, and the counters agree with the model.
        REQUIRE(with.counters.columns_added > 0);
        REQUIRE(with.counters.columns_added <= kCap);
        REQUIRE(grown.model.num_vars() - base_vars ==
                static_cast<size_t>(with.counters.columns_added));
        // The answer is a real solution of the grown model.
        REQUIRE(with.best_state.values.size() == grown.model.num_vars());
        REQUIRE(feasible_at(grown.model, with.best_state));
    }
}

// ---------------------------------------------------------------------------
// Budget.
// ---------------------------------------------------------------------------

TEST_CASE("no pricing call starts past the deadline, and the budget holds", "[column]") {
    // A pricer that spends whatever time it is told is left -- the worst
    // behaviour the contract allows -- on a model that never finishes, priced
    // after every batch. The batch in flight at the deadline ends there, so
    // without the guard its pricing call would start past it.
    Model m = infeasible_pair();
    auto log = std::make_shared<std::vector<RecordingGenerator::Call>>();
    class Spender : public ColumnGenerator {
    public:
        explicit Spender(std::shared_ptr<std::vector<RecordingGenerator::Call>> log)
            : log_(std::move(log)) {}
        void price(const PricingContext& ctx, PricingEvent why, ModelExtension& /*ext*/) override {
            const auto now = std::chrono::steady_clock::now();
            log_->push_back({why, ctx.batches, ctx.remaining_seconds, now});
            // Sleep a slice of what is left, so several calls happen and the last
            // lands near the deadline.
            const double slice = std::min(ctx.remaining_seconds, 0.02);
            std::this_thread::sleep_for(std::chrono::duration<double>(slice));
        }
        [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
            return std::make_unique<Spender>(*this);
        }

    private:
        std::shared_ptr<std::vector<RecordingGenerator::Call>> log_;
    };
    SearchConfig cfg;
    cfg.column_generator = std::make_shared<Spender>(log);
    cfg.pricing_period = 1;
    constexpr double kLimit = 0.4;
    const auto started = std::chrono::steady_clock::now();
    const SearchResult r = solve(m, kLimit, 11, true, nullptr, nullptr, 3, nullptr, cfg);
    const auto deadline = started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<double>(kLimit));

    REQUIRE(r.termination == TerminationReason::TimeLimit);
    REQUIRE(log->size() > 2);
    for (const auto& c : *log) {
        // The engine's own reading, which is the claim. The test's clock started
        // slightly before solve()'s, so its deadline gets a sliver of slack.
        REQUIRE(c.remaining > 0.0);
        REQUIRE(c.at < deadline + std::chrono::milliseconds(10));
    }
    REQUIRE(r.counters.pricing_calls == static_cast<int64_t>(log->size()));
    REQUIRE(r.counters.pricing_seconds > 0.0);
    // The run honours its wall clock: the pricer's sleeps are bounded by what it
    // was told was left, so the overrun is one sleep slice plus one batch.
    REQUIRE(r.time_seconds < kLimit + 0.25);
}

// ---------------------------------------------------------------------------
// No generator: nothing changes.
// ---------------------------------------------------------------------------

TEST_CASE("pricing settings without a generator leave the trajectory bit-identical", "[column]") {
    const CuttingStock cs = u120_00();
    for (const uint64_t seed : {1U, 9U}) {
        CuttingModel a = build_trivial(cs);
        CuttingModel b = build_trivial(cs);
        SearchConfig plain = iteration_budget(30'000);
        SearchConfig armed = plain;
        armed.pricing_period = 1;
        armed.price_on_new_best = true;
        armed.max_generated_columns = 5;
        armed.column_retire_age = 1;
        const SearchResult ra =
            solve(a.model, 0.0, seed, true, nullptr, nullptr, 3, nullptr, plain);
        const SearchResult rb =
            solve(b.model, 0.0, seed, true, nullptr, nullptr, 3, nullptr, armed);
        REQUIRE(ra.best_state.values == rb.best_state.values);
        REQUIRE(ra.iterations == rb.iterations);
        REQUIRE(ra.perturbations == rb.perturbations);
        REQUIRE(ra.objective == rb.objective);
        REQUIRE(rb.counters.pricing_calls == 0);
    }
}

TEST_CASE("a generator that stages nothing leaves the trajectory bit-identical", "[column]") {
    // Stronger than the test above: every pricing site runs -- the early resync,
    // the context, the aging pass -- and still nothing may move.
    const CuttingStock cs = u120_00();
    for (const uint64_t seed : {2U, 13U}) {
        CuttingModel a = build_trivial(cs);
        CuttingModel b = build_trivial(cs);
        SearchConfig plain = iteration_budget(30'000);
        SearchConfig armed = plain;
        auto log = std::make_shared<std::vector<RecordingGenerator::Call>>();
        armed.column_generator = std::make_shared<RecordingGenerator>(log);
        armed.pricing_period = 2;
        armed.price_on_new_best = true;
        armed.column_retire_age = 1;
        const SearchResult ra =
            solve(a.model, 0.0, seed, true, nullptr, nullptr, 3, nullptr, plain);
        const SearchResult rb =
            solve(b.model, 0.0, seed, true, nullptr, nullptr, 3, nullptr, armed);
        REQUIRE(rb.counters.pricing_calls > 0);
        REQUIRE(ra.best_state.values == rb.best_state.values);
        REQUIRE(ra.iterations == rb.iterations);
        REQUIRE(ra.perturbations == rb.perturbations);
        REQUIRE(ra.objective == rb.objective);
    }
}

// ---------------------------------------------------------------------------
// Column pool: cap, duplicates, retirement.
// ---------------------------------------------------------------------------

TEST_CASE("an extension past the column cap is refused whole", "[column]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    const size_t base_vars = cm.model.num_vars();
    // Two columns per call against a cap of three: the first call lands, and
    // every later one has room for one column only, so each is refused whole --
    // and, since the room never reaches zero, the generator keeps being asked.
    class TwoAtATime : public ColumnGenerator {
    public:
        explicit TwoAtATime(std::vector<int32_t> rows) : rows_(std::move(rows)) {}
        void price(const PricingContext& /*ctx*/, PricingEvent /*why*/,
                   ModelExtension& ext) override {
            for (int k = 0; k < 2; ++k) {
                const int32_t x = ext.int_var(0, 1);
                ext.append_to_sum(rows_[0], ext.prod(ext.constant(1.0), x));
            }
        }
        [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
            return std::make_unique<TwoAtATime>(*this);
        }

    private:
        std::vector<int32_t> rows_;
    };
    SearchConfig cfg = iteration_budget(20'000);
    cfg.column_generator = std::make_shared<TwoAtATime>(cm.row_sums);
    cfg.pricing_period = 1;
    cfg.max_generated_columns = 3;
    const SearchResult r = solve(cm.model, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(r.counters.columns_added == 2);
    REQUIRE(cm.model.num_vars() == base_vars + 2);
    REQUIRE(r.counters.extensions_refused == r.counters.pricing_calls - 1);
    REQUIRE(r.counters.pricing_calls > 2);
}

TEST_CASE("a full column pool stops pricing", "[column]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    class OneAtATime : public ColumnGenerator {
    public:
        explicit OneAtATime(std::vector<int32_t> rows) : rows_(std::move(rows)) {}
        void price(const PricingContext& /*ctx*/, PricingEvent /*why*/,
                   ModelExtension& ext) override {
            const int32_t x = ext.int_var(0, 1);
            ext.append_to_sum(rows_[0], ext.prod(ext.constant(1.0), x));
        }
        [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
            return std::make_unique<OneAtATime>(*this);
        }

    private:
        std::vector<int32_t> rows_;
    };
    SearchConfig cfg = iteration_budget(30'000);
    cfg.column_generator = std::make_shared<OneAtATime>(cm.row_sums);
    cfg.pricing_period = 1;
    cfg.max_generated_columns = 4;
    const SearchResult r = solve(cm.model, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(r.counters.batches > 10);
    REQUIRE(r.counters.columns_added == 4);
    REQUIRE(r.counters.pricing_calls == 4);
    REQUIRE(r.counters.extensions_refused == 0);
}

TEST_CASE("ColumnSignatureSet is exact and order-insensitive", "[column]") {
    ColumnSignatureSet set;
    REQUIRE(set.insert({{2, 1.0}, {0, 3.0}}, 1.0));
    // The same column listed in another order, with a split coefficient and a
    // zero entry, is the same signature.
    REQUIRE_FALSE(set.insert({{0, 1.0}, {2, 1.0}, {0, 2.0}, {5, 0.0}}, 1.0));
    REQUIRE(set.contains({{0, 3.0}, {2, 1.0}}, 1.0));
    // Any difference in cost, row or coefficient is a different column.
    REQUIRE(set.insert({{0, 3.0}, {2, 1.0}}, 2.0));
    REQUIRE(set.insert({{0, 3.0}, {3, 1.0}}, 1.0));
    REQUIRE(set.insert({{0, 3.0}, {2, 1.5}}, 1.0));
    REQUIRE(set.size() == 4);
}

TEST_CASE("ColumnPool retires only columns at their lower bound everywhere it looks", "[column]") {
    Model m;
    const int32_t base = m.int_var(0, 5);
    m.add_constraint(m.leq(m.sum({base}), m.constant(5.0)));
    m.close();
    ModelExtension ext(m);
    const int32_t c0 = ext.int_var(0, 3);
    const int32_t c1 = ext.int_var(0, 3);
    const int32_t c2 = ext.int_var(0, 3);
    ext.add_constraint(ext.leq(ext.sum({c0, c1, c2}), ext.constant(9.0)));
    const ExtensionResult res = m.extend(ext);

    ColumnPool pool(10, 2);
    pool.on_added(res);
    REQUIRE(pool.remaining() == 7);
    REQUIRE(pool.added() == 3);

    const int32_t v0 = -(c0 + 1);
    const int32_t v1 = -(c1 + 1);
    const int32_t v2 = -(c2 + 1);
    m.var_mut(v1).value = 2.0;  // c1 in use in the current assignment
    Model::State kept = m.copy_state();
    kept.values[static_cast<size_t>(v1)] = 0.0;
    kept.values[static_cast<size_t>(v2)] = 1.0;  // c2 in use by a kept state

    REQUIRE(pool.age(m, {&kept}).empty());  // c0 reaches age 1
    REQUIRE(pool.age(m, {&kept}) == std::vector<int32_t>{v0});
    REQUIRE(pool.retired() == 1);
    REQUIRE(pool.live() == 2);
    // Back at the bound everywhere, the other two start counting from zero.
    m.var_mut(v1).value = 0.0;
    kept.values[static_cast<size_t>(v2)] = 0.0;
    REQUIRE(pool.age(m, {&kept}).empty());
    REQUIRE(pool.age(m, {&kept}) == std::vector<int32_t>{v1, v2});
    REQUIRE(pool.retired() == 3);
}

TEST_CASE("retired columns are pinned and out of FJ's reach", "[column]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    const size_t base_vars = cm.model.num_vars();
    SearchConfig cfg = iteration_budget(40'000);
    cfg.column_generator = std::make_shared<KnapsackPricer>(cs, cm.row_sums, cm.objective_sum);
    cfg.pricing_period = 3;
    cfg.column_retire_age = 2;
    const SearchResult r = solve(cm.model, 0.0, 4, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(r.feasible);
    REQUIRE(r.counters.columns_retired > 0);
    int64_t pinned = 0;
    for (size_t v = base_vars; v < cm.model.num_vars(); ++v) {
        const Variable& var = cm.model.var(static_cast<int32_t>(v));
        if (var.ub == var.lb) {
            ++pinned;
            REQUIRE(r.best_state.values[v] == var.lb);
        }
    }
    REQUIRE(pinned == r.counters.columns_retired);
    REQUIRE(feasible_at(cm.model, r.best_state));
}

TEST_CASE("FeasibilityJump::retire refuses an unpinned variable and drops a pinned one",
          "[column]") {
    Model m;
    const int32_t a = m.int_var(0, 3);
    const int32_t b = m.bool_var();
    m.add_constraint(
        m.geq(m.sum({m.prod(m.constant(1.0), a), m.prod(m.constant(1.0), b)}), m.constant(2.0)));
    m.close();
    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    const int32_t bid = -(b + 1);
    REQUIRE_THROWS_AS(fj.retire({bid}), std::invalid_argument);  // Bool on [0, 1]
    REQUIRE_THROWS_AS(fj.retire({99}), std::out_of_range);
    m.var_mut(bid).value = 0.0;
    m.var_mut(bid).ub = 0.0;
    full_evaluate(m);
    fj.retire({bid});
    for (int i = 0; i < 10; ++i) {
        (void)fj.batch(100);
        fj.perturb(1.0);
    }
    REQUIRE(m.var(bid).value == 0.0);  // never moved again, not even by a kick
}

// ---------------------------------------------------------------------------
// The incumbent stays honest.
// ---------------------------------------------------------------------------

TEST_CASE("a row that cuts off the incumbent demotes it", "[column]") {
    // Once a feasible point exists, the generator adds the row x_0 <= d_0 - 1.
    // With the trivial patterns only x_0 covers size 0, so every feasible point
    // so far violates it -- and so does every point of the grown model, which is
    // now infeasible. The incumbent recorded before the cut must not survive it.
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    class Cut : public ColumnGenerator {
    public:
        explicit Cut(int d0) : d0_(d0) {}
        void price(const PricingContext& ctx, PricingEvent /*why*/, ModelExtension& ext) override {
            if (done_ || ctx.incumbent == nullptr) {
                return;
            }
            ext.add_constraint(ext.leq(-1, ext.constant(d0_ - 1)));  // handle -1 is x_0
            done_ = true;
        }
        [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
            return std::make_unique<Cut>(*this);
        }

    private:
        int d0_;
        bool done_ = false;
    };
    SearchConfig cfg = iteration_budget(100'000);
    cfg.column_generator = std::make_shared<Cut>(cs.demand[0]);
    cfg.pricing_period = 1;
    const SearchResult r = solve(cm.model, 0.0, 3, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(r.counters.rows_added == 1);
    REQUIRE(r.counters.incumbents_revalidated == 1);
    // x_0 can no longer cover demand_0, and nothing else holds that size, so
    // the grown model is infeasible -- and the result must say so rather than
    // hand back the cut-off incumbent as a solution.
    REQUIRE_FALSE(r.feasible);
    REQUIRE_FALSE(feasible_at(cm.model, r.best_state));
}

// ---------------------------------------------------------------------------
// Frozen models and the portfolio.
// ---------------------------------------------------------------------------

TEST_CASE("solve refuses a generator on a frozen model", "[column]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    cm.model.freeze();
    SearchConfig cfg = iteration_budget(1000);
    cfg.column_generator = std::make_shared<KnapsackPricer>(cs, cm.row_sums, cm.objective_sum);
    REQUIRE_THROWS_AS(solve(cm.model, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg),
                      std::invalid_argument);
    // ...and private_copy() is the way through.
    Model mine = cm.model.private_copy();
    REQUIRE_FALSE(mine.is_frozen());
    const SearchResult r = solve(mine, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(r.iterations > 0);
    REQUIRE(cm.model.num_vars() == cs.sizes.size());  // the shared structure never grew
}

TEST_CASE("the portfolio refuses a generator through a model factory", "[column][parallel]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    SearchConfig cfg;
    cfg.column_generator = std::make_shared<KnapsackPricer>(cs, cm.row_sums, cm.objective_sum);
    ParallelSearch ps(2);
    ParallelConfig pc;
    pc.n_threads = 2;
    const Model copy = cm.model;
    REQUIRE_THROWS_AS(
        ps.solve([&copy]() { return copy; }, 0.2, 1, cfg, nullptr, nullptr, nullptr, pc),
        std::invalid_argument);
}

namespace {
// Counts clones and records every instance that priced, so the portfolio test can
// see that each worker priced with its own copy. The registry is the TEST's
// shared state, guarded by its own mutex; the engine never touches it.
struct CloneRegistry {
    std::atomic<int> clones{0};
    std::mutex mutex;
    std::set<const void*> pricers;
};

class TrackedPricer : public KnapsackPricer {
public:
    TrackedPricer(const CuttingStock& cs, const CuttingModel& cm,
                  std::shared_ptr<CloneRegistry> reg)
        : KnapsackPricer(cs, cm.row_sums, cm.objective_sum), reg_(std::move(reg)) {}
    void price(const PricingContext& ctx, PricingEvent why, ModelExtension& ext) override {
        {
            const std::scoped_lock lock(reg_->mutex);
            reg_->pricers.insert(this);
        }
        KnapsackPricer::price(ctx, why, ext);
    }
    [[nodiscard]] std::unique_ptr<ColumnGenerator> clone() const override {
        reg_->clones.fetch_add(1, std::memory_order_relaxed);
        return std::make_unique<TrackedPricer>(*this);
    }

private:
    std::shared_ptr<CloneRegistry> reg_;
};
}  // namespace

TEST_CASE("each portfolio worker grows its own model with its own generator",
          "[column][parallel]") {
    const CuttingStock cs = u120_00();
    CuttingModel cm = build_trivial(cs);
    const size_t base_vars = cm.model.num_vars();
    auto reg = std::make_shared<CloneRegistry>();
    SearchConfig cfg;
    cfg.column_generator = std::make_shared<TrackedPricer>(cs, cm, reg);
    cfg.pricing_period = 5;
    cfg.max_generated_columns = 200;
    constexpr int kWorkers = 3;
    ParallelSearch ps(kWorkers);
    ParallelConfig pc;
    pc.n_threads = kWorkers;
    const SearchResult r = ps.solve(cm.model, 1.0, 17, cfg, nullptr, nullptr, nullptr, pc);

    // One clone per worker (one solve each, no restarts), and every worker priced
    // with a distinct instance.
    REQUIRE(reg->clones.load() == kWorkers);
    REQUIRE(reg->pricers.size() == static_cast<size_t>(kWorkers));
    REQUIRE(r.counters.portfolio_restarts == 0);
    // The master came back as the winner's grown model, open, and the answer
    // indexes it.
    REQUIRE_FALSE(cm.model.is_frozen());
    REQUIRE(cm.model.num_vars() > base_vars);
    REQUIRE(r.best_state.values.size() == cm.model.num_vars());
    REQUIRE(r.feasible);
    REQUIRE(feasible_at(cm.model, r.best_state));
    REQUIRE(r.objective < static_cast<double>(cs.total_items()));
    REQUIRE(r.counters.pricing_calls > 0);
}
