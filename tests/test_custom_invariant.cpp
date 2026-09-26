// User code inside the DAG: `NodeOp::Custom` and `CustomInvariant` (#166).
//
// Every fixture here computes its value FROM ITS OWN COMMITTED BELIEF about the
// inputs rather than by re-reading all of them, which is the point: an invariant
// that re-summed everything on every call would be correct under any probe
// protocol at all, and would test nothing. Here a `commit()` the engine owes and
// does not pay, or a `rollback()` it pays twice, desynchronises the belief from
// the assignment and shows up as a WRONG VALUE against `full_evaluate` -- not as
// a slow one.
//
// That is also why the property test keeps a second model and re-derives from
// scratch there: the assertion is "incremental agrees with from-scratch after
// every step", and only a from-scratch pass on a DIFFERENT invariant instance
// can say so.

#include "cbls/custom_invariant.h"
#include "cbls/dag_ops.h"
#include "cbls/expr.h"
#include "cbls/io.h"
#include "cbls/model.h"
#include "cbls/moves.h"
#include "cbls/pool.h"
#include "cbls/rng.h"
#include "cbls/violation.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace cbls;

namespace {

// Shared across every clone of one invariant, so a test can ask what the engine
// did without reaching into a particular instance.
//
// Atomic because the portfolio test's workers write it from their own threads;
// `serials` is mutex-guarded and is written at most once per instance, so the
// lock does NOT serialise the invariant's own state and a shared instance would
// still be a race ThreadSanitizer reports.
struct CallLog {
    std::atomic<int> evaluate{0};
    std::atomic<int> delta{0};
    std::atomic<int> commit{0};
    std::atomic<int> rollback{0};
    std::atomic<int> clone{0};
    std::atomic<int> next_serial{0};
    std::mutex mu;
    std::set<int> serials;  // guarded by mu: which instances did any work

    void reset() {
        evaluate = 0;
        delta = 0;
        commit = 0;
        rollback = 0;
        clone = 0;
        const std::lock_guard<std::mutex> guard(mu);
        serials.clear();
    }
    [[nodiscard]] size_t distinct_workers() {
        const std::lock_guard<std::mutex> guard(mu);
        return serials.size();
    }
};

// value = sum of the scalar inputs + sum over each structured input's elements
// of (element + 1).
//
// `delta` re-reads only the inputs `changed` names and folds their difference
// into the committed value, so its cost is in the inputs that moved rather than
// in the node's arity. `partial` is 1.0 for a scalar input and unknown (NaN) for
// a structured one, which is the honest answer.
class BeliefSum : public CustomInvariant {
public:
    explicit BeliefSum(std::shared_ptr<CallLog> log)
        : log_(std::move(log)), serial_(log_->next_serial.fetch_add(1)) {}

    double evaluate(const InvariantInputs& in) override {
        log_->evaluate.fetch_add(1);
        note_active();
        const int32_t n = in.size();
        scalars_.assign(static_cast<size_t>(n), 0.0);
        elements_.assign(static_cast<size_t>(n), {});
        double total = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            if (in.is_structured_input(i)) {
                const ConstSpan<int32_t> el = in.elements(i);
                elements_[static_cast<size_t>(i)].assign(el.begin(), el.end());
                total += element_sum(el);
            } else {
                scalars_[static_cast<size_t>(i)] = in.value(i);
                total += in.value(i);
            }
        }
        value_ = total;
        staged_value_ = total;
        staged_.clear();
        return total;
    }

    double delta(const InvariantInputs& in, ConstSpan<int32_t> changed) override {
        log_->delta.fetch_add(1);
        note_active();
        staged_value_ = value_;
        staged_.clear();
        for (const int32_t i : changed) {
            const auto slot = static_cast<size_t>(i);
            Staged entry;
            entry.index = i;
            if (in.is_structured_input(i)) {
                const ConstSpan<int32_t> el = in.elements(i);
                entry.structured = true;
                entry.elements.assign(el.begin(), el.end());
                staged_value_ += element_sum(el) - element_sum(elements_[slot]);
            } else {
                entry.value = in.value(i);
                staged_value_ += in.value(i) - scalars_[slot];
            }
            staged_.push_back(std::move(entry));
        }
        return staged_value_;
    }

    void commit() override {
        log_->commit.fetch_add(1);
        value_ = staged_value_;
        for (const Staged& entry : staged_) {
            const auto slot = static_cast<size_t>(entry.index);
            if (entry.structured) {
                elements_[slot] = entry.elements;
            } else {
                scalars_[slot] = entry.value;
            }
        }
        staged_.clear();
    }

    void rollback() override {
        log_->rollback.fetch_add(1);
        staged_value_ = value_;
        staged_.clear();
    }

    double partial(const InvariantInputs& in, int32_t i) override {
        if (in.is_structured_input(i)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0;
    }

    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        log_->clone.fetch_add(1);
        auto copy = std::make_unique<BeliefSum>(*this);
        copy->serial_ = log_->next_serial.fetch_add(1);
        copy->noted_ = false;
        return copy;
    }

private:
    struct Staged {
        int32_t index = -1;
        bool structured = false;
        double value = 0.0;
        std::vector<int32_t> elements;
    };

    template <typename Range>
    static double element_sum(const Range& elements) {
        double total = 0.0;
        for (const int32_t e : elements) {
            total += static_cast<double>(e) + 1.0;
        }
        return total;
    }

    // Recorded once per instance, so the mutex is out of the way of the state
    // below it -- see CallLog.
    void note_active() {
        if (noted_) {
            return;
        }
        noted_ = true;
        const std::lock_guard<std::mutex> guard(log_->mu);
        log_->serials.insert(serial_);
    }

    std::shared_ptr<CallLog> log_;
    int serial_ = -1;
    bool noted_ = false;
    // The committed belief. Desynchronise it and the value goes wrong.
    double value_ = 0.0;
    std::vector<double> scalars_;
    std::vector<std::vector<int32_t>> elements_;
    // Staged by the last `delta`, owed a commit() or a rollback().
    double staged_value_ = 0.0;
    std::vector<Staged> staged_;
};

// A model whose every DAG position holds the custom node: it is a constraint
// body, it feeds an expression, and that expression is the objective.
//
//   c = custom(b, k, f, L)
//   minimize c + 100        (an expression over c)
//   c <= 2                  (c as a constraint body)
//
// The bound is 2 rather than something comfortable so that the row's VIOLATION
// changes across the probes below, not just its residual: `per_constraint_
// violation_delta` reports only rows whose violation moved, and a row that is
// satisfied at every assignment the test visits would make it report nothing.
struct Fixture {
    Model model;
    std::shared_ptr<CallLog> log;
    int32_t b = 0;
    int32_t k = 0;
    int32_t f = 0;
    int32_t list = 0;
    int32_t custom_node = -1;
    int32_t row = -1;
};

Fixture build_fixture(const std::shared_ptr<CallLog>& log) {
    Fixture fx;
    fx.log = log;
    Model& m = fx.model;
    fx.b = m.bool_var("b");
    fx.k = m.int_var(0, 9, "k");
    fx.f = m.float_var(-3.0, 3.0, "f");
    fx.list = m.list_var(6, 0, 6, ListInit::Random, "L");
    fx.custom_node =
        m.custom({fx.b, fx.k, fx.f, fx.list}, std::make_unique<BeliefSum>(log), "belief_sum");
    fx.row = m.leq(fx.custom_node, m.constant(2.0));
    m.add_constraint(fx.row);
    m.minimize(m.sum({fx.custom_node, m.constant(100.0)}));
    m.close();
    return fx;
}

// What the fixture's custom node is worth at the model's current assignment,
// computed here rather than by the invariant.
double expected_value(const Model& m, const Fixture& fx) {
    double total = m.var(handle_to_var_id(fx.b)).value + m.var(handle_to_var_id(fx.k)).value +
                   m.var(handle_to_var_id(fx.f)).value;
    for (const int32_t e : m.var(handle_to_var_id(fx.list)).elements) {
        total += static_cast<double>(e) + 1.0;
    }
    return total;
}

}  // namespace

TEST_CASE("a custom node is an ordinary node in every DAG position", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    Model& m = fx.model;

    SECTION("it evaluates where it sits") {
        REQUIRE_THAT(m.node_value(fx.custom_node), WithinAbs(expected_value(m, fx), 1e-12));
    }

    SECTION("the expression above it sees its value") {
        REQUIRE_THAT(m.node_value(m.objective_id()),
                     WithinAbs(expected_value(m, fx) + 100.0, 1e-12));
    }

    SECTION("the constraint row above it sees its value") {
        REQUIRE_THAT(m.node_value(fx.row), WithinAbs(expected_value(m, fx) - 2.0, 1e-12));
    }

    SECTION("as the objective directly") {
        Model direct;
        const int32_t x = direct.int_var(0, 10, "x");
        const int32_t c = direct.custom({x}, std::make_unique<BeliefSum>(log), "obj");
        direct.minimize(c);
        direct.close();
        direct.var_mut(handle_to_var_id(x)).value = 7.0;
        delta_evaluate(direct, {handle_to_var_id(x)});
        REQUIRE_THAT(direct.node_value(direct.objective_id()), WithinAbs(7.0, 1e-12));
    }
}

TEST_CASE("full_evaluate is the invariant's reset point", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    // close() already ran one full_evaluate over every node.
    REQUIRE(log->evaluate.load() == 1);
    REQUIRE(log->delta.load() == 0);

    full_evaluate(fx.model);
    REQUIRE(log->evaluate.load() == 2);
    REQUIRE_THAT(fx.model.node_value(fx.custom_node),
                 WithinAbs(expected_value(fx.model, fx), 1e-12));
}

TEST_CASE("the move path calls delta, never evaluate", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    Model& m = fx.model;
    const int32_t kid = handle_to_var_id(fx.k);

    log->reset();
    m.var_mut(kid).value = 5.0;
    delta_evaluate(m, &kid, 1);

    CHECK(log->evaluate.load() == 0);
    CHECK(log->delta.load() == 1);
    CHECK(log->commit.load() == 1);
    CHECK(log->rollback.load() == 0);
    REQUIRE_THAT(m.node_value(fx.custom_node), WithinAbs(expected_value(m, fx), 1e-12));
}

TEST_CASE("one scalar probe costs one delta and one rollback", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    Model& m = fx.model;
    ViolationManager vm(m);
    const int32_t kid = handle_to_var_id(fx.k);
    const double before = m.node_value(fx.custom_node);

    log->reset();
    const double d = vm.weighted_violation_delta(kid, 9.0);

    // This is criterion 2 of #166: the per-candidate scalar probe is bracketed,
    // so the invariant sees ONE incremental call and ONE rollback -- not two
    // deltas, and not a re-evaluation from scratch.
    CHECK(log->delta.load() == 1);
    CHECK(log->rollback.load() == 1);
    CHECK(log->commit.load() == 0);
    CHECK(log->evaluate.load() == 0);
    // The probe is a pure counterfactual: the node is back where it was.
    REQUIRE_THAT(m.node_value(fx.custom_node), WithinAbs(before, 1e-12));
    REQUIRE(std::isfinite(d));

    SECTION("and the sparse variant is bracketed too") {
        log->reset();
        const auto per_row = m.per_constraint_violation_delta(kid, 9.0);
        CHECK(log->delta.load() == 1);
        CHECK(log->rollback.load() == 1);
        CHECK(log->commit.load() == 0);
        CHECK(log->evaluate.load() == 0);
        REQUIRE_THAT(m.node_value(fx.custom_node), WithinAbs(before, 1e-12));
        REQUIRE_FALSE(per_row.empty());  // the custom node really is in the row
    }

    SECTION("and many probes in a row leave the belief intact") {
        for (int j = 0; j <= 9; ++j) {
            (void)vm.weighted_violation_delta(kid, static_cast<double>(j));
        }
        CHECK(log->rollback.load() == 11);  // 1 above + 10 here
        CHECK(log->commit.load() == 0);
        // A committed move after all those probes must still land on the right
        // value, which it only can if every rollback restored the belief.
        m.var_mut(kid).value = 4.0;
        delta_evaluate(m, &kid, 1);
        REQUIRE_THAT(m.node_value(fx.custom_node), WithinAbs(expected_value(m, fx), 1e-12));
    }
}

TEST_CASE("delta_evaluate and full_evaluate agree over a random move sequence", "[custom]") {
    auto live_log = std::make_shared<CallLog>();
    auto ref_log = std::make_shared<CallLog>();
    Fixture live = build_fixture(live_log);
    Fixture ref = build_fixture(ref_log);

    const int32_t bid = handle_to_var_id(live.b);
    const int32_t kid = handle_to_var_id(live.k);
    const int32_t fid = handle_to_var_id(live.f);
    const int32_t lid = handle_to_var_id(live.list);
    ViolationManager vm(live.model);

    // Same starting assignment in both models.
    ref.model.restore_state(live.model.copy_state());
    full_evaluate(ref.model);

    RNG rng(20250926);
    int structural_steps = 0;
    for (int step = 0; step < 400; ++step) {
        const int64_t kind = rng.integers(0, 4);
        if (kind == 0) {
            live.model.var_mut(bid).value = static_cast<double>(rng.integers(0, 2));
            delta_evaluate(live.model, &bid, 1);
        } else if (kind == 1) {
            live.model.var_mut(kid).value = static_cast<double>(rng.integers(0, 10));
            delta_evaluate(live.model, &kid, 1);
        } else if (kind == 2) {
            live.model.var_mut(fid).value = rng.uniform(-3.0, 3.0);
            delta_evaluate(live.model, &fid, 1);
        } else {
            // A positional ElementEdit, the representation the structural batch
            // actually builds (#164) -- not a whole-vector replacement.
            const auto len = static_cast<int32_t>(live.model.var(lid).elements.size());
            std::vector<Move> moves;
            generate_standard_moves(live.model.var(lid), rng, moves, nullptr);
            if (!moves.empty()) {
                const auto pick =
                    static_cast<size_t>(rng.integers(0, static_cast<int64_t>(moves.size())));
                apply_move(live.model, moves[pick]);
                delta_evaluate(live.model, &lid, 1);
                ++structural_steps;
            }
            (void)len;
        }

        // Interleave the bracketed probe, which is the thing most likely to
        // leave the belief stale if the protocol is wrong.
        if (step % 3 == 0) {
            (void)vm.weighted_violation_delta(kid, static_cast<double>(rng.integers(0, 10)));
        }

        // From scratch, on a different invariant instance, from the same
        // assignment.
        ref.model.restore_state(live.model.copy_state());
        full_evaluate(ref.model);

        REQUIRE_THAT(live.model.node_value(live.custom_node),
                     WithinAbs(ref.model.node_value(ref.custom_node), 1e-9));
        REQUIRE_THAT(live.model.node_value(live.row),
                     WithinAbs(ref.model.node_value(ref.row), 1e-9));
        REQUIRE_THAT(live.model.node_value(live.model.objective_id()),
                     WithinAbs(ref.model.node_value(ref.model.objective_id()), 1e-9));
    }
    // The List half of criterion 1 is only covered if List edits actually
    // happened -- a probe that connects nothing proves nothing.
    REQUIRE(structural_steps > 20);
    REQUIRE(live_log->delta.load() > 0);
    REQUIRE(live_log->rollback.load() > 0);
}

TEST_CASE("a custom node's partial reaches the AD path", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    Model& m = fx.model;

    // d(custom)/d(f) == 1.0, which BeliefSum reports for a scalar input.
    REQUIRE_THAT(compute_partial(m, fx.custom_node, handle_to_var_id(fx.f)), WithinAbs(1.0, 1e-12));
    // The List input's partial is NaN ("unknown"), which the engine reads as 0
    // rather than propagating.
    const double structured = compute_partial(m, fx.custom_node, handle_to_var_id(fx.list));
    REQUIRE(std::isfinite(structured));
    REQUIRE_THAT(structured, WithinAbs(0.0, 1e-12));
    // And the row above it differentiates through the custom node.
    REQUIRE_THAT(compute_partial(m, fx.row, handle_to_var_id(fx.f)), WithinAbs(1.0, 1e-12));
}

TEST_CASE("copying a model clones its invariants rather than sharing them", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    Model& original = fx.model;
    const double before = original.node_value(fx.custom_node);
    const int clones_before = log->clone.load();

    Model copy(original);
    REQUIRE(log->clone.load() == clones_before + 1);
    REQUIRE(&copy.custom_invariant(0) != &original.custom_invariant(0));

    // Move the copy. The original's invariant must not have heard about it --
    // which is exactly what a shared instance would get wrong, and what would
    // otherwise show up only as a portfolio trajectory divergence.
    const int32_t kid = handle_to_var_id(fx.k);
    copy.var_mut(kid).value = 9.0;
    delta_evaluate(copy, &kid, 1);

    REQUIRE_THAT(original.node_value(fx.custom_node), WithinAbs(before, 1e-12));
    original.var_mut(kid).value = 1.0;
    delta_evaluate(original, &kid, 1);
    REQUIRE_THAT(original.node_value(fx.custom_node),
                 WithinAbs(expected_value(original, fx), 1e-12));
    REQUIRE_THAT(copy.node_value(fx.custom_node), WithinAbs(expected_value(copy, fx), 1e-12));

    SECTION("and a frozen master shares its DAG while keeping its own invariant") {
        Model frozen = fx.model;
        frozen.freeze();
        Model replica(frozen);
        REQUIRE(replica.is_frozen());
        REQUIRE(&replica.node(fx.custom_node) == &frozen.node(fx.custom_node));
        REQUIRE(&replica.custom_invariant(0) != &frozen.custom_invariant(0));
    }
}

TEST_CASE("a portfolio gives every worker its own invariant", "[custom][parallel]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);
    fx.model.freeze();

    log->reset();
    ParallelSearch ps(2);
    const SearchResult r = ps.solve(fx.model, 0.4, 7);

    // Two workers, each with its own clone, and both of them did work. If the
    // engine handed one instance to both threads, the non-atomic belief inside
    // BeliefSum would be a data race -- which is what the ThreadSanitizer run
    // of this test case checks, and what this count can only suggest.
    CHECK(log->clone.load() >= 2);
    CHECK(log->distinct_workers() >= 2);
    CHECK(log->delta.load() > 0);
    REQUIRE(std::isfinite(r.time_seconds));
}

TEST_CASE("a custom node cannot be serialised, and says which node", "[custom][io]") {
    auto log = std::make_shared<CallLog>();
    Fixture fx = build_fixture(log);

    std::ostringstream out;
    try {
        save_model(fx.model, out);
        FAIL("save_model accepted a model holding a custom node");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        CHECK(msg.find("belief_sum") != std::string::npos);
        CHECK(msg.find("n" + std::to_string(fx.custom_node)) != std::string::npos);
    }
    // Nothing was written: the refusal comes before the first line, so an
    // existing file is not replaced by a prefix of a model.
    REQUIRE(out.str().empty());

    SECTION("an unnamed node is still identified") {
        Model m;
        const int32_t x = m.int_var(0, 1, "x");
        const int32_t c = m.custom({x}, std::make_unique<BeliefSum>(log));
        m.add_constraint(m.leq(c, m.constant(0.0)));
        m.close();
        std::ostringstream unnamed;
        try {
            save_model(m, unnamed);
            FAIL("save_model accepted an unnamed custom node");
        } catch (const std::runtime_error& e) {
            CHECK(std::string(e.what()).find("<unnamed>") != std::string::npos);
        }
    }
}

TEST_CASE("Model::custom refuses what it cannot build", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Model m;
    const int32_t x = m.int_var(0, 1, "x");

    REQUIRE_THROWS_AS(m.custom({x}, nullptr, "null"), std::invalid_argument);
    REQUIRE_THROWS_AS(m.custom({-99}, std::make_unique<BeliefSum>(log), "bad"), std::out_of_range);
    REQUIRE_THROWS_AS(m.custom({12345}, std::make_unique<BeliefSum>(log), "bad"),
                      std::out_of_range);
    // Nothing was registered by any of the three.
    REQUIRE_FALSE(m.has_custom_nodes());

    const int32_t c = m.custom({x}, std::make_unique<BeliefSum>(log), "ok");
    m.add_constraint(m.leq(c, m.constant(5.0)));
    m.close();
    REQUIRE(m.has_custom_nodes());
    REQUIRE(m.custom_name(0) == "ok");
    REQUIRE_THROWS_AS(m.custom_invariant(1), std::out_of_range);
    REQUIRE_THROWS_AS(m.custom_name(-1), std::out_of_range);

    m.freeze();
    REQUIRE_THROWS_AS(m.custom({x}, std::make_unique<BeliefSum>(log), "frozen"), std::logic_error);
}

TEST_CASE("the Expr form of a custom node composes", "[custom]") {
    auto log = std::make_shared<CallLog>();
    Model m;
    const Expr x = m.Int(0, 10, "x");
    const Expr y = m.Float(0.0, 5.0, "y");
    const Expr c = m.Custom({x, y}, std::make_unique<BeliefSum>(log), "sum");
    m.add_constraint(c <= m.Constant(100.0));
    m.minimize(c * m.Constant(2.0));
    m.close();

    x.var_mut().value = 3.0;
    y.var_mut().value = 1.5;
    delta_evaluate(m, {x.var_id(), y.var_id()});
    REQUIRE_THAT(m.node_value(c.handle), WithinAbs(4.5, 1e-12));
    REQUIRE_THAT(m.node_value(m.objective_id()), WithinAbs(9.0, 1e-12));
}

TEST_CASE("the default CustomInvariant delta is a from-scratch evaluate", "[custom]") {
    // An invariant with no incremental form at all: the base-class `delta`
    // forwards to `evaluate`, which is what an author who has nothing cheaper
    // should be able to ship.
    class PlainSquare : public CustomInvariant {
    public:
        double evaluate(const InvariantInputs& in) override {
            ++calls;
            const double v = in.value(0);
            return v * v;
        }
        [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
            return std::make_unique<PlainSquare>(*this);
        }
        int calls = 0;
    };

    Model m;
    const int32_t x = m.int_var(0, 6, "x");
    const int32_t c = m.custom({x}, std::make_unique<PlainSquare>(), "square");
    m.add_constraint(m.leq(c, m.constant(100.0)));
    m.close();

    const int32_t xid = handle_to_var_id(x);
    m.var_mut(xid).value = 5.0;
    delta_evaluate(m, &xid, 1);
    REQUIRE_THAT(m.node_value(c), WithinAbs(25.0, 1e-12));

    // The default `partial` is NaN, read as zero -- so the node contributes no
    // gradient rather than a NaN one.
    REQUIRE_THAT(compute_partial(m, c, xid), WithinAbs(0.0, 1e-12));
}
