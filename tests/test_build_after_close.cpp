// A closed model does not grow (#173). Before the builders refused a closed
// model, a node or row they appended after close() was never placed in the
// topological order, so no evaluation computed it and solve() reported feasible
// over a violated row. The one internal post-close addition is the objective
// row, which `add_objective_soft_constraint` appends and rebuilds for -- and
// which `freeze()` and the first `solve()` of an objective model both run, so a
// ViolationManager or FeasibilityJump built before it is one row short.
//
// `tests/python/test_build_after_close.py` is the binding half.

#include "cbls/custom_invariant.h"
#include "cbls/dag_ops.h"
#include "cbls/expr.h"
#include "cbls/feasibility_jump.h"
#include "cbls/model.h"
#include "cbls/search.h"
#include "cbls/violation.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

using namespace cbls;

namespace {

// A pure sum of its inputs: enough of a CustomInvariant to reach `Model::custom`.
class InputSum : public CustomInvariant {
public:
    double evaluate(const InvariantInputs& in) override {
        double total = 0.0;
        for (int32_t i = 0; i < in.size(); ++i) {
            total += in.value(i);
        }
        return total;
    }
    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        return std::make_unique<InputSum>(*this);
    }
};

void require_valid_topo_order(const Model& m) {
    const std::vector<int32_t>& order = m.topo_order();
    REQUIRE(order.size() == m.num_nodes());
    std::vector<int32_t> seen(m.num_nodes(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        const int32_t nid = order[i];
        REQUIRE(nid >= 0);
        REQUIRE(static_cast<size_t>(nid) < m.num_nodes());
        REQUIRE(seen[nid] == 0);
        seen[nid] = 1;
        REQUIRE(m.topo_position(nid) == static_cast<int32_t>(i));
        for (const ChildRef& child : m.children(m.nodes()[nid])) {
            if (!child.is_var) {
                // A child must already have been emitted.
                REQUIRE(seen[child.id] == 1);
            }
        }
    }
}

// Everything a refused builder could have touched, so each refusal can be
// checked to leave the model exactly as it found it.
struct Footprint {
    size_t vars = 0;
    size_t nodes = 0;
    size_t constraints = 0;
    int32_t objective = -1;
    bool maximizing = false;
    size_t sequences = 0;
    size_t partitions = 0;

    explicit Footprint(const Model& m)
        : vars(m.num_vars()),
          nodes(m.num_nodes()),
          constraints(m.constraint_ids().size()),
          objective(m.objective_id()),
          maximizing(m.is_maximizing()),
          sequences(m.var_sequences().size()),
          partitions(m.list_partitions().size()) {}

    bool operator==(const Footprint& o) const {
        return vars == o.vars && nodes == o.nodes && constraints == o.constraints &&
               objective == o.objective && maximizing == o.maximizing && sequences == o.sequences &&
               partitions == o.partitions;
    }
};

struct ClosedFixture {
    Model m;
    int32_t x = 0;    // Int [0, 10]
    int32_t y = 0;    // Int [0, 10]
    int32_t l1 = 0;   // List, universe 4, unpartitioned
    int32_t l2 = 0;   // List, universe 4, unpartitioned
    int32_t row = 0;  // x + y, a node built before close()
    int32_t cut = 0;  // x >= 5, a node built before close() but never added

    ClosedFixture() {
        x = m.int_var(0, 10, "x");
        y = m.int_var(0, 10, "y");
        l1 = m.list_var(4, 0, 4, ListInit::Empty, "l1");
        l2 = m.list_var(4, 0, 4, ListInit::Empty, "l2");
        row = m.sum({x, y});
        cut = m.geq(x, m.constant(5.0));
        m.add_constraint(m.leq(row, m.constant(20.0)));
        m.add_constraint(m.leq(m.lambda_sum(l1, [](int e) { return e; }), m.constant(100.0)));
        m.add_constraint(m.leq(m.lambda_sum(l2, [](int e) { return e; }), m.constant(100.0)));
        m.close();
    }
};

void require_refused(ClosedFixture& f, const std::function<void()>& build) {
    const Footprint before(f.m);
    REQUIRE_THROWS_MATCHES(
        build(), std::logic_error,
        Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("model is closed")));
    REQUIRE(Footprint(f.m) == before);
}

}  // namespace

TEST_CASE("solving an unclosed model without an objective closes it and solves it", "[closed]") {
    // Without an objective there is no objective row, so nothing rebuilt the
    // derived indices: the run read an unbuilt variable-to-constraint index and
    // failed with "var id out of range" instead of solving the model.
    Model m;
    const int32_t x = m.int_var(0, 10, "x");
    m.add_constraint(m.geq(x, m.constant(5.0)));
    REQUIRE_FALSE(m.is_closed());
    SearchConfig cfg;
    cfg.max_iterations = 2000;
    const SearchResult r = solve(m, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(m.is_closed());
    REQUIRE(r.feasible);
    m.restore_state(r.best_state);
    REQUIRE(m.var(handle_to_var_id(x)).value >= 5.0);
    REQUIRE_THROWS_AS(m.add_constraint(m.leq(x, m.constant(9.0))), std::logic_error);
}

TEST_CASE("solving an unclosed objective model closes it, so later builders refuse", "[closed]") {
    // The first solve of an objective model appends the objective row and runs
    // the same rebuild close() does. It used to leave closed_ false, so a row
    // added afterwards was accepted, never evaluated, and the next solve
    // reported feasible over it -- #173's wrong answer without a close() call.
    Model m;
    const int32_t x = m.float_var(0.0, 10.0, "x");
    const int32_t five = m.constant(5.0);
    m.minimize(m.sum({x}));
    REQUIRE_FALSE(m.is_closed());
    SearchConfig cfg;
    cfg.max_iterations = 200;
    (void)solve(m, 0.0, 1, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE(m.is_closed());
    REQUIRE_THROWS_AS(m.add_constraint(m.geq(x, five)), std::logic_error);
}

TEST_CASE("add_constraint after close is refused rather than silently unevaluated", "[closed]") {
    // The issue's repro: the row x >= 5 arrived after close(), was never
    // evaluated, and solve() returned feasible at x == 0.
    Model m;
    const int32_t x = m.int_var(0, 10, "x");
    m.add_constraint(m.leq(x, m.constant(20.0)));
    const int32_t five = m.constant(5.0);  // built before close, so only the row is new
    m.close();

    const size_t rows = m.constraint_ids().size();
    REQUIRE_THROWS_AS(m.add_constraint(m.geq(x, five)), std::logic_error);
    REQUIRE_THROWS_AS(m.constant(5.0), std::logic_error);
    REQUIRE(m.constraint_ids().size() == rows);
}

TEST_CASE("every variable builder refuses a closed model", "[closed]") {
    ClosedFixture f;
    require_refused(f, [&] { (void)f.m.bool_var(); });
    require_refused(f, [&] { (void)f.m.int_var(0, 1); });
    require_refused(f, [&] { (void)f.m.float_var(0.0, 1.0); });
    require_refused(f, [&] { (void)f.m.list_var(3); });
    require_refused(f, [&] { (void)f.m.list_var(3, 0, 3); });
    require_refused(f, [&] { (void)f.m.set_var(3); });
    require_refused(f, [&] { (void)f.m.Bool(); });
    require_refused(f, [&] { (void)f.m.Int(0, 1); });
    require_refused(f, [&] { (void)f.m.Float(0.0, 1.0); });
    require_refused(f, [&] { (void)f.m.List(3); });
    require_refused(f, [&] { (void)f.m.List(3, 0, 3); });
    require_refused(f, [&] { (void)f.m.Set(3); });
}

TEST_CASE("every expression builder refuses a closed model", "[closed]") {
    ClosedFixture f;
    Model& m = f.m;
    const int32_t x = f.x;
    const int32_t y = f.y;
    require_refused(f, [&] { (void)m.constant(1.0); });
    require_refused(f, [&] { (void)m.Constant(1.0); });
    require_refused(f, [&] { (void)m.neg(x); });
    require_refused(f, [&] { (void)m.sum({x, y}); });
    require_refused(f, [&] { (void)m.prod(x, y); });
    require_refused(f, [&] { (void)m.div_expr(x, y); });
    require_refused(f, [&] { (void)m.pow_expr(x, y); });
    require_refused(f, [&] { (void)m.min_expr({x, y}); });
    require_refused(f, [&] { (void)m.max_expr({x, y}); });
    require_refused(f, [&] { (void)m.abs_expr(x); });
    require_refused(f, [&] { (void)m.sin_expr(x); });
    require_refused(f, [&] { (void)m.cos_expr(x); });
    require_refused(f, [&] { (void)m.tan_expr(x); });
    require_refused(f, [&] { (void)m.exp_expr(x); });
    require_refused(f, [&] { (void)m.log_expr(x); });
    require_refused(f, [&] { (void)m.sqrt_expr(x); });
    require_refused(f, [&] { (void)m.signpower_expr(x, y); });
    require_refused(f, [&] { (void)m.tanh_expr(x); });
    require_refused(f, [&] { (void)m.if_then_else(f.cut, x, y); });
    require_refused(f, [&] { (void)m.at(f.l1, x); });
    require_refused(f, [&] { (void)m.count(f.l1); });
    require_refused(f, [&] { (void)m.leq(x, y); });
    require_refused(f, [&] { (void)m.eq_expr(x, y); });
    require_refused(f, [&] { (void)m.geq(x, y); });
    require_refused(f, [&] { (void)m.neq(x, y); });
    require_refused(f, [&] { (void)m.lt(x, y); });
    require_refused(f, [&] { (void)m.gt(x, y); });
    require_refused(f, [&] { (void)m.lambda_sum(f.l1, [](int e) { return e; }); });
    require_refused(f, [&] {
        (void)m.pair_lambda_sum(f.l1, [](int a, int b) { return a + b; }, PairMode::Cyclic);
    });
    require_refused(f, [&] {
        (void)m.pair_lambda_sum(
            f.l1, [](int a, int b) { return a + b; }, [](int e) { return e; },
            [](int e) { return e; });
    });
    require_refused(f, [&] { (void)m.custom({x, y}, std::make_unique<InputSum>(), "c"); });
    require_refused(f, [&] { (void)m.Custom({Expr{&m, x}}, std::make_unique<InputSum>(), "c"); });
    // The operator overloads reach the same builders.
    require_refused(f, [&] { (void)(Expr{&m, x} + Expr{&m, y}); });
    require_refused(f, [&] { (void)(Expr{&m, x} <= 3.0); });
}

TEST_CASE("constraints, objectives and declarations refuse a closed model", "[closed]") {
    ClosedFixture f;
    Model& m = f.m;
    require_refused(f, [&] { m.add_constraint(f.cut); });
    require_refused(f, [&] { m.add_constraint(Expr{&m, f.cut}); });
    require_refused(f, [&] { m.minimize(f.row); });
    require_refused(f, [&] { m.minimize(Expr{&m, f.row}); });
    require_refused(f, [&] { m.maximize(f.row); });
    require_refused(f, [&] { m.maximize(Expr{&m, f.row}); });
    require_refused(f, [&] { m.add_var_sequence({f.x, f.y}); });
    require_refused(f, [&] { (void)m.add_list_partition({f.l1, f.l2}, Cover::AtMostOnce); });
}

TEST_CASE("a closed model still takes per-variable and search writes", "[closed]") {
    // The refusal is for STRUCTURE. What a search writes -- a variable's value,
    // the objective bound, a state restore -- is per-model state and stays open.
    ClosedFixture f;
    Model& m = f.m;
    const Footprint before(m);
    Variable& vx = m.var_mut(handle_to_var_id(f.x));
    vx.value = 7.0;
    delta_evaluate(m, {handle_to_var_id(f.x)});
    REQUIRE_THAT(m.node_value(f.row), Catch::Matchers::WithinAbs(7.0, 1e-12));
    const Model::State state = m.copy_state();
    m.restore_state(state);
    full_evaluate(m);
    REQUIRE(Footprint(m) == before);
}

TEST_CASE("the internal objective row still grows a closed model", "[closed]") {
    // add_objective_soft_constraint is the one internal post-close growth path
    // (the first solve on an objective model). It must not trip the refusal the
    // public builders take, and must produce the same row it always did: a
    // Const holding the bound, then Leq(objective, bound), appended last.
    Model m;
    const int32_t x = m.int_var(0, 10, "x");
    const int32_t y = m.int_var(0, 10, "y");
    const int32_t obj = m.sum({x, y});
    m.add_constraint(m.geq(obj, m.constant(3.0)));
    m.minimize(obj);
    m.close();
    const size_t nodes = m.num_nodes();
    const size_t rows = m.constraint_ids().size();

    REQUIRE_NOTHROW(m.add_objective_soft_constraint());
    REQUIRE(m.num_nodes() == nodes + 2);
    REQUIRE(m.constraint_ids().size() == rows + 1);
    REQUIRE(m.objective_bound_node() == static_cast<int32_t>(nodes));
    REQUIRE(m.node(static_cast<int32_t>(nodes)).op == NodeOp::Const);
    const ExprNode& row = m.node(static_cast<int32_t>(nodes + 1));
    REQUIRE(row.op == NodeOp::Leq);
    REQUIRE(std::isinf(m.node(static_cast<int32_t>(nodes)).const_value));
    const ConstSpan<ChildRef> kids = m.children(row);
    REQUIRE(kids.size() == 2);
    REQUIRE((!kids[0].is_var && kids[0].id == obj));
    REQUIRE((!kids[1].is_var && kids[1].id == m.objective_bound_node()));
    REQUIRE(m.constraint_ids().back() == row.id);
    REQUIRE(m.objective_constraint_idx() == static_cast<int32_t>(rows));
    require_valid_topo_order(m);  // the row is placed, not just appended

    const SearchResult r = solve(m, 0.2, 1);
    REQUIRE(r.feasible);
    REQUIRE_THAT(r.objective, Catch::Matchers::WithinAbs(3.0, 1e-9));
}

namespace {

// A closed model with an objective, whose objective row has NOT been added yet:
// the state a ViolationManager or FeasibilityJump sees if it is built before
// `freeze()` or the first `solve()`.
struct ObjectiveFixture {
    Model m;
    int32_t a = 0;

    ObjectiveFixture() {
        a = m.bool_var();
        const int32_t lhs = m.sum({m.prod(m.constant(1.0), a)});
        m.add_constraint(m.geq(lhs, m.constant(2.0)));  // never satisfiable
        m.minimize(lhs);
        m.close();
    }
};

}  // namespace

TEST_CASE("a ViolationManager built before the objective row refuses to read", "[closed]") {
    // Its weights and violation cache are one entry per row as the model stood.
    // The objective row makes the model one row longer, and every read indexes
    // both by constraint index -- `bump_weights` WRITES -- so it throws rather
    // than overreading the heap.
    ObjectiveFixture f;
    ViolationManager vm(f.m);
    REQUIRE_NOTHROW(vm.total_violation());

    f.m.add_objective_soft_constraint();

    std::vector<double> snapshot;
    REQUIRE_THROWS_AS(vm.total_violation(), std::logic_error);
    REQUIRE_THROWS_AS(vm.augmented_objective(), std::logic_error);
    REQUIRE_THROWS_AS(vm.snapshot_violations(snapshot), std::logic_error);
    REQUIRE_THROWS_AS(vm.bump_weights(), std::logic_error);
    REQUIRE_THROWS_AS(vm.weighted_violation_delta(0, 1.0), std::logic_error);

    // A manager built after the row reads normally.
    ViolationManager fresh(f.m);
    REQUIRE(fresh.weights.size() == f.m.constraint_ids().size());
    REQUIRE_NOTHROW(fresh.total_violation());
    REQUIRE_NOTHROW(fresh.bump_weights());
}

TEST_CASE("an FJ built before the objective row refuses to run", "[closed]") {
    // Its per-row tables -- `violated_`, `is_linear_`, `vars_of_constraint_`, the
    // linear scorer's slots -- are sized at construction and indexed by the
    // model's current row count, unchecked. The entry points a driver calls once
    // per batch check the counts instead.
    ObjectiveFixture f;
    ViolationManager vm(f.m);
    RNG rng(17);
    FeasibilityJump fj(f.m, vm, rng);
    fj.begin(true);
    REQUIRE_FALSE(fj.batch(20));

    f.m.add_objective_soft_constraint();

    REQUIRE_THROWS_AS(fj.batch(20), std::logic_error);
    REQUIRE_THROWS_AS(fj.resync(), std::logic_error);
    REQUIRE_THROWS_AS(fj.reset_weights(), std::logic_error);
    REQUIRE_THROWS_AS(fj.begin(false), std::logic_error);
    REQUIRE_THROWS_AS(fj.perturb(0.5), std::logic_error);
    REQUIRE_THROWS_AS(fj.apply_novelty_jump(), std::logic_error);
    REQUIRE_THROWS_AS(fj.run(), std::logic_error);

    // Both rebuilt after the row: runs.
    ViolationManager vm2(f.m);
    FeasibilityJump fj2(f.m, vm2, rng);
    fj2.begin(true);
    REQUIRE_NOTHROW(fj2.batch(20));
}
