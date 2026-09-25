// What `Model::freeze()` buys and what it costs: a frozen model's structure is
// shared by every copy of it, which is how a portfolio replicates a model without
// replicating the DAG (#157) -- and the price is that the structure can no longer
// change.
//
// Sharing is checked by ADDRESS, through `node()`, which hands out a reference
// into the structure. That is the only observable that distinguishes "shared" from
// "copied and identical", and the distinction is the entire point of the change:
// a copy-on-write detach, or a freeze that quietly deep-copied, would leave every
// behavioural test green and the memory saving gone.

#include "cbls/dag_ops.h"
#include "cbls/inner_solver.h"
#include "cbls/lns.h"
#include "cbls/model.h"
#include "cbls/pool.h"
#include "cbls/search.h"
#include "cbls/verify.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace cbls;

namespace {

// x + y <= 10, minimize x + y. Small, but every piece the sharing has to get
// right is present: a variable feeding a constraint row, and an objective the
// soft-constraint row is built from.
Model build_two_var_model(int32_t& x, int32_t& y, int32_t& row) {
    Model m;
    x = m.int_var(0, 20, "x");
    y = m.int_var(0, 20, "y");
    const int32_t total = m.sum({x, y});
    row = m.leq(total, m.constant(10.0));
    m.add_constraint(row);
    m.minimize(total);
    m.close();
    return m;
}

}  // namespace

TEST_CASE("a frozen model shares its structure with every copy", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);
    REQUIRE_FALSE(master.is_frozen());

    master.freeze();
    REQUIRE(master.is_frozen());

    Model a = master;  // the copy is the thing under test
    Model b = master;
    REQUIRE(a.is_frozen());
    REQUIRE(b.is_frozen());

    // One DAG, three models. This is the acceptance criterion: no worker copies
    // the nodes, the edges or the derived indices.
    REQUIRE(&a.node(0) == &master.node(0));
    REQUIRE(&b.node(0) == &master.node(0));
    REQUIRE(a.children(a.node(row)).begin() == master.children(master.node(row)).begin());
    REQUIRE(a.topo_order().data() == master.topo_order().data());
    REQUIRE(a.constraint_ids().data() == master.constraint_ids().data());

    // The per-model side is NOT shared -- that is what makes two workers
    // independent.
    REQUIRE(a.node_values().data() != master.node_values().data());
    REQUIRE(a.variables().data() != master.variables().data());
}

TEST_CASE("an open model is still deep-copied, structure and all", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);

    Model copy = master;
    REQUIRE_FALSE(copy.is_frozen());
    REQUIRE(&copy.node(0) != &master.node(0));

    // And the copy can still be extended without touching the original, which is
    // the reason an open model keeps deep-copying.
    const size_t before = master.num_nodes();
    copy.add_constraint(copy.leq(copy.constant(1.0), copy.constant(2.0)));
    REQUIRE(master.num_nodes() == before);
    REQUIRE(copy.num_nodes() == before + 3);  // two constants and the row
}

TEST_CASE("two replicas sharing a structure cannot observe each other's values", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);
    master.freeze();

    Model a = master;
    Model b = master;
    const int32_t xid = handle_to_var_id(x);
    const int32_t yid = handle_to_var_id(y);

    a.var_mut(xid).value = 3.0;
    a.var_mut(yid).value = 4.0;
    delta_evaluate(a, {xid, yid});

    b.var_mut(xid).value = 9.0;
    b.var_mut(yid).value = 8.0;
    delta_evaluate(b, {xid, yid});

    // Each replica sees its own assignment, its own node values and its own row
    // residual. Nothing leaks either way, and the master is untouched.
    REQUIRE_THAT(a.node_value(row), WithinAbs(3.0 + 4.0 - 10.0, 1e-12));
    REQUIRE_THAT(b.node_value(row), WithinAbs(9.0 + 8.0 - 10.0, 1e-12));
    REQUIRE_THAT(master.node_value(row), WithinAbs(0.0 + 0.0 - 10.0, 1e-12));
    REQUIRE(a.var(xid).value == 3.0);
    REQUIRE(b.var(xid).value == 9.0);
    REQUIRE(master.var(xid).value == 0.0);

    // Re-evaluating one replica from scratch does not disturb the other, which is
    // what a restart does after `restore_state`.
    full_evaluate(a);
    REQUIRE_THAT(b.node_value(row), WithinAbs(9.0 + 8.0 - 10.0, 1e-12));
}

TEST_CASE("each replica keeps its own objective bound", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);
    // freeze() folds the objective into the constraint set, exactly where solve()
    // would have -- that is the freeze point, and the reason it is not close().
    master.freeze();
    REQUIRE(master.has_objective_constraint());
    const int32_t obj_row =
        master.constraint_ids()[static_cast<size_t>(master.objective_constraint_idx())];

    Model a = master;
    Model b = master;
    const int32_t xid = handle_to_var_id(x);

    a.var_mut(xid).value = 2.0;
    full_evaluate(a);
    b.var_mut(xid).value = 2.0;
    full_evaluate(b);

    a.set_objective_bound(5.0);
    b.set_objective_bound(1.0);

    // The bound was the one piece of per-worker mutable state living in shared
    // storage: `set_objective_bound` used to write the RHS Const node's
    // `const_value`. With it still there, these two assertions read each other's
    // bound -- and the second `full_evaluate` below silently resets both.
    REQUIRE(a.objective_bound() == 5.0);
    REQUIRE(b.objective_bound() == 1.0);
    REQUIRE_THAT(a.node_value(obj_row), WithinAbs(2.0 - 5.0, 1e-12));
    REQUIRE_THAT(b.node_value(obj_row), WithinAbs(2.0 - 1.0, 1e-12));

    // A restart re-evaluates the whole DAG. Each replica must come back with ITS
    // bound rather than the +inf the shared Const node carries.
    full_evaluate(a);
    full_evaluate(b);
    REQUIRE_THAT(a.node_value(obj_row), WithinAbs(2.0 - 5.0, 1e-12));
    REQUIRE_THAT(b.node_value(obj_row), WithinAbs(2.0 - 1.0, 1e-12));

    // Releasing it makes the row inert again -- `obj - inf`, satisfied by a wide
    // margin -- which is how solve() opens a re-solve. Only this replica's.
    a.set_objective_bound(std::numeric_limits<double>::infinity());
    full_evaluate(a);
    REQUIRE(a.node_value(obj_row) < 0.0);
    REQUIRE_THAT(b.node_value(obj_row), WithinAbs(2.0 - 1.0, 1e-12));
}

TEST_CASE("a replica copies the master's node values rather than re-deriving them", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);
    const int32_t xid = handle_to_var_id(x);
    master.freeze();  // values consistent here: x = 0, so the row reads 0 - 10

    // Deliberately STALE: the variable moves and nothing re-evaluates, so the
    // master's cached row still says -10 while its assignment says 6. A copy that
    // re-derived would produce -4; only one that copies the array reproduces -10.
    //
    // That asymmetry is the whole point of the case. Evaluating the master first
    // would make every assertion below hold whether the copy constructor
    // re-evaluated or not, which is what an earlier revision of this test did.
    master.var_mut(xid).value = 6.0;

    Model replica = master;
    REQUIRE(replica.var(xid).value == 6.0);
    REQUIRE_THAT(replica.node_value(row), WithinAbs(0.0 - 10.0, 1e-12));
    REQUIRE(replica.node_values() == master.node_values());

    // And it is a copy, not a view: evaluating the replica leaves the master stale.
    full_evaluate(replica);
    REQUIRE_THAT(replica.node_value(row), WithinAbs(6.0 - 10.0, 1e-12));
    REQUIRE_THAT(master.node_value(row), WithinAbs(0.0 - 10.0, 1e-12));
}

TEST_CASE("a frozen model refuses every structural change", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model m = build_two_var_model(x, y, row);
    m.freeze();

    // Variable creation.
    REQUIRE_THROWS_AS(m.bool_var(), std::logic_error);
    REQUIRE_THROWS_AS(m.int_var(0, 1), std::logic_error);
    REQUIRE_THROWS_AS(m.float_var(0.0, 1.0), std::logic_error);
    REQUIRE_THROWS_AS(m.list_var(3), std::logic_error);
    REQUIRE_THROWS_AS(m.set_var(3), std::logic_error);
    // Expression creation, through both node builders.
    REQUIRE_THROWS_AS(m.constant(1.0), std::logic_error);
    REQUIRE_THROWS_AS(m.neg(row), std::logic_error);
    REQUIRE_THROWS_AS(m.sum({row, row}), std::logic_error);
    REQUIRE_THROWS_AS(m.lambda_sum(x, [](int) { return 0.0; }), std::logic_error);
    // Constraints, objective, sequences, sizing and re-closing.
    REQUIRE_THROWS_AS(m.add_constraint(row), std::logic_error);
    REQUIRE_THROWS_AS(m.minimize(row), std::logic_error);
    REQUIRE_THROWS_AS(m.maximize(row), std::logic_error);
    REQUIRE_THROWS_AS(m.add_var_sequence({x, y}), std::logic_error);
    REQUIRE_THROWS_AS(m.reserve(100, 100), std::logic_error);
    REQUIRE_THROWS_AS(m.close(), std::logic_error);

    // Nothing was appended by any of the refusals.
    REQUIRE(m.num_vars() == 2);
    // x+y, the Leq's RHS const, the row, then freeze()'s two for the objective row.
    REQUIRE(m.num_nodes() == 5);

    // And what a search does is still allowed.
    m.var_mut(handle_to_var_id(x)).value = 1.0;
    delta_evaluate(m, {handle_to_var_id(x)});
    m.set_objective_bound(3.0);
    const Model::State state = m.copy_state();
    m.restore_state(state);
    full_evaluate(m);
}

TEST_CASE("add_objective_soft_constraint stays a no-op on a frozen model", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model m = build_two_var_model(x, y, row);
    m.freeze();
    const size_t nodes = m.num_nodes();

    // `solve()` calls this on every model it is handed, so its idempotent return
    // has to come before the frozen check -- otherwise no portfolio worker could
    // start on the replica `freeze()` just prepared.
    m.add_objective_soft_constraint();
    REQUIRE(m.num_nodes() == nodes);
    REQUIRE(m.has_objective_constraint());
}

TEST_CASE("freeze closes an open model and is idempotent", "[share]") {
    Model m;
    const int32_t x = m.int_var(0, 5, "x");
    m.add_constraint(m.leq(x, m.constant(3.0)));
    REQUIRE_FALSE(m.is_closed());

    m.freeze();
    REQUIRE(m.is_closed());
    REQUIRE(m.is_frozen());
    REQUIRE_FALSE(m.topo_order().empty());
    // No objective, so no objective row -- freeze() adds one only where solve()
    // would have.
    REQUIRE_FALSE(m.has_objective_constraint());

    const size_t nodes = m.num_nodes();
    m.freeze();
    REQUIRE(m.num_nodes() == nodes);
}

TEST_CASE("a frozen model solves, and solving it changes nothing structural", "[share]") {
    // A Float variable and an intensify hook on purpose: a frozen 2-var Int model
    // never reaches the continuous paths -- FloatIntensifyHook's Newton steps and
    // `compute_all_partials` -- and those are where a structural write would be
    // least expected and most damaging.
    Model m;
    const int32_t a = m.float_var(0.0, 10.0, "a");
    const int32_t b = m.int_var(0, 10, "b");
    const int32_t total = m.sum({a, b});
    m.add_constraint(m.geq(total, m.constant(3.0)));
    m.minimize(total);
    m.close();
    m.freeze();
    const size_t nodes = m.num_nodes();
    const size_t constraints = m.constraint_ids().size();

    // solve() calls add_objective_soft_constraint() and set_objective_bound() on
    // whatever model it is handed. Both have to be no-ops-or-allowed on a frozen
    // model, or a portfolio over a shared structure could not run at all.
    FloatIntensifyHook hook;
    LNS lns(0.3);
    SearchConfig cfg;
    cfg.max_iterations = 400;
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/7, /*use_fj=*/true, &hook, &lns,
                                 /*lns_interval=*/2, nullptr, cfg);
    REQUIRE(r.feasible);
    REQUIRE(m.num_nodes() == nodes);
    REQUIRE(m.constraint_ids().size() == constraints);
}

// The portfolio entry point, not just the `Model` mechanism above. Without this
// the delivery vehicle for #157 has no coverage at all: deleting `master.freeze()`
// from `ParallelSearch::solve` would leave every other test green while each worker
// silently deep-copied the DAG and the whole memory saving disappeared -- exactly
// the "fails and reports nothing" mode copy-on-write was rejected for.
//
// Sharing is observed through the `hook_factory`, which `ParallelSearch` calls on
// the worker's own thread with the worker's own `Model&` (a PYTHON hook_factory
// gets a copy instead; a C++ one does not). Comparing `&m.node(0)` against the
// master's is the same address test the cases above use.
TEST_CASE("the portfolio hands every worker a replica of one shared structure", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);
    const size_t nodes_before = master.num_nodes();

    std::mutex seen_mutex;
    std::vector<const ExprNode*> seen;
    std::vector<const double*> values_seen;
    auto hook_factory = [&](Model& m) -> std::shared_ptr<InnerSolverHook> {
        const std::scoped_lock guard(seen_mutex);
        seen.push_back(&m.node(0));
        values_seen.push_back(m.node_values().data());
        return std::make_shared<FloatIntensifyHook>();
    };

    // Iteration-budgeted rather than wall-clocked: under `ctest -j$(nproc)` a
    // sub-second deadline can expire during thread creation, which makes
    // solve_portfolio return `feasible == false` with nothing wrong.
    SearchConfig cfg;
    cfg.max_iterations = 500;
    ParallelConfig par;
    par.n_threads = 4;
    ParallelSearch ps(4);
    const SearchResult r =
        ps.solve(master, /*time_limit=*/0.0, /*seed=*/42, cfg, hook_factory, nullptr, nullptr, par);
    REQUIRE(r.feasible);

    // The master was frozen on the calling thread, and the objective row went on
    // once -- not once per worker.
    REQUIRE(master.is_frozen());
    REQUIRE(master.has_objective_constraint());
    REQUIRE(master.num_nodes() == nodes_before + 2);

    const std::scoped_lock guard(seen_mutex);
    REQUIRE(seen.size() == 4);
    for (size_t i = 0; i < seen.size(); ++i) {
        // One DAG for the whole portfolio...
        REQUIRE(seen[i] == &master.node(0));
        // ...and a value array that is not the master's. Compared against the
        // MASTER only: it outlives every worker, where two workers' arrays are
        // only distinct while both are alive, and a worker that returns early
        // frees an allocation the next one may legitimately reuse. Mutual
        // independence is pinned without threads above.
        REQUIRE(values_seen[i] != master.node_values().data());
    }
}

// The three-argument master overload, which the two production callers do not use.
// It delegates to the factory overload, so all this has to pin is the freeze and a
// result -- but without it the overload is untested code.
TEST_CASE("the simple master overload freezes and solves", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model master = build_two_var_model(x, y, row);
    const size_t nodes_before = master.num_nodes();
    ParallelSearch ps(2);
    const SearchResult r = ps.solve(master, /*time_limit=*/1.0, /*seed=*/9);
    REQUIRE(r.feasible);
    // Behavioural rather than a flag read: `close()` in place of `freeze()` adds no
    // objective row, so the node count is what catches that revert.
    REQUIRE(master.num_nodes() == nodes_before + 2);
    REQUIRE(master.has_objective_constraint());
    REQUIRE(master.is_frozen());
}

// Structured variables and the shared lambda tables, which the scalar models above
// never touch -- and which `setcover`'s `Set` encoding is made of. Two things to
// pin: `Lambda` reads a replica's OWN `elements`, and a frozen model's shared
// `std::function` table is still callable from each replica.
TEST_CASE("a frozen Set model's replicas read their own elements through a Lambda", "[share]") {
    Model master;
    const int32_t universe = 4;
    const int32_t s = master.set_var(universe, /*min_size=*/0, /*max_size=*/universe, "s");
    // Sum of (element + 1) over the chosen subset: distinguishes any two subsets,
    // so a replica reading a peer's elements would show up as the wrong value.
    const int32_t weighted = master.lambda_sum(s, [](int element) { return element + 1.0; });
    const int32_t row = master.leq(weighted, master.constant(100.0));
    master.add_constraint(row);
    master.close();
    master.freeze();

    Model a = master;
    Model b = master;
    const int32_t sid = handle_to_var_id(s);
    a.var_mut(sid).elements = {0, 1};  // 1 + 2
    b.var_mut(sid).elements = {2, 3};  // 3 + 4
    full_evaluate(a);
    full_evaluate(b);

    REQUIRE_THAT(a.node_value(weighted), WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(b.node_value(weighted), WithinAbs(7.0, 1e-12));
    // One lambda table, two answers.
    REQUIRE(&a.lambda_func(0) == &master.lambda_func(0));
    REQUIRE(&b.lambda_func(0) == &master.lambda_func(0));
    // And the structural mutator that would grow that shared table still refuses.
    REQUIRE_THROWS_AS(a.lambda_sum(s, [](int) { return 0.0; }), std::logic_error);
}

// Self-assignment and the two cross-state assignments, none of which the cases
// above reach. The copy assignment is `Model tmp(other); *this = std::move(tmp);`,
// so a missing self-check would destroy `*this` before reading it.
TEST_CASE("Model assignment survives self-assignment and either freeze state", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model open_model = build_two_var_model(x, y, row);
    const int32_t xid = handle_to_var_id(x);
    open_model.var_mut(xid).value = 4.0;
    full_evaluate(open_model);

    Model& alias = open_model;
    open_model = alias;  // self-assignment
    REQUIRE(open_model.num_vars() == 2);
    REQUIRE(open_model.var(xid).value == 4.0);
    REQUIRE_FALSE(open_model.is_frozen());

    Model frozen = build_two_var_model(x, y, row);
    frozen.freeze();

    // frozen -> open: the target becomes frozen and shares.
    Model target = build_two_var_model(x, y, row);
    target = frozen;
    REQUIRE(target.is_frozen());
    REQUIRE(&target.node(0) == &frozen.node(0));

    // open -> frozen: the target becomes open again, with its own structure.
    Model target2 = frozen;
    REQUIRE(target2.is_frozen());
    target2 = open_model;
    REQUIRE_FALSE(target2.is_frozen());
    REQUIRE(&target2.node(0) != &open_model.node(0));
    REQUIRE(target2.var(xid).value == 4.0);
}

// `verify_model` re-evaluates every node and compares against the stored value.
// The objective row's RHS is the one `Const` whose stored value is not its
// `const_value`, so a tightened bound is exactly where the two could disagree --
// and at +inf the comparison is `|inf - inf|` = NaN, which is never > tol and so
// checks nothing.
TEST_CASE("verify_model agrees with a tightened objective bound", "[share]") {
    int32_t x = 0;
    int32_t y = 0;
    int32_t row = 0;
    Model m = build_two_var_model(x, y, row);
    m.freeze();
    m.var_mut(handle_to_var_id(x)).value = 2.0;
    // ORDER IS LOAD-BEARING. `full_evaluate` writes exactly `evaluate(node, model)`
    // into the value array, and `check_dag_consistency` compares the two -- so
    // running it AFTER the tightening makes them equal by construction whatever
    // `evaluate`'s `Const` arm reads, including an arm that ignores
    // `objective_bound()` entirely. Evaluating first and tightening second leaves
    // the RHS node stored at 5.0 where an arm reading `const_value` would
    // recompute +inf, so the check has something to catch.
    full_evaluate(m);
    m.set_objective_bound(5.0);

    const VerifyResult result = verify_model(m, 1e-9);
    for (const VerifyError& e : result.errors) {
        REQUIRE(e.kind != VerifyError::Kind::DagConsistency);
    }
}
