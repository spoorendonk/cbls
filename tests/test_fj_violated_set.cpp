// FeasibilityJump's incremental violated-row state (#174): the dense list of
// violated rows and the per-variable count of active violated rows are kept up
// to date at every site that moves a row in or out of V, instead of being
// recomputed by a sweep. These tests pin that state against a from-scratch
// recomputation from the model's node values and the live GLS weights, through
// every kind of change: committed moves, weight bumps (including a decay that
// deactivates a violated row), weights masked from outside between batches,
// Novelty apply/undo, retirement, and Model::extend / on_extended.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/model_extension.h>
#include <cstdint>
#include <random>
#include <vector>

using namespace cbls;

namespace {

// is_violated's threshold in src/feasibility_jump.cpp; NaN counts as violated.
constexpr double kTol = 1e-9;

bool fresh_violated(const Model& m, size_t ci) {
    return !(m.node_value(m.constraint_ids()[ci]) <= kTol);
}

// The from-scratch recomputation.
void require_consistent(const FeasibilityJump& fj, const Model& m, const ViolationManager& vm) {
    const size_t nc = m.constraint_ids().size();
    std::vector<int32_t> expected_rows;
    for (size_t c = 0; c < nc; ++c) {
        const bool v = fresh_violated(m, c);
        CAPTURE(c);
        REQUIRE(fj.row_violated(static_cast<int32_t>(c)) == v);
        if (v) {
            expected_rows.push_back(static_cast<int32_t>(c));
        }
    }
    REQUIRE(fj.violated_rows() == expected_rows);
    for (size_t v = 0; v < m.num_vars(); ++v) {
        int32_t expected = 0;
        for (const int32_t c : m.constraints_of_var(static_cast<int32_t>(v))) {
            if (fresh_violated(m, static_cast<size_t>(c)) &&
                vm.weights[static_cast<size_t>(c)] > 0.0) {
                ++expected;
            }
        }
        CAPTURE(v);
        REQUIRE(fj.active_violated_rows_of(static_cast<int32_t>(v)) == expected);
    }
}

// A random sparse integer model with more rows than it can satisfy at once, so
// V keeps changing under the search rather than emptying on the first batch.
// Returns the Sum node of every row, for extensions to grow.
std::vector<int32_t> build_random_rows(Model& m, uint32_t seed, int num_vars, int num_rows) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> pick_var(0, num_vars - 1);
    std::uniform_int_distribution<int> pick_coef(1, 3);
    std::uniform_int_distribution<int> pick_sign(0, 1);
    std::uniform_int_distribution<int> pick_rhs(-2, 6);
    std::vector<int32_t> handles;
    handles.reserve(static_cast<size_t>(num_vars));
    for (int i = 0; i < num_vars; ++i) {
        handles.push_back(m.int_var(0, 4));
    }
    std::vector<int32_t> sums;
    for (int r = 0; r < num_rows; ++r) {
        std::vector<int32_t> terms;
        std::vector<int> used;
        while (used.size() < 4) {
            const int v = pick_var(gen);
            if (std::find(used.begin(), used.end(), v) != used.end()) {
                continue;
            }
            used.push_back(v);
            const double coef = (pick_sign(gen) != 0 ? 1.0 : -1.0) * pick_coef(gen);
            terms.push_back(m.prod(m.constant(coef), handles[static_cast<size_t>(v)]));
        }
        const int32_t sum = m.sum(terms);
        sums.push_back(sum);
        const double rhs = pick_rhs(gen);
        m.add_constraint(r % 2 == 0 ? m.leq(sum, m.constant(rhs)) : m.geq(sum, m.constant(rhs)));
    }
    return sums;
}

GFJConfig small_batch_config() {
    GFJConfig cfg;
    cfg.two_phase = false;
    cfg.time_limit = 0.0;
    cfg.unproductive_iterations = 0;
    return cfg;
}

}  // namespace

TEST_CASE("FJ's violated set matches a recompute through moves and weight bumps",
          "[fj][violated_set]") {
    // Short batches, checked after each: every batch mixes committed moves
    // (update_var) and GLS bumps, and rho alternates between the two values
    // solve() samples.
    Model m;
    build_random_rows(m, 3, 30, 60);
    m.close();
    ViolationManager vm(m);
    RNG rng(5);
    FeasibilityJump fj(m, vm, rng, small_batch_config());
    fj.begin(true);
    require_consistent(fj, m, vm);
    bool saw_violated = false;
    for (int b = 0; b < 200; ++b) {
        fj.set_rho(b % 2 == 0 ? 0.95 : 1.0);
        (void)fj.batch(7);
        require_consistent(fj, m, vm);
        saw_violated = saw_violated || !fj.violated_rows().empty();
    }
    REQUIRE(saw_violated);  // not vacuous: V was non-empty at some check
}

TEST_CASE("FJ's violated set follows a decay that deactivates a violated row",
          "[fj][violated_set]") {
    // The GLS update decays FIRST and bumps only a row whose DECAYED weight is
    // still > 0, so a violated row decayed to 0 is left at 0: still in V, but no
    // longer active. The bump must therefore re-read the weight of every row in V
    // rather than trust the bit it set when the row entered. rho = 0 makes that
    // happen to every violated row at the first bump of the batch,
    // deterministically. Under FJ's lazy decay (#175) it is the only way: a
    // positive rho never takes a positive weight to 0 (LazyWeightDecay floors it).
    Model m;
    build_random_rows(m, 17, 30, 60);
    m.close();
    ViolationManager vm(m);
    RNG rng(9);
    FeasibilityJump fj(m, vm, rng, small_batch_config());
    fj.begin(true);
    int deactivated = 0;
    for (int round = 0; round < 30; ++round) {
        fj.set_rho(0.95);
        (void)fj.batch(20);
        require_consistent(fj, m, vm);
        fj.set_rho(0.0);
        (void)fj.batch(3);
        require_consistent(fj, m, vm);
        for (const int32_t c : fj.violated_rows()) {
            if (vm.weights[static_cast<size_t>(c)] == 0.0) {
                ++deactivated;
            }
        }
        fj.reset_weights();
        require_consistent(fj, m, vm);
    }
    REQUIRE(deactivated > 0);  // the case this test exists for did occur
}

TEST_CASE("FJ's violated set picks up weights masked and unmasked between batches",
          "[fj][violated_set]") {
    // run()'s two-phase mask sets weights to 0 from outside FJ's bookkeeping,
    // and a caller can do the same between batches. The counts must follow the
    // live weights by the time the next batch has run, however few moves it made.
    Model m;
    build_random_rows(m, 23, 30, 60);
    m.close();
    ViolationManager vm(m);
    RNG rng(13);
    FeasibilityJump fj(m, vm, rng, small_batch_config());
    fj.begin(true);
    RNG pick(99);  // separate from the search's stream
    const auto nc = static_cast<int64_t>(vm.weights.size());
    bool masked_a_violated_row = false;
    for (int b = 0; b < 150; ++b) {
        for (int k = 0; k < 6; ++k) {
            const auto c = static_cast<size_t>(pick.integers(0, nc));
            masked_a_violated_row =
                masked_a_violated_row ||
                (fj.row_violated(static_cast<int32_t>(c)) && vm.weights[c] > 0.0);
            vm.weights[c] = vm.weights[c] > 0.0 ? 0.0 : 1.0;  // flip the mask
        }
        vm.invalidate_cache();
        (void)fj.batch(1);
        require_consistent(fj, m, vm);
    }
    REQUIRE(masked_a_violated_row);
}

TEST_CASE("FJ's violated set survives Novelty apply and undo", "[fj][violated_set]") {
    // apply_novelty_jump moves variables and reverts most of them while it
    // searches; each of those writes V through the same path as update_var. It
    // leaves V describing the assignment it ends on, which is checked BEFORE the
    // resync the caller then owes -- a resync would rebuild it and hide a lapse.
    Model m;
    build_random_rows(m, 31, 30, 60);
    m.close();
    ViolationManager vm(m);
    RNG rng(21);
    FeasibilityJump fj(m, vm, rng, small_batch_config());
    fj.begin(true);
    int novelty_calls_with_violation = 0;
    for (int b = 0; b < 60; ++b) {
        (void)fj.batch(20);
        const bool had_violation = !fj.violated_rows().empty();
        const std::vector<double> before = m.copy_state().values;
        (void)fj.apply_novelty_jump();
        require_consistent(fj, m, vm);
        if (had_violation && m.copy_state().values != before) {
            ++novelty_calls_with_violation;
        }
        fj.resync();
        require_consistent(fj, m, vm);
    }
    REQUIRE(novelty_calls_with_violation > 0);  // Novelty did commit moves
}

TEST_CASE("FJ's violated set follows Model::extend through on_extended", "[fj][violated_set]") {
    // New columns entering existing rows -- some of them violated and active, so
    // their variable lists grow while they are counted -- and new rows over old
    // and new variables. Checked straight after on_extended and again after
    // further batches on the grown model.
    Model m;
    const std::vector<int32_t> sums = build_random_rows(m, 57, 20, 40);
    m.close();
    ViolationManager vm(m);
    RNG rng(8);
    FeasibilityJump fj(m, vm, rng, small_batch_config());
    fj.begin(true);
    (void)fj.batch(25);
    require_consistent(fj, m, vm);

    for (int round = 0; round < 3; ++round) {
        // Grow a counted row if there is one: that is the case whose count
        // depends on the list being re-walked after the merge.
        std::vector<int32_t> targets;
        for (const int32_t c : fj.violated_rows()) {
            if (vm.weights[static_cast<size_t>(c)] > 0.0) {
                targets.push_back(c);
            }
        }
        REQUIRE_FALSE(targets.empty());
        ModelExtension ext(m);
        const int32_t col = ext.int_var(0, 4);
        ext.set_initial(col, 0.0);  // leave the grown rows' values where they were
        for (size_t k = 0; k < targets.size() && k < 3; ++k) {
            ext.append_to_sum(sums[static_cast<size_t>(targets[k])],
                              ext.prod(ext.constant(1.0), col));
        }
        // A new row over an old variable and the new column, violated at once
        // (both are at most 4). Handle -1 is variable 0.
        const int32_t old_var = -1;
        ext.add_constraint(ext.geq(
            ext.sum({ext.prod(ext.constant(1.0), old_var), ext.prod(ext.constant(1.0), col)}),
            ext.constant(20.0)));
        const ExtensionResult res = m.extend(ext);
        vm.on_extended(res);
        fj.on_extended(res);
        require_consistent(fj, m, vm);
        for (int b = 0; b < 10; ++b) {
            (void)fj.batch(5);
            require_consistent(fj, m, vm);
        }
    }
}

TEST_CASE("FJ's violated set recounts an incidence row missing from touched_constraints",
          "[fj][violated_set]") {
    // ExtensionResult promises every row in new_incidences is also in
    // touched_constraints or new, and Model::extend keeps that promise. But
    // on_extended's validation does not enforce it -- ExtensionResult is a plain
    // struct -- so the counts are kept right without it: a counted row whose
    // variable list the merge grows is uncounted against the old list and
    // recounted against the merged one whether or not it is in `rows`. This
    // hands on_extended a real extension with that row dropped from
    // touched_constraints. The new column starts at 0, so the row's value, and
    // hence its in-V bit, is unchanged -- only the recount is at stake.
    Model m;
    const std::vector<int32_t> sums = build_random_rows(m, 57, 20, 40);
    m.close();
    ViolationManager vm(m);
    RNG rng(8);
    FeasibilityJump fj(m, vm, rng, small_batch_config());
    fj.begin(true);
    (void)fj.batch(25);

    int32_t target = -1;
    for (const int32_t c : fj.violated_rows()) {
        if (vm.weights[static_cast<size_t>(c)] > 0.0) {
            target = c;
            break;
        }
    }
    REQUIRE(target >= 0);  // a counted row to grow

    ModelExtension ext(m);
    const int32_t col = ext.int_var(0, 4);
    ext.set_initial(col, 0.0);
    ext.append_to_sum(sums[static_cast<size_t>(target)], ext.prod(ext.constant(1.0), col));
    ExtensionResult res = m.extend(ext);
    const auto it =
        std::find(res.touched_constraints.begin(), res.touched_constraints.end(), target);
    REQUIRE(it != res.touched_constraints.end());
    res.touched_constraints.erase(it);
    const int32_t new_var = res.first_new_var;
    REQUIRE(std::any_of(res.new_incidences.begin(), res.new_incidences.end(),
                        [&](const std::pair<int32_t, int32_t>& inc) {
                            return inc.first == target && inc.second == new_var;
                        }));

    vm.on_extended(res);
    fj.on_extended(res);
    require_consistent(fj, m, vm);
    REQUIRE(fj.active_violated_rows_of(new_var) > 0);  // it sits in the counted row
}
