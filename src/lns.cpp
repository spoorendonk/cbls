#include "cbls/lns.h"

#include "cbls/dag_ops.h"
#include "cbls/randomize.h"
#include "cbls/search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cbls {

LNS::LNS(double destroy_fraction) : destroy_fraction_(destroy_fraction) {}

// Out of line so the vtable has a single translation unit to live in.
LNS::~LNS() = default;

// Lexicographic search-state key (real violation, objective): feasibility first,
// then objective — mirroring solve()'s record_best. Real violation excludes the
// artificial `obj <= bound` soft constraint (when present) so the objective is
// not double-counted, the way the dead augmented_objective() accept rule did.
static std::pair<double, double> state_key(const Model& model) {
    const int32_t obj_ci = model.has_objective_constraint() ? model.objective_constraint_idx() : -1;
    const auto& cids = model.constraint_ids();
    double real_violation = 0.0;
    for (size_t i = 0; i < cids.size(); ++i) {
        if (static_cast<int32_t>(i) == obj_ci) {
            continue;
        }
        const double v = model.node(cids[i]).value;
        // NaN must be handled before max(): std::max(0.0, NaN) is 0.0, which
        // would score a NaN-poisoned repair as perfectly feasible and accept it
        // over a genuinely better incumbent. Same guard as
        // clamped_node_violation and search.cpp's max_real_violation.
        if (std::isnan(v)) {
            real_violation = std::numeric_limits<double>::infinity();
            break;
        }
        real_violation += std::max(0.0, v);
    }
    double obj = model.objective_id() >= 0 ? model.node(model.objective_id()).value : 0.0;
    return {real_violation, obj};
}

bool LNS::destroy_repair(Model& model, ViolationManager& vm, RNG& rng, double repair_time_limit) {
    auto old_key = state_key(model);
    auto saved_state = model.copy_state();

    // Collect variables to destroy
    std::vector<int32_t> var_indices;

    const auto& seqs = model.var_sequences();
    if (!seqs.empty()) {
        // Sequence-aware destroy: pick k random sequences
        int n_seqs = static_cast<int>(seqs.size());
        int k = std::max(1, static_cast<int>(std::ceil(n_seqs * destroy_fraction_)));
        std::vector<int32_t> seq_indices(n_seqs);
        std::iota(seq_indices.begin(), seq_indices.end(), 0);
        rng.shuffle(seq_indices);
        seq_indices.resize(k);

        // Mark which vars are in sequences
        std::vector<bool> in_seq(model.num_vars(), false);
        for (const auto& s : seqs) {
            for (int32_t v : s.var_ids) {
                in_seq[v] = true;
            }
        }

        // Destroy all vars in chosen sequences
        for (int32_t si : seq_indices) {
            for (int32_t v : seqs[si].var_ids) {
                var_indices.push_back(v);
            }
        }

        // Also destroy a proportional fraction of non-sequence vars
        std::vector<int32_t> non_seq;
        for (int32_t i = 0; i < static_cast<int32_t>(model.num_vars()); ++i) {
            if (!in_seq[i]) {
                non_seq.push_back(i);
            }
        }
        int n_non_seq_destroy = std::max(0, static_cast<int>(non_seq.size() * destroy_fraction_));
        rng.shuffle(non_seq);
        non_seq.resize(n_non_seq_destroy);
        var_indices.insert(var_indices.end(), non_seq.begin(), non_seq.end());
    } else {
        // Uniform random destroy (no sequences registered)
        int n_destroy = std::max(1, static_cast<int>(model.num_vars() * destroy_fraction_));
        var_indices.resize(model.num_vars());
        std::iota(var_indices.begin(), var_indices.end(), 0);
        rng.shuffle(var_indices);
        var_indices.resize(n_destroy);
    }

    // Randomize destroyed variables. ListOrder::Perturb because destroy/repair
    // works on an incumbent: a List keeps its elements and gets a new order,
    // which is what this loop did before the three copies were merged.
    for (int32_t idx : var_indices) {
        randomize_var(model.var_mut(idx), rng, ListOrder::Perturb);
    }

    full_evaluate(model);

    // Repair via FJ-NL
    fj_nl_initialize(model, vm, 2000, &rng, repair_time_limit);

    auto new_key = state_key(model);

    if (new_key < old_key) {
        return true;
    } else {
        model.restore_state(saved_state);
        full_evaluate(model);
        return false;
    }
}

int LNS::destroy_repair_cycle(Model& model, ViolationManager& vm, RNG& rng, int n_rounds,
                              double repair_time_limit) {
    int improvements = 0;
    for (int i = 0; i < n_rounds; ++i) {
        if (destroy_repair(model, vm, rng, repair_time_limit)) {
            improvements++;
        }
    }
    return improvements;
}

}  // namespace cbls
