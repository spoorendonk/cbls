#include "cbls/lns.h"
#include "cbls/dag_ops.h"
#include "cbls/search.h"
#include <cmath>

namespace cbls {

LNS::LNS(double destroy_fraction) : destroy_fraction_(destroy_fraction) {}

bool LNS::destroy_repair(Model& model, ViolationManager& vm, RNG& rng) {
    double old_F = vm.augmented_objective();
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
        for (const auto& s : seqs)
            for (int32_t v : s.var_ids)
                in_seq[v] = true;

        // Destroy all vars in chosen sequences
        for (int32_t si : seq_indices)
            for (int32_t v : seqs[si].var_ids)
                var_indices.push_back(v);

        // Also destroy a proportional fraction of non-sequence vars
        std::vector<int32_t> non_seq;
        for (int32_t i = 0; i < static_cast<int32_t>(model.num_vars()); ++i)
            if (!in_seq[i])
                non_seq.push_back(i);
        int n_non_seq_destroy = std::max(0,
            static_cast<int>(non_seq.size() * destroy_fraction_));
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

    // Randomize destroyed variables
    for (int32_t idx : var_indices) {
        auto& var = model.var_mut(idx);
        switch (var.type) {
        case VarType::Bool:
            var.value = static_cast<double>(rng.integers(0, 2));
            break;
        case VarType::Int:
            var.value = static_cast<double>(rng.integers(
                static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
            break;
        case VarType::Float:
            var.value = rng.uniform(var.lb, var.ub);
            break;
        case VarType::List:
            rng.shuffle(var.elements);
            break;
        case VarType::Set: {
            int size = static_cast<int>(rng.integers(var.min_size, var.max_size + 1));
            var.elements = rng.choice(var.universe_size, size);
            break;
        }
        }
    }

    full_evaluate(model);

    // Repair via FJ-NL
    fj_nl_initialize(model, vm, 2000, &rng);

    double new_F = vm.augmented_objective();

    if (new_F < old_F) {
        return true;
    } else {
        model.restore_state(saved_state);
        full_evaluate(model);
        return false;
    }
}

int LNS::destroy_repair_cycle(Model& model, ViolationManager& vm,
                               RNG& rng, int n_rounds) {
    int improvements = 0;
    for (int i = 0; i < n_rounds; ++i) {
        if (destroy_repair(model, vm, rng)) {
            improvements++;
        }
    }
    return improvements;
}

}  // namespace cbls
