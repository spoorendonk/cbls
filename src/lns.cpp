#include "cbls/lns.h"
#include "cbls/dag_ops.h"
#include "cbls/search.h"

namespace cbls {

LNS::LNS(double destroy_fraction) : destroy_fraction_(destroy_fraction) {}

bool LNS::destroy_repair(Model& model, ViolationManager& vm, RNG& rng) {
    double old_F = vm.augmented_objective();
    auto saved_state = model.copy_state();

    // Select variables to destroy
    int n_destroy = std::max(1, static_cast<int>(model.num_vars() * destroy_fraction_));
    std::vector<int32_t> var_indices(model.num_vars());
    std::iota(var_indices.begin(), var_indices.end(), 0);
    rng.shuffle(var_indices);
    var_indices.resize(n_destroy);

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
