#pragma once

#include "model.h"
#include "violation.h"

namespace cbls {

class InnerSolverHook {
public:
    virtual ~InnerSolverHook() = default;

    // Called with mutable model + violation manager.
    // Hook mutates model directly (var values + delta_evaluate).
    virtual void solve(Model& model, ViolationManager& vm) = 0;
};

// Generic Float intensification: coordinate-descent sweeps over all Float vars
// using Newton steps on violated constraints + gradient steps on objective.
class FloatIntensifyHook : public InnerSolverHook {
public:
    int max_sweeps = 3;
    double initial_step_size = 0.1;
    int max_line_search_steps = 5;
    int max_multi_var_constraints = 5;

    void solve(Model& model, ViolationManager& vm) override;
};

}  // namespace cbls
