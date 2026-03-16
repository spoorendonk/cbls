#pragma once

#include "model.h"
#include "violation.h"
#include <vector>
#include <utility>

namespace cbls {

class InnerSolverHook {
public:
    virtual ~InnerSolverHook() = default;

    struct Result {
        std::vector<std::pair<int32_t, double>> assignments;  // var_id -> new value
    };

    // Called with mutable model + violation manager.
    // Hook may read var values, compute partials, etc.
    // Returns assignments for variables it optimizes.
    virtual Result solve(Model& model, ViolationManager& vm) = 0;
};

// Generic Float intensification: coordinate-descent sweeps over all Float vars
// using Newton steps on violated constraints + gradient steps on objective.
class FloatIntensifyHook : public InnerSolverHook {
public:
    int max_sweeps = 3;
    double step_size = 0.1;

    Result solve(Model& model, ViolationManager& vm) override;
};

}  // namespace cbls
