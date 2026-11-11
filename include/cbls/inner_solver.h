#pragma once

#include "model.h"
#include "stop.h"
#include "violation.h"

#include <cstdint>
#include <vector>

namespace cbls {

class InnerSolverHook {
public:
    virtual ~InnerSolverHook() = default;

    // Called on each new feasible solution; mutates the model directly (var
    // values + delta_evaluate). last_changed_vars: var IDs of the last accepted
    // jump, empty when untracked (after a perturbation or LNS repair).
    //
    // stop (#191): requested once the deadline passes, a peer stops the
    // portfolio, or the host cancels. The search cannot preempt this call, so
    // a hook that can outlast the budget must poll it and return promptly,
    // leaving every committed value delta-evaluated. With no wall clock it
    // reads no clock, so an iteration-budgeted run stays bit-reproducible; the
    // default (direct calls) is never requested. Valid only until solve()
    // returns: the hook's own threads may poll it during the call, not after.
    //
    // A parameter, not a second virtual, so a subclass overriding only one
    // cannot be silently bypassed. An old-signature override fails to compile
    // if marked `override` or if it left a direct subclass abstract; UNMARKED
    // in a FloatIntensifyHook subclass it compiles and is hidden, flagged only
    // by -Woverloaded-virtual (GCC 15 -Wall, clang-tidy), not use-override.
    virtual void solve(Model& model, ViolationManager& vm,
                       const std::vector<int32_t>& last_changed_vars = {}, StopRef stop = {}) = 0;
};

// Generic Float intensification: coordinate-descent sweeps over all Float vars
// using Newton steps on violated constraints + gradient steps on objective.
class FloatIntensifyHook : public InnerSolverHook {
public:
    int max_sweeps = 3;
    double initial_step_size = 0.1;
    int max_line_search_steps = 5;
    int max_multi_var_constraints = 5;

    // Polls `stop` before each sweep's first Float variable, then every
    // vars_per_poll(model) Float variables, and before every multi-variable
    // Newton step; returns at the first raised poll. Each unit between polls
    // commits or rolls back whole, so an early return leaves a consistent
    // model. A never-raised `stop` leaves the descent unchanged.
    void solve(Model& model, ViolationManager& vm,
               const std::vector<int32_t>& last_changed_vars = {}, StopRef stop = {}) override;

    // Float variables between two polls: 65536 / max(rows, nodes), clamped to
    // [1, 256] (argument in src/inner_solver.cpp). Public so tests can predict
    // the poll sequence.
    static int vars_per_poll(const Model& model);
};

}  // namespace cbls
