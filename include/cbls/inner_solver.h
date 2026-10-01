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

    // Called with mutable model + violation manager on each new feasible
    // solution. Hook mutates model directly (var values + delta_evaluate).
    // last_changed_vars: var IDs changed in the last accepted jump (empty when
    // the caller does not track them, e.g. after a perturbation or LNS repair).
    //
    // stop (#191): the search passes its own stop condition -- requested once
    // the wall-clock deadline has passed, a peer worker has stopped the
    // portfolio, or the host has cancelled. The search cannot preempt a
    // synchronous call, so a hook whose work can outlast the remaining budget
    // must poll it and return as soon as it can once it is requested, leaving
    // the model consistent (every committed value delta-evaluated). Before it
    // existed FloatIntensifyHook ran a 20s MIPfeas solve ~10s over budget.
    // On a run with no wall clock it reads no clock, so an iteration-budgeted
    // run stays bit-reproducible. The default is never requested, which is what
    // a direct call outside a search gets. `stop` refers to the caller's state and
    // is valid only until solve() returns; it may be polled from the hook's own
    // threads during the call, but must not be kept beyond it.
    //
    // Added as a parameter of the one virtual rather than as a second virtual
    // beside it, deliberately: with two, a subclass of FloatIntensifyHook that
    // overrode only the stop-less one was silently bypassed by the search. Here
    // an old-signature override fails to compile only if it is marked
    // `override`, or if it was a direct subclass's only implementation of this
    // pure virtual (the class stays abstract). An UNMARKED old-signature
    // override in a FloatIntensifyHook subclass still compiles and is silently
    // hidden. -Woverloaded-virtual flags it (GCC 15's -Wall and clang-tidy's
    // clang-diagnostic-overloaded-virtual, both checked); modernize-use-override
    // does not, since the function overrides nothing.
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

    // Polls `stop` before the first Float variable of each sweep and then every
    // vars_per_poll(model) Float variables, and before every multi-variable
    // Newton step; returns at the first raised poll. Each unit between polls
    // commits or rolls back completely, so an early return leaves a consistent
    // model. With `stop` never raised the descent is exactly what it was before
    // the polls existed.
    void solve(Model& model, ViolationManager& vm,
               const std::vector<int32_t>& last_changed_vars = {}, StopRef stop = {}) override;

    // Float variables descended between two polls of `stop` on this model:
    // 65536 / max(rows, nodes) (src/inner_solver.cpp carries the argument), between 1
    // and 256. Public so a test can predict the poll sequence exactly.
    static int vars_per_poll(const Model& model);
};

}  // namespace cbls
