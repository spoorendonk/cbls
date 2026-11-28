# Plugging in a problem-specific solver: `InnerSolverHook`

cbls's search is generic: it moves single variables to cut weighted
violation. Where a problem has structure a dedicated algorithm exploits
better, an economic dispatch for a fixed commitment, say, or a TSP move
inside a route, an `InnerSolverHook` lets that algorithm run on the
search's points without forking the engine. The engine stays problem-blind.
The hook is C++ code next to the model that builds it.

`FloatIntensifyHook` (`include/cbls/inner_solver.h`, `src/inner_solver.cpp`)
is the generic hook, which the mipfeas, minlplib and uc-chped runners pass by
default; `solve()` itself runs none unless given one. The unit-commitment hook in `benchmarks/uc-chped/uc_hook.h` is the
worked example below. The header is the contract; this page explains it.

## The interface

```cpp
class InnerSolverHook {
public:
    virtual ~InnerSolverHook() = default;
    virtual void solve(Model& model, ViolationManager& vm,
                       const std::vector<int32_t>& last_changed_vars = {},
                       StopRef stop = {}) = 0;
    [[nodiscard]] virtual bool wants_infeasible() const { return false; }
};
```

`solve()` reads the search's current point from `model`, may change any
variable's value, and must return with the model fully evaluated.
`last_changed_vars` holds the last accepted jump's variables, and is empty when
nothing tracked them; the search itself passes `{}`. `stop` is the run's stop
condition (see [Stopping](#stopping)).

## When the search calls it

- **After every batch that ends real-feasible.** Real-feasible means
  `ViolationManager::max_real_violation()` is within
  `SearchConfig::feasibility_tolerance`. The objective-bound row does not
  count, since it is the search's target and not a constraint. The point has
  already been offered to the incumbent by then.
- **With `wants_infeasible()` true, after every other batch too**, with no cap.
  This is opt-in (#454). It is for a hook that can *repair* a point, where the
  search rarely lands feasible on its own. Because nothing caps the calls, such
  a hook budgets itself (the UC hook caps itself by a share of its wall time,
  below).
- **Not once the run is stopping:** a hook is not *started* once the deadline
  has passed, a peer has stopped the portfolio, or the host has cancelled.
  Under `max_iterations`, the batch that uses up the budget can still be
  followed by one call.

`SearchCounters::inner_solver_calls` counts every call, and
`inner_solver_infeasible_calls` counts those after an infeasible batch. A
call that does nothing still costs one resync and counts as a call.

## What the search does with the result

- **It records the result only if it is real-feasible and strictly better than
  this worker's incumbent,** or if it is the worker's first feasible point at
  all, whatever its objective (a feasibility witness). `model.objective_bound()` sits below that
  incumbent: it is the target the search pushes toward, not the acceptance
  test.
- **It never rolls the point back.** It resyncs and continues from whatever the
  hook left, unless a diversification kick due in the same pass replaces the
  point. So a hook that cannot improve or repair the point should leave it as
  it found it. The UC hook restores the point value for value when the model
  refuses its write-back.
- **On a run that ends infeasible,** the closest approach reported comes from
  batch ends before the hook. The hook's own result counts only if a later
  batch ends on it unchanged.

## Stopping

The call is synchronous, so the search cannot pre-empt it. `stop.requested()`
turns true once the deadline passes, a peer stops the portfolio, or the host
cancels the run. A hook whose work can outlast the budget **must** poll `stop`,
return promptly, and leave every value it committed evaluated. Under an
iteration budget, with no wall clock, `stop` reads no clock, so a hook that
reads none either keeps the run bit-reproducible. `stop` is valid only until
`solve()` returns.

## Passing it in

A single search takes a raw pointer:

```cpp
MyHook hook(/* problem data, variable handles */);
SearchResult r = cbls::solve(model, time_limit, seed, &hook, /*lns=*/nullptr,
                             /*lns_interval=*/3, /*callback=*/nullptr, config);
```

`ParallelSearch` takes a factory. It calls the factory once per worker, not
once per restart, and each call must return a distinct object: the search never
locks a hook, so a shared one would race.

```cpp
auto hook_factory = [&](cbls::Model&) -> std::shared_ptr<cbls::InnerSolverHook> {
    return std::make_shared<MyHook>(/* ... */);
};
ps.solve(model, time_limit, seed, config, hook_factory, lns_factory, callback, par);
```

A worker keeps its hook across pool restarts, while its incumbent and bound
start over. Any state the hook carries must stay valid across a restart, so
document what carries over.

Hooks are C++ only: there is no nanobind trampoline for a Python subclass.

## Worked example: the unit-commitment hook

UC-CHPED (`docs/benchmarks/uc-chped/README.md`) decides an on/off commitment
`y[u][t]` and a dispatch `p[u][t]` per unit and period, with valve-point
fuel costs. cbls searches both. For a fixed `y` the dispatch is a small
per-period problem, and for a fixed dispatch rule a commitment local search
is cheap to score. `UCHook` (#445, #452) adds both:

1. **Read** the point's `y` and `p` through variable handles kept from model
   construction (`UCModel::y`, `UCModel::p`). Handles stay valid on every
   worker's copy of the model, so each worker's hook reads the same handles.
2. **Classify** the point with the search's own test,
   `vm.max_real_violation() <= feasibility_tolerance`, not
   `vm.is_feasible()`, which counts the objective-bound row (#454). Hook and
   engine then agree on what feasible means.
3. **Budget infeasible calls.** It sets `wants_infeasible()` from
   `--uc-infeasible`. An infeasible call is skipped once the hook's own time
   exceeds `UCHookConfig::infeasible_share` of the wall time since it was
   constructed, or when its commitment repeats the last call's.
4. **Repair** (`--uc-repair`) a commitment that breaks min up/down or capacity,
   or skip the call.
5. **Search:** a first-improvement descent over each unit's on/off blocks,
   scored by re-dispatching only the periods a move touches, then an iterated
   local search (`--uc-ils`, a slice of the row's budget per call) that polls
   `stop`.
6. **Write back** the new `(y, p)` only if it beats the point's cost. If the
   point was infeasible, it must beat the best cost the hook has seen instead.
   The hook re-evaluates the model and keeps the write-back only if the model
   holds it real-feasible and cheaper. Otherwise it restores the point read,
   value for value.

Each part of the contract shows up in that sequence:
- **Self-budgeting:** step 3.
- **Polling `stop`:** steps 4-6. A stopped call is not remembered as searched,
  so the next one may retry it.
- **Leaving the point:** step 6's rollback.
- **State across restarts:** `UCHook`'s header says what it keeps and why.
  `best_seen_` stays at least the worker's incumbent, so after a restart it is
  only a stricter bar.
- **Reproducibility:** the hook's throttle and ILS read the wall clock, so the
  runner refuses `--uc-hook` under `--no-time-limit`.

### Running it, and checking what it writes

The hook is off unless `--uc-hook ed|full` is passed. `ed` only re-dispatches;
`full` adds the commitment search. The published protocol never runs it, and
the runner refuses `--uc-hook` when `--out` is the published table. Every knob
is recorded in the row's `search_config`.

```sh
build/cbls_uc_chped benchmarks/instances/uc-chped --instance ucp13 --periods 3 \
  --demand-equality --verify --out /tmp/o.csv --solution-out /tmp/s.jsonl \
  --uc-hook full --uc-infeasible --uc-repair --uc-ils 0.05
.venv/bin/python benchmarks/uc-chped/check_solutions.py \
  benchmarks/instances/uc-chped /tmp/s.jsonl
```

`--solution-out` writes one JSON line of `(y, p)` per measured row that has an
objective, and none for a row that ends infeasible or that the verifier vetoed.
`check_solutions.py`, written from Pedroso's formulation and not from cbls's
model, recomputes feasibility and cost from the instance file alone. That
check is the hook's evidence chain: a cost the checker cannot reproduce is not
a result. `tests/python/test_uc_chped_hook.py` runs the pair.

## Writing a new hook: checklist

- Classify points with `vm.max_real_violation()` at the search's tolerance.
- Return with the model fully evaluated (`full_evaluate(model)`, then
  `vm.invalidate_cache()`, after bulk writes).
- Leave the point unchanged when you cannot improve it.
- Poll `stop` in anything that can outlast the budget.
- With `wants_infeasible()`, budget yourself.
- Make one object per worker, and document the state that crosses a pool
  restart.
- Read no clock, or document that the run is no longer bit-reproducible.
