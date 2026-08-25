#pragma once

#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "moves.h"
#include "randomize.h"
#include "rng.h"
#include "violation.h"

#include <limits>

namespace cbls {

struct SearchConfig {
    // Keep the assignment the caller handed in, whole: suppresses both the
    // List/Set randomisation and FeasibilityJump's closest-to-zero scalar start.
    // Used by epoch restarts and by callers supplying their own start (including
    // a randomised one — see initialize_random). LNS repair gets the same effect
    // through a different knob: it calls fj_nl_initialize, which sets
    // GFJConfig::set_initial_x = false rather than going through SearchConfig.
    bool skip_init = false;
    // Total GLS iterations (not batches). 0 = unlimited (bounded by time_limit).
    // Checked at batch boundaries, so SearchResult::iterations may exceed this
    // by up to batch_iterations - 1.
    int64_t max_iterations = 0;
    bool use_fj = true;
    int lns_interval = 3;

    // ViolationLS batch outer loop (Algorithm 6).
    int64_t batch_iterations = 1000;  // GLS iterations per batch
    int perturbation_period = 100;    // batches without improvement before perturbing
    // How much of the model a diversification kick moves: each scalar variable
    // is randomised with this probability, and each List/Set variable gets
    // clamp(round(p * |elements|), 1, |elements|) random structural moves
    // (#111) — the floor of one is what keeps a kick on a structural model from
    // being a no-op, and it means every structure moves on every kick whatever p
    // is, p = 0 included. So p governs how much of each structure moves, not
    // which structures move; there is no way to turn the structural half down.
    // Note k counts MOVES, not displaced slots: list_2opt reverses a sub-range
    // (mean ~n/3), so on a positionally-read List (`at`) k = 0.1n rewrites most
    // positions. In the adjacency terms pair_lambda_sum reads, p = 0.1 breaks
    // ~26% of adjacent pairs. If all of that happens to move nothing, one
    // variable is moved anyway, so a kick is never a no-op (#109). The
    // "every structure, every kick" part holds only while the wall-clock budget
    // lasts: a kick that runs into the deadline stops between moves and leaves
    // the remaining structures alone, having moved at least one (#115).
    double perturbation_probability = 0.1;
    // Structural batch (P4): instead of a scalar Feasibility/Novelty Jump batch,
    // sweep the List/Set variables trying typed structural moves (swap / 2-opt /
    // relocate / or-opt / set add-remove-swap) and keep any that reduce weighted
    // violation. FJ only jumps scalar variables, so list/set-structured models
    // need this to improve their structural assignment. <0 picks an automatic
    // default: 0.33 when the model has List/Set variables, 0.0 otherwise.
    double structural_batch_probability = -1.0;
    // Novelty Jump is implemented, wired, and unit-tested, but OFF by default:
    // its per-batch cost is not yet bounded tightly enough for the large
    // continuous benchmarks (it burns the time budget there). Enable + tune
    // (probability, work budget, when-stuck-only) in P5 (#70); the paper uses
    // 0.5 with deterministic-time-bounded batches.
    bool use_compound_moves = false;        // run Novelty Jump batches (else FJ only)
    double novelty_jump_probability = 0.5;  // P(a batch is Novelty Jump)

    // A constraint counts as satisfied when its violation is <= this. Absolute,
    // applied to the constraint node's violation value (for an equality row that
    // is |lhs - rhs|), so on models whose constraint bodies are large in
    // magnitude the effective requirement is tighter than it looks.
    // See kDefaultFeasibilityTolerance for why 1e-6.
    double feasibility_tolerance = kDefaultFeasibilityTolerance;
};

/// Why `solve()`'s outer loop stopped. Exactly one of these ends every run.
///
/// This exists so a test can prove *which* budget bound a run, instead of
/// inferring it from elapsed time. A test that gives a small wall-clock budget
/// and asserts on the work done is silently inert unless it also checks that the
/// clock is what stopped the run — the failure mode that made the previous
/// `fj_nl_initialize` time-limit test pass whether or not the limit was honoured
/// (#104).
///
/// It is not test-only scaffolding: it is the qualifier on `time_seconds`. The
/// CLI reports it in both output formats, so a run that exhausted its budget is
/// distinguishable from one that converged inside it — which is exactly what a
/// reader of the per-instance wall times published under epic #87 needs in order
/// to read them correctly.
enum class TerminationReason {
    /// The wall-clock deadline from `solve()`'s `time_limit` expired.
    TimeLimit,
    /// `SearchConfig::max_iterations` was reached (GLS iterations, or the batch
    /// count when structural/novelty batches stall the iteration counter).
    IterationLimit,
    /// Pure-feasibility model (no objective): the first feasible solution is the
    /// answer, so the search stopped on finding one.
    Feasible,
    /// Neither a wall-clock budget nor an iteration budget was set, so the loop
    /// returned immediately having done no work rather than spinning forever.
    NoBudget,
};

/// Stable snake_case token for a TerminationReason ("time_limit",
/// "iteration_limit", "feasible", "no_budget"). Machine-readable — it is the
/// value the CLI writes to the JSONL `termination` field — and used verbatim in
/// the human output too, so there is exactly one spelling to keep in step with
/// the enum. Returns a static string; never null.
const char* termination_reason_name(TerminationReason reason);

struct SearchResult {
    /// Objective at `best_state`, or `+inf` when there is nothing to report.
    /// `+inf` does NOT imply infeasible: a feasible point on which the objective
    /// overflows to +inf/NaN is recorded as a feasibility witness and returned
    /// with `feasible = true` and this left at `+inf` (issue #100). Test
    /// `feasible` for solvedness and `std::isfinite(objective)` before using the
    /// value; the CLI prints "no finite objective at this assignment" and the
    /// JSONL record emits `"objective": null` for that case.
    double objective = std::numeric_limits<double>::infinity();
    /// Whether `best_state` satisfies every real constraint. A property of the
    /// constraints alone — independent of whether the objective is finite there.
    bool feasible = false;
    Model::State best_state;
    int64_t iterations = 0;
    double time_seconds = 0.0;
    /// Which budget ended the run — the qualifier on `iterations` and
    /// `time_seconds` above. See TerminationReason.
    TerminationReason termination = TerminationReason::NoBudget;
    /// Largest violation over the real constraints at `best_state` — i.e. the
    /// residual of the assignment actually returned (<= the feasibility
    /// tolerance when `feasible`). On an infeasible run `best_state` is the
    /// search's closest approach to the feasible region, so this distinguishes
    /// a numerical near-miss from a run that never got near it, and the caller
    /// can inspect the model to see *which* constraints remain violated.
    double best_violation = std::numeric_limits<double>::infinity();
    /// Whether the Float escape probe was armed when the run ended (#117). A
    /// latch sampled at exit, NOT a count of armings: the probe is armed once the
    /// search is stuck and disarmed on the next improvement, so `true` means "this
    /// run ended stuck" while `false` does *not* mean "never armed" — a run that
    /// armed and then found a new best reports `false`. Exposed so the regression
    /// tests for the two arming conditions can observe them without timing the
    /// call. Single-`solve()` only: `ParallelSearch`'s live aggregation paths
    /// compose the result field by field and drop this (and `best_violation`), so
    /// it reads `false` there. The one exception is `solve_portfolio`'s
    /// no-feasible-solution fallback, which copies a worker's whole `SearchResult`
    /// and so carries either value -- currently unreachable, since `SolutionPool`
    /// always inserts, but it is a struct copy and not a field-by-field compose.
    bool escape_probe_armed = false;
};

struct SolveProgress {
    int64_t iteration = 0;
    double time_seconds = 0.0;
    double objective = std::numeric_limits<double>::infinity();
    /// Weighted violation over **all** rows, the artificial `obj <= bound` row
    /// included — so it is routinely positive on a feasible assignment, since
    /// the bound is tightened below the incumbent objective on every
    /// improvement. Do not read it as "zero whenever feasible".
    ///
    /// It reads ~1e30 while the search sits on a feasible point whose objective
    /// is not finite: the bound there is a finite sentinel (see record_best,
    /// #116) and the row's violation is the engine's blowup clamp. That pairing
    /// of `feasible = true` with an enormous violation is expected, not a bug.
    double total_violation = 0.0;
    bool feasible = false;
    bool new_best = false;
    int perturbations = 0;  // diversification kicks so far (ViolationLS)
};

class SolveCallback {
public:
    virtual ~SolveCallback();
    virtual void on_progress(const SolveProgress& p) = 0;
};

/// Randomise **every** variable, scalars included.
///
/// `solve()` does NOT call this: FeasibilityJump owns the scalar start (see
/// `initialize_structured_random`). It remains available for callers who
/// deliberately want a randomised scalar start — combine it with
/// `SearchConfig::skip_init = true`, which makes `solve()` keep the assignment it
/// is handed instead of re-initialising it.
///
/// Safe on an unbounded domain: the draw goes through `randomize_var`, which
/// samples a finite in-domain window instead of the raw bounds (#112).
void initialize_random(Model& model, RNG& rng);

/// Randomise only the structured (List, Set) variables, leaving every scalar
/// untouched. This is what `solve()` calls: FeasibilityJump's
/// `begin(set_initial_x)` initialises the scalars to the domain value closest to
/// zero (the published Feasibility Jump start), so randomising them here as well
/// would only be overwritten (#108).
void initialize_structured_random(Model& model, RNG& rng);

/// Returns the number of GLS iterations the repair pass actually spent.
///
/// Comparing it against `max_iterations` tells the caller whether the pass
/// converged / exhausted its iteration budget or was cut short by `time_limit`.
/// Without that, "did the clock stop this?" is only answerable by timing the
/// call, which is why the test that was supposed to cover it could not tell the
/// difference (#104).
int64_t fj_nl_initialize(Model& model, ViolationManager& vm, int max_iterations = 10000,
                         RNG* rng = nullptr, double time_limit = 2.0);

/// `time_limit <= 0` disables the wall clock entirely: the run is then bounded by
/// `config.max_iterations` alone and is fully deterministic for a given seed
/// (what the tests rely on). With neither budget set the call returns immediately
/// having done no work, rather than looping forever.
SearchResult solve(Model& model, double time_limit = 10.0, uint64_t seed = 42, bool use_fj = true,
                   InnerSolverHook* hook = nullptr, LNS* lns = nullptr, int lns_interval = 3,
                   SolveCallback* callback = nullptr, const SearchConfig& config = {});

}  // namespace cbls
