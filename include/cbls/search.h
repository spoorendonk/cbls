#pragma once

#include "counters.h"
#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "move_generator.h"
#include "moves.h"
#include "randomize.h"
#include "rng.h"
#include "stop.h"
#include "violation.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace cbls {

/// Defined in `solution_pool.h`, which this header deliberately does NOT
/// include: `solve()` only needs the pointer type, and the full definition
/// would push `<atomic>`, `<mutex>`, `<optional>` and `SolutionPool` itself into
/// every translation unit that includes `search.h`. A caller that actually
/// builds one includes `cbls/pool.h` (or `cbls/solution_pool.h`), as
/// `ParallelSearch` does.
struct SearchCoordination;

/// Defined in `tracer.h`. Forward-declared for the same reason
/// `SearchCoordination` is: `SearchConfig` holds only a pointer, and a caller
/// that actually builds one includes `cbls/tracer.h` (or `cbls/cbls.h`, or
/// `cbls/pool.h`, which needs the definition for `tracer_factory`).
class Tracer;

struct SearchConfig {
    // Keep the assignment the caller handed in, whole: suppresses both the
    // List/Set randomisation and FeasibilityJump's closest-to-zero scalar start.
    // Used by portfolio restarts and by callers supplying their own start (including
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
    // default: 0.33 when the batch would build at least one generator -- an
    // entry in `move_generators`, or a List/Set variable with
    // `default_structural_generators` on -- and 0.0 otherwise. Keying it on
    // List/Set presence alone would arm a batch that builds nothing.
    double structural_batch_probability = -1.0;

    // ---- structural batch: what proposes moves, and how one is chosen (#165) --
    //
    // Extra move generators, on top of the built-in per-variable List/Set ones.
    // A domain move (an inter-route exchange, a block move, an ejection chain),
    // a move over several variables, a restricted one: all of them plug in here
    // rather than being special-cased inside the engine.
    //
    // CLONED PER WORKER. Every portfolio worker builds its own StructuralBatch,
    // which clones each generator, so a generator holding a cache or a cursor
    // holds per-worker state. The instances registered here are never mutated by
    // the search -- which is what the `const` in the pointer type says -- so
    // several searches may share one registration safely.
    std::vector<std::shared_ptr<const MoveGenerator>> move_generators;
    // Whether the built-in List/Set generators are registered too. On by
    // default: a custom generator is a peer of the built-ins, not a replacement
    // for them. Turn it off to run a model purely on custom moves.
    bool default_structural_generators = true;
    // How the batch turns a generator's candidates into a commit. The default
    // is bit-for-bit the pre-#165 rule; see StructuralSelection.
    StructuralSelection structural_selection = StructuralSelection::FirstImprovingSample;
    // Candidates scored per generator under BestOfSample / ViolationGuided.
    // Inert under the default policy, which takes exactly one generate() call.
    //
    // 8 is a NEUTRAL PLACEHOLDER, not a measured choice, and no result in this
    // repo is derived from it: the two policies it feeds are opt-in and off by
    // default. A caller that cares should sweep it on its own instances:
    // benchmarks/setcover/ab_selection.sh A/Bs the POLICIES at this default and
    // has no --sample-size flag of its own, so sweeping the size itself means
    // driving cbls_setcover --sample-size directly.
    int structural_sample_size = 8;
    // Optional granular neighbourhood handed to the BUILT-IN generators, making
    // a move's target one of the moved element's nearest neighbours instead of a
    // uniform draw (Toth & Vigo 2003). Null -- the default -- keeps the uniform
    // draw and therefore the pre-#165 trajectory. Shared, not cloned: it is
    // immutable, and copying a k-nearest list per worker is the cost sharing the
    // model's structure exists to avoid (#157).
    std::shared_ptr<const NeighbourList> structural_neighbours;
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

    // End a Feasibility-Jump batch that has run this many GLS iterations without
    // reducing the unweighted violation of the REAL rows, and take the
    // diversification kick as due when that happens rather than waiting out
    // perturbation_period (#102). Before the first feasible solution no batch
    // ever "improves", so without this the kick cadence is a fixed
    // perturbation_period * batch_iterations iterations with no feedback from
    // the search at all -- 100 000 by default, which MINLPLib st_e40 spends 92%
    // of inside a limit cycle it has no way out of. <= 0 restores the old fixed
    // cadence.
    //
    // The kick this buys is only the kick: it does not arm the Float escape
    // probe and does not reset the stagnation counter, so `perturbation_period`
    // still means what it says and the probe stays a last resort (#107). See the
    // kick site in solve(). Forwarded to GFJConfig::unproductive_iterations,
    // whose comment says what the default 300 is and is not.
    //
    // TWO THINGS BOUND IT, and neither is visible from this field alone:
    //
    //   1. `stagnation >= perturbation_period / 20` -- a second witness from the
    //      outer loop. The exit's own measure is not trusted on its own, because
    //      it sums the REAL rows and cannot see the artificial `obj <= bound`
    //      row; the outer loop's own count of non-improving batches is what
    //      confirms the search has actually stopped getting anywhere.
    //   2. Once a feasible solution exists the kick draws only its cheap half.
    //      A diversification kick is either a perturb (microseconds) or an LNS
    //      destroy-repair bounded by min(2.0, remaining()) SECONDS. After the
    //      first feasible solution the measure goes blind -- the real rows sit at
    //      a positive equilibrium traded against the objective row, so "no new
    //      all-time low" is the normal state of a batch that is working
    //      perfectly -- and an LNS repair launched on that signal is both
    //      expensive and, on a converged continuous model, rejected outright.
    //      Measured on MINLPLib ex8_6_1: three such repairs took 4.7s of a 10s
    //      budget and cost ~20 gap points. The perturb half is kept, because
    //      st_e40 uses exactly those post-feasible kicks to hop between its 52
    //      feasible integer combinations and loses its BKS on half the seeds
    //      without them. See `SearchResult::lns_repairs`.
    //
    //      Those hops are no longer a walk: since #158 each kick departs from
    //      `kick_origin()` rather than from where the last one landed, so the
    //      52 combinations are explored in a star around the incumbent instead
    //      of a chain. That was expected to cost st_e40 its BKS and it does
    //      not. Whole 50-instance roster, 10s, FOUR PAIRED SEEDS, scored by
    //      ablation_report.py against each instance's own floor derived from
    //      the control's across-seed spread: st_e40 is 0.00 gap and 4/4
    //      feasible on BOTH arms, and not one of the 32 scored instances moved
    //      outside its floor (typical +/-9.89 gap-to-BKS points, median delta
    //      +0.00). 50 on the roster, 48 comparable (elec25/elec50 are
    //      documented failures on every arm), 32 of those scorable -- a floor
    //      needs a measurable control spread, which 16 do not have. See
    //      docs/architecture.md for why no floor is imputed onto them.
    //
    //      Read the history here as a lesson about method, which is more use
    //      than the number. An earlier pass reported "st_e40 BKS 8/8 -> 6/8,
    //      nvs02 12.1% -> 20.2%" and those figures were an artefact: the three
    //      instances were SELECTED as the largest movers in a two-seed run and
    //      then re-measured on their own, which is winner's curse. At four
    //      paired seeds with a measured floor, nvs02 is +3.77 against a floor
    //      of 34.25 and nvs14 +8.33 against 41.28 -- both inside the noise.
    //      Don't re-derive a per-instance cost here without a floor.
    //
    int64_t unproductive_iterations = 300;

    // ---- host integration (#169): cancellation and tracing ------------------
    //
    // Both are NON-OWNING views of objects the CALLER keeps alive for the whole
    // solve, and both are copied wherever a SearchConfig is -- which is what
    // makes them reach a portfolio worker's restart, since `run_worker` copies
    // this struct per restart.
    //
    // They live here rather than as trailing `solve()` parameters for one
    // concrete reason: the Python `solve` wrapper drops `solve()`'s trailing
    // `SearchCoordination*`, and a parameter that a binding silently drops is
    // the defect #169 exists to fix. A field is bound once and cannot be
    // dropped by accident.

    // The host's cancellation channel. Default: nothing attached, and the run is
    // bounded by `time_limit` and `max_iterations` alone. A raised stop ends the
    // run at the next batch boundary with `TerminationReason::Cancelled`. See
    // `StopRef`, and note the lifetime rule there.
    StopRef stop;

    // The host's event sink, or null. Null is the default and costs one null
    // compare per event site. PER WORKER under a portfolio -- do not hand the
    // same `Tracer` to several workers unless it is itself thread-safe; use
    // `ParallelConfig::tracer_factory`, which builds one per worker. See
    // `Tracer` for the granularity contract and for what attaching one costs.
    Tracer* tracer = nullptr;
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
enum class TerminationReason : std::uint8_t {
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
    /// A peer worker ended the run: `SearchCoordination::stop` was set while this
    /// search still had budget left. Only reachable from `ParallelSearch`, which
    /// raises the flag when one worker has answered the question outright -- a
    /// pure-feasibility model's first feasible solution. Distinct from
    /// `TimeLimit` on purpose: a worker cancelled 0.2s into a 60s budget did not
    /// run out of clock, and `time_seconds` must not be read as though it did.
    Stopped,
    /// The HOST cancelled: `SearchConfig::stop` (or `ParallelConfig::stop`) was
    /// requested while the run still had budget. Distinct from `Stopped`, which
    /// is a PEER WORKER ending the run from inside `ParallelSearch` -- a host
    /// integrating cbls as a component needs to tell "I cancelled it" from "it
    /// finished early on its own", and both from "it ran out of clock". Takes
    /// precedence over `Stopped` when both are true, because the host's cancel
    /// is the outer cause; `Feasible` still outranks both, since a worker that
    /// solved the model finished rather than stopped.
    Cancelled,
};

/// Stable snake_case token for a TerminationReason ("time_limit",
/// "iteration_limit", "feasible", "no_budget", "stopped", "cancelled"). Machine-readable — it is
/// the value the CLI writes to the JSONL `termination` field — and used verbatim in the human
/// output too, so there is exactly one spelling to keep in step with the enum. Returns a static
/// string; never null.
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
    /// call. This is the ONE field `ParallelSearch` still drops: it composes
    /// its result from the pool's best solution plus per-worker sums, and a
    /// latch can be neither summed nor attributed to the worker whose state won
    /// the pool. It reads `false` there with no exception, the no-solution
    /// fallback included, since that returns a default `SearchResult`.
    /// `best_violation` is no longer in this class -- a pooled `Solution`
    /// carries the residual of the state it holds, and the aggregation reads
    /// it.
    bool escape_probe_armed = false;

    /// Diversification kicks taken during the run -- the counter
    /// `SolveProgress::perturbations` reports, sampled at exit. Both routes are
    /// counted: the `perturbation_period` one and #102's unproductive-batch one.
    /// Exposed so a regression test can bound how OFTEN the search diversifies
    /// without reading its internals; a test that can only see the trajectory
    /// cannot tell a suppressed kick from a merely delayed one.
    /// `ParallelSearch` SUMS this over its workers, and within a worker over
    /// its restarts, the way `iterations` is summed: the counter describes work
    /// DONE. It is therefore not comparable to a single run's count at the same
    /// wall time. It read 0 under a portfolio until the aggregation began
    /// assembling it, which made an ablation read as "the mechanism never
    /// fired" on a run that used it thousands of times.
    int perturbations = 0;

    /// LNS destroy-repair cycles run during the run. Separate from
    /// `perturbations` because the two halves of a diversification kick cost
    /// wildly different amounts: a perturb is microseconds, an LNS repair is
    /// bounded by `min(2.0, remaining())` SECONDS and on a converged continuous
    /// model its result is usually rejected outright. #102's unproductive route
    /// draws only the cheap half once a feasible solution exists, and this is
    /// what lets a regression test assert that without timing the run.
    ///
    /// This counts repairs ATTEMPTED. `lns_repairs_accepted` below is the
    /// subset whose `destroy_repair` returned true; read the two together,
    /// because a nonzero count here says only that LNS spent budget, not that
    /// it helped.
    /// Summed over workers and restarts by `ParallelSearch`, as `perturbations`
    /// above is.
    int lns_repairs = 0;

    /// The subset of `lns_repairs` whose repair was ACCEPTED -- the calls where
    /// `LNS::destroy_repair` returned true. It records the RETURN, not a rule:
    /// the built-in `LNS` returns true exactly when the repaired state beat the
    /// incumbent on the lexicographic (real violation, objective) key and was
    /// therefore kept instead of rolled back, but `destroy_repair` is a virtual
    /// extension point and an override decides for itself what its `true`
    /// means.
    ///
    /// Separate from `lns_repairs` because "LNS ran" and "LNS helped" are
    /// different questions and only the first was answerable before #150. The
    /// `unproductive_iterations` note above records the measurement that makes
    /// the difference worth publishing: on a converged continuous model a
    /// post-feasible repair is rejected outright, so an arm can burn seconds
    /// per kick at an acceptance rate of zero and nothing in the result record
    /// would say so.
    ///
    /// Never greater than `lns_repairs`. A rejected repair is NOT free and is
    /// not a no-op either -- it draws from the RNG and rolls the state back --
    /// so `lns_repairs_accepted == 0` does not make an LNS arm equivalent to a
    /// no-LNS one; it only says the budget bought nothing.
    ///
    /// The bar rose at #158: `diversify()` restores `kick_origin()` before
    /// calling `destroy_repair`, so the key the repair is scored against is that
    /// point's rather than a drifted one's. At `--threads 1` that origin IS the
    /// incumbent, which makes the sentence above about beating "the incumbent"
    /// literal; under a portfolio it may instead be the state `adopt_from_pool`
    /// installed, which is a worse point by construction.
    ///
    /// What a rejection costs inverted at the same time -- in the ASSIGNMENT,
    /// not in the cost: the paragraph above still holds, a rejection always drew
    /// from the RNG and always spent the repair budget. Before #158 the snapshot
    /// was the drifted point the kick started from, so a rollback left the
    /// assignment exactly where it was. Now the snapshot is the origin, so a
    /// rollback MOVES the assignment, from wherever it had drifted to the
    /// origin, and perturbs nothing. So the kick is not a diversification
    /// either: the search re-descends from a point it has already converged on,
    /// with the GLS weights reset.
    ///
    /// Summed over workers and restarts by `ParallelSearch`, as `perturbations`
    /// above is.
    int lns_repairs_accepted = 0;

    /// The objective at the FIRST feasible point this run recorded, and the
    /// seconds it took to reach it (#149). Observational only: nothing in the
    /// search reads either field back, so the trajectory is the one the run
    /// without them would have taken. The one added cost is a single
    /// `steady_clock::now()` at the first feasible point, latched thereafter --
    /// which is why `docs/architecture.md`'s "reads no clock at all" invariant
    /// for an iteration-budgeted run now names three reads rather than two.
    ///
    /// "First feasible" is the moment `have_feasible` first turns true --
    /// before the inner-solver polish that follows in the same batch, and
    /// before any bound tightening. That is deliberate: the question these two
    /// fields exist to answer is where the search ARRIVES in the feasible
    /// region, as against how far the descent afterwards carries it, and a
    /// polished value is already the second of those.
    ///
    /// `time_to_first_feasible` is the authoritative "did this run ever record
    /// a feasible point" cell: it is NaN if and only if none was recorded.
    /// `first_feasible_objective` may be NaN or +/-inf on a run that DID reach
    /// feasibility -- the first feasible point can be the non-finite-objective
    /// witness of #100 -- so test the time, not the objective, and test
    /// `std::isfinite(first_feasible_objective)` before using the value.
    ///
    /// On a model with NO objective it is `0.0`, not NaN: the loop's
    /// `current_obj()` reports a finite 0.0 there, exactly as `objective`
    /// reports 0.0 for such a model. A consumer that mixes objective-free
    /// models into an aggregate must exclude them on the model, not on this
    /// field -- a 0.0 here is indistinguishable from a real objective of zero.
    ///
    /// NaN rather than `objective`'s `+inf` for "nothing to report", because
    /// +inf is a value the #100 witness path genuinely produces here and the
    /// two readings must not collide.
    ///
    /// `ParallelSearch` records both: the earliest worker AND restart to reach
    /// feasibility, with the objective that same run reached -- the pair means
    /// nothing split across two. The time is shifted onto the PORTFOLIO's
    /// clock, because a worker's own `SearchResult` times from that worker's
    /// `solve()` start and a worker restarts: unshifted, a late restart's own
    /// 0.001s would be reported for a portfolio most of the way through its
    /// budget. NaN still means no worker reached a feasible point.
    double first_feasible_objective = std::numeric_limits<double>::quiet_NaN();
    /// Seconds from the start of `solve()` to the first feasible point. See
    /// `first_feasible_objective` above; the two are recorded together and are
    /// NaN together on a run that never reached feasibility.
    double time_to_first_feasible = std::numeric_limits<double>::quiet_NaN();

    /// Where the run spent its work: batches by kind, the structural sweep's
    /// acceptance, the inner solver's cost, portfolio restarts (#169). The
    /// fields above answer "how much" and "how good"; this answers "on what".
    ///
    /// Summed across workers AND across a worker's restarts by `ParallelSearch`,
    /// exactly as `perturbations` is -- through `SearchCounters::merge`, which
    /// both aggregation sites call so the two cannot drift apart. It describes
    /// work DONE, so a portfolio's totals are not comparable to a single run's
    /// at the same wall time.
    ///
    /// Observational only. Nothing in the search reads a counter back, and
    /// filling them adds no clock read to a run without a wall-clock budget --
    /// see `SearchCounters::inner_solver_seconds` for the one field that gate
    /// costs.
    SearchCounters counters;
};

/// One progress row. Under `ParallelSearch` a row is a HYBRID by design and has
/// to be read as one: `time_seconds` is on the PORTFOLIO's clock and `objective`
/// / `new_best` describe the PORTFOLIO's incumbent -- so the stream is monotone
/// in both and its last row matches the returned result, which is what a harness
/// integrating it as a step function needs. `iteration`, `total_violation`,
/// `feasible` and `perturbations` stay the REPORTING worker's own, and
/// consecutive rows come from different workers, so those four are neither
/// monotone nor a rate, and `total_violation` is not the residual of the point
/// whose `objective` the row carries. Note `perturbations` here is one worker's
/// count where `SearchResult::perturbations` is the portfolio's sum. See
/// `PortfolioProgress` in src/pool.cpp.
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
///
/// `coord` is `ParallelSearch`'s cross-worker channel: non-null only there. With
/// it null -- every call site outside that class -- the search reads and writes
/// nothing shared and its trajectory is bit-identical to the run without the
/// parameter. `tests/test_parallel.cpp` pins that. Three of the four benchmark
/// runners are single-threaded and always pass null; `benchmarks/mipfeas` does
/// too at its default `--threads 1`, which is the arm every published MIPfeas
/// figure was measured on and the one that keeps the same-algorithm comparison
/// about the engine rather than about the portfolio.
SearchResult solve(Model& model, double time_limit = 10.0, uint64_t seed = 42, bool use_fj = true,
                   InnerSolverHook* hook = nullptr, LNS* lns = nullptr, int lns_interval = 3,
                   SolveCallback* callback = nullptr, const SearchConfig& config = {},
                   SearchCoordination* coord = nullptr);

}  // namespace cbls
