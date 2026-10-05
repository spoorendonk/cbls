#pragma once

#include "linear_jump.h"
#include "model.h"
#include "rng.h"
#include "violation.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace cbls {

// Generalised Feasibility Jump (ViolationLS, Davies et al. CPAIOR 2024,
// Algorithms 1-3). Drives the Model's current assignment X toward feasibility
// by repeatedly applying the best of a sampled set of improving single-variable
// "jumps", with Guided Local Search (GLS) weight bumping on stagnation.
//
// State maps to the paper's S = <G, X, W, V, Q, J>:
//   G = Model (graph), X = Model variable values, W = ViolationManager::weights,
//   V = violated constraints, Q = scan set of candidate vars, J = JumpTable.
//
// Only scalar variables (Bool/Int/Float) are jumped; List/Set variables are
// left untouched (they are handled by structural moves, P4).

// Per-variable cached jump: the best value to move the variable to and the
// resulting reduction in weighted violation (score = -W.deltaG(v, jump_value)).
// A positive score means an improving move exists. Entries are lazily
// invalidated when a neighbouring variable changes (paper Algorithm 1).
class JumpTable {
public:
    explicit JumpTable(size_t num_vars) : entries_(num_vars) {}

    [[nodiscard]] bool valid(int32_t var_id) const { return entries_[var_id].valid; }
    void invalidate(int32_t var_id) { entries_[var_id].valid = false; }
    void invalidate_all() {
        for (auto& e : entries_) {
            e.valid = false;
        }
    }
    void set(int32_t var_id, double jump_value, double score) {
        entries_[var_id] = {jump_value, score, true};
    }
    [[nodiscard]] double jump_value(int32_t var_id) const { return entries_[var_id].jump_value; }
    [[nodiscard]] double score(int32_t var_id) const { return entries_[var_id].score; }
    /// Multiply every cached score by `factor` (#175). A score is linear in the
    /// weights it was computed under, so when `LazyWeightDecay` folds a factor
    /// into the stored weights the cached scores move with them and stay
    /// comparable to the ones computed afterwards. O(#vars); called only when a
    /// factor is actually folded, never per bump.
    void scale_scores(double factor) {
        for (auto& e : entries_) {
            if (e.valid) {
                e.score *= factor;
            }
        }
    }

private:
    struct Entry {
        double jump_value = 0.0;
        double score = 0.0;
        bool valid = false;
    };
    std::vector<Entry> entries_;
};

// The best jump for a single scalar variable: jump_value minimises the weighted
// violation delta over the variable's domain, and score = -delta (>0 improving).
struct JumpResult {
    double jump_value = 0.0;
    double score = 0.0;
};

// Compute the best jump for `var_id` under the per-constraint `weights`: the
// value minimising the weighted violation delta over a small candidate set, and
// score = −delta (>0 improving). For Float variables the candidates are
// gradient-informed — a Newton step toward each violated constraint's root
// (reverse-mode AD) plus the domain midpoint and endpoints. A single call is not
// a converged 1-D minimiser; the GLS loop iterates these cheap jumps. Passing
// the GLS weights gives the Feasibility-Jump score; passing the novelty weights
// gives the Novelty-Jump (W') argmin. `var_id` must be scalar (Bool/Int/Float).
// `allow_escape_probe` opts a Float at a stationary point into a local
// two-sided probe. Off by default: it is a last resort, not a steady-state
// behaviour — see the comment on the probe in feasibility_jump.cpp.
//
// `linear`, when given, scores the candidates in closed form wherever every
// weighted row of G_v is linear -- an affine comparison or an affine bare body
// (see linear_jump.h) -- and falls back to `Model::weighted_violation_delta`
// otherwise. Same candidates and the
// same first-seen-minimum rule; the scores agree to rounding, not to the bit, so
// the selected jump is guaranteed identical only on integral data (where both
// are exact) -- on fractional data an ulp can flip a near-tie.
// It also supplies the Float Newton candidates' row partials where its cached
// slope is bit-identical to `compute_partial`, so those candidates do not move.
JumpResult compute_var_jump(Model& model, const std::vector<double>& weights, int32_t var_id,
                            bool allow_escape_probe = false, LinearJumpScorer* linear = nullptr);

// Guided Local Search weight update (paper Algorithm 3, lines 8-10): decay all
// weights by rho, then bump every currently-violated constraint by 1. Weights
// of constraints masked to 0 (e.g. non-linear constraints in the linear phase)
// stay 0 under decay and are never bumped while satisfied.
//
// This is the EAGER form, O(#constraints) per call: every weight is multiplied
// and every row's violation read. FeasibilityJump no longer calls it -- its bump
// uses LazyWeightDecay below (#175), the same update in a different
// representation -- and it stays as the public reference the lazy form is
// tested against.
void gls_update_weights(ViolationManager& vm, double rho);

// The GLS weight decay of ViolationLS (Davies et al., CPAIOR 2024, Algorithm 3),
// represented lazily (#175). THE ALGORITHM'S WEIGHTS ARE UNCHANGED; only how
// they are stored moves. With a global scale s, the stored vector holds
// w'_c = w_c / s, so
//
//   decay by rho:   s <- s * rho                      O(1), not O(#rows)
//   bump row c:     w'_c <- w'_c + 1/s  (if w'_c > 0)  O(1) per violated row
//   effective w_c:  s * w'_c
//
// which is `w <- rho * w; w += 1 on the active violated rows` in exact
// arithmetic, and agrees with the eager form to rounding, not to the bit. It is
// what CP-SAT's violation_ls does too (it grows the bump instead of shrinking a
// scale, the same thing). At rho = 1 with s = 1 the step is exactly 1.0 and the
// two forms ARE bit-identical. Trajectories differ by more than rounding all the
// same: the eager form left a cached jump score the bump did not invalidate at
// its pre-decay scale, stale by rho^-k against fresh ones, while a decay leaves
// every cached score exact in the scaled space (docs/architecture.md, #175).
//
// SCALE-FREE READERS. Every consumer FJ has inside its GLS loop only compares
// weighted sums with each other or with 0 (a jump is improving iff its score is
// > 0, the best of a sample is the largest), and multiplying every weight by the
// same s > 0 multiplies every such sum by s. So the loop reads the stored w'
// directly and never pays for s. `active()` reads `w' > 0`, which is
// `w > 0` because s > 0.
//
// RENORMALISATION. s may not leave [kMinScale, kMaxScale]: a decay that would
// take it out instead FOLDS s * rho into every stored weight (O(#rows)) and
// resets s = 1. The bound is about RANGE, not precision -- the lazy form rounds
// no worse than the eager one, which re-rounds every weight on every decay --
// and 1e30 leaves room: a stored weight is at most 1e30 times its effective
// value and a residual is clamped at kInfPenalty = 1e30, so a weighted term is at
// most ~1e60 times the effective weight -- which from begin()/reset_weights' 1 is
// below 1 / (1 - rho) = 20 at rho = 0.95 and grows by one per bump only in
// rho = 1 batches, where s never moves from 1. Nowhere near overflow; a weight a
// caller sets huge overflows the eager form just the same. At rho = 0.95 a fold
// happens once per ceil(log(1e-30) / log(0.95)) = 1347 decays, more than a
// default 1000-iteration batch can make, so inside solve() at the default
// batch_iterations it does not fire at all; at rho = 1 it never does. A rho the scale cannot absorb
// (0, negative, NaN, or anything that underflows s past the bound in one step) takes the same fold,
// which is then the eager update (to the bit when s = 1 at that point).
//
// WEIGHT 0 STAYS EXACTLY 0, and a positive weight stays positive. A masked row
// stores 0, and 0 / s, 0 + nothing and 0 * s are all 0. A positive stored
// weight times a positive fold factor that underflows is floored at the
// smallest subnormal instead of becoming 0, because decay by rho > 0 cannot
// reach 0 in exact arithmetic and a row at 0 is masked for good (the bump skips
// it). The eager form agrees at the rho values solve() draws: at 0.95 a
// repeatedly decayed weight sticks at 9 subnormal ulps (w * 0.95 rounds back to
// w there) and never reaches 0. Only a fold by exactly 0 (rho = 0) zeroes a
// positive weight, as the eager `w * 0` does.
//
// Between calls that fold, the stored vector is NOT the effective weights. The
// owner must `materialise` before anything outside its scale-free readers looks
// at them; FeasibilityJump does so on every exit from its GLS loop, so
// ViolationManager::weights means effective weights whenever FJ is not running.
class LazyWeightDecay {
public:
    static constexpr double kMinScale = 1e-30;
    static constexpr double kMaxScale = 1e30;

    /// Decay every weight in `w` by rho. Returns the factor folded into the
    /// stored weights, 1.0 when none was: the caller must multiply anything
    /// else it keeps in the scaled space (cached jump scores) by it.
    double decay(std::vector<double>& w, double rho);
    /// The bump of row c: effective w_c += 1, unless the row is masked (0).
    void bump(std::vector<double>& w, size_t c) const {
        if (w[c] > 0.0) {
            w[c] += step_;
        }
    }
    /// Row c's effective weight, s * w'_c.
    [[nodiscard]] double effective(const std::vector<double>& w, size_t c) const {
        return scale_ * w[c];
    }
    /// Fold s into the stored weights so they ARE the effective weights, and
    /// reset s = 1. O(#rows) unless s is already 1, when it touches nothing.
    /// Returns the factor folded (1.0 if none), as `decay` does.
    double materialise(std::vector<double>& w);
    [[nodiscard]] double scale() const { return scale_; }

private:
    double fold(std::vector<double>& w, double factor);

    double scale_ = 1.0;
    double step_ = 1.0;  // 1 / scale_, so a bump is one add
};

struct GFJConfig {
    int sample_size_linear = 5;   // best-of-N sampling, linear phase (paper)
    int sample_size_general = 3;  // best-of-N sampling, general phase (paper)
    double rho = 0.95;            // GLS decay; caller samples from {0.95, 1.0} per batch
    bool two_phase = true;        // GLS on linear submodel first, then full model
    bool set_initial_x = true;    // set X[v] to the domain value closest to 0 first
    int64_t max_iterations = 0;   // 0 = unbounded (bounded by time_limit)
    double time_limit = 0.0;      // seconds; 0 = no limit
    // End a *batch* that has run this many GLS iterations without reducing the
    // unweighted violation of the REAL rows below its best so far for that
    // batch. <= 0 disables the check. See the comment on unweighted_violation_
    // for the measure and what it was calibrated against. Applies only to the
    // batch API: the single-shot gls()/run() path has no outer loop to hand
    // control back to, so an early exit there would simply be giving up.
    //
    // What 300 is, honestly. It is a three-point grid search — {100, 300, 1000}
    // — over ONE roster (MINLPLib, 50 instances) at ONE contended 2s budget and
    // ONE seed. 100 lost feasibility on kall_ellipsoids_tc02b; 1000 stopped
    // solving st_e40, the instance the mechanism was written for, so the ceiling
    // is set by the motivating case rather than independently. It is not a tuned
    // optimum. #145 re-ran the grid on the 50-instance held-out MINLPLib roster at
    // 10s, three seeds (benchmarks/instances/minlplib/HELDOUT.md): 1000 was worse,
    // 100 not resolvable against 300. That says nothing about 60s, and nothing
    // establishes that 300 transfers off MINLPLib.
    //
    // It is also DIMENSIONLESS on a quantity whose natural scale is not. An
    // iteration count says nothing about how much violation a model can shed per
    // iteration, and "300 iterations without a new minimum" means something very
    // different on a 4-variable instance than on one with 10^5 rows. This is the
    // same scale-invariance complaint search.cpp records against
    // perturbation_period, which counts BATCHES for a threshold that is really a
    // duration. A progress-rate or budget-fraction formulation would not have
    // it; neither has been measured. Re-tuning by grid search would only move
    // the number, not fix the shape.
    int64_t unproductive_iterations = 300;
};

enum class GFJStatus : std::uint8_t { Feasible, Unsolved };

class FeasibilityJump {
public:
    FeasibilityJump(Model& model, ViolationManager& vm, RNG& rng, GFJConfig config = {});

    // Run GLS until feasible, or until the iteration/time budget is exhausted
    // (standalone construction / single-shot use).
    GFJStatus run();

    // Batch API for the ViolationLS outer loop (Algorithm 6). The caller owns
    // the loop: begin() once, then batch() repeatedly, calling reset_weights()
    // on a new best (after tightening the objective bound) and perturb() on
    // stagnation. set_rho() re-randomises the GLS decay between batches.
    void begin(bool set_initial_x);
    bool batch(int64_t batch_iterations);  // true if feasible (no active violated)
    // Whether the last batch() ended because it had stopped reducing the real
    // rows' violation, rather than because it exhausted its iteration budget or
    // the deadline. Only meaningful straight after a batch() call; cleared at
    // each batch entry. Governed by GFJConfig::unproductive_iterations.
    [[nodiscard]] bool batch_stuck() const { return batch_stuck_; }
    void reset_weights();  // W <- 1 and rebuild the scan set
    void resync();         // rebuild the scan set from current state, keep weights
    // Randomise each jumpable var w.p. p, then apply
    // clamp(round(p*|elements|), 1, |elements|) random structural moves to each
    // List/Set var (#111); if all of that moved nothing, force one variable — a
    // scalar if any can move, else a structure — so the kick is never a no-op
    // (#109). The fallback leaves large models bit-identical to plain p-draws,
    // and a model without List/Set variables keeps its exact draw sequence.
    void perturb(double probability);
    void set_rho(double rho) { config_.rho = rho; }
    // Armed by the search loop once it has stagnated; see solve().
    void set_escape_probe(bool on) { escape_probe_ = on; }
    // The armed state, so the caller (and its regression tests) can observe the
    // arming decision directly instead of inferring it from a trajectory.
    [[nodiscard]] bool escape_probe() const { return escape_probe_; }
    // Arm/disarm the unproductive-batch exit (GFJConfig::unproductive_iterations)
    // for the next batch. Armed at begin(); solve() re-decides it before every
    // batch from its own stagnation count -- see the SECOND WITNESS paragraph on
    // unweighted_violation_ for why the measure alone is not enough to end a
    // batch on.
    void set_watch_progress(bool on) { watch_progress_ = on; }
    // The running progress measure: unweighted violation of the active REAL rows
    // (see unweighted_violation_). Read-only observability for the regression
    // test that pins the incremental accumulator against a fresh recomputation;
    // the search itself does not consult it.
    [[nodiscard]] double unweighted_violation() const { return unweighted_violation_; }

    /// Read-only observability for the incremental violated-row state (#174), in
    /// the same spirit as `unweighted_violation()`: the tests that pin it against a
    /// from-scratch recomputation read it here, the search never does.
    ///
    ///  - `row_violated(ci)`: row `ci` is in V (the violated set FJ tracks);
    ///  - `violated_rows()`: V sorted ascending -- a sorted copy; the search keeps
    ///    V in flip order and never sorts it;
    ///  - `active_violated_rows_of(v)`: how many rows of `vars_of_constraint_` that
    ///    list `v` are violated AND active (weight > 0) -- the count
    ///    `participates_in_active_violated` reads. Meaningful for a jumpable
    ///    variable; a structured one is in no row's list and reads 0.
    ///    "Active" is as of the row's last reconcile: a weight a caller changes
    ///    between batches shows up here only once the next batch, resync or
    ///    reset has re-read the rows in V.
    ///
    /// `row_violated` and `active_violated_rows_of` throw `std::out_of_range` on
    /// an index the model has no row or variable for.
    [[nodiscard]] bool row_violated(int32_t ci) const;
    [[nodiscard]] std::vector<int32_t> violated_rows() const;
    [[nodiscard]] int32_t active_violated_rows_of(int32_t var_id) const;

    /// Read-only check of the two scan sets' bookkeeping (#206), for tests:
    /// true iff Q (apply_jump's) and Q' (Novelty's) each hold no duplicate and
    /// their membership flags are set exactly for the vars they hold, and no
    /// var on Novelty's compound-move stack is in Q'. Both sets are swap-removed
    /// from by index, so a removal at the wrong index either strands a flag or
    /// leaves the chosen var in Q' while it sits on the stack. Meaningful
    /// between calls, not inside one. O(|V| + |Q| + |Q'|).
    [[nodiscard]] bool scan_sets_consistent() const;

    /// The jump table's cached entry for `var_id` -- jump value and score as
    /// apply_jump last computed them -- or nullopt when the entry is invalid and
    /// the next draw of the variable will recompute it. Read-only observability
    /// for the tests that pin which moves invalidate whose entries (#210); the
    /// search never calls it. Throws `std::out_of_range` on an index the model
    /// has no variable for.
    [[nodiscard]] std::optional<JumpResult> cached_jump(int32_t var_id) const;
    /// How many times the objective row's neighbour walk was skipped as inert
    /// (#210) -- by update_var, and by Novelty's re-queue after a move or undo
    /// -- since construction. Diagnostics, and the pin on each site's wiring.
    [[nodiscard]] int64_t objective_walks_skipped() const { return objective_skips_fj_; }
    [[nodiscard]] int64_t novelty_objective_walks_skipped() const {
        return objective_skips_novelty_;
    }

    [[nodiscard]] bool all_satisfied() const;
    [[nodiscard]] int64_t iterations() const {
        return iterations_;
    }  // total GLS iterations since begin()

    // ---- Deadline-check tuning (#113) ----
    //
    // The GLS loop checks the wall clock on a stride sized in *time*, not in
    // iterations: one stride costs at most kStrideBudgetFraction of the budget,
    // or one (atomic) GLS iteration, whichever is larger. That is the batch's
    // worst-case overrun. See the long comment on gls_loop for the measurements.
    static constexpr double kStrideBudgetFraction = 1.0 / 64.0;
    static constexpr int64_t kStrideGrowth = 8;  // max growth per adjustment
    // Hard iteration cap on the stride, and the reason the worst case is
    // bounded at all. A time-sized stride alone is not enough: the shrink can
    // only be APPLIED at a check, and the next check is a whole stride away, so
    // a stride grown while iterations were cheap is spent in full once they turn
    // expensive — the tuner goes silent exactly when it is needed. Measured at
    // 18.7x over a 1s budget on a model whose cost jumps mid-run, against 2.8x
    // for the fixed stride this replaced. 64 is that fixed stride, so the worst
    // case is now no worse than the code being replaced, while the time-based
    // shrink still delivers #113's case (136x -> 2.5x at a 0.05s budget).
    // Priced at ~1.2% throughput on cheap iterations, which is all that letting
    // the stride grow past 64 was ever buying.
    static constexpr int64_t kMaxDeadlineStride = 64;
    // Pure function of the last measurement, exposed so the tuner can be tested
    // directly — in particular that it shrinks, not only grows.
    static int64_t next_deadline_stride(int64_t stride, double elapsed_seconds,
                                        double target_seconds);
    // How the deadline is currently being observed. `deadline_checks()` counts
    // clock reads made INSIDE THE GLS LOOP; `deadline_check_stride()` is the
    // live stride. Note the kick's structural pass (#111) reads the clock on its
    // own path and is not counted here — see `structural_kick_checks()` below —
    // so a zero is evidence about this loop and not about the whole engine. Both
    // paths are gated on `has_deadline_`, which is what actually delivers
    // determinism.
    [[nodiscard]] int64_t deadline_checks() const { return deadline_checks_; }
    [[nodiscard]] int64_t deadline_check_stride() const { return deadline_stride_; }

    // ---- The structural kick's own deadline bound (#115) ----
    //
    // How the last perturb() observed the deadline inside its structural pass,
    // in the unit that pass advances in: one structural MOVE (one move-set
    // generation plus one apply, O(|elements| + universe) element copies).
    // `structural_kick_moves()` is the moves that pass applied, which is the
    // quantity the bound is about and is observable without a clock;
    // `structural_kick_checks()` is 0 for a run with no wall clock, the direct
    // evidence that no clock read reached control flow.
    //
    // Two things the counts deliberately exclude, so read them as being about the
    // strided pass rather than about the whole kick. perturb()'s never-a-no-op
    // fallback (force_structural_move) applies a real structural move that
    // kick_moves_ does not count, so a kick's true worst case is one move more
    // than reported. And arm_structural_kick() reads the clock once per kick
    // without counting it, which is also why a deadline-armed scalar-only model
    // now pays one steady_clock::now() per kick where it paid none before.
    [[nodiscard]] int64_t structural_kick_moves() const { return kick_moves_; }
    [[nodiscard]] int64_t structural_kick_checks() const { return kick_checks_; }
    /// The closed-form linear scorer this object scores jumps with -- its
    /// fast/fallback counters, and a way for tests to check its scores.
    /// `prepare` on it touches only its own scratch and lazily built
    /// rows, never the assignment, so calling it between batches is harmless.
    [[nodiscard]] const LinearJumpScorer& linear_scorer() const { return linear_; }
    [[nodiscard]] LinearJumpScorer& linear_scorer() { return linear_; }
    [[nodiscard]] int64_t structural_kick_stride() const { return kick_stride_; }

    // Novelty Jump (paper Algorithms 4-5): a bounded-backtracking compound-move
    // search that escapes local optima single-variable FJ cannot (chained-
    // invariant fixes). One stand-alone ApplyNoveltyJump (Algorithm 4) from a
    // fresh W' and Q', capped at novelty_work_budget() applied moves. Commits
    // the improving compound move(s) it finds (left applied) and returns true
    // if it reaches feasibility, else leaves any committed moves applied and
    // returns false. Call from a local optimum with violated_/weights current
    // (e.g. right after begin() or a stalled batch); the caller must resync()
    // afterwards. Uses novelty weights W' = kCompoundDiscount*W for constraints
    // not violated at entry, full W for those violated at entry. The search
    // runs novelty_batch() instead.
    bool apply_novelty_jump();
    /// One Novelty Jump BATCH (#209): GLS with ApplyNoveltyJump as its move
    /// (ViolationLS Algorithm 6 line 22, Algorithm 3 with M = Algorithm 4).
    /// Each time a whole ApplyNoveltyJump finds no compound move at its largest
    /// discrepancy budget, the GLS weights are decayed and bumped exactly as
    /// batch()'s are, and the search goes on; W' and Q' are set up once per
    /// batch and maintained incrementally from there. The batch applies at most
    /// `batch_iterations / 3` moves (a compound move's legs and the moves it
    /// explores and undoes each count once, undos not at all; a weight bump
    /// counts as one too), since a Novelty move scores its whole sample of 3
    /// afresh where an FJ iteration reads cached jumps -- see the definition;
    /// <= 0 sets no limit, as for batch(), and leaves the deadline to end it.
    /// Returns true if no active
    /// constraint is violated. Leaves FJ's own state (V, Q, jump table) current,
    /// so the caller needs no resync() -- unlike apply_novelty_jump(). Charges
    /// nothing to iterations(); see novelty_moves() and friends.
    bool novelty_batch(int64_t batch_iterations);
    /// Moves (applies, not undos) the last apply_novelty_jump() made, and the
    /// cap on that number. Observability for the regression test that pins
    /// the cap (#206): a sibling loop that kept going after the cap ran out
    /// is what let one call outlive a 20s budget.
    [[nodiscard]] int64_t novelty_moves_last_call() const { return nj_moves_this_call_; }
    [[nodiscard]] static constexpr int64_t novelty_work_budget() { return kNoveltyWorkBudget; }
    /// Cumulative Novelty engagement since construction, over both entry
    /// points: moves applied (undos not counted), compound moves committed,
    /// and GLS weight bumps a Novelty batch made. Diagnostics: the search
    /// never reads them back.
    [[nodiscard]] int64_t novelty_moves() const { return novelty_moves_; }
    [[nodiscard]] int64_t novelty_commits() const { return novelty_commits_; }
    [[nodiscard]] int64_t novelty_weight_bumps() const { return novelty_bumps_; }
    /// The novelty weight W' of row `ci` as Novelty last left it -- read-only
    /// observability for the tests that pin its incremental upkeep (#209).
    /// Throws `std::out_of_range` on a row the model does not have, or before
    /// any Novelty call has sized W'.
    [[nodiscard]] double novelty_weight(int32_t ci) const;
    /// Read-only check of W's incremental upkeep (#209), for tests: true iff
    /// every row NOT noted for the next level reset holds its level-start
    /// novelty weight as of the current assignment and weights --
    /// W[c] if the row is violated, kCompoundDiscount * W[c] otherwise -- and
    /// the noted list agrees with its flags. That is the invariant that lets a
    /// level reset visit only the noted rows. Meaningful between calls, and
    /// vacuously true before the first Novelty call. O(#rows).
    [[nodiscard]] bool novelty_weights_consistent() const;
    /// How many times Q' has been seeded from V's rows since construction:
    /// once per apply_novelty_jump() or novelty_batch() since #209, where it
    /// used to be once per committed compound move and per discrepancy level.
    [[nodiscard]] int64_t novelty_scan_set_seeds() const { return novelty_seeds_; }

private:
    // One GLS pass over the constraints whose weight is currently > 0 (the
    // "active" set). Returns Feasible if all active constraints are satisfied.
    GFJStatus gls(int sample_size);
    // GLS inner loop reusing current state, bounded by a per-call iteration
    // limit (<=0 for none) plus the global budget/deadline.
    GFJStatus gls_loop(int sample_size, int64_t batch_iter_limit);
    // gls_loop's body. The weights are in LazyWeightDecay's scaled space while it
    // runs; gls_loop materialises them on every way out.
    GFJStatus gls_loop_scaled(int sample_size, int64_t batch_iter_limit);
    [[nodiscard]] bool any_active_violated() const;
    // How a batch reports its own end when it ran out of budget rather than out
    // of work: Feasible only if nothing active is still violated. Used at every
    // budget exit of gls_loop so they cannot drift apart. Re-grounds first, so
    // the verdict is read off re-summed rows (#188).
    [[nodiscard]] GFJStatus batch_end_status() {
        reground_drifted_rows();
        return any_active_violated() ? GFJStatus::Unsolved : GFJStatus::Feasible;
    }
    // ---- The incremental Sums' drift (#188) ----
    //
    // Is row c's verdict undecided by its Sum's drift? Only a row that reads an
    // incremental Sum directly (row_slot_) can be, and only while that Sum
    // is drifting (IncSumState::drifting): then its real residual
    // lies within drift_bound (plus the rounding of the comparison itself) of
    // `residual`, and the question is whether that interval straddles kTol.
    [[nodiscard]] bool verdict_undecided(int32_t c, double residual) const;
    // Put row c on uncertain_rows_ if its verdict is undecided now. Called
    // wherever a row's value or its Sum's drift changes inside a batch --
    // update_var's rows -- so the list holds every undecided row.
    void note_row_certainty(int32_t c, double residual);
    // The local-minimum gate. Re-sums the Sum under every row on
    // uncertain_rows_ that is still undecided, and settles the rows that moved.
    // Returns true when a row entered or left V: the "no improving jump"
    // conclusion was reached on values that are no longer the model's, so the
    // caller re-samples instead of bumping or declaring feasibility.
    bool settle_undecided_rows();
    // Re-sums every Sum that drifted since the last call and settles the rows
    // that moved. Returns true when a row entered or left V. The end of every
    // batch, the start of Novelty Jump, and every Feasible verdict call it.
    bool reground_drifted_rows();
    // After reground_inc_sum(s) of slot_scratch_: settle each of their rows
    // whose value moved as update_var settles a row -- the unweighted total, V,
    // and its variables' cached jumps and scan-set membership -- from the
    // "before" values captured in row_before_. Returns true if V changed.
    bool settle_regrounded_rows();
    // Snapshot the rows of `slot` into row_before_ ahead of its re-sum.
    void capture_rows_of_slot(int32_t slot);
    void build_row_slots();
    // No improving jump exists anywhere in the scan set: bump the GLS weights of
    // the violated constraints and re-queue their variables, so the next
    // iteration scores them against the new penalty landscape.
    void bump_weights_and_requeue();
    // Fold the lazy decay's scale into vm_.weights and the cached jump scores, so
    // both are in effective terms again (#175). Every exit from gls_loop calls it,
    // exceptional ones included: outside the loop the weights are public.
    void materialise_weights();
    // Fold the iteration just completed into the batch's progress state:
    // `batch_best_violation` is the running minimum of the unweighted real-row
    // violation and `unproductive_streak_` counts iterations since it last
    // moved. Returns true when the streak has hit its limit AND an exact
    // re-grounding confirms the batch really is stuck, i.e. the batch must end.
    // `watch_progress` is the caller's arming decision, computed once per batch.
    bool track_batch_progress(double& batch_best_violation, bool watch_progress);
    // Read the clock and re-size the deadline stride. Returns true if the
    // deadline has passed. Only called when the countdown has expired, so it is
    // off the per-iteration path; see the comment above gls_loop.
    bool deadline_passed_and_retune();
    // Recompute unweighted_violation_ from scratch: sum of the finite positive
    // residuals of the active REAL rows (objective row excluded, see that
    // member). This is the only thing that re-grounds the incremental
    // accumulator, so every place the accumulator's value is allowed to decide
    // something calls it first. Three sites:
    //
    //   * gls_loop entry, so a batch never inherits the previous batch's
    //     accumulated rounding. rebuild_violated_and_scan_set does NOT cover
    //     this: consecutive non-improving FJ batches call gls_loop directly with
    //     no rebuild in between, so across a long stagnant run the accumulator
    //     would otherwise never be re-grounded at all;
    //   * the unproductive-streak limit, before batch_stuck_ is set — the one
    //     consequential read, so it is made on an exact sum rather than on a
    //     drifted one;
    //   * rebuild_violated_and_scan_set, which resynchronises the loop after a
    //     mutation made outside update_var (the novelty jump, the structural
    //     pass, perturb, LNS).
    void refresh_unweighted_violation();
    // ApplyJump (Algorithm 2): sample up to `sample_size` vars from Q, apply the
    // best improving jump via update_var. Returns false if none improves.
    bool apply_jump(int sample_size);
    // UpdateVar (Algorithm 1): commit X[v] <- jump, refresh V, invalidate
    // neighbour jumps, replenish Q.
    void update_var(int32_t var_id);
    // Whether a move that took the objective's VALUE from `before` to `after`
    // leaves every cached jump, and every Novelty verdict, of the row's
    // other variables as it was -- so update_var and nj_requeue_neighbours can
    // skip the row's neighbour walk (#210). See the definition for the
    // predicate and the regime each answer wins and loses in.
    [[nodiscard]] bool objective_row_inert(double before, double after);
    // The objective's value as the DAG holds it now; 0.0 without an objective row.
    [[nodiscard]] double objective_value() const;
    // Whether every jumpable column of the objective row holds a value inside
    // its declared box -- the premise of the finite-bound skip. O(objective
    // support); run by rebuild_violated_and_scan_set.
    [[nodiscard]] bool objective_columns_in_box() const;

    [[nodiscard]] bool active(int32_t constraint_idx) const;  // weight > 0
    [[nodiscard]] bool jumpable(int32_t var_id) const;        // scalar var
    // Uniformly chosen jumpable var with a domain of at least two values — the
    // one perturb() falls back to when its per-variable draws moved nothing.
    // -1 if the model has no such variable, in which case a kick that changes
    // nothing is the correct outcome.
    int32_t pick_forced_perturb_var();
    // The List/Set half of a diversification kick: perturb() cannot reach them
    // through jumpable(), so each structural variable gets a run of
    // clamp(round(p * |elements|), 1, |elements|) random typed structural moves
    // instead (#111) — plus, for a member of a ListPartition, that partition's
    // inter-list moves anchored on it, which is what lets a kick move an
    // all-empty partition (#164). Returns true if any variable's elements NET changed —
    // by set equality for a Set, whose elements are unordered. Draws no random
    // numbers at all on a model without List/Set variables. Deadline-bounded
    // between MOVES, not between variables (#115); see kick_past_deadline().
    bool perturb_structural(double probability);
    // Reset the structural kick's move counter and stride tuner. Called once per
    // perturb_structural(), so a kick never inherits a stride another kick grew.
    void arm_structural_kick();
    // True when the structural pass must stop: the deadline has passed, observed
    // on a stride counted in structural moves. Never true before the pass has
    // applied a move, so a deadline already crossed on entry cannot turn a kick
    // into the no-op #109/#111 exist to prevent.
    bool kick_past_deadline();
    // Apply one structural move to some List/Set variable that can take one —
    // the structural peer of pick_forced_perturb_var(), for a kick that would
    // otherwise change nothing on a model with no movable scalar. False if
    // every structure is a dead end.
    bool force_structural_move();
    [[nodiscard]] bool participates_in_active_violated(int32_t var_id) const {
        return active_violated_of_var_[static_cast<size_t>(var_id)] > 0;
    }
    void rebuild_violated_and_scan_set();
    // ---- Incremental violated-row state (#174); see violated_rows_ ----
    // Move row `c` into or out of V. When it stays in (or enters), reconciles its
    // counted bit with the LIVE active(c), so a weight change is picked up at the
    // next evaluation of the row whatever caused it.
    void set_violated(int32_t c, bool now);
    // Bring row c's counted bit in line with violated && active(c), adjusting
    // active_violated_of_var_ over the row's variable list if it changes.
    void reconcile_counted(int32_t c);
    // Re-derive the list, the positions, the counted bits and the per-variable
    // counts from violated_'s in-V bits (which it does not re-evaluate). One
    // O(#constraints + nonzeros of the counted rows) sweep.
    void rebuild_violated_index();
    // reconcile_counted over every row in V: O(|V|). Run at every gls_loop entry,
    // since weights may have changed outside this object's view between batches
    // (bump_weights_and_requeue does the same inline, row by row).
    void reconcile_all_counted();
    // Drop row c's contribution to active_violated_of_var_ if it is counted,
    // leaving it in V (kInV kept).
    void uncount(int32_t c);
    void set_initial_assignment();
    void compute_linear_constraints();
    /// Throw unless every per-row and per-variable table here, and the
    /// ViolationManager's weights, are sized for the model as it is NOW. See the
    /// definition for why it is checked per batch rather than per row.
    void require_tables_in_step() const;
    void enqueue(int32_t var_id);

    // Novelty Jump internals (Algorithm 5). A candidate var with its W'-argmin
    // jump and both scores (original-weight `score`, novelty-weight
    // `novelty_score`).
    struct NoveltyPick {
        int32_t var = -1;
        double jump = 0.0;
        double score = 0.0;          // -W . deltaG(v, jump)
        double novelty_score = 0.0;  // -W' . deltaG(v, jump)
    };
    // How an ApplyNoveltyJump (Algorithm 4) ended.
    enum class NoveltyOutcome : uint8_t {
        Feasible,   // a committed compound move left no active row violated
        LocalMin,   // no compound move at the largest discrepancy budget: bump
        OutOfWork,  // the work bound or the deadline cut the search short
    };
    // Set up W', Q' and the stack for a run of Novelty: the per-BATCH O(#rows)
    // W' init and O(nnz(V)) seed, which nothing inside the run repeats.
    void begin_novelty_run();
    void init_novelty_weights();
    void seed_novelty_scan_set();
    void nj_enqueue(int32_t var_id);
    // `objective_before`: the objective's value before v's move or undo
    // (objective_value()), for the inert-row skip.
    void nj_requeue_neighbours(int32_t v, double objective_before);
    NoveltyPick select_novelty_var(double s_m, double s_c);
    bool novelty_jump_search(double s_m, int budget);
    // Algorithm 4 over the state begin_novelty_run() set up, incrementally.
    NoveltyOutcome novelty_descent();
    // Algorithm 4 lines 3-5 for the next discrepancy level, in O(rows whose W'
    // moved off its level-start value) rather than O(#rows); see the definition.
    void reset_changed_novelty_weights();
    void note_novelty_weight_changed(int32_t c);
    // Algorithm 3 lines 8-12 inside a Novelty batch.
    void novelty_bump_weights();
    // Multiply W' by a lazy-decay fold factor, under LazyWeightDecay::fold's rule.
    void fold_novelty_weights(double factor);
    void clear_novelty_stack();

    Model& model_;
    ViolationManager& vm_;
    RNG& rng_;
    GFJConfig config_;

    JumpTable jumps_;
    // The GLS decay's lazy scale (#175). s != 1 only inside gls_loop; see
    // LazyWeightDecay and materialise_weights.
    LazyWeightDecay weight_decay_;
    // ---- V, the violated set, kept incrementally (#174) ----
    //
    // Per constraint, two bits: kInV (the row is violated, i.e. in V) and
    // kCounted (it is in V AND was active -- weight > 0 -- when last evaluated,
    // and so is counted in active_violated_of_var_). `violated_[c] != 0` still
    // reads "in V", because kCounted is never set without kInV.
    //
    // Every write goes through set_violated, reconcile_counted, uncount or
    // rebuild_violated_index (which re-derives everything from the raw kInV bits
    // rebuild_violated_and_scan_set fills in). The derived structures are:
    //   - violated_rows_ + violated_pos_: V as a dense list with a position index
    //     (-1 = absent), swap-removed on the way out;
    //   - active_violated_of_var_: per variable, the number of COUNTED rows whose
    //     vars_of_constraint_ list names it. participates_in_active_violated is
    //     then `> 0`, O(1) instead of O(|G_v|).
    //
    // WHERE IT WINS AND WHERE IT LOSES. The scans it replaces were O(#rows) per
    // weight bump, per Novelty seed and per any_active_violated, and
    // update_var paid O(|G_vp|) per neighbour vp -- the two-hop nonzeros, per
    // committed move. Now any_active_violated, the Novelty seed's scan and the
    // bump's scan are each O(|V|), plus the variable lists of the rows they
    // queue from. They visit V in its list order, which is deterministic -- a
    // function of the flip history alone -- but not ascending: #174 sorted V
    // before each of them to stay bit-identical with the whole-row sweep, which
    // made a bump O(|V| log |V|) capped at O(#rows); once #175 gave up
    // bit-identity that sort bought nothing, and it is gone. The bump itself
    // used to stay O(#rows) through gls_update_weights; #175 made the decay
    // lazy (LazyWeightDecay), so a bump is now O(|V|) plus the nonzeros of V's
    // counted rows it requeues, plus one O(#rows + #vars) fold per gls_loop
    // exit that decayed (and, in an unlimited gls()/run() loop, one per 1347
    // decays at rho = 0.95).
    // A Novelty batch as a whole likewise stays O(#rows), through
    // init_novelty_weights and the rebuild it ends with -- once per batch since
    // #209, which made W' and Q' incremental inside it (see the Novelty section
    // of feasibility_jump.cpp).
    //
    // The price is a constant per row FLIP: a push or a swap-remove, and a walk
    // of the flipped row's variable list to adjust the counts. In update_var that
    // is a walk its neighbour loop makes over the same row anyway, so a move adds
    // at most one more pass of the one-hop loop it already runs. Novelty's apply
    // AND undo pay it too, where they used to write one byte per row, although
    // Novelty never reads the counts -- bounded by kNoveltyWorkBudget moves per
    // call, and accepted to keep one write path. So it loses where almost every
    // row flips on almost every move, and on Novelty probes over long rows (the
    // objective row lists every objective column).
    //
    // Memory, per worker: 4 B per row (violated_pos_), up to 4 B per row more
    // for violated_rows_' capacity (it keeps the largest V it has held), and 4 B
    // per variable.
    //
    // "Active" is read live wherever this state is evaluated -- set_violated
    // reconciles the row it writes, bump_weights_and_requeue reconciles each row
    // in V against its bumped weight, and reconcile_all_counted re-reads every
    // row in V at each gls_loop entry -- so a weight changed between batches, or
    // by the bump, is picked up. A weight changed from outside WHILE gls_loop runs
    // would not be; nothing does that. This bookkeeping reads the weight only of
    // a row in V or entering it; other readers -- refresh_unweighted_violation,
    // update_var's residual over G_v, init_novelty_weights -- still read rows
    // outside V. Under the lazy decay (#175) the first two read `w' > 0`, which is
    // `w > 0`, and init_novelty_weights runs outside the GLS loop, where the
    // weights have been materialised.
    static constexpr uint8_t kInV = 1;
    static constexpr uint8_t kCounted = 2;
    std::vector<uint8_t> violated_;                // per constraint: kInV | kCounted
    std::vector<int32_t> violated_rows_;           // V, dense
    std::vector<int32_t> violated_pos_;            // per constraint: index in violated_rows_, or -1
    std::vector<int32_t> active_violated_of_var_;  // per var: counted rows listing it
    std::vector<uint8_t> in_queue_;                // per var: in Q
    std::vector<int32_t> queue_;                   // scan set Q (vars with possibly-positive score)
    // ---- The incremental Sums' drift (#188) ----
    // Row -> the slot of the incremental Sum it reads directly, or -1; and the
    // inverse, slot -> its rows, as CSR (an MPS range row is two rows over one
    // Sum). Built once, in the constructor: 4 B per row and per slot.
    std::vector<int32_t> row_slot_;
    std::vector<uint32_t> slot_rows_begin_;
    std::vector<int32_t> slot_rows_;
    // Rows whose verdict was undecided by drift when last evaluated, each once
    // (in_uncertain_). Emptied by settle_undecided_rows and the re-groundings.
    std::vector<int32_t> uncertain_rows_;
    std::vector<uint8_t> in_uncertain_;
    // Scratch: the slots a gate or a re-grounding re-summed, and (row, value
    // before) for each of their rows.
    std::vector<int32_t> slot_scratch_;
    std::vector<std::pair<int32_t, double>> row_before_;
    std::vector<uint8_t> is_linear_;  // per constraint
    // Closed-form scoring over linear rows (affine comparisons and affine bare
    // bodies). Its per-row eligibility is maintained wherever is_linear_ is:
    // compute_linear_constraints (the constructor).
    LinearJumpScorer linear_;
    std::vector<std::vector<int32_t>> vars_of_constraint_;  // constraint idx -> jumpable vars (G_c)
    // Arm/disarm the deadline and reset the stride tuner (both entry points).
    void arm_deadline();

    std::chrono::steady_clock::time_point deadline_;
    // Wall-clock deadline observation state (#113). All of it is untouched, and
    // the clock unread, while has_deadline_ is false.
    std::chrono::steady_clock::time_point last_deadline_check_;
    int64_t deadline_stride_ = 1;     // iterations between clock reads
    int64_t deadline_countdown_ = 1;  // iterations left until the next one
    int64_t deadline_checks_ = 0;     // clock reads made inside the GLS loop
    // The same observation state for the structural kick (#115), kept separate
    // because the two loops advance in different units — GLS iterations there,
    // structural moves here — and interleave: a kick runs between batches, so
    // sharing one tuner would have each mis-size the other's stride. Also
    // untouched, and the clock unread, while has_deadline_ is false.
    std::chrono::steady_clock::time_point last_kick_check_;
    int64_t kick_stride_ = 1;     // structural moves between clock reads
    int64_t kick_countdown_ = 1;  // moves left until the next one
    int64_t kick_checks_ = 0;     // clock reads made inside the structural pass
    int64_t kick_moves_ = 0;      // structural moves the last kick applied
    bool has_deadline_ = false;
    int64_t iterations_ = 0;
    // Armed by the search loop once it is stuck: after `perturbation_period`
    // batches without improvement, or -- with a wall clock -- after a quarter of
    // the budget with no new best (#117). Cleared on every new best. Gates the Float escape probe
    // so it stays a last resort rather than a steady-state behaviour.
    //
    // Those two remain the ONLY arming routes. #102's unproductive-batch exit
    // deliberately does not add a third: it fires on an iteration count, so on a
    // model with no reachable feasible point the first batch already trips it,
    // and arming from there would leave the probe on from ~300 iterations into
    // the run onward -- which is the always-on regime #107 measured at 9x. Its
    // one mitigation, disarming on improvement, is no defence when the arming
    // condition recurs every batch. So an unproductive batch takes the
    // diversification kick early and nothing else; the probe still waits for
    // `perturbation_period` non-improving batches. See the kick site in
    // search.cpp, which is careful not to reset the stagnation counter that
    // clock runs on.
    bool escape_probe_ = false;

    // ---- Unproductive-batch detection (#102) ----
    //
    // A GLS batch runs batch_iterations (default 1000) iterations whether or not
    // it is achieving anything, and the outer loop only diversifies after
    // perturbation_period (default 100) non-improving batches. Before the first
    // feasible solution no batch ever improves, so that is a fixed cadence of
    // one diversification per 100 000 GLS iterations, with no feedback from the
    // search at all.
    //
    // Measured on MINLPLib st_e40 (#102): the search falls into a limit cycle
    // within ~20 iterations -- x1 flipping between two values and x3 hopping
    // between the roots of the four rows that contain it -- and then spends 92%
    // of a 155 000-iteration run on weight bumps that change nothing, visiting
    // 3 of that instance's 343 integer combinations and none of its 52 feasible
    // ones. It is not stuck at a fixed point (the weight bump does keep flipping
    // which move is improving), so only a progress measure detects it.
    //
    // The GLS weight dynamics cannot break that cycle under EITHER rho. solve()
    // draws rho from {0.95, 1.0} per batch (sample_rho), and the two draws fail
    // for different reasons -- worth writing down, because "it only happens at
    // rho = 1" would be a much smaller finding than this is.
    //
    //   * rho = 1.0. gls_update_weights is then `w += 1` on every violated row,
    //     so weights grow without bound; the trace has them at 375/324. The two
    //     rows trapping x3 are violated together and contain it with coefficient
    //     magnitude 1, so each bump adds the same +1 to both and the difference
    //     that decides x3's move is invariant. The two candidate deltas stayed
    //     pinned at exactly +47.5625 and +35.5286 for the whole run, and
    //     compute_var_jump's `fv < best_f` (best_f = 0 at the current value)
    //     rejects a positive delta.
    //   * rho = 0.95. `w *= 0.95; w += 1` is a contraction: a permanently
    //     violated row converges to the fixed point 1/(1 - 0.95) = 20 instead of
    //     growing, and a row that stops being violated decays geometrically
    //     toward zero -- though not to exactly 0.0: w * 0.95 rounds back to w at
    //     9 subnormal ulps, so the eager form sticks there, and LazyWeightDecay
    //     floors a positive weight above 0 (#175). The deltas therefore do not
    //     stay pinned -- they decay to (numerically) 0. But `fv < best_f` is
    //     STRICT, so a zero delta is rejected too, and the cycle holds for the
    //     opposite arithmetic reason.
    //
    // Either way no jump is ever improving, the loop bumps and re-bumps, and
    // nothing in FJ can tell that the assignment has stopped moving anywhere.
    //
    // So: end a batch that has gone GFJConfig::unproductive_iterations iterations
    // without pushing the measure below its best so far for that batch. This
    // bounds what a single unproductive batch can consume; it never caps a batch
    // that is still descending toward feasibility, because any new minimum
    // resets the count. The outer loop still owns what happens next -- this only
    // stops the GLS loop from burning the whole stagnation window before the
    // outer loop is allowed to look. See GFJConfig::unproductive_iterations for
    // what its default is and is not.
    //
    // ONE REGIME IS EXCLUDED, and it is not a corner case. The measure sums the
    // REAL rows only (see below for why), so once they are all satisfied it is
    // identically zero and cannot improve on itself. Read naively the exit would
    // then fire on EVERY batch of the objective-descent phase -- a stall
    // detector that is unconditionally true, which is the same shape of defect
    // as measuring a whole sum that a clamped row swallows. A batch whose
    // measure is zero is therefore never declared stuck: the search there is
    // descending against the artificial objective row, which this measure
    // deliberately cannot see, and having no signal is not evidence of being
    // stuck. `perturbation_period` keeps owning that regime, as it did before.
    //
    // THE SECOND WITNESS (watch_progress_). Excluding a measure that reads
    // exactly zero is necessary and nowhere near sufficient, and #102's ex8_6_1
    // regression is what that costs. Two things go wrong once a feasible
    // solution exists:
    //
    //   * the plateau is at POSITIVE violation, not at zero. The bound is
    //     tightened on every new best, FJ pulls the assignment off the
    //     real-feasible set to chase the objective row, and the real rows settle
    //     at an equilibrium the measure can see but cannot interpret --
    //     0.016 to 0.51 on ex8_6_1, three orders above kTol. batch_best_violation
    //     is a running MINIMUM, so "unproductive_iterations without a new
    //     all-time low" is the normal state of any such plateau;
    //   * on a model whose real rows are EQUALITIES an exactly-zero sum is
    //     essentially never observed anyway (residual |body(x)|), so the
    //     zero test is dead code there. ex8_6_1 is 45 nonlinear equalities over
    //     75 continuous variables.
    //
    // Measured, 10s at seed 42: the run improved its incumbent on 152 of its
    // first 162 batches and was then declared stuck on nine of the rest. Three
    // of the nine kicks drew LNS (lns_interval = 3), each bounded by
    // min(2.0, remaining()), and those three alone consumed 4.7s of the 10s
    // budget. 8.2k GLS iterations instead of 32k, objective -3.02 instead of
    // -8.62.
    //
    // So the exit needs a witness the measure cannot supply, and the outer loop
    // already has one: its own count of consecutive non-improving batches.
    // solve() arms this flag only once that count reaches a fraction of
    // `perturbation_period`, which makes the mechanism an ACCELERATION of the
    // stagnation window rather than a replacement for it -- one kick per
    // (that fraction + 1) batches instead of one per 100, where leaving it
    // unwitnessed gave one per ~300 iterations. A search improving its incumbent
    // every few batches never arms it at all, and is then bit-identical to a run
    // with the mechanism switched off. Arming rather than merely suppressing the
    // KICK is deliberate: an armed exit still ends batches at
    // unproductive_iterations, and on ex8_6_1 those early exits alone -- six of
    // them, no kick at all -- still cost about six gap points through the
    // changed batch cadence.
    //
    // ---- What the measure is, and why it is that ----
    //
    // Sum of the finite positive residuals of the active REAL rows.
    //
    // UNWEIGHTED. The weighted total is what the loop descends, but the GLS bump
    // raises it on every stagnant iteration without the assignment moving, so it
    // rises and falls for reasons that have nothing to do with progress.
    //
    // REAL rows only -- the artificial `obj <= bound` row is excluded. This is
    // not a refinement, it is the difference between working and not. While
    // #116's sentinel bound is installed (a feasible region containing a
    // non-finite objective: elec25/elec50 are exactly this) that row's residual
    // clamps to kInfPenalty = 1e30, where one ulp is ~1.4e14. Summed in, it
    // swallows every O(1) real row, no real improvement can move the total by
    // one bit, and EVERY batch would be declared stuck at exactly iteration 300
    // for the whole sentinel window no matter what the search was doing. That is
    // #100's defect and #118's, and search.cpp records the invariant it teaches:
    // anything comparing two assignments by violation must difference PER
    // CONSTRAINT. Two things carry that discipline here. The accumulator is
    // BUILT per constraint in update_var -- each row's own before/after are
    // differenced, so an unchanged row contributes an exact 0 rather than a
    // rounding of the whole sum -- and the clamped row is excluded outright,
    // which is how max_real_violation and LNS::state_key are safe (the comment
    // in search.cpp calls that "safe by exclusion"). Note a running MINIMUM over
    // a trajectory cannot be written as a single per-constraint difference the
    // way the structural batch's pairwise accept test could; excluding the row is
    // what makes the running form sound.
    //
    // FINITE positive residuals only. A real row can evaluate to +inf or NaN on
    // a non-convex body. Both contribute 0, on the same predicate in the fresh
    // recomputation and in the incremental update, which keeps the two
    // definitions identical and stops a single inf from turning the accumulator
    // into a NaN it can never leave (inf subtracted, then inf added back).
    //
    // ---- Why it does not drift into nonsense ----
    //
    // It is an incremental accumulator, so it rounds. #118 is that defect in its
    // pure form -- "phantom improvements" from differencing two readings taken at
    // different points in a drift cycle -- and search.cpp records that an
    // ABSOLUTE epsilon cannot filter it (x - 1e-12 == x for every x > 2^14).
    // Downward drift here would reset the streak and stop the mechanism firing;
    // upward drift would fire it early. Three things bound it:
    //
    //   1. re-grounding. refresh_unweighted_violation() runs at every gls_loop
    //      entry, so accumulated error can never exceed one batch's worth of
    //      roundings (~1e-13 relative at the default 1000 iterations) instead of
    //      a whole run's;
    //   2. a RELATIVE floor on what counts as progress (kProgressRelEps), four
    //      orders above that drift bound and far below any genuine row
    //      improvement;
    //   3. confirmation. The streak limit does not set batch_stuck_ on the
    //      accumulator's word: it recomputes exactly and re-tests. Only an exact
    //      sum ends a batch, and the recomputation re-grounds the accumulator as
    //      a side effect, so a poisoned or drifted one heals at the next limit
    //      rather than persisting. It costs one O(rows) sweep per
    //      unproductive_iterations iterations, which is why it is affordable at
    //      the decision point but not on every iteration.
    //
    // Relative floor on what counts as a new minimum; see (2) above.
    static constexpr double kProgressRelEps = 1e-9;
    double unweighted_violation_ = 0.0;
    // Armed at begin() and re-decided by solve() before every batch; see THE
    // SECOND WITNESS above. A caller driving batch() directly (the unit tests)
    // gets the mechanism armed throughout, which is what those tests want.
    bool watch_progress_ = true;
    int64_t unproductive_streak_ = 0;
    bool batch_stuck_ = false;
    // Index of the artificial `obj <= bound` row in constraint_ids(), or -1 on a
    // model with no objective. Fixed at close(); cached because the accumulator
    // consults it on the hot path.
    int32_t objective_ci_ = -1;
    // The objective row's max single-variable variation (OR-Tools'
    // row_max_variations; LinearJumpScorer::row_max_variation) over its
    // jumpable variables, built on the first objective_row_inert call that
    // needs it. NotAffine: the row has no slopes, and the skip never applies.
    // See objective_row_inert.
    enum class ObjectiveSlopes : uint8_t { Unbuilt, Built, NotAffine };
    ObjectiveSlopes objective_slopes_ = ObjectiveSlopes::Unbuilt;
    double objective_max_variation_ = 0.0;
    // objective_columns_in_box() as of the last rebuild_violated_and_scan_set.
    // False switches the finite-bound skip off until the next rebuild: a value
    // outside its box (reachable through the API: a Python `Variable.value`
    // write, then `skip_init`) can move a jump further than M. FJ's own moves
    // stay inside the box, so a rebuild is the only point it can change.
    bool objective_in_box_ = true;
    int64_t objective_skips_fj_ = 0;       // see objective_walks_skipped()
    int64_t objective_skips_novelty_ = 0;  // see novelty_objective_walks_skipped()

    // Novelty Jump state (Algorithms 4-5).
    static constexpr double kCompoundDiscount = 1.0 / 1024.0;  // epsilon (OR-tools value)
    static constexpr int64_t kNoveltyWorkBudget = 256;  // max moves applied per apply_novelty_jump
    static constexpr int64_t kNoveltySample = 3;        // vars select_novelty_var keeps (paper §4)
    int64_t nj_work_remaining_ = 0;                     // bounds compound-move search cost
    int64_t nj_moves_this_call_ = 0;                    // see novelty_moves_last_call()
    int64_t novelty_moves_ = 0;                         // see novelty_moves()
    int64_t novelty_commits_ = 0;                       // see novelty_commits()
    int64_t novelty_bumps_ = 0;                         // see novelty_weight_bumps()
    int64_t novelty_seeds_ = 0;                         // see novelty_scan_set_seeds()
    std::vector<double> novelty_weights_;               // W'
    // Rows whose W' may differ from its level-start value (W for a violated
    // row, kCompoundDiscount * W otherwise), each once: OR-Tools'
    // compound_weight_changed. See reset_changed_novelty_weights.
    std::vector<int32_t> nw_changed_;
    std::vector<uint8_t> nw_in_changed_;  // per row: on nw_changed_
    std::vector<int32_t> nj_queue_;       // novelty scan set Q
    std::vector<uint8_t> nj_in_queue_;    // per var: in the novelty scan set
    std::vector<uint8_t> on_stack_;       // per var: on the compound-move stack (the paper's T)
    struct StackMove {
        int32_t var;
        double old_value;
    };
    std::vector<StackMove> move_stack_;
};

}  // namespace cbls
