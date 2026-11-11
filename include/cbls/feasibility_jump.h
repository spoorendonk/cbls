#pragma once

#include "linear_jump.h"
#include "model.h"
#include "rng.h"
#include "violation.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cbls {

// Generalised Feasibility Jump (ViolationLS, Davies et al. CPAIOR 2024,
// Algorithms 1-3): apply the best of a sampled set of improving
// single-variable jumps, with GLS weight bumping on stagnation. Paper state
// S = <G, X, W, V, Q, J>: G = Model, X = variable values, W =
// ViolationManager::weights, V = violated constraints, Q = scan set, J =
// JumpTable. Only Bool/Int/Float are jumped; List/Set are left to structural
// moves.

// Score of a jump that improves only the objective (#218): the smallest
// positive double. Its f is 0, so JumpTable::scale_scores leaves it alone.
inline constexpr double kObjectiveTieScore = std::numeric_limits<double>::denorm_min();

// Per-variable cached jump: best value and score = -W.deltaG(v, jump_value)
// (> 0 improving). Lazily invalidated when a neighbouring variable changes
// (paper Algorithm 1).
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
    /// Multiply every cached score by `factor` when LazyWeightDecay folds a
    /// factor into the weights (#175): a score is linear in the weights.
    /// O(#vars), only on a fold. `keep_objective_ties` (#218) skips
    /// kObjectiveTieScore entries, which a factor <= 0.5 would round to 0.
    void scale_scores(double factor, bool keep_objective_ties = false) {
        for (auto& e : entries_) {
            if (e.valid && (!keep_objective_ties || e.score != kObjectiveTieScore)) {
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

// Best jump for one scalar variable; score = -delta (> 0 improving).
struct JumpResult {
    double jump_value = 0.0;
    double score = 0.0;
};

// Best jump for `var_id` (must be Bool/Int/Float) under `weights`: the
// candidate minimising the weighted violation delta, score = -delta. Float
// candidates: a Newton step toward each violated row's root (reverse-mode AD)
// plus the domain midpoint and endpoints; one call is not a converged 1-D
// minimiser. GLS weights give the FJ score, novelty weights the Novelty-Jump
// argmin. `allow_escape_probe` adds a two-sided probe at a Float stationary
// point (a last resort; see feasibility_jump.cpp).
//
// `linear` scores candidates in closed form where every weighted row of G_v
// is linear (see linear_jump.h), else falls back to
// Model::weighted_violation_delta. Same candidates and first-seen-minimum
// rule; scores agree to rounding, so the selected jump is identical only on
// integral data. `tol`: a row within it is satisfied and offers no Newton
// candidate.
//
// `objective_coef` (var_id's coefficient in an affine objective) turns on the
// tie-break (#218): among candidates with exactly the minimum f, the largest
// objective decrease wins, lexicographically, so it never overrides a
// violation decrease. An objective-only win scores kObjectiveTieScore. Between
// variables, apply_jump breaks an exact score tie on the objective gain.
JumpResult compute_var_jump(Model& model, const std::vector<double>& weights, int32_t var_id,
                            bool allow_escape_probe = false, LinearJumpScorer* linear = nullptr,
                            double tol = kDefaultFeasibilityTolerance,
                            const double* objective_coef = nullptr);

// GLS weight update (paper Algorithm 3, lines 8-10): decay all weights by rho,
// then bump every violated row by 1; rows masked to 0 stay 0. Eager,
// O(#constraints); FJ uses LazyWeightDecay (#175), and this stays as the
// reference it is tested against.
void gls_update_weights(ViolationManager& vm, double rho,
                        double tol = kDefaultFeasibilityTolerance);

// The GLS decay of Algorithm 3, stored lazily (#175); the algorithm's weights
// are unchanged. With global scale s the stored vector holds w'_c = w_c / s:
//
//   decay by rho:   s <- s * rho                      O(1), not O(#rows)
//   bump row c:     w'_c <- w'_c + 1/s  (if w'_c > 0)  O(1) per violated row
//   effective w_c:  s * w'_c
//
// Equal to the eager form to rounding (bit-identical at rho = 1, s = 1), as in
// CP-SAT's violation_ls, but trajectories differ: eager left cached scores
// stale (#175). Cached scores stay exact in the scaled space across a decay
// (docs/architecture.md).
//
// Scale-free readers: FJ's GLS loop only compares weighted sums with each
// other or 0, so it reads w' directly. `active()` reads w' > 0 (s > 0).
//
// Renormalisation: s stays in [kMinScale, kMaxScale]; a decay that would leave
// it folds s * rho into every stored weight (O(#rows)) and resets s = 1. A
// weighted term is at most ~1e60x its effective weight, far from overflow; at
// rho = 0.95 a fold happens once per 1347 decays. A rho the scale cannot
// absorb (0, negative, NaN, one-step underflow) takes the fold, i.e. the eager
// update.
//
// Weight 0 stays exactly 0 and a positive weight stays positive: an
// underflowing fold floors at the smallest subnormal, since a 0 row is masked
// for good. Only rho = 0 zeroes a positive weight, as eager `w * 0` does.
//
// Between folds the stored vector is not the effective weights: the owner
// must `materialise` before anything else reads them. FJ does so on every
// exit from its GLS loop.
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
    // Best-of-N ApplyJump sample: ViolationLS §3 uses 5 for a Feasibility Jump
    // in either phase (#211); 3 is its Novelty sample (kNoveltySample).
    int sample_size_linear = 5;
    int sample_size_general = 5;
    double rho = 0.95;  // GLS decay; caller samples from {0.95, 1.0} per batch
    // A row with residual <= this is satisfied: out of V, never bumped, no
    // Newton candidate. solve() sets it to SearchConfig::feasibility_tolerance
    // (#211).
    double tolerance = kDefaultFeasibilityTolerance;
    // OR-Tools' use of the objective before the first feasible point (#218):
    // ComputeScore's epsilon (a violation tie goes to the candidate lowering
    // the objective most; see compute_var_jump) and ShouldScan's
    // HasBetterObjectiveValue (a variable not at its objective-better bound stays
    // in Q). Off here (the pre-#218 path bit for bit); solve() turns it on via
    // SearchConfig::fj_objective_tie_break (#226). Needs an affine objective,
    // read once at construction; otherwise inert. FJ batches only: Novelty and
    // LNS repair do not take it.
    bool objective_tie_break = false;
    bool two_phase = true;       // GLS on linear submodel first, then full model
    bool set_initial_x = true;   // set X[v] to the domain value closest to 0 first
    int64_t max_iterations = 0;  // 0 = unbounded (bounded by time_limit)
    double time_limit = 0.0;     // seconds; 0 = no limit
    // End a batch after this many GLS iterations without reducing the real
    // rows' unweighted violation below the batch's best; <= 0 disables. Batch
    // API only: gls()/run() have no outer loop to return to. See
    // unweighted_violation_. 300 is a coarse grid pick on MINLPLib, not a tuned
    // optimum, and is dimensionless on a quantity whose scale is not (#102,
    // #145).
    int64_t unproductive_iterations = 300;
};

enum class GFJStatus : std::uint8_t { Feasible, Unsolved };

class FeasibilityJump {
public:
    FeasibilityJump(Model& model, ViolationManager& vm, RNG& rng, GFJConfig config = {});

    // Run GLS until feasible or the iteration/time budget is spent (single-shot
    // use).
    GFJStatus run();

    // Batch API for the ViolationLS outer loop (Algorithm 6). The caller owns
    // the loop: begin() once, then batch() repeatedly, calling reset_weights()
    // on a new best (after tightening the objective bound) and perturb() on
    // stagnation. set_rho() re-randomises the GLS decay between batches.
    void begin(bool set_initial_x);
    bool batch(int64_t batch_iterations);  // true if feasible (no active violated)
    // Whether the last batch() ended unproductive (GFJConfig::
    // unproductive_iterations) rather than on budget or deadline. Valid only
    // straight after batch(); cleared at batch entry.
    [[nodiscard]] bool batch_stuck() const { return batch_stuck_; }
    void reset_weights();  // W <- 1 and rebuild the scan set
    void resync();         // rebuild the scan set from current state, keep weights
    // Randomise each jumpable var w.p. p, then apply
    // clamp(round(p*|elements|), 1, |elements|) random structural moves to each
    // List/Set var (#111); if nothing moved, force one variable, a scalar if
    // any can move (#109). The fallback leaves large models, and models
    // without List/Set, bit-identical to plain p-draws.
    void perturb(double probability);
    void set_rho(double rho) { config_.rho = rho; }
    // Armed by the search loop once it has stagnated; see solve().
    void set_escape_probe(bool on) { escape_probe_ = on; }
    // The armed state, for the caller and its tests.
    [[nodiscard]] bool escape_probe() const { return escape_probe_; }
    // Arm/disarm the unproductive-batch exit for the next batch. Armed at
    // begin(); solve() re-decides it before every batch (see the second-witness
    // note on unweighted_violation_).
    void set_watch_progress(bool on) { watch_progress_ = on; }
    // Unweighted violation of the active real rows (see
    // unweighted_violation_). For tests; the search does not read it.
    [[nodiscard]] double unweighted_violation() const { return unweighted_violation_; }

    /// Incremental violated-row state, for tests (#174); the search never reads
    /// these.
    ///  - `row_violated(ci)`: row `ci` is in V;
    ///  - `violated_rows()`: a sorted copy of V (kept in flip order);
    ///  - `active_violated_rows_of(v)`: rows listing `v` that are violated and
    ///    active (weight > 0) as of their last reconcile; 0 for a structured
    ///    variable.
    /// `row_violated` and `active_violated_rows_of` throw std::out_of_range on an
    /// unknown index.
    [[nodiscard]] bool row_violated(int32_t ci) const;
    [[nodiscard]] std::vector<int32_t> violated_rows() const;
    [[nodiscard]] int32_t active_violated_rows_of(int32_t var_id) const;

    /// For tests (#206): true iff Q and Q' (Novelty's) hold no duplicate, their
    /// membership flags match their contents, and no var on Novelty's
    /// compound-move stack is in Q'. Valid between calls. O(|V| + |Q| + |Q'|).
    [[nodiscard]] bool scan_sets_consistent() const;

    /// Compare every incremental structure against a from-scratch recompute
    /// (#225); one line per disagreement, empty when consistent. For tests via
    /// SearchAuditor; valid between calls. Checks node values, V (its index,
    /// counted bits and per-variable counts), every valid jump entry's value and
    /// score, and that no improving jump is missing from Q. Not the unweighted
    /// progress accumulator, which rounds by design.
    /// Values agree within 1e-9 * (1 + |fresh|), plus a drifting Sum's tracked
    /// bound; scores within 1e-7 * (1 + |cached| + |fresh|), since the closed
    /// form agrees with the probe only to rounding.
    /// O(#nodes + #rows + jumpable vars x |G_v|) and a model copy per call.
    [[nodiscard]] std::vector<std::string> audit() const;

    /// The cached jump entry for `var_id`, or nullopt when invalid. For tests
    /// (#210); throws std::out_of_range on an unknown index.
    [[nodiscard]] std::optional<JumpResult> cached_jump(int32_t var_id) const;
    /// Objective-row neighbour walks skipped as inert since construction (#210,
    /// #209): update_var's, and Novelty's re-queues. Diagnostics.
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
    // The GLS loop reads the clock on a stride sized in time: at most
    // kStrideBudgetFraction of the budget, or one GLS iteration if larger.
    // That is a batch's worst-case overrun.
    static constexpr double kStrideBudgetFraction = 1.0 / 64.0;
    static constexpr int64_t kStrideGrowth = 8;  // max growth per adjustment
    // Iteration cap on the stride, and why the overrun is bounded: a stride
    // grown while iterations were cheap is spent in full once they turn
    // expensive, since a shrink applies only at a check. 64 was the old fixed
    // stride, so the worst case is no worse than before (#113). Costs ~1.2%
    // throughput on cheap iterations (#113).
    static constexpr int64_t kMaxDeadlineStride = 64;
    // Pure function of the last measurement; exposed for testing.
    static int64_t next_deadline_stride(int64_t stride, double elapsed_seconds,
                                        double target_seconds);
    // `deadline_checks()` counts clock reads inside the GLS loop (the kick's
    // structural pass counts its own: structural_kick_checks());
    // `deadline_check_stride()` is the live stride. Both paths read the clock
    // only under `has_deadline_`.
    [[nodiscard]] int64_t deadline_checks() const { return deadline_checks_; }
    [[nodiscard]] int64_t deadline_check_stride() const { return deadline_stride_; }

    // ---- The structural kick's own deadline bound (#115) ----
    //
    // The last perturb()'s structural pass, counted in structural moves (one
    // move-set generation plus one apply). `structural_kick_checks()` is 0 on a
    // run with no wall clock. Excluded: force_structural_move's fallback move (a
    // kick's worst case is one move more) and arm_structural_kick's one clock
    // read per kick.
    [[nodiscard]] int64_t structural_kick_moves() const { return kick_moves_; }
    [[nodiscard]] int64_t structural_kick_checks() const { return kick_checks_; }
    /// The closed-form linear scorer (its counters; tests check its scores).
    /// `prepare` touches only its own scratch, never the assignment.
    [[nodiscard]] const LinearJumpScorer& linear_scorer() const { return linear_; }
    [[nodiscard]] LinearJumpScorer& linear_scorer() { return linear_; }
    [[nodiscard]] int64_t structural_kick_stride() const { return kick_stride_; }

    // Novelty Jump (paper Algorithms 4-5): one stand-alone ApplyNoveltyJump
    // (Algorithm 4) from a fresh W' and Q', capped at novelty_work_budget()
    // applied moves, each charging 3 to iterations(). Committed compound moves
    // stay applied; returns true if feasible. Call from a local optimum with
    // violated_/weights current; the caller must resync() afterwards. W' =
    // kCompoundDiscount*W on rows satisfied at entry, W on violated ones. The
    // search uses novelty_batch() instead.
    bool apply_novelty_jump();
    /// One Novelty Jump batch (#209): GLS with ApplyNoveltyJump as its move
    /// (ViolationLS Algorithm 6 line 22, Algorithm 3 with M = Algorithm 4).
    /// When ApplyNoveltyJump finds no compound move at its largest discrepancy
    /// budget, the GLS weights are decayed and bumped as in batch(). W' and Q'
    /// are set up once per batch and kept incrementally. Each applied move costs
    /// kNoveltySample = 3 iterations (undos 0), each bump 1; bounded by
    /// `batch_iterations` (<= 0: none), the deadline (polled per move and bump)
    /// and GFJConfig::max_iterations. Returns true if no active row is violated.
    /// Leaves V, Q and the jump table current: no resync() needed.
    bool novelty_batch(int64_t batch_iterations);
    /// Moves (applies, not undos) the last apply_novelty_jump() made, and the
    /// cap on them (#206).
    [[nodiscard]] int64_t novelty_moves_last_call() const { return nj_moves_this_call_; }
    [[nodiscard]] static constexpr int64_t novelty_work_budget() { return kNoveltyWorkBudget; }
    /// Cumulative Novelty moves applied, compound moves committed and GLS bumps
    /// since construction. Diagnostics.
    [[nodiscard]] int64_t novelty_moves() const { return novelty_moves_; }
    [[nodiscard]] int64_t novelty_commits() const { return novelty_commits_; }
    [[nodiscard]] int64_t novelty_weight_bumps() const { return novelty_bumps_; }
    /// Row `ci`'s novelty weight W' as Novelty last left it, for tests (#209).
    /// Throws std::out_of_range on an unknown row or before W' is sized.
    [[nodiscard]] double novelty_weight(int32_t ci) const;
    /// For tests (#209): true iff every row not noted for the next level reset
    /// holds its level-start W' (W[c] if violated, kCompoundDiscount * W[c]
    /// otherwise) and the noted list matches its flags; the invariant that lets
    /// a level reset visit only noted rows. Valid between calls. O(#rows).
    [[nodiscard]] bool novelty_weights_consistent() const;
    /// Times Q' has been seeded from V's rows since construction: once per
    /// apply_novelty_jump() or novelty_batch() (#209).
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
    // A budget exit's status: Feasible only if nothing active is violated, read
    // after re-grounding (#188). Every budget exit of gls_loop uses it.
    [[nodiscard]] GFJStatus batch_end_status() {
        reground_drifted_rows();
        return any_active_violated() ? GFJStatus::Unsolved : GFJStatus::Feasible;
    }
    // ---- The incremental Sums' drift (#188) ----
    //
    // Whether row c's verdict is undecided: it reads a drifting incremental Sum
    // directly (row_slot_) and the interval of width drift_bound around
    // `residual` straddles GFJConfig::tolerance (0 for a domain row, #258).
    [[nodiscard]] bool verdict_undecided(int32_t c, double residual) const;
    [[nodiscard]] bool domain_verdict_undecided(double residual, double drift_bound) const;
    // Put row c on uncertain_rows_ if undecided; called wherever a row's value
    // or its Sum's drift changes in a batch.
    void note_row_certainty(int32_t c, double residual);
    // The local-minimum gate: re-sum the Sum under each still-undecided row and
    // settle the rows that moved. True when V changed, so the caller re-samples
    // instead of bumping or declaring feasibility.
    bool settle_undecided_rows();
    // Re-sum every Sum that drifted since the last call and settle moved rows;
    // true when V changed. Called at batch end, Novelty entry and every
    // Feasible verdict.
    bool reground_drifted_rows();
    // After reground_inc_sum(s) of slot_scratch_: settle each moved row as
    // update_var does, from row_before_. True if V changed.
    bool settle_regrounded_rows();
    // Snapshot the rows of `slot` into row_before_ ahead of its re-sum.
    void capture_rows_of_slot(int32_t slot);
    void build_row_slots();
    // No improving jump in the scan set: bump the violated rows' weights and
    // re-queue their variables.
    void bump_weights_and_requeue();
    // Fold the lazy decay's scale into vm_.weights and the cached scores
    // (#175); every exit from gls_loop calls it, exceptional ones included.
    void materialise_weights();
    // Fold the last iteration into the batch's progress: running minimum of the
    // unweighted real-row violation and the streak since it moved. True when
    // the streak hits its limit and an exact re-grounding confirms it.
    bool track_batch_progress(double& batch_best_violation, bool watch_progress);
    // Read the clock and re-size the stride; true if the deadline passed. Only
    // on countdown expiry, off the per-iteration path.
    bool deadline_passed_and_retune();
    // Recompute unweighted_violation_ from scratch (finite positive residuals
    // of active real rows). The only re-grounding of the accumulator; called at
    // gls_loop entry, before batch_stuck_ is set, and in
    // rebuild_violated_and_scan_set.
    void refresh_unweighted_violation();
    // ApplyJump (Algorithm 2): sample up to `sample_size` vars from Q, apply the
    // best improving jump via update_var. Returns false if none improves.
    bool apply_jump(int sample_size);
    // UpdateVar (Algorithm 1): commit X[v] <- jump, refresh V, invalidate
    // neighbour jumps, replenish Q.
    void update_var(int32_t var_id);
    // Whether moving the objective's value from `before` to `after` leaves every
    // cached jump and Novelty verdict of the row's other variables unchanged,
    // so the neighbour walk can be skipped (#210). See the definition for the
    // cost argument.
    [[nodiscard]] bool objective_row_inert(double before, double after);
    // The objective's value as the DAG holds it now; 0.0 without an objective row.
    [[nodiscard]] double objective_value() const;
    // Whether every jumpable objective column lies in its declared box (the
    // finite-bound skip's premise). O(objective support).
    [[nodiscard]] bool objective_columns_in_box() const;

    [[nodiscard]] bool active(int32_t constraint_idx) const;  // weight > 0
    [[nodiscard]] bool jumpable(int32_t var_id) const;        // scalar var
    // Uniform jumpable var with at least two domain values, perturb()'s
    // fallback; -1 if none (then a no-op kick is correct).
    int32_t pick_forced_perturb_var();
    // The List/Set half of a kick (#111): clamp(round(p * |elements|), 1,
    // |elements|) typed moves per structure, plus a ListPartition member's
    // inter-list moves (#164). True if any variable's elements net changed (set
    // equality for a Set). Draws nothing on a model without List/Set.
    // Deadline-bounded between moves (#115).
    bool perturb_structural(double probability);
    // Reset the kick's move counter and stride tuner, once per pass.
    void arm_structural_kick();
    // True when the structural pass must stop on the deadline (strided in
    // moves). Never before the first move, so a kick is never a no-op (#109).
    bool kick_past_deadline();
    // Apply one structural move to some List/Set that can take one (the
    // structural pick_forced_perturb_var); false if none can.
    bool force_structural_move();
    [[nodiscard]] bool participates_in_active_violated(int32_t var_id) const {
        return active_violated_of_var_[static_cast<size_t>(var_id)] > 0;
    }
    void rebuild_violated_and_scan_set();
    // ---- Incremental violated-row state (#174); see violated_rows_ ----
    // Move row `c` into or out of V; staying in, reconcile its counted bit with
    // the live active(c).
    void set_violated(int32_t c, bool now);
    // Bring row c's counted bit in line with violated && active(c), adjusting
    // active_violated_of_var_ over the row's variable list if it changes.
    void reconcile_counted(int32_t c);
    // Re-derive the list, positions, counted bits and per-variable counts from
    // violated_'s in-V bits. O(#constraints + nonzeros of counted rows).
    void rebuild_violated_index();
    // reconcile_counted over V, O(|V|); at every gls_loop entry, since weights
    // may change between batches.
    void reconcile_all_counted();
    // Drop row c's contribution to active_violated_of_var_ if it is counted,
    // leaving it in V (kInV kept).
    void uncount(int32_t c);
    void set_initial_assignment();
    void compute_linear_constraints();
    /// Throw unless every per-row and per-variable table, and the
    /// ViolationManager's weights, are sized for the model as it is now
    /// (checked per batch; see the definition).
    void require_tables_in_step() const;
    // audit()'s three parts (#225): node values, V and its counts, and the jump
    // table with Q's completeness; audit_admits is Q's rule read off `fresh`.
    void audit_nodes(const Model& fresh, std::vector<std::string>& out) const;
    void audit_violated_set(std::vector<std::string>& out) const;
    void audit_counts(const std::vector<int32_t>& counts, size_t in_v,
                      std::vector<std::string>& out) const;
    void audit_jump_table(Model& fresh, std::vector<std::string>& out) const;
    [[nodiscard]] bool audit_admits(int32_t v, const Model& fresh) const;
    void enqueue(int32_t var_id);

    // Novelty Jump internals (Algorithm 5): a candidate var, its W'-argmin jump
    // and both scores.
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
    // Set up W', Q' and the stack for a Novelty run: the per-batch O(#rows) W'
    // init and O(nnz(V)) seed.
    void begin_novelty_run();
    void init_novelty_weights();
    void seed_novelty_scan_set();
    void nj_enqueue(int32_t var_id);
    // `objective_before`: the objective's value before v's move or undo
    // (objective_value()), for the inert-row skip.
    void nj_requeue_neighbours(int32_t v, double objective_before);
    NoveltyPick select_novelty_var(double s_m, double s_c);
    bool novelty_jump_search(double s_m, int budget);
    // Algorithm 5 lines 11-16 and 20-21: one leg applied, and one undone.
    void nj_apply(int32_t v, double jump);
    void nj_undo(int32_t v, double old_value);
    // Algorithm 4 over the state begin_novelty_run() set up, incrementally.
    NoveltyOutcome novelty_descent();
    // Algorithm 4 lines 3-5 for the next discrepancy level, in O(rows whose W'
    // moved off its level-start value) rather than O(#rows).
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
    // The GLS decay's lazy scale (#175); s != 1 only inside gls_loop.
    LazyWeightDecay weight_decay_;
    // ---- V, the violated set, kept incrementally (#174) ----
    //
    // Per constraint two bits: kInV (in V) and kCounted (in V and active,
    // weight > 0, when last evaluated; counted in active_violated_of_var_).
    // kCounted implies kInV, so `violated_[c] != 0` reads "in V". Written only
    // through set_violated, reconcile_counted, uncount and
    // rebuild_violated_index. Derived:
    //   - violated_rows_ + violated_pos_: V as a dense list with a position index
    //     (-1 = absent), swap-removed on the way out;
    //   - active_violated_of_var_: per variable, the counted rows whose
    //     vars_of_constraint_ list names it, so participates_in_active_violated
    //     is O(1) instead of O(|G_v|).
    //
    // Cost: any_active_violated, the Novelty seed and the bump scan are O(|V|)
    // plus the variable lists of the rows they queue from, instead of
    // O(#rows), + one O(#rows + #vars) fold per gls_loop exit that decayed; V
    // is visited in flip order (deterministic, not sorted). A
    // Novelty batch stays O(#rows) once per batch (#209). The price is a
    // constant per row flip (a push or swap-remove plus a walk of the row's
    // variable list), paid by Novelty's apply and undo too; it loses where
    // almost every row flips on every move, and on Novelty probes over long rows.
    // Memory per worker: 4 B per row, up to 4 B more per row of violated_rows_
    // capacity, and 4 B per variable.
    //
    // "Active" is read live: set_violated, the bump and reconcile_all_counted
    // (every gls_loop entry) re-read it, so a weight changed between batches is
    // picked up. Nothing may change a weight while gls_loop runs.
    static constexpr uint8_t kInV = 1;
    static constexpr uint8_t kCounted = 2;
    std::vector<uint8_t> violated_;                // per constraint: kInV | kCounted
    std::vector<int32_t> violated_rows_;           // V, dense
    std::vector<int32_t> violated_pos_;            // per constraint: index in violated_rows_, or -1
    std::vector<int32_t> active_violated_of_var_;  // per var: counted rows listing it
    std::vector<uint8_t> in_queue_;                // per var: in Q
    std::vector<int32_t> queue_;                   // scan set Q (vars with possibly-positive score)
    // ---- The incremental Sums' drift (#188) ----
    // Row -> slot of the incremental Sum it reads directly, or -1; and slot ->
    // rows as CSR (an MPS range row is two rows over one Sum). Built in the
    // constructor: 4 B per row and per slot.
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
    // Closed-form scoring over linear rows; its eligibility is set with
    // is_linear_, in compute_linear_constraints.
    LinearJumpScorer linear_;
    std::vector<std::vector<int32_t>> vars_of_constraint_;  // constraint idx -> jumpable vars (G_c)
    // Arm/disarm the deadline and reset the stride tuner (both entry points).
    void arm_deadline();

    std::chrono::steady_clock::time_point deadline_;
    // Wall-clock deadline observation (#113); untouched, and the clock unread,
    // while has_deadline_ is false.
    std::chrono::steady_clock::time_point last_deadline_check_;
    int64_t deadline_stride_ = 1;     // iterations between clock reads
    int64_t deadline_countdown_ = 1;  // iterations left until the next one
    int64_t deadline_checks_ = 0;     // clock reads made inside the GLS loop
    // The same for the structural kick (#115), separate because the loops
    // advance in different units and interleave. Also untouched while
    // has_deadline_ is false.
    std::chrono::steady_clock::time_point last_kick_check_;
    int64_t kick_stride_ = 1;     // structural moves between clock reads
    int64_t kick_countdown_ = 1;  // moves left until the next one
    int64_t kick_checks_ = 0;     // clock reads made inside the structural pass
    int64_t kick_moves_ = 0;      // structural moves the last kick applied
    bool has_deadline_ = false;
    int64_t iterations_ = 0;
    // Armed by the search loop once stuck: after `perturbation_period` batches
    // without improvement, or a quarter of a wall-clock budget with no new best
    // (#117); cleared on every new best. Gates the Float escape probe, keeping it
    // a last resort. The unproductive-batch exit (#102) is deliberately not a
    // third route: it would leave the probe always on (#107).
    bool escape_probe_ = false;

    // ---- Unproductive-batch detection (#102) ----
    //
    // Without it, before the first feasible solution the kick cadence is a
    // fixed one per perturbation_period * batch_iterations iterations. GLS
    // weights alone cannot break a limit cycle under either rho (#102), so a
    // batch ends after GFJConfig::unproductive_iterations iterations without a
    // new minimum of this measure; any new minimum resets the count, so a
    // descending batch is never cut.
    //
    // A batch whose measure is zero is never declared stuck: the measure sums
    // real rows only, so once they are satisfied it cannot improve, and the
    // search there descends against the objective row. Exact zero is not
    // enough after feasibility either (a positive equilibrium; equalities
    // never reach 0), so the exit also needs the outer loop's witness
    // (watch_progress_): solve() arms it only after perturbation_period / 20
    // non-improving batches, making it an acceleration of the stagnation window,
    // not a replacement. A search that keeps improving never arms it and is
    // bit-identical to one without the mechanism.
    //
    // The measure: sum of the finite positive residuals of the active real
    // rows. Unweighted, because GLS bumps move the weighted total without the
    // assignment moving. Real rows only: the `obj <= bound` row can clamp at
    // kInfPenalty = 1e30 (#116), which would swallow every real row (#100,
    // #118). Built per constraint in update_var (each row's own before/after
    // differenced), so an unchanged row contributes exactly 0. +inf/NaN
    // residuals contribute 0, the same in the fresh and incremental forms.
    //
    // Drift is bounded three ways: re-grounding at every gls_loop entry (error
    // stays within one batch's roundings); a relative progress floor
    // (kProgressRelEps); and an exact recompute before batch_stuck_ is set,
    // one O(rows) sweep per unproductive_iterations iterations.
    //
    // Relative floor on what counts as a new minimum.
    static constexpr double kProgressRelEps = 1e-9;
    double unweighted_violation_ = 0.0;
    // Armed at begin() and re-decided by solve() before every batch (the
    // second witness above); stays armed for a caller driving batch() directly.
    bool watch_progress_ = true;
    int64_t unproductive_streak_ = 0;
    bool batch_stuck_ = false;
    // Index of the `obj <= bound` row in constraint_ids(), or -1 without an
    // objective; fixed at close(), cached for the hot path.
    int32_t objective_ci_ = -1;
    // The objective row's max single-variable variation over its jumpable
    // variables (OR-Tools' row_max_variations), built on first need by
    // objective_row_inert. NotAffine: no slopes, the skip never applies.
    enum class ObjectiveSlopes : uint8_t { Unbuilt, Built, NotAffine };
    ObjectiveSlopes objective_slopes_ = ObjectiveSlopes::Unbuilt;
    // Per-variable coefficient of an affine objective (#218), read at
    // construction under GFJConfig::objective_tie_break; empty = off.
    std::vector<double> objective_coef_;
    // compute_var_jump's tie-break handle for var_id; null when off.
    [[nodiscard]] const double* objective_tie(int32_t var_id) const {
        return (config_.objective_tie_break && !objective_coef_.empty())
                   ? &objective_coef_[static_cast<size_t>(var_id)]
                   : nullptr;
    }
    // Whether var_id can still lower the objective inside its box (OR-Tools'
    // HasBetterObjectiveValue): what keeps it in Q under objective_tie_break.
    [[nodiscard]] bool objective_scan(int32_t var_id) const;
    // Q's membership rule: a variable of an active violated row, or, under
    // objective_tie_break, one objective_scan admits.
    [[nodiscard]] bool should_scan(int32_t var_id) const {
        return participates_in_active_violated(var_id) ||
               (config_.objective_tie_break && objective_scan(var_id));
    }
    double objective_max_variation_ = 0.0;
    // objective_columns_in_box() as of the last rebuild_violated_and_scan_set.
    // False disables the finite-bound skip until the next rebuild: an
    // out-of-box value (via the API plus `skip_init`) can move a jump further
    // than M. FJ's own moves stay in the box.
    bool objective_in_box_ = true;
    int64_t objective_skips_fj_ = 0;       // see objective_walks_skipped()
    int64_t objective_skips_novelty_ = 0;  // see novelty_objective_walks_skipped()

    // Novelty Jump state (Algorithms 4-5).
    static constexpr double kCompoundDiscount = 1.0 / 1024.0;  // epsilon (OR-tools value)
    static constexpr int64_t kNoveltyWorkBudget = 256;  // max moves applied per apply_novelty_jump
    static constexpr int64_t kNoveltySample = 3;        // vars select_novelty_var keeps (paper §4)
    int64_t nj_work_remaining_ = 0;                     // bounds compound-move search cost
    int64_t nj_moves_this_call_ = 0;                    // see novelty_moves_last_call()
    int64_t nj_move_cost_ = 1;             // work units a move takes from nj_work_remaining_
    int64_t novelty_moves_ = 0;            // see novelty_moves()
    int64_t novelty_commits_ = 0;          // see novelty_commits()
    int64_t novelty_bumps_ = 0;            // see novelty_weight_bumps()
    int64_t novelty_seeds_ = 0;            // see novelty_scan_set_seeds()
    std::vector<double> novelty_weights_;  // W'
    // Rows whose W' may differ from its level-start value, each once (OR-Tools'
    // compound_weight_changed). See reset_changed_novelty_weights.
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
