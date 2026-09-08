#pragma once

#include "model.h"
#include "rng.h"
#include "violation.h"

#include <chrono>
#include <cstdint>
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
JumpResult compute_var_jump(Model& model, const std::vector<double>& weights, int32_t var_id,
                            bool allow_escape_probe = false);

// Guided Local Search weight update (paper Algorithm 3, lines 8-10): decay all
// weights by rho, then bump every currently-violated constraint by 1. Weights
// of constraints masked to 0 (e.g. non-linear constraints in the linear phase)
// stay 0 under decay and are never bumped while satisfied.
void gls_update_weights(ViolationManager& vm, double rho);

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
    // optimum, has no second roster behind it, and nothing here establishes that
    // 300 transfers off MINLPLib.
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
    [[nodiscard]] int64_t structural_kick_stride() const { return kick_stride_; }

    // Novelty Jump (paper Algorithms 4-5): a bounded-backtracking compound-move
    // search that escapes local optima single-variable FJ cannot (chained-
    // invariant fixes). Commits the improving compound move(s) it finds (left
    // applied) and returns true if it reaches feasibility, else leaves any
    // committed moves applied and returns false. Call from a local optimum with
    // violated_/weights current (e.g. right after begin() or a stalled batch);
    // the caller must resync() afterwards. Uses novelty weights W' =
    // kCompoundDiscount*W for constraints not violated at entry, full W for
    // those violated at entry.
    bool apply_novelty_jump();

private:
    // One GLS pass over the constraints whose weight is currently > 0 (the
    // "active" set). Returns Feasible if all active constraints are satisfied.
    GFJStatus gls(int sample_size);
    // GLS inner loop reusing current state, bounded by a per-call iteration
    // limit (<=0 for none) plus the global budget/deadline.
    GFJStatus gls_loop(int sample_size, int64_t batch_iter_limit);
    [[nodiscard]] bool any_active_violated() const;
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
    // instead (#111). Returns true if any variable's elements NET changed —
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
    [[nodiscard]] bool participates_in_active_violated(int32_t var_id) const;
    void rebuild_violated_and_scan_set();
    void set_initial_assignment();
    void compute_linear_constraints();
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
    void init_novelty_weights();
    void seed_novelty_scan_set();
    void nj_enqueue(int32_t var_id);
    NoveltyPick select_novelty_var(double s_m, double s_c);
    bool novelty_jump_search(double s_m, int budget);

    Model& model_;
    ViolationManager& vm_;
    RNG& rng_;
    GFJConfig config_;

    JumpTable jumps_;
    std::vector<uint8_t> violated_;   // per constraint: in V
    std::vector<uint8_t> in_queue_;   // per var: in Q
    std::vector<int32_t> queue_;      // scan set Q (vars with possibly-positive score)
    std::vector<int32_t> examined_;   // scratch: distinct vars sampled in one apply_jump
    std::vector<uint8_t> is_linear_;  // per constraint
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
    //     growing, and a row that stops being violated decays geometrically to
    //     zero (and to exactly 0.0, at which point active() masks it out
    //     entirely). The deltas therefore do not stay pinned -- they decay to
    //     exactly 0. But `fv < best_f` is STRICT, so a zero delta is rejected
    //     too, and the cycle holds for the opposite arithmetic reason.
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
    // way structural_pass's pairwise accept test could; excluding the row is
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

    // Novelty Jump state (Algorithms 4-5).
    static constexpr double kCompoundDiscount = 1.0 / 1024.0;  // epsilon (OR-tools value)
    static constexpr int64_t kNoveltyWorkBudget = 256;  // max moves applied per apply_novelty_jump
    int64_t nj_work_remaining_ = 0;                     // bounds compound-move search cost
    std::vector<double> novelty_weights_;               // W'
    std::vector<int32_t> nj_queue_;                     // novelty scan set Q
    std::vector<uint8_t> nj_in_queue_;                  // per var: in the novelty scan set
    std::vector<uint8_t> on_stack_;  // per var: on the compound-move stack (the paper's T)
    struct StackMove {
        int32_t var;
        double old_value;
    };
    std::vector<StackMove> move_stack_;
};

}  // namespace cbls
