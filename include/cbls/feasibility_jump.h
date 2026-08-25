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

    bool valid(int32_t var_id) const { return entries_[var_id].valid; }
    void invalidate(int32_t var_id) { entries_[var_id].valid = false; }
    void invalidate_all() {
        for (auto& e : entries_) {
            e.valid = false;
        }
    }
    void set(int32_t var_id, double jump_value, double score) {
        entries_[var_id] = {jump_value, score, true};
    }
    double jump_value(int32_t var_id) const { return entries_[var_id].jump_value; }
    double score(int32_t var_id) const { return entries_[var_id].score; }

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
};

enum class GFJStatus { Feasible, Unsolved };

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
    void reset_weights();                  // W <- 1 and rebuild the scan set
    void resync();                         // rebuild the scan set from current state, keep weights
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
    bool escape_probe() const { return escape_probe_; }
    bool all_satisfied() const;
    int64_t iterations() const { return iterations_; }  // total GLS iterations since begin()

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
    int64_t deadline_checks() const { return deadline_checks_; }
    int64_t deadline_check_stride() const { return deadline_stride_; }

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
    int64_t structural_kick_moves() const { return kick_moves_; }
    int64_t structural_kick_checks() const { return kick_checks_; }
    int64_t structural_kick_stride() const { return kick_stride_; }

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
    bool any_active_violated() const;
    // ApplyJump (Algorithm 2): sample up to `sample_size` vars from Q, apply the
    // best improving jump via update_var. Returns false if none improves.
    bool apply_jump(int sample_size);
    // UpdateVar (Algorithm 1): commit X[v] <- jump, refresh V, invalidate
    // neighbour jumps, replenish Q.
    void update_var(int32_t var_id);

    bool active(int32_t constraint_idx) const;  // weight > 0
    bool jumpable(int32_t var_id) const;        // scalar var
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
    bool participates_in_active_violated(int32_t var_id) const;
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
    bool escape_probe_ = false;

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
