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
    void perturb(double probability);      // randomise each jumpable var w.p. p
    void set_rho(double rho) { config_.rho = rho; }
    // Armed by the search loop once it has stagnated; see solve().
    void set_escape_probe(bool on) { escape_probe_ = on; }
    bool all_satisfied() const;
    int64_t iterations() const { return iterations_; }  // total GLS iterations since begin()

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
    std::chrono::steady_clock::time_point deadline_;
    bool has_deadline_ = false;
    int64_t iterations_ = 0;
    // Armed by the search loop after `perturbation_period` batches without
    // improvement, cleared on every new best. Gates the Float escape probe so
    // it stays a last resort rather than a steady-state behaviour.
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
