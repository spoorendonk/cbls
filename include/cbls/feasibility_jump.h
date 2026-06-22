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

// Compute the best jump for `var_id` under the current weights `vm`. For Float
// variables the domain minimisation is a 1-D convex search (golden section) on
// the weighted violation (the paper assumes per-variable convexity). `var_id`
// must be a scalar (Bool/Int/Float) variable.
JumpResult compute_var_jump(Model& model, const ViolationManager& vm, int32_t var_id);

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

    // Run GLS until feasible, or until the iteration/time budget is exhausted.
    GFJStatus run();

private:
    // One GLS pass over the constraints whose weight is currently > 0 (the
    // "active" set). Returns Feasible if all active constraints are satisfied.
    GFJStatus gls(int sample_size);
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
};

}  // namespace cbls
