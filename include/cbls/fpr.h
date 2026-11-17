#pragma once

// Fix-propagate-repair (Salvagnin, Roberti, Fischetti, MPC 2024), LP-free:
// one attempt fixes integral variables one at a time under `DomainPropagator`,
// after Fig. 1 and spoorendonk/mip-heuristics' `fpr_attempt_step` (#425).
// Repair is the caller's: it is told whether the mode wants one (#426).

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace cbls {

class DomainPropagator;
class LinearRowStore;
class Model;
class RNG;

/// Variable ranking (paper Table 1). Clique-cover orders need a clique table
/// and an LP centre, which cbls has neither of, so they are out.
enum class FprVarOrder : std::uint8_t {
    LR,          ///< formulation (id) order
    Type,        ///< binary, then other integral, then Float; id order within
    Random,      ///< Type's buckets, each shuffled
    Locks,       ///< Type's buckets, each by max(up, down) locks, descending
    DomainSize,  ///< dynamic: the smallest unfixed integral domain at each node
};

/// Value selection (paper Table 2, LP-free), on the variable's current box.
enum class FprValueRule : std::uint8_t {
    Up,         ///< the upper bound
    Random,     ///< uniform in the box, rounded on an integral variable
    GoodObj,    ///< the bound that lowers the (min-form) objective; lb at zero cost
    BadObj,     ///< the other one; lb at zero cost
    LooseDyn,   ///< the direction with fewer dynamic locks; ties as GoodObj
    Incumbent,  ///< the variable's current value, clamped to the box
};

/// Framework modes (paper Section 3). In-tree repair is deferred (#419), so
/// Dfsrep searches as Dfs and differs only in asking for leaf repair.
enum class FprMode : std::uint8_t {
    Diveprop,  ///< propagate, never backtrack, leaf repair
    Dfs,       ///< propagate, backtrack on a conflict, no repair
    Dfsrep,    ///< as Dfs, then leaf repair
};

struct FprConfig {
    FprVarOrder order = FprVarOrder::Type;
    FprValueRule value = FprValueRule::GoodObj;
    FprMode mode = FprMode::Dfs;
};

/// mip-heuristics' `kInitialFprConfigs` (paper Section 6.3, Class 1) less its
/// Dive and RepairSearch entries, with typecl read as type (no clique table):
/// Badobjcl Dfs, Locks2 Dfs, Locks Dfsrep, Badobjcl Dfsrep, Random (typecl /
/// random) Diveprop, Domsize Dfs.
inline constexpr std::array<FprConfig, 6> kFprPortfolio{{
    {FprVarOrder::Type, FprValueRule::BadObj, FprMode::Dfs},
    {FprVarOrder::Locks, FprValueRule::LooseDyn, FprMode::Dfs},
    {FprVarOrder::LR, FprValueRule::LooseDyn, FprMode::Dfsrep},
    {FprVarOrder::Type, FprValueRule::BadObj, FprMode::Dfsrep},
    {FprVarOrder::Type, FprValueRule::Random, FprMode::Diveprop},
    {FprVarOrder::DomainSize, FprValueRule::LooseDyn, FprMode::Dfs},
}};

/// Entry `(worker + attempt) mod 6` of `kFprPortfolio`, as mip-heuristics'
/// `FprWorker` rotates: each worker visits every entry once before wrapping.
[[nodiscard]] FprConfig fpr_portfolio_config(int64_t worker, int64_t attempt);

struct FprResult {
    /// Per variable id; a structured variable's entry is unused (0).
    std::vector<double> values;
    /// The dive fixed every integral variable (a leaf). A Dfs that ran out of
    /// nodes or siblings is not complete; its values are still filled.
    bool complete = false;
    /// The mode repairs at the leaf (Diveprop, Dfsrep).
    bool wants_repair = false;
    int64_t nodes = 0;      ///< nodes popped, at most `num_vars + 1`
    int64_t conflicts = 0;  ///< fixings propagation refuted
};

/// One worker's FPR attempts over a closed, evaluated model, a store built
/// from it and a propagator over that store. The propagator is the caller's:
/// call `propagate_root` once at depth 0 first (the root tightenings), and
/// `run` leaves it at the depth it found it.
///
/// Integral variables are branched; Floats are never branched: a Float with a
/// nonzero cost takes its cost-preferred bound, one without its current value
/// (the reference's `cont_fallback`), clamped to its final box. An integral
/// variable in no linear row is fixed by the value rule with nothing to
/// propagate.
///
/// Two deviations, both forced by `DomainPropagator`'s rule that a conflict is
/// backtracked: a trivial rounding propagation refutes is skipped, and a
/// Diveprop fixing propagation refutes is undone there and kept outside it
/// (the reference carries on in the refuted state; #427). After a refuted
/// root, `run` branches nothing and reports `complete = false`.
///
/// Costs: the constructor is O(vars + nnz) plus one reverse sweep of the
/// objective. `run` is at most `num_vars + 1` nodes, each a `fix` plus
/// O(order) for the static orders' cursor scan (amortised O(1) along a dive)
/// or O(changed bounds * log) for DomainSize's lazy heap, plus O(vars) to
/// order and fill. Memory: O(vars) plus the propagator's trail.
class FprSearch {
public:
    FprSearch(const Model& model, const LinearRowStore& store, DomainPropagator& prop);

    /// The root pass: the propagator's `propagate_all` at depth 0. Its verdict
    /// is kept, since a conflict leaves no mark on the bounds: false refutes
    /// the model's linear rows, and every later `run` is then a bare fill.
    bool propagate_root();

    /// One attempt; `attempt` > 0 shuffles the first 30% of the order, as
    /// mip-heuristics does on later attempts. Values read the model's current
    /// values (Incumbent, the Float fill); the model is not written.
    void run(const FprConfig& cfg, int64_t attempt, RNG& rng, FprResult& out);

    /// The min-form objective's partial per variable at construction, 0 without
    /// an objective: a linear objective's costs.
    [[nodiscard]] double cost(int32_t var) const { return cost_[static_cast<size_t>(var)]; }
    [[nodiscard]] int32_t up_locks(int32_t var) const {
        return up_locks_[static_cast<size_t>(var)];
    }
    [[nodiscard]] int32_t down_locks(int32_t var) const {
        return down_locks_[static_cast<size_t>(var)];
    }
    /// The static order `run` branches in (before the attempt shuffle), over
    /// scalar variables; DomainSize starts from Type's.
    void static_order(FprVarOrder order, RNG& rng, std::vector<int32_t>& out) const;
    /// The value `rule` picks for `var` on `[lo, hi]` at the propagator's
    /// current activities.
    [[nodiscard]] double choose_value(int32_t var, double lo, double hi, FprValueRule rule,
                                      RNG& rng) const;

private:
    struct Node {
        int32_t var;
        double value;
        int32_t level;  // the checkpoint holding the parent's state
        size_t cursor;  // the static order's scan position below this node
    };

    [[nodiscard]] bool fixed(int32_t var) const;
    [[nodiscard]] double loose_dyn(int32_t var, double lo, double hi) const;
    [[nodiscard]] double alternative(int32_t var, double preferred) const;
    void trivial_fixings();
    void heap_push(int32_t var);
    void heap_push_changed(int32_t level);
    [[nodiscard]] int32_t next_var(bool dynamic, size_t& cursor);
    void push_children(int32_t var, size_t cursor, bool backtracks, const FprConfig& cfg, RNG& rng);
    void dive(const FprConfig& cfg, RNG& rng, FprResult& out);
    void fill(const FprConfig& cfg, RNG& rng, FprResult& out) const;

    const Model& model_;
    const LinearRowStore& store_;
    DomainPropagator& prop_;
    std::vector<int32_t> scalars_;  // scalar variable ids, ascending
    std::vector<uint8_t> integral_;
    std::vector<uint8_t> binary_;
    /// Every row of the variable is in the store: its locks are all its locks,
    /// so trivial rounding is safe for the rows cbls has.
    std::vector<uint8_t> linear_only_;
    std::vector<double> cost_;
    std::vector<int32_t> up_locks_;
    std::vector<int32_t> down_locks_;
    double inf_;           // the propagator's infinity: a bound at or past it is infinite
    bool root_ok_ = true;  // `propagate_root`'s verdict
    /// Each static order by `FprVarOrder`, built once at construction from a
    /// fixed-seed RNG, as mip-heuristics precomputes one per strategy before
    /// any attempt, so an attempt does not depend on what ran before it.
    std::array<std::vector<int32_t>, 5> orders_;

    // Per attempt, reused.
    std::vector<std::pair<int32_t, double>> trivial_;
    std::vector<int32_t> order_;
    std::vector<Node> stack_;
    std::vector<std::pair<double, int32_t>> heap_;  // (domain size, var), min-heap
    std::vector<int32_t> changed_;
    std::vector<uint8_t> forced_;  // Diveprop: fixed outside the propagator
    std::vector<double> forced_value_;
    int64_t node_cap_ = 0;
};

}  // namespace cbls
