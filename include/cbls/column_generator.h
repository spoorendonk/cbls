#pragma once

#include "counters.h"
#include "dag.h"
#include "model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cbls {

class ModelExtension;
struct ExtensionResult;
class ViolationManager;

/// Duplicate detection for generated columns (#168): a set of column
/// SIGNATURES, where a signature is the column's cost and its (row, coefficient)
/// list.
///
/// A helper the GENERATOR calls, not a check the engine makes behind its back.
/// The engine cannot recover a column's coefficients from a `ModelExtension` in
/// general -- a column is an arbitrary DAG recording, `prod(constant(a), x)` in
/// one model and a bare `x`, a `neg`, or a nonlinear term in another -- so it has
/// no canonical form to hash. The generator does: it built the coefficients
/// before staging them. So the engine OWNS one of these per `solve()`, hands it
/// over in `PricingContext::signatures`, and it persists across every pricing
/// call of that solve. A generator that wants the base model's columns counted
/// too registers them on its first call, which is the one moment it knows it is
/// looking at the base model.
///
/// Exact, not probabilistic: the hash only buckets, and a bucket hit compares
/// the whole signature, so a hash collision is never read as a duplicate. The
/// signature is canonicalised first -- rows sorted ascending, a repeated row's
/// coefficients summed, zero coefficients dropped -- so two orderings of one
/// column are one signature. Coefficients and cost compare BITWISE; a generator
/// computing the same column through two roundings gets two signatures, which is
/// the conservative failure (a duplicate admitted, never a distinct column
/// refused).
class ColumnSignatureSet {
public:
    /// Register a column. Returns true if it was new, false if an identical
    /// signature is already registered (in which case nothing changes).
    bool insert(std::vector<std::pair<int32_t, double>> coefficients, double cost);
    /// Whether an identical signature is registered.
    [[nodiscard]] bool contains(std::vector<std::pair<int32_t, double>> coefficients,
                                double cost) const;
    [[nodiscard]] size_t size() const noexcept { return set_.size(); }

private:
    struct Signature {
        double cost = 0.0;
        std::vector<std::pair<int32_t, double>> coefficients;  // canonical
    };
    struct Hash {
        size_t operator()(const Signature& s) const noexcept;
    };
    struct Equal {
        bool operator()(const Signature& a, const Signature& b) const noexcept;
    };
    static Signature canonical(std::vector<std::pair<int32_t, double>> coefficients, double cost);

    std::unordered_set<Signature, Hash, Equal> set_;
};

/// What a `ColumnGenerator` sees at a pricing call (#168). Every member is valid
/// for the duration of `price()` only.
///
/// THE CALL IS AT A SAFE POINT BETWEEN BATCHES: no GLS iteration is in flight,
/// `model`'s node values are consistent with its variables (a structural batch's
/// pending resync has been paid first), and `weights` is the live GLS weight
/// vector itself, not a copy -- `weights.begin() == vm.weights.data()`.
struct PricingContext {
    /// The model, holding the CURRENT assignment -- the point the search stands
    /// on, which is not necessarily the incumbent. Read the assignment through
    /// `model.var(v).value`; there is no separate copy of it, because taking one
    /// would be an O(#vars) allocation per call for a field most pricers ignore.
    const Model& model;
    /// The GLS weight `W[c]` per constraint index into `model.constraint_ids()`,
    /// exactly `ViolationManager::weights`. Existing rows keep these across every
    /// extension (`ViolationManager::on_extended`), which is what lets them act as
    /// prices: they measure how hard the current columns find each row.
    ///
    /// NOT LP duals and carrying no optimality certificate. Two properties to know
    /// before reading them as prices: they are reset to 1.0 on every new best and
    /// every diversification kick (the paper's GLS restart, `reset_weights`), so a
    /// call just after one of those sees a flat vector; and a satisfied row decays
    /// by rho (0.95 or 1.0, sampled per batch), so "small" means "satisfied
    /// lately", not "slack".
    ConstSpan<double> weights;
    /// The `obj <= bound` row's index into `weights`, or -1 on a model with no
    /// objective. `weights[objective_constraint_idx]` is the objective's own GLS
    /// weight, and `W[i] / W_obj` is the price scale: a column with cost `c_p` and
    /// row coefficients `a_ip` changes the weighted violation by roughly
    /// `W_obj * c_p - sum_i W[i] * a_ip` over the rows it helps, so a pricer looks
    /// for columns where that is negative.
    int32_t objective_constraint_idx = -1;
    /// The objective row's current RHS (+inf before the first feasible point).
    double objective_bound = 0.0;
    /// The best feasible assignment this solve has recorded, or null before the
    /// first one. A progress row (`SolveCallback`) may already have reported its
    /// objective; if an extension then changes that objective or cuts the point
    /// off, the engine re-derives it (`SearchCounters::incumbents_revalidated`)
    /// and the returned result reflects that, but no correcting row is emitted. Sized for the model
    /// AS IT IS NOW -- the engine pads it after every extension -- so `Model::restore_state` would
    /// accept it.
    const Model::State* incumbent = nullptr;
    /// The incumbent's objective, or +inf with no incumbent.
    double incumbent_objective = 0.0;
    /// Batches completed so far in this solve.
    int64_t batches = 0;
    /// Seconds since `solve()` started, or NaN on a run with no wall clock: an
    /// iteration-budgeted run reads no clock at all, so that its trajectory -- a
    /// generator's decisions included -- is reproducible on any machine.
    double elapsed_seconds = 0.0;
    /// Seconds left before the wall-clock deadline, or +inf when the run has no
    /// wall clock (an iteration-budgeted run). A generator must return within
    /// this: the engine does not start a call past the deadline, but it cannot
    /// interrupt one.
    double remaining_seconds = 0.0;
    /// How many more VARIABLES this solve may still add
    /// (`SearchConfig::max_generated_columns` minus those already added). An
    /// extension that stages more than this is refused whole, so a generator
    /// that cannot stay under it wastes its call -- and any signature it
    /// registered for that call stays registered (see `signatures`).
    int64_t columns_remaining = 0;
    /// This solve's duplicate registry; see `ColumnSignatureSet`. The engine
    /// never un-registers anything: a signature registered for an extension that
    /// is then refused (over `columns_remaining`) or abandoned (the generator
    /// threw) marks a column the model does not have. So register only what you
    /// stage, and stage only what fits.
    ColumnSignatureSet& signatures;
    /// The violation manager `weights` belongs to, for a pricer that wants the
    /// cached per-row violations too. `weights` above is exactly its `weights`.
    const ViolationManager& violations;
};

/// A pricing oracle: proposes new columns (and optionally rows) from the GLS
/// weights while the search runs (#168). The local-search analogue of column
/// generation -- a heuristic, not an exact method.
///
/// Registered on `SearchConfig::column_generator`. The engine calls `price` at
/// the events `SearchConfig` enables, applies whatever it staged through
/// `Model::extend` (#167), grows the violation manager and Feasibility Jump with
/// the model -- existing rows keep their GLS weights -- and continues. A new
/// column is an ordinary Bool/Int/Float variable, so FJ can move it on the very
/// next batch.
///
/// CLONED PER `solve()`: every `solve()` -- so every portfolio worker, and every
/// worker restart -- calls `clone()` once on the registered prototype and prices
/// with its own copy, so a generator holding a cache or a cursor holds it per
/// search. The prototype is never mutated by the engine, which is what the
/// `const` in `SearchConfig`'s pointer says -- but `clone()` IS called on it from
/// several worker threads at once under `ParallelSearch`, so it must be safe to
/// call concurrently (a `const` member that only reads is).
class ColumnGenerator {
public:
    ColumnGenerator() = default;
    ColumnGenerator(const ColumnGenerator&) = default;
    ColumnGenerator& operator=(const ColumnGenerator&) = default;
    ColumnGenerator(ColumnGenerator&&) = default;
    ColumnGenerator& operator=(ColumnGenerator&&) = default;
    virtual ~ColumnGenerator();

    /// Stage new variables, terms on existing `Sum` rows and new rows into `ext`,
    /// which was built against `ctx.model` as it is now. Staging nothing is a
    /// valid answer and costs the engine nothing further.
    ///
    /// Must respect `ctx.remaining_seconds` and `ctx.columns_remaining`. A throw
    /// propagates out of `solve()`, exactly as a throwing `InnerSolverHook` does,
    /// with nothing this call STAGED applied (nothing is applied until `price`
    /// returns). The engine's own bookkeeping for the call has run by then: any
    /// column `ColumnPool` aged out at this call was retired -- pinned at its
    /// lower bound in the model -- before `price` was invoked. Retirement pins
    /// outlive `solve()`: the caller's model comes back with them.
    ///
    /// New variables must be SCALAR -- `ModelExtension` offers nothing else, and
    /// says why.
    virtual void price(const PricingContext& ctx, PricingEvent why, ModelExtension& ext) = 0;

    /// A copy carrying this generator's configuration. Called on the registered
    /// prototype, possibly from several threads at once; see the class comment.
    /// Must not return null.
    [[nodiscard]] virtual std::unique_ptr<ColumnGenerator> clone() const = 0;
};

/// The engine's book of the columns pricing added to ONE solve: the cap, and
/// the aging that retires columns which stay at their lower bound (#168).
///
/// Public, rather than a detail of `src/search.cpp`, so its two rules can be
/// tested directly instead of inferred from a search trajectory.
///
/// RETIREMENT IS PERMANENT AND IS NOT REMOVAL. `Model::extend` cannot remove a
/// variable (#167's non-goal), so a retired column stays in the model, pinned:
/// its upper bound is set to its lower bound and Feasibility Jump drops it from
/// its scan tables (`FeasibilityJump::retire`). What that buys is scan cost, not
/// memory -- the column still occupies its slot in G_v, its terms still sit in
/// their `Sum` rows, and it still counts against the cap. Its signature stays
/// registered, so the generator cannot re-add it either; a column worth
/// reviving was not worth retiring, and `column_retire_age` is the knob.
class ColumnPool {
public:
    /// `cap` is the most VARIABLES pricing may add over the solve (<= 0 means no
    /// room at all). `retire_age` is how many consecutive pricing calls a column
    /// must sit at its lower bound before it is retired; 0 disables retirement.
    ColumnPool(int64_t cap, int retire_age);

    /// Room left under the cap.
    [[nodiscard]] int64_t remaining() const noexcept { return cap_ - added_; }
    /// Record the variables an applied extension added. Every one of them is a
    /// column from here on, whatever its type.
    void on_added(const ExtensionResult& ext);
    /// Age every live column once and return the ones due for retirement now,
    /// ascending. A column is aged only while it sits at its lower bound in the
    /// CURRENT assignment and in every state in `keep` -- the incumbent and any
    /// other point the search may return to -- because retiring it pins it
    /// there, and a state that uses it would stop being restorable to a point
    /// the search can still move. Any other value resets its age to 0. The
    /// returned columns are marked retired; the caller pins them.
    std::vector<int32_t> age(const Model& model, const std::vector<const Model::State*>& keep);

    [[nodiscard]] int64_t added() const noexcept { return added_; }
    [[nodiscard]] int64_t retired() const noexcept { return retired_; }
    [[nodiscard]] int64_t live() const noexcept { return added_ - retired_; }

private:
    struct Entry {
        int32_t var = -1;
        int age = 0;
        bool retired = false;
    };
    int64_t cap_;
    int retire_age_;
    int64_t added_ = 0;
    int64_t retired_ = 0;
    std::vector<Entry> entries_;
};

}  // namespace cbls
