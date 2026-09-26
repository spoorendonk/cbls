#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace cbls {

enum class VarType : uint8_t { Bool, Int, Float, List, Set };

/// True for the types that carry an `elements` permutation/subset rather than a
/// scalar `value`. The complement — Bool/Int/Float — is exactly what
/// `FeasibilityJump::jumpable()` accepts, and `solve()` relies on the two
/// partitioning VarType to initialise every variable exactly once (#108): FJ sets
/// the scalars, `initialize_structured_random` sets the rest.
///
/// Both are whitelists rather than one being `!other`, so a VarType added later
/// has to opt in on each side. Nothing catches it if you forget: the build uses
/// no `-Wall`/`-Wswitch`, so a new type would silently be neither initialised nor
/// jumped. Add it here and to `jumpable()` in the same change.
constexpr bool is_structured(VarType type) {
    return type == VarType::List || type == VarType::Set;
}

/// How `initialize_structured_random` fills a List (#164).
///
/// It decides the STARTING assignment only; the move set can reach every other
/// assignment from any of them. A member of a partition is filled as part of
/// that partition (`randomize_list_partition`), which reads `ListInit` only
/// under `Cover::AtMostOnce` -- an `Exact` cover is maintained by the moves
/// rather than by a penalty row, so it has to hold at the first assignment and
/// the members are always laid out complete. See `Model::add_list_partition`.
enum class ListInit : uint8_t {
    /// `elements == [0, 1, ..., universe-1]`, and therefore a permutation.
    /// Requires `min_size == max_size == universe_size`, which is exactly what
    /// `list_var(n)` builds -- and is what keeps that call's randomisation on
    /// `rng.permutation(max_size)`, the pre-#164 draw verbatim.
    Identity,
    /// `elements == []`. Draws no random numbers. Requires `min_size == 0`,
    /// since otherwise it is a starting assignment outside the List's own
    /// length window.
    Empty,
    /// A uniformly random subset of a uniformly random admissible size, in a
    /// uniformly random order.
    Random,
};

struct Variable {
    int32_t id = -1;
    VarType type = VarType::Float;
    double value = 0.0;
    double lb = 0.0;
    double ub = 0.0;
    std::string name;
    std::vector<int32_t> elements;  // List/Set current elements
    // The three size fields mean the same thing for both structured types
    // (#164): `elements` is drawn from the universe {0..universe_size-1} and its
    // size stays within [min_size, max_size]. A List additionally keeps its
    // elements DISTINCT and ORDERED, where a Set's stored order carries no
    // meaning. `list_var(n)` is the special case universe == min == max == n,
    // i.e. a permutation, which is all a List could be before #164.
    int32_t universe_size = 0;  // List/Set: universe {0..n-1}
    int32_t min_size = 0;       // List/Set: minimum length/cardinality
    int32_t max_size = 0;       // List/Set: maximum length/cardinality
    /// List only: how `initialize_structured_random` fills it.
    ListInit list_init = ListInit::Identity;
    /// List only: set by `Model::add_list_partition`, and the reason
    /// `list_moves` emits no `list_insert`/`list_remove` for this variable. Its
    /// membership is shared with the partition's other lists, so an intra-list
    /// insert would duplicate an element a sibling holds, and an intra-list
    /// remove would drop an element out of an `Exact` cover. The partition's own
    /// generator owns both halves instead (`generate_partition_moves`).
    bool partitioned = false;
    // The nodes that read this variable are `Model::dependents(id)`: a slice of
    // one flat array the model owns, not a vector per variable (#156).
};

/// A read-only view of a contiguous run of `T` -- what `Model::children`,
/// `Model::parents`, `Model::dependents` and `Model::constraints_of_var` hand
/// out, each a slice of one flat array the model owns. C++17 has no `std::span`,
/// and the four accessors need nothing beyond iteration, a size and an index.
///
/// Invalidated by whatever reallocates the array it points into: appending a
/// node for `children`, and the rebuild in `close()` /
/// `add_objective_soft_constraint()` for the other three. No caller holds one
/// across either.
///
/// `operator[]` asserts its bound. That is not decoration: an index past a
/// node's children now lands on the NEXT node's children inside one heap block,
/// which AddressSanitizer cannot see, where the per-node vector this replaced
/// was a heap overflow it reported. The assert costs nothing under NDEBUG.
template <typename T>
class ConstSpan {
public:
    constexpr ConstSpan() noexcept = default;
    constexpr ConstSpan(const T* data, size_t size) noexcept : data_(data), size_(size) {}

    [[nodiscard]] constexpr const T* begin() const noexcept { return data_; }
    [[nodiscard]] constexpr const T* end() const noexcept { return data_ + size_; }
    [[nodiscard]] constexpr size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr const T& operator[](size_t i) const noexcept {
        assert(i < size_);
        return data_[i];
    }

private:
    const T* data_ = nullptr;
    size_t size_ = 0;
};

enum class NodeOp : uint8_t {
    Const,
    Neg,
    Sum,
    Prod,
    Div,
    Pow,
    Min,
    Max,
    Abs,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    SignPower,
    Tanh,
    If,
    At,
    Count,
    Lambda,
    PairLambda,
    Leq,
    Eq,
    Geq,
    Neq,
    Lt,
    Gt
};

struct ChildRef {
    int32_t id = -1;
    bool is_var = false;
};

/// One DAG node. It owns no heap block: its children are the slice
/// `[child_begin, child_begin + child_count)` of the model's flat child array,
/// read through `Model::children(node)`, and its parents are
/// `Model::parents(id)` (#156).
///
/// Offsets rather than pointers, so that copying a `Model` -- which is how a
/// portfolio replicates a model per worker -- needs nothing rebased, whether it
/// deep-copies the structure or shares it (#157).
///
/// It carries NO evaluation result: a node's current value lives in the model's
/// per-model `node_values()` array, read through `Model::node_value(id)`. That
/// is what lets the whole node array sit in the immutable `ModelStructure` that
/// portfolio replicas share, where a `value` field would be one worker's search
/// state in storage every worker reads (#157).
struct ExprNode {
    int32_t id = -1;
    NodeOp op = NodeOp::Const;
    double const_value = 0.0;
    uint32_t child_begin = 0;
    uint32_t child_count = 0;
    int32_t lambda_func_id = -1;  // index into ModelStructure::lambda_funcs
};

/// How a `PairLambda` node closes its chain of consecutive pairs.
enum class PairMode : uint8_t {
    Open,   ///< e_0-e_1, ..., e_{n-2}-e_{n-1}. The original behaviour.
    Cyclic  ///< the Open pairs plus e_{n-1}-e_0, for n >= 2.
};

/// Everything a `PairLambda` node needs beyond its pair function: the closing
/// rule and the two optional fixed-endpoint terms.
///
/// It is a SIDE TABLE in `ModelStructure`, parallel to `pair_lambda_funcs` and
/// keyed by the same `ExprNode::lambda_func_id`, rather than three more
/// `NodeOp` enumerators. `src/dag.cpp`'s two dispatch tables are 28 cases wide
/// and already carry a cognitive-complexity suppression each; one variant per
/// closing rule crossed with head/tail presence would be six more cases in both
/// of them for no gain, since every variant evaluates through the same loop.
///
/// `head_id` and `tail_id` index `ModelStructure::lambda_funcs` -- the same
/// table `lambda_sum` fills -- or are -1 for "no term". They are therefore
/// shared across portfolio workers on exactly the terms the `lambda_funcs`
/// comment states.
struct PairLambdaSpec {
    PairMode mode = PairMode::Open;
    int32_t head_id = -1;  // index into ModelStructure::lambda_funcs, or -1
    int32_t tail_id = -1;  // index into ModelStructure::lambda_funcs, or -1
};

/// Residual of `a <= b`, i.e. `a - b`, with the IEEE `inf - inf` indeterminacy
/// resolved by the comparison the residual stands for — but *only* where the
/// infinity is a written bound rather than an arithmetic overflow.
///
/// `a_is_const` / `b_is_const` say whether that side is a literal `Const` node.
/// That flag is the whole point of this function, because an infinity means two
/// completely different things depending on where it came from:
///
///  - **Sentinel.** A `Const` +inf is the standard "this side is absent"
///    idiom: `a <= +inf` and `-inf <= b` hold for every a/b, including a
///    non-finite one. Plain `a - b` yields NaN when both sides are infinite
///    with the same sign, and the violation machinery reads NaN as a maximal
///    violation — so a *vacuous* row would be reported as the worst-violated
///    row in the model. That is what destroys the feasibility signal when the
///    objective is folded in as `obj <= bound`, the objective overflows to
///    +inf, and the bound is still at its initial +inf (issue #100).
///
///  - **Overflow.** An infinity computed by an expression means only "this
///    quantity left double range". `exp(1000) <= exp(720)` is a genuinely
///    violated row, and both sides evaluate to +inf; treating it as vacuous
///    would silently pass an assignment we have no evidence for. Overflow
///    therefore keeps the NaN, which the clamp turns into a maximal violation.
///    This preserves the invariant the rest of the engine defends explicitly
///    (see the NaN guards in ViolationManager, LNS and the search loop).
///
/// Only the side that would make the row vacuous is consulted: +inf on both
/// sides is vacuous when the *upper* bound is the sentinel, -inf on both sides
/// when the *lower* one is. Every other input is the plain difference, which is
/// already right: +inf for (a=+inf, b=-inf), -inf for (a=-inf, b=+inf), and NaN
/// whenever either side is NaN.
///
/// A sentinel-vacuous row returns 0.0 — satisfied, sitting exactly on the
/// boundary. `Lt`/`Gt` then add their strictness epsilon (1e-9) on top, so a
/// vacuous strict row reads as violated by 1e-9, which is *inside* the default
/// feasibility tolerance of 1e-6 and therefore still counts as satisfied. That
/// is not special to infinities: it is the pre-existing epsilon convention, and
/// a finite `1.0 < 1.0` behaves identically.
///
/// Scope: used by `Leq`/`Geq`/`Lt`/`Gt`. `Eq`/`Neq` deliberately keep plain
/// `|a - b|`, so `x == y` at inf/inf stays NaN (maximally violated). An
/// infinity is a *bound* idiom; asserting a quantity is exactly infinite is not
/// something the engine can evaluate, and leaving it unevaluable is the safe
/// reading.
inline double comparison_residual(double a, double b, bool a_is_const, bool b_is_const) {
    if (a == b && std::isinf(a)) {
        const bool sentinel = (a > 0.0) ? b_is_const : a_is_const;
        if (sentinel) {
            return 0.0;
        }
    }
    return a - b;
}

// Forward declaration
class Model;

double evaluate(const ExprNode& node, const Model& model);
double local_derivative(const ExprNode& node, int child_idx, const Model& model);

}  // namespace cbls
