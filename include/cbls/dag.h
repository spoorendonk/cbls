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

/// Default tolerance below which a constraint counts as satisfied, on the
/// node's *absolute* residual (|lhs - rhs| for an equality). Matches SCIP's
/// `numerics/feastol` and `verify_model`. Much tighter is meaningless in double
/// precision (a 1e4 body would need ~13 digits). Here, not in violation.h,
/// because `Model::set_domain_tolerance` defaults domain margins from it.
inline constexpr double kDefaultFeasibilityTolerance = 1e-6;

enum class VarType : uint8_t { Bool, Int, Float, List, Set };

/// True for the types carrying `elements` rather than a scalar `value`. The
/// complement is exactly `FeasibilityJump::jumpable()`, and `solve()` relies on
/// the two partitioning VarType to initialise every variable once (#108).
/// Both are `==` whitelists, not switches, so no -Wswitch catches a new type:
/// it would silently be neither initialised nor jumped. Add it here and to
/// `jumpable()` in the same change.
constexpr bool is_structured(VarType type) {
    return type == VarType::List || type == VarType::Set;
}

/// How `initialize_structured_random` fills a List (#164).
///
/// The STARTING assignment only; the moves reach the rest from any. A
/// partition member is filled by `randomize_list_partition`, which reads it
/// only under `Cover::AtMostOnce`: an `Exact` cover is kept by the moves, not a
/// penalty row, so it must hold from the start (`Model::add_list_partition`).
enum class ListInit : uint8_t {
    /// `elements == [0, 1, ..., universe-1]`, a permutation. Requires
    /// `min_size == max_size == universe_size`, as `list_var(n)` builds, whose
    /// randomisation stays `rng.permutation(max_size)` (#164).
    Identity,
    /// `elements == []`; draws no random numbers. Requires `min_size == 0`
    /// (else it starts outside the length window).
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
    // For both structured types (#164), `elements` is drawn from
    // {0..universe_size-1} with size in [min_size, max_size]. A List's elements
    // are DISTINCT and ORDERED; a Set's stored order means nothing.
    // `list_var(n)` is universe == min == max == n, a permutation.
    int32_t universe_size = 0;  // List/Set: universe {0..n-1}
    int32_t min_size = 0;       // List/Set: minimum length/cardinality
    int32_t max_size = 0;       // List/Set: maximum length/cardinality
    /// List only: how `initialize_structured_random` fills it.
    ListInit list_init = ListInit::Identity;
    /// List only, set by `Model::add_list_partition`: `list_moves` then emits
    /// no `list_insert`/`list_remove`, which would duplicate a sibling's
    /// element or drop one from an `Exact` cover. `generate_partition_moves`
    /// owns both.
    bool partitioned = false;
    // Its readers are `Model::dependents(id)`, a slice of a model-owned array.
};

/// A read-only view of a contiguous run of `T` (C++17 has no `std::span`):
/// what `Model::children`, `parents`, `dependents` and `constraints_of_var`
/// hand out, each a slice of one flat model-owned array.
///
/// Invalidated by whatever reallocates that array: appending a node for
/// `children`, the rebuild in `close()` / `add_objective_soft_constraint()`
/// for the other three. No caller holds one across either.
///
/// `operator[]` asserts its bound because an overrun lands silently on the
/// NEXT slice of the same heap block, invisible to AddressSanitizer. Free
/// under NDEBUG.
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
    Gt,
    // A new op shows up as -Wswitch / `clang-diagnostic-switch` in the
    // `default:`-free tables of `src/dag.cpp` and `src/io.cpp` (#171) -- not
    // in `if`/`==` chains, nor in `node_is_affine` (src/feasibility_jump.cpp),
    // whose deliberate `default:` silently classes it non-affine. So
    // `grep -rn 'NodeOp::' src/ include/ python/` for the rest.
    //
    // The six below (#186) are piecewise constant or opaque functors: local
    // derivative 0, never affine. FJ instead offers the plateau edges of
    // Ceil/Floor/Round and the index values of Element, reached through
    // `carries_slope` ops, as jump candidates (`compute_var_jump`).
    /// `table[i]` or `table[i][j]` over a numeric table, each index any scalar,
    /// truncated toward zero -- 1 or 2 children, every one an index. The table
    /// is `ModelStructure::element_tables[lambda_func_id]`. See
    /// `Model::element` for the index rule.
    Element,
    /// `std::ceil` of its one child.
    Ceil,
    /// `std::floor` of its one child.
    Floor,
    /// `std::round` of its one child: half-way cases away from zero.
    Round,
    /// `lambda_sum` whose functor also reads scalar children: children[0] is
    /// the List/Set variable, children[1..] the `extra` scalars, whose current
    /// values the functor receives. `lambda_extra_funcs[lambda_func_id]`.
    LambdaExtra,
    /// `pair_lambda_sum` whose functor also reads scalar children, laid out as
    /// `LambdaExtra`. `pair_lambda_extra_funcs[lambda_func_id]`, closing rule
    /// in `pair_lambda_extra_modes` at the same index.
    PairLambdaExtra
};

struct ChildRef {
    int32_t id = -1;
    bool is_var = false;
};

/// One DAG node, owning no heap block: children are the slice
/// `[child_begin, child_begin + child_count)` of the model's flat child array
/// (`Model::children(node)`), parents are `Model::parents(id)` (#156). Offsets,
/// not pointers, so copying a `Model` per portfolio worker rebases nothing,
/// deep-copied or shared (#157).
///
/// NO evaluation result: values live in the per-model `node_values()`
/// (`Model::node_value(id)`), so the node array can sit in the immutable
/// `ModelStructure` replicas share (#157).
struct ExprNode {
    int32_t id = -1;
    NodeOp op = NodeOp::Const;
    /// The incremental Sum (#177, #188), set by `Model::close()`: `kIncSum` on
    /// a `Sum` that `commit_scalar_move` moves by its terms' changes rather
    /// than re-summing, `kFeedsIncSum` on a term of one. A Sum qualifies when
    /// no term is a Sum or named twice and every reader is a top-level
    /// comparison against a variable or constant (reasons:
    /// `Model::classify_incremental_sums`). Its per-model state
    /// (`Model::inc_sum_state`) is at slot `lambda_func_id`. Sits in padding
    /// after `op` (no memory) and is read from the already-loaded node (no
    /// extra load per dirty node).
    uint8_t inc_sum_flags = 0;
    static constexpr uint8_t kIncSum = 1;
    static constexpr uint8_t kFeedsIncSum = 2;
    /// A `Const`'s literal; on `Lt`/`Gt`, the row's strictness margin (see
    /// `strict_comparison_margin`). Unused by every other op.
    double const_value = 0.0;
    uint32_t child_begin = 0;
    uint32_t child_count = 0;
    /// Which table entry this node's user code lives in, by op:
    ///
    ///  - `Lambda`, and a `PairLambda`'s head/tail: `ModelStructure::lambda_funcs`
    ///  - `PairLambda`: `pair_lambda_funcs` and the parallel `pair_lambda_specs`
    ///  - `Element`: `ModelStructure::element_tables` (#186)
    ///  - `LambdaExtra`: `lambda_extra_funcs`; `PairLambdaExtra`:
    ///    `pair_lambda_extra_funcs` and the parallel `pair_lambda_extra_modes`
    ///  - a `Sum` with `kIncSum`: its slot in `Model::inc_sum_state()` and
    ///    `ModelStructure::inc_sum_nodes` (#188)
    ///
    /// One field for all, since no node is two of them and a model can have
    /// 4.3M nodes. -1 for every other op.
    int32_t lambda_func_id = -1;
};

// `inc_sum_flags` sits in padding: pinned so a field added later cannot grow
// every node of a 4.3M-node model without someone deciding to (#177).
static_assert(sizeof(ExprNode) == 32, "ExprNode grew; see inc_sum_flags");

/// How a `PairLambda` node closes its chain of consecutive pairs.
enum class PairMode : uint8_t {
    Open,   ///< e_0-e_1, ..., e_{n-2}-e_{n-1}. The original behaviour.
    Cyclic  ///< the Open pairs plus e_{n-1}-e_0, for n >= 2.
};

/// Everything a `PairLambda` node needs beyond its pair function: the closing
/// rule and the two optional fixed-endpoint terms.
///
/// A SIDE TABLE in `ModelStructure`, parallel to `pair_lambda_funcs` and keyed
/// by `ExprNode::lambda_func_id`, not more `NodeOp`s: every variant runs the
/// same loop, and `src/dag.cpp`'s two dispatch tables (each already under a
/// complexity suppression) would gain six cases apiece for nothing.
///
/// `head_id`/`tail_id` index `ModelStructure::lambda_funcs` (as `lambda_sum`
/// fills it), or -1 for no term; shared across workers on that table's terms.
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
///  - **Sentinel.** A `Const` +-inf means "this side is absent": `a <= +inf`
///    and `-inf <= b` hold for every a/b. Plain `a - b` gives NaN at same-sign
///    infinities, read as maximal violation, so a *vacuous* row would rank
///    worst -- destroying the feasibility signal when the folded `obj <= bound`
///    overflows against the initial +inf bound (#100).
///
///  - **Overflow.** A computed infinity only left double range:
///    `exp(1000) <= exp(720)` is genuinely violated. It keeps the NaN, which
///    the clamp makes a maximal violation, as the engine's NaN guards expect.
///
/// Only the side that would make the row vacuous is consulted (+inf on both
/// with `b` a literal, -inf on both with `a` one). Everything else is the plain
/// difference, already right: +-inf for opposite infinities, NaN for a NaN.
///
/// A sentinel-vacuous row returns 0.0, on the boundary. `Lt`/`Gt` add their
/// margin; an inf constant is not integral, so it is 1e-9, inside the default
/// 1e-6 tolerance: a vacuous strict row stays satisfied.
///
/// Used by `Leq`/`Geq`/`Lt`/`Gt`. `Eq`/`Neq` keep plain `|a - b|`, so `x == y`
/// at inf/inf stays NaN: infinity is a bound idiom, and "exactly infinite" is
/// not evaluable, so it is safest left maximally violated.
inline double comparison_residual(double a, double b, bool a_is_const, bool b_is_const) {
    if (a == b && std::isinf(a)) {
        const bool sentinel = (a > 0.0) ? b_is_const : a_is_const;
        if (sentinel) {
            return 0.0;
        }
    }
    return a - b;
}

/// Whether a node of this op is a comparison row (`Leq`/`Geq`/`Lt`/`Gt`/`Eq`):
/// its value is a residual of its two children, not an expression in them. A
/// constraint whose node is anything else is a bare body, read as `expr <= 0`.
/// One predicate for every place that splits rows so (#192, #190).
inline bool is_comparison_op(NodeOp op) {
    return op == NodeOp::Leq || op == NodeOp::Geq || op == NodeOp::Lt || op == NodeOp::Gt ||
           op == NodeOp::Eq;
}

/// The margin `Lt`/`Gt` add to `comparison_residual`, so a strict row is
/// violated by it at equality. Fixed per node by `Model::lt`/`gt` in
/// `ExprNode::const_value`; `evaluate` and `linear_jump.h` both read it there,
/// so they reproduce one residual.
///
/// Integral sides (`Model::is_integral`) get 1: over integers `a < b` IS
/// `a - b + 1 <= 0`, exactly (#211; an epsilon inside the 1e-6 tolerance let
/// `x < y` accept x = y). Any other strict row gets `kStrictComparisonEps`
/// and so behaves as `a <= b` up to the tolerance: strictness over a continuous
/// side has no closest feasible point, which is why MPS and NL carry no strict
/// rows. A caller who needs a gap writes `x + delta <= y`.
inline constexpr double kStrictComparisonEps = 1e-9;
inline constexpr double kIntegralStrictMargin = 1.0;
constexpr double strict_comparison_margin(bool integral_sides) {
    return integral_sides ? kIntegralStrictMargin : kStrictComparisonEps;
}

// Forward declaration
class Model;

double evaluate(const ExprNode& node, const Model& model);
double local_derivative(const ExprNode& node, int child_idx, const Model& model);

/// Element, Ceil, Floor, Round: the piecewise-constant ops whose plateaus FJ
/// offers as jump candidates (#186).
bool is_breakpoint_op(NodeOp op) noexcept;

/// Whether the breakpoint walk (#186) carries a slope through `op`: every op
/// with a meaningful `local_derivative`, i.e. all but the structural ops, user
/// code, the breakpoint ops themselves and Const. A default-free switch in
/// src/dag.cpp, so a new op must choose.
bool carries_slope(NodeOp op) noexcept;

}  // namespace cbls
