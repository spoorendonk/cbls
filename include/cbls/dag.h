#pragma once

#include <cmath>
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
inline constexpr bool is_structured(VarType type) {
    return type == VarType::List || type == VarType::Set;
}

struct Variable {
    int32_t id = -1;
    VarType type = VarType::Float;
    double value = 0.0;
    double lb = 0.0;
    double ub = 0.0;
    std::string name;
    std::vector<int32_t> elements;       // List/Set current elements
    int32_t universe_size = 0;           // Set: universe {0..n-1}
    int32_t min_size = 0;                // Set: minimum cardinality
    int32_t max_size = 0;                // Set/List: maximum cardinality
    std::vector<int32_t> dependent_ids;  // ExprNode IDs that depend on this var
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

struct ExprNode {
    int32_t id = -1;
    NodeOp op = NodeOp::Const;
    double value = 0.0;
    double const_value = 0.0;
    std::vector<ChildRef> children;
    std::vector<int32_t> parent_ids;
    int32_t lambda_func_id = -1;  // index into Model::lambda_funcs_
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
