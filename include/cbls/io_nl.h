#pragma once

// AMPL NL reader interface for CBLS.
//
// The NL format is the AMPL solver interface "nl" file. This reader handles
// both variants: text (header begins with 'g') and binary ('b'), the latter
// only in the host's byte order. The format is publicly specified by David Gay,
// "Writing .nl Files" (https://ampl.github.io/nlwrite.pdf) and "Hooking Your
// Solver to AMPL" (https://ampl.com/REFS/hooking2.pdf); this is an original
// implementation from those public specifications and carries no third-party
// license.
//
// The reader populates the minimal POD `NlProblem` defined here. The
// NL-to-Model adapter (`nl_to_model`, in src/io/nl_to_model.cpp) builds a closed
// CBLS `Model` from an `NlProblem`, skipping (not throwing on) instances whose
// operator set is unsupported.

#include "bound_propagation.h"
#include "model.h"
#include "presolve.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cbls {

inline constexpr double kNlInf = std::numeric_limits<double>::infinity();

/// One expression-graph node, stored in postfix-free prefix form: each segment
/// of the file is parsed into a small tree of these. Children are indices into
/// the owning expression's `nodes` vector.
enum class NlNodeKind : uint8_t {
    Num,  ///< numeric constant (value in `num`)
    Var,  ///< variable reference (index in `index`)
    Op,   ///< operator (opcode in `opcode`, children in `children`)
};

struct NlExprNode {
    NlNodeKind kind = NlNodeKind::Num;
    double num = 0.0;               ///< Num: constant value
    int32_t index = -1;             ///< Var: variable index
    int32_t opcode = -1;            ///< Op: AMPL opcode (see nl_reader.cpp table)
    std::vector<int32_t> children;  ///< Op: child node indices into the same expr
};

/// A nonlinear expression: a flat node pool with a designated root. An empty
/// pool (root == -1) means "no nonlinear part" (the segment was absent / 0).
struct NlExpr {
    std::vector<NlExprNode> nodes;
    int32_t root = -1;
    [[nodiscard]] bool empty() const { return root < 0; }
};

/// Bound/constraint type codes from the NL `r` (constraint) and `b` (variable)
/// segments. The numeric encoding matches the NL spec exactly.
enum class NlBoundType : uint8_t {
    Range = 0,  ///< lower <= body <= upper   (two finite bounds)
    Upper = 1,  ///< body <= upper            (<=)
    Lower = 2,  ///< body >= lower            (>=)
    Free = 3,   ///< no bound
    Equal = 4,  ///< body == value
};

struct NlVarBound {
    NlBoundType type = NlBoundType::Free;
    double lower = -kNlInf;
    double upper = kNlInf;
};

struct NlConBound {
    NlBoundType type = NlBoundType::Free;
    double lower = -kNlInf;
    double upper = kNlInf;
};

/// One linear coefficient (variable index, value).
struct NlLinTerm {
    int32_t var = -1;
    double coef = 0.0;
};

struct NlConstraint {
    NlExpr nonlinear;               ///< nonlinear part (may be empty)
    std::vector<NlLinTerm> linear;  ///< linear part from the `J` segment
    NlConBound bound;
};

struct NlObjective {
    bool maximize = false;          ///< NL O-segment sense (0=min, 1=max)
    NlExpr nonlinear;               ///< nonlinear part (may be empty)
    std::vector<NlLinTerm> linear;  ///< linear part from the `G` segment
};

/// A defined variable (an NL `V` segment, AMPL's common subexpression): its value
/// is `sum(linear) + nonlinear`. Expressions reference it as `v<n_vars + k>`.
struct NlDefinedVar {
    bool defined = false;           ///< false until its `V` segment is read
    std::vector<NlLinTerm> linear;  ///< linear part (terms may name other defined vars)
    NlExpr nonlinear;               ///< nonlinear part (may be the constant 0)
};

struct NlProblem {
    std::string name;

    int32_t n_vars = 0;
    int32_t n_cons = 0;
    int32_t n_objs = 0;
    /// Defined-variable count from the header's common-expression line
    /// (comb + comc + como + comc1 + como1). Not columns: the model inlines them.
    int32_t n_defined_vars = 0;
    /// Total integer/binary variable count from the NL header's discrete-variable
    /// line (nbv + niv + nlvbi + nlvci + nlvoi).
    int32_t n_discrete_vars = 0;

    /// Per-column integrality, size n_vars: 1 if NL column j is an integer or
    /// binary variable. Recovered from the header counts plus Gay's variable
    /// ordering (see `nl_reader.cpp`); `sum(var_is_discrete) == n_discrete_vars`.
    std::vector<uint8_t> var_is_discrete;

    std::vector<NlVarBound> var_bounds;     ///< size n_vars (from `b`)
    std::vector<double> initial_x;          ///< size n_vars; from `x`, NaN if unset
    std::vector<NlConstraint> constraints;  ///< size n_cons
    std::vector<NlObjective> objectives;    ///< size n_objs
    /// size n_defined_vars; entry k is NL variable `n_vars + k`.
    std::vector<NlDefinedVar> defined_vars;
};

/// Read an AMPL NL file, text ('g' header) or binary ('b'). Throws
/// std::runtime_error on a malformed file or a binary one in a byte order
/// other than the host's.
NlProblem read_nl(const std::string& filename);

/// Parse NL from an already-loaded buffer, text or binary (used by tests with
/// inline fixtures). `name` labels the resulting problem.
NlProblem parse_nl(const std::string& text, const std::string& name = "");

/// Build a closed CBLS `Model` from an `NlProblem`.
///
/// One CBLS variable per NL column (per *kept* column with presolve on; see
/// `var_handles`): Int for columns flagged in `var_is_discrete`, Float
/// otherwise. A defined variable (`V` segment) is not a column: it is inlined,
/// built once as an expression node that every use shares. Each constraint body
/// (`nonlinear + linear`) is translated to one or two CBLS comparison
/// constraints from its `r` bound. The first objective is used; its sense
/// drives minimize/maximize.
///
/// Unsupported operators do NOT throw: `supported` is set false and a reason is
/// appended to `skipped_reasons`. On an unsupported instance the returned model
/// is left un-closed and should not be solved.
struct NlToModelOptions {
    /// Derive implied bounds from the purely linear rows before falling back on
    /// the clamps below (see `bound_propagation.h`). An implied bound cannot cut
    /// off a feasible point; a clamp can, so it is only ever consulted last.
    bool propagate_bounds = true;
    /// Fixed-point iteration cap for that pass; see `BoundPropagationOptions`.
    int max_propagation_passes = 10;
    double inf_clamp = 1.0e9;  ///< ±inf variable bounds clamped to this magnitude
    /// ±inf bounds on an *integer* column clamp to this instead. A ±1e9 integer
    /// domain is not a searchable space; unbounded integers in practice mean
    /// "small non-negative count", so a far tighter box is the useful default.
    double int_inf_clamp = 1.0e6;
    /// Off by default; off, the model is the one built without presolve. On,
    /// the linear rows, the column boxes and the objective's linear part go
    /// through `presolve` (presolve.h). A linear row is one whose nonlinear
    /// part is absent or a lone constant `k` (AMPL writes `n0` for every
    /// linear row), read with its sides moved by `-k`; it must also have a
    /// linear part, a finite side and only in-range columns. Every other row
    /// is kept verbatim ("opaque"), and every column it or any expression can
    /// reach is frozen: one in an expression, in a defined variable, or in
    /// the linear part of an opaque row. So no expression is remapped.
    /// `propagate_bounds` and `propagation.max_passes` are taken from the two
    /// fields above, so propagation runs once, inside presolve.
    PresolveOptions presolve;
};

struct NlToModelResult {
    Model model;
    /// `var_handles[i]` is the CBLS variable handle for NL column `i` without
    /// presolve, and for *reduced* column `i` with it: map a point back with
    /// `original_values`. Code that assumes one handle per NL column must check
    /// the size first.
    std::vector<int32_t> var_handles;
    /// `constraint_node_ids[i]` is the *primary* constraint node for NL row `i`
    /// (the second node of a range row is added to the model but not recorded):
    /// the upper side of a range. -1 for a Free row and a row presolve removed;
    /// with presolve a range row may have lost a side it implied.
    std::vector<int32_t> constraint_node_ids;
    int32_t objective_node_id = -1;  ///< -1 if no objective
    bool supported = true;           ///< false if any operator was unsupported
    std::vector<std::string> skipped_reasons;
    /// Outcome of the propagation pass without presolve; all-zero when it was
    /// disabled or ran inside presolve (counted in `presolve_stats` there).
    BoundPropagationStats bound_stats;
    /// Columns whose bounds a clamp still had to supply *after* propagation,
    /// i.e. where the unsound fallback was actually used.
    int n_clamped_columns = 0;
    /// All-zero (status `Unchanged`) without presolve. `Infeasible` means
    /// presolve proved the linear part empty and the model was built without it.
    PresolveStats presolve_stats;
    /// Non-empty when presolve refused the input (its validation message); the
    /// model was then built without presolve.
    std::string presolve_rejected;
    /// Maps reduced values back to the NL columns; empty (n_original 0) unless
    /// the model was built from a presolved problem.
    Postsolve postsolve;
    /// NL column of each reduced column; empty unless built from presolve.
    std::vector<int32_t> reduced_to_original;

    /// The model's current point as one value per NL column: the variable
    /// values, mapped back through `postsolve` when the model was presolved.
    /// `model` must be this result's model (or a copy of it).
    [[nodiscard]] std::vector<double> original_values(const Model& model) const;
};

NlToModelResult nl_to_model(const NlProblem& prob, const NlToModelOptions& opts = {});

}  // namespace cbls
