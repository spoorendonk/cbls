#pragma once

// AMPL NL (text format) reader interface for CBLS.
//
// The NL format is the AMPL solver interface "nl" file. This reader handles the
// TEXT variant only (header begins with 'g'); the binary variant ('b') is
// rejected with a clear error. The format is publicly specified by David Gay,
// "Writing .nl Files" (https://ampl.github.io/nlwrite.pdf) and "Hooking Your
// Solver to AMPL" (https://ampl.com/REFS/hooking2.pdf); this is an original
// implementation from those public specifications and carries no third-party
// license.
//
// The reader populates the minimal POD `NlProblem` defined here. The
// NL-to-Model adapter (`nl_to_model`, in src/io/nl_to_model.cpp) builds a closed
// CBLS `Model` from an `NlProblem`, skipping (not throwing on) instances whose
// operator set is unsupported.

#include "model.h"

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
    bool empty() const { return root < 0; }
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

struct NlProblem {
    std::string name;

    int32_t n_vars = 0;
    int32_t n_cons = 0;
    int32_t n_objs = 0;

    std::vector<NlVarBound> var_bounds;     ///< size n_vars (from `b`)
    std::vector<double> initial_x;          ///< size n_vars; from `x`, NaN if unset
    std::vector<NlConstraint> constraints;  ///< size n_cons
    std::vector<NlObjective> objectives;    ///< size n_objs
};

/// Read an AMPL NL file in TEXT format ('g' header). Throws std::runtime_error
/// on a binary header ('b') or a malformed file.
NlProblem read_nl(const std::string& filename);

/// Parse NL text from an already-loaded buffer (used by tests with inline
/// fixtures). `name` labels the resulting problem.
NlProblem parse_nl(const std::string& text, const std::string& name = "");

/// Build a closed CBLS `Model` from an `NlProblem`.
///
/// One CBLS variable per NL column (Float for all — MINLPLib discrete vars are
/// handled by clamping to int domains when integrality is declared, but the NL
/// reader here does not yet surface integrality, so all are continuous). Each
/// constraint body (`nonlinear + linear`) is translated to one or two CBLS
/// comparison constraints from its `r` bound. The first objective is used; its
/// sense drives minimize/maximize.
///
/// Unsupported operators do NOT throw: `supported` is set false and a reason is
/// appended to `skipped_reasons`. On an unsupported instance the returned model
/// is left un-closed and should not be solved.
struct NlToModelOptions {
    double inf_clamp = 1.0e9;  ///< ±inf variable bounds clamped to this magnitude
};

struct NlToModelResult {
    Model model;
    /// `var_handles[i]` is the CBLS variable handle for NL column `i`.
    std::vector<int32_t> var_handles;
    /// `constraint_node_ids[i]` is the *primary* constraint node for NL row `i`
    /// (the second node of a range row is added to the model but not recorded).
    std::vector<int32_t> constraint_node_ids;
    int32_t objective_node_id = -1;  ///< -1 if no objective
    bool supported = true;           ///< false if any operator was unsupported
    std::vector<std::string> skipped_reasons;
};

NlToModelResult nl_to_model(const NlProblem& prob, const NlToModelOptions& opts = {});

}  // namespace cbls
