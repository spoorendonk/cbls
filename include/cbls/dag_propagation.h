#pragma once

// A model's nonlinear rows over monotone ops, for HC4-style interval
// propagation in the `DomainPropagator` (#424). Off unless a caller hands the
// store to the propagator (`SearchConfig::fpr_dag_propagation`).

#include "cbls/dag.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cbls {

class Model;

/// A closed interval of the extended reals, the propagator's DAG scratch.
struct DagInterval {
    double lo;
    double hi;
};

/// Every row whose residual subtree uses only `Const, Neg, Sum, Prod, Min,
/// Max, Exp, Log` over scalar variables, each as a flat list of its nodes in
/// model topological order (children before parents), built once from the
/// closed model.
///
/// Kept: Leq, Geq, Lt, Gt, Eq and a bare body, read as the residual
/// `p - q` (Geq/Gt swapped, as everywhere) lying in `[lo, hi]`: Eq
/// `[-tol, tol]`, the rest `[-inf, tol - margin]`, the margin being Lt/Gt's
/// strictness margin. Skipped: the objective row (soft), domain rows (#258:
/// their residual adds a margin), Neq, every affine row (`LinearRowStore`
/// propagates those), a row with any other op or a structured variable, and
/// a row reading the objective bound's Const (per-worker state).
///
/// Memory: per row O(its subtree), a shared subexpression stored once per
/// row reading it -- 12 B per node and 16 B per child, plus 4 B per distinct
/// variable twice (row list, variable index). A DAG whose rows all read one
/// large shared subexpression costs rows x its size; the propagator is
/// opt-in, so only callers that asked for it pay.
class DagRowStore {
public:
    enum class RefKind : uint8_t { kVar, kNode, kConst };
    /// A child: `kVar` names a slot of the row's variables, `kNode` a node of
    /// the row (index from the row's `node_begin`), `kConst` a literal.
    struct Ref {
        RefKind kind = RefKind::kConst;
        int32_t index = 0;
        double value = 0.0;
    };
    struct Node {
        NodeOp op = NodeOp::Const;
        uint32_t child_begin = 0;  ///< into `children()`
        uint32_t child_count = 0;
    };
    struct Row {
        int32_t constraint = -1;  ///< the model's constraint index
        Ref p;
        Ref q;
        double lo = 0.0;  ///< bounds on `p - q`
        double hi = 0.0;
        uint32_t node_begin = 0;  ///< nodes `[node_begin, node_end)` of `nodes()`
        uint32_t node_end = 0;
        uint32_t var_begin = 0;  ///< slots `[var_begin, var_end)` of `row_vars()`
        uint32_t var_end = 0;
    };

    /// `row_tolerance` (finite, >= 0) widens every row as `LinearRowStore`'s
    /// does. The model must be closed and evaluated (`full_evaluate`): the
    /// affine test reads node values. Throws `std::logic_error` otherwise.
    explicit DagRowStore(const Model& model, double row_tolerance = 0.0);

    [[nodiscard]] int32_t num_rows() const noexcept { return static_cast<int32_t>(rows_.size()); }
    [[nodiscard]] int32_t num_vars() const noexcept {
        return static_cast<int32_t>(var_row_start_.size()) - 1;
    }
    [[nodiscard]] const Row& row(int32_t k) const { return rows_[static_cast<size_t>(k)]; }
    [[nodiscard]] const std::vector<Node>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<Ref>& children() const noexcept { return children_; }
    /// The model variable of each row's slot.
    [[nodiscard]] const std::vector<int32_t>& row_vars() const noexcept { return row_vars_; }
    /// Variable `v`'s rows are `[var_row_start(v), var_row_start(v + 1))` of `var_rows()`.
    [[nodiscard]] size_t var_row_start(int32_t v) const {
        return var_row_start_[static_cast<size_t>(v)];
    }
    [[nodiscard]] const std::vector<int32_t>& var_rows() const noexcept { return var_rows_; }
    /// Nodes plus children: a scan's work unit, for the propagator's valve.
    [[nodiscard]] int64_t size() const noexcept {
        return static_cast<int64_t>(nodes_.size() + children_.size());
    }
    /// The most nodes (variable slots) any one row has: scratch sizes.
    [[nodiscard]] size_t max_row_nodes() const noexcept { return max_row_nodes_; }
    [[nodiscard]] size_t max_row_vars() const noexcept { return max_row_vars_; }

private:
    void index_rows_by_variable(int32_t n_vars);

    std::vector<Row> rows_;
    std::vector<Node> nodes_;
    std::vector<Ref> children_;
    std::vector<int32_t> row_vars_;
    std::vector<size_t> var_row_start_;
    std::vector<int32_t> var_rows_;
    size_t max_row_nodes_ = 0;
    size_t max_row_vars_ = 0;
};

}  // namespace cbls
