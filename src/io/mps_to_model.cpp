// Adapter: MpsProblem -> closed CBLS Model.

#include "cbls/bound_propagation.h"
#include "cbls/expr.h"
#include "cbls/io_mps.h"
#include "cbls/model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cbls {

namespace {

// Supply a finite bound where the column has none, so the engine has a box to
// search. This is the *fallback*, reached only where propagation derived no
// implied bound: unlike an implied bound it is not entailed by the constraints,
// so it can cut off feasible points. A bound that is finite — declared in the
// file or derived by propagation — is therefore honoured as written, however
// wide; only "no bound" is replaced.
double clamp_lo(double lb, double inf_clamp) {
    return (std::isnan(lb) || is_unbounded_below(lb)) ? -inf_clamp : lb;
}
double clamp_hi(double ub, double inf_clamp) {
    return (std::isnan(ub) || is_unbounded_above(ub)) ? inf_clamp : ub;
}

/// A row body's two-sided bounds. Returned as a pair rather than written
/// through two `double&` out-parameters: `lo` and `hi` have the same type, so a
/// transposed argument list is silently accepted by the compiler and inverts the
/// constraint's sense. Named fields make that transposition unrepresentable.
struct RowBounds {
    double lo = -kMpsInf;
    double hi = kMpsInf;
};

/// The row's body bounds `lo <= body <= hi`, matching exactly the constraints
/// the adapter goes on to build from the same sense/rhs/range triple.
RowBounds row_bounds(const MpsRow& r) {
    const double rng = std::abs(r.range);
    RowBounds b;
    switch (r.sense) {
        case MpsRowSense::L:
            b.hi = r.rhs;
            b.lo = r.range != 0.0 ? r.rhs - rng : -kMpsInf;
            break;
        case MpsRowSense::G:
            b.lo = r.rhs;
            b.hi = r.range != 0.0 ? r.rhs + rng : kMpsInf;
            break;
        case MpsRowSense::E:
            b.lo = r.range < 0.0 ? r.rhs + r.range : r.rhs;
            b.hi = r.range > 0.0 ? r.rhs + r.range : r.rhs;
            break;
    }
    return b;
}

/// The constraint matrix in CSR form, plus the objective row's nonzero indices.
/// The objective row (`row_idx == -1`) is deliberately outside the CSR: it
/// carries no bounds, so propagation must not see it, and its terms are gathered
/// separately when the objective expression is built.
struct RowMatrix {
    std::vector<int32_t> row_start;  ///< n_rows + 1 offsets into cols/coefs
    std::vector<int32_t> cols;
    std::vector<double> coefs;
    std::vector<int> obj_nz;  ///< indices into prob.nonzeros of the objective row
};

/// Group the flat MPS nonzero triplets by row. One flat CSR rather than a
/// vector-per-row: it is the single structure both the expression builder and
/// bound propagation read, so the matrix is laid out once instead of being
/// grouped once and copied again.
RowMatrix build_row_matrix(const MpsProblem& prob, int n_rows) {
    RowMatrix mat;
    mat.row_start.assign(static_cast<std::size_t>(n_rows) + 1, 0);
    for (const auto& nz : prob.nonzeros) {
        if (nz.row_idx >= 0 && nz.row_idx < n_rows) {
            ++mat.row_start[static_cast<std::size_t>(nz.row_idx) + 1];
        }
    }
    for (int i = 0; i < n_rows; ++i) {
        mat.row_start[static_cast<std::size_t>(i) + 1] +=
            mat.row_start[static_cast<std::size_t>(i)];
    }
    mat.cols.resize(static_cast<std::size_t>(mat.row_start[n_rows]));
    mat.coefs.resize(static_cast<std::size_t>(mat.row_start[n_rows]));
    std::vector<int32_t> fill(mat.row_start.begin(), mat.row_start.end() - 1);
    for (int k = 0; k < static_cast<int>(prob.nonzeros.size()); ++k) {
        const auto& nz = prob.nonzeros[k];
        if (nz.row_idx == -1) {
            mat.obj_nz.push_back(k);
        } else if (nz.row_idx >= 0 && nz.row_idx < n_rows) {
            const auto at = static_cast<std::size_t>(fill[nz.row_idx]++);
            mat.cols[at] = nz.col_idx;
            mat.coefs[at] = nz.value;
        }
    }
    return mat;
}

/// Derive implied column bounds from the rows. The rows are handed to
/// propagation as *views* into the caller's CSR arrays: the constraint matrix of
/// a large MIP runs to hundreds of megabytes and must not be duplicated here.
BoundPropagationStats tighten_column_bounds(const MpsProblem& prob, const RowMatrix& mat,
                                            const MpsToModelOptions& opts, std::vector<double>& lb,
                                            std::vector<double>& ub,
                                            const std::vector<uint8_t>& integral) {
    std::vector<LinearRow> rows(prob.rows.size());
    for (std::size_t i = 0; i < prob.rows.size(); ++i) {
        LinearRow& row = rows[i];
        const RowBounds rb = row_bounds(prob.rows[i]);
        row.lo = rb.lo;
        row.hi = rb.hi;
        row.nnz = mat.row_start[i + 1] - mat.row_start[i];
        row.cols = mat.cols.data() + mat.row_start[i];
        row.coefs = mat.coefs.data() + mat.row_start[i];
    }
    BoundPropagationOptions popts;
    popts.max_passes = opts.max_propagation_passes;
    return propagate_bounds(rows, integral, lb, ub, popts);
}

/// The per-column search box handed to variable creation, and which columns are
/// integral. `lb`/`ub` are still in MPS units here: the `inf_clamp` fallback is
/// applied per column when the variable itself is created.
struct ColumnBoxes {
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<uint8_t> integral;
};

/// The declared column boxes, tightened by propagation when it is enabled.
/// `stats` records the outcome; it is left all-zero (bar the verdict) when
/// propagation proved the relaxation empty and its result was discarded.
ColumnBoxes column_boxes(const MpsProblem& prob, const RowMatrix& mat,
                         const MpsToModelOptions& opts, BoundPropagationStats& stats) {
    const auto n_cols = prob.vars.size();
    ColumnBoxes box;
    box.lb.resize(n_cols);
    box.ub.resize(n_cols);
    box.integral.resize(n_cols);
    for (std::size_t j = 0; j < n_cols; ++j) {
        // A NaN bound is "no bound", which is how clamp_lo/clamp_hi already
        // read it. Sanitising here keeps those guards reachable and keeps the two
        // adapters symmetric — without it, `propagate_bounds` rejects the NaN and
        // the same file would build under --no-propagate-bounds but not by
        // default, a divergence that has nothing to do with propagation.
        box.lb[j] = std::isnan(prob.vars[j].lb) ? -kMpsInf : prob.vars[j].lb;
        box.ub[j] = std::isnan(prob.vars[j].ub) ? kMpsInf : prob.vars[j].ub;
        box.integral[j] = prob.vars[j].kind == MpsVarKind::Continuous ? 0 : 1;
    }
    if (!opts.propagate_bounds) {
        return box;
    }
    const std::vector<double> raw_lb = box.lb;
    const std::vector<double> raw_ub = box.ub;
    stats = tighten_column_bounds(prob, mat, opts, box.lb, box.ub, box.integral);
    if (stats.infeasible) {
        // Propagation proved the linear relaxation empty. That is either a
        // genuinely infeasible instance or numerical trouble; either way the
        // honest thing is to hand the search the box the file declared and
        // let it report what it finds, rather than a derived empty one.
        box.lb = raw_lb;
        box.ub = raw_ub;
        // Nothing was applied, so the counts must not say otherwise; only
        // the verdict survives.
        stats = BoundPropagationStats{};
        stats.infeasible = true;
    }
    return box;
}

/// One CBLS variable built from one MPS column, and whether `inf_clamp` or the
/// int32 clip had to narrow it. The flag travels with the handle because the
/// caller counts *columns* that were narrowed, not narrowings.
struct ColumnVar {
    int32_t handle = 0;
    bool clamped = false;
};

/// Map one MPS column onto a CBLS variable of the matching type, substituting
/// `inf_clamp` where the column has no bound at all. Throws if the column's
/// domain is empty once its kind's rounding has been applied.
ColumnVar add_column(Model& m, const MpsVar& v, double box_lb, double box_ub,
                     const MpsToModelOptions& opts) {
    ColumnVar out;
    const double lb = clamp_lo(box_lb, opts.inf_clamp);
    const double ub = clamp_hi(box_ub, opts.inf_clamp);
    // One column, one count. The int32 clip below can narrow the *same*
    // column again, and this is a count of columns, not of narrowings.
    out.clamped = lb != box_lb || ub != box_ub;
    if (lb > ub) {
        throw std::runtime_error("MPS column " + v.name + " has lb > ub after clamping");
    }
    // Note: bool_var / int_var / float_var return already-encoded
    // negative variable handles, *not* raw var ids.
    if (v.kind == MpsVarKind::Binary) {
        // Binary: enforce {0,1}. CBLS' bool_var has fixed [0,1] bounds;
        // use it whenever the MPS bounds align with {0,1}, otherwise
        // fall back to int_var with the explicit bounds.
        const int ilb = static_cast<int>(std::lround(std::min(1.0, std::max(0.0, lb))));
        const int iub = static_cast<int>(std::lround(std::max(0.0, std::min(1.0, ub))));
        if (ilb > iub) {
            throw std::runtime_error("MPS binary column " + v.name +
                                     " has empty integer domain after rounding");
        }
        out.handle = (ilb == 0 && iub == 1) ? m.bool_var(v.name) : m.int_var(ilb, iub, v.name);
        return out;
    }
    if (v.kind == MpsVarKind::Integer) {
        // Integer: round bounds inward to nearest integers. The rounding
        // stays in double: a finite bound is honoured however wide now, and
        // anything below the 1e20 sentinel can reach here — `long long`
        // cannot represent all of that, and the conversion would be UB.
        const double dlb = std::ceil(lb);
        const double dub = std::floor(ub);
        if (dlb > dub) {
            throw std::runtime_error("MPS integer column " + v.name +
                                     " has empty integer domain after rounding");
        }
        // CBLS int_var takes int — clip to the int32 range, on *both* sides
        // of both bounds: a column bounded entirely above INT_MAX would
        // otherwise keep an unclipped lower bound and invert. That is a
        // representational limit, not an implied bound, so where it bites it
        // narrows the column and counts as clamped.
        constexpr auto kIntLo = static_cast<double>(std::numeric_limits<int>::min());
        constexpr auto kIntHi = static_cast<double>(std::numeric_limits<int>::max());
        out.clamped = out.clamped || dlb < kIntLo || dub > kIntHi;
        const double ilb = std::min(std::max(dlb, kIntLo), kIntHi);
        const double iub = std::min(std::max(dub, kIntLo), kIntHi);
        out.handle = m.int_var(static_cast<int>(ilb), static_cast<int>(iub), v.name);
        return out;
    }
    out.handle = m.float_var(lb, ub, v.name);
    return out;
}

/// Build `sum_j coef_j * x_j` as a CBLS sum node, from parallel (col, coef)
/// arrays. Constraint rows pass a slice of the CSR; the objective row, which the
/// CSR excludes, passes arrays gathered from its own index list.
int32_t build_lin_expr(Model& m, const std::vector<int32_t>& var_handles, const int32_t* cols,
                       const double* coefs, int32_t nnz) {
    std::vector<int32_t> terms;
    terms.reserve(static_cast<std::size_t>(nnz));
    for (int32_t t = 0; t < nnz; ++t) {
        const int32_t var_handle = var_handles[cols[t]];
        const double value = coefs[t];
        if (value == 1.0) {
            terms.push_back(var_handle);
        } else if (value == -1.0) {
            terms.push_back(m.neg(var_handle));
        } else {
            terms.push_back(m.prod(m.constant(value), var_handle));
        }
    }
    if (terms.empty()) {
        return m.constant(0.0);
    }
    if (terms.size() == 1) {
        return terms[0];
    }
    return m.sum(terms);
}

/// Translate one row's sense (with optional range) into the one or two CBLS
/// constraints it stands for, and return the primary constraint node.
///
///   L: lhs <= rhs;            range -> lhs >= rhs - |range|
///   G: lhs >= rhs;            range -> lhs <= rhs + |range|
///   E: lhs == rhs (range>0 -> [rhs,rhs+r], range<0 -> [rhs+r,rhs])
int32_t add_row_constraints(Model& m, const MpsRow& r, int32_t lhs) {
    const int32_t rhs_node = m.constant(r.rhs);
    const double rng = r.range;
    int32_t cn = -1;
    switch (r.sense) {
        case MpsRowSense::L:
            cn = m.leq(lhs, rhs_node);
            m.add_constraint(cn);
            if (rng != 0.0) {
                m.add_constraint(m.geq(lhs, m.constant(r.rhs - std::abs(rng))));
            }
            break;
        case MpsRowSense::G:
            cn = m.geq(lhs, rhs_node);
            m.add_constraint(cn);
            if (rng != 0.0) {
                m.add_constraint(m.leq(lhs, m.constant(r.rhs + std::abs(rng))));
            }
            break;
        case MpsRowSense::E:
            if (rng > 0.0) {
                // `hi` before the geq below, so the DAG node ids this row
                // creates are unchanged from before the split.
                const int32_t hi = m.constant(r.rhs + rng);
                m.add_constraint(m.geq(lhs, rhs_node));
                cn = m.leq(lhs, hi);
                m.add_constraint(cn);
            } else if (rng < 0.0) {
                m.add_constraint(m.geq(lhs, m.constant(r.rhs + rng)));  // rng < 0
                cn = m.leq(lhs, rhs_node);
                m.add_constraint(cn);
            } else {
                cn = m.eq_expr(lhs, rhs_node);
                m.add_constraint(cn);
            }
            break;
    }
    return cn;
}

/// Set the model objective from the 'N' row's terms plus any RHS offset, and
/// return its node id. Returns -1 when the file declared neither.
int32_t add_objective(Model& m, const MpsProblem& prob, const RowMatrix& mat,
                      const std::vector<int32_t>& var_handles) {
    if (mat.obj_nz.empty() && prob.objective_offset == 0.0) {
        return -1;
    }
    std::vector<int32_t> obj_cols;
    std::vector<double> obj_coefs;
    obj_cols.reserve(mat.obj_nz.size());
    obj_coefs.reserve(mat.obj_nz.size());
    for (int k : mat.obj_nz) {
        obj_cols.push_back(prob.nonzeros[k].col_idx);
        obj_coefs.push_back(prob.nonzeros[k].value);
    }
    int32_t obj_node = mat.obj_nz.empty()
                           ? m.constant(0.0)
                           : build_lin_expr(m, var_handles, obj_cols.data(), obj_coefs.data(),
                                            static_cast<int32_t>(obj_cols.size()));
    if (prob.objective_offset != 0.0) {
        obj_node = m.sum({obj_node, m.constant(prob.objective_offset)});
    }
    // `Model::minimize` rejects raw variable handles. If the linear
    // objective collapsed to a single var, wrap it in a sum node.
    if (obj_node < 0) {
        obj_node = m.sum({obj_node});
    }
    m.minimize(obj_node);
    return obj_node;
}

}  // namespace

MpsToModelResult mps_to_model(const MpsProblem& prob, const MpsToModelOptions& opts) {
    if (prob.maximize) {
        throw std::runtime_error(
            "MPS: OBJSENSE MAX is not supported by mps_to_model (CBLS expects "
            "minimisation). Negate the objective coefficients in the input or extend "
            "this adapter.");
    }

    MpsToModelResult result;
    Model& m = result.model;

    const int n_cols = static_cast<int>(prob.vars.size());
    const int n_rows = static_cast<int>(prob.rows.size());

    const RowMatrix mat = build_row_matrix(prob, n_rows);

    // Implied bounds run before variable creation, so the derived box is what
    // the engine sees.
    const ColumnBoxes box = column_boxes(prob, mat, opts, result.bound_stats);

    result.var_handles.reserve(n_cols);
    for (int j = 0; j < n_cols; ++j) {
        const auto col = static_cast<std::size_t>(j);
        const ColumnVar cv = add_column(m, prob.vars[col], box.lb[col], box.ub[col], opts);
        if (cv.clamped) {
            ++result.n_clamped_columns;
        }
        result.var_handles.push_back(cv.handle);
    }

    result.constraint_node_ids.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const int32_t lhs = build_lin_expr(
            m, result.var_handles, mat.cols.data() + mat.row_start[i],
            mat.coefs.data() + mat.row_start[i], mat.row_start[i + 1] - mat.row_start[i]);
        result.constraint_node_ids.push_back(add_row_constraints(m, prob.rows[i], lhs));
    }

    result.objective_node_id = add_objective(m, prob, mat, result.var_handles);

    m.close();
    return result;
}

}  // namespace cbls
