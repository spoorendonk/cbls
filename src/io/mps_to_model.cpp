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
#include <utility>
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

/// The node and child-reference counts the expression builder will spend on a
/// matrix with coefficients `coefs` and an objective with coefficients
/// `obj_coefs`, over `n_rows` rows, reserved on `m` up front. See the call in
/// `build_plain` for why this is counted rather than guessed.
template <typename ObjCoefs>
void reserve_model(Model& m, std::size_t n_cols, std::size_t n_rows,
                   const std::vector<double>& coefs, const ObjCoefs& obj_coefs) {
    std::size_t term_nodes = 0;
    std::size_t term_child_refs = 0;
    const auto count_term = [&term_nodes, &term_child_refs](double c) {
        if (c == 1.0) {
            term_child_refs += 1;  // the variable is used directly
            return;
        }
        term_nodes += (c == -1.0) ? 1 : 2;
        term_child_refs += (c == -1.0) ? 2 : 3;
    };
    for (const double c : coefs) {
        count_term(c);
    }
    obj_coefs(count_term);
    // Per row: at most the Sum, two comparisons (two children each) and their
    // two bound constants. Columns add no nodes -- `add_column` makes only the
    // variable, its box is the variable's bounds -- so they reserve none. The
    // slack covers the objective's offset constant and wrapping Sum, and the
    // `obj <= bound` row solve() appends. All small constant factors on counts
    // already known, so the total is close rather than lavish.
    m.reserve(n_cols, term_nodes + (5 * n_rows) + 16, term_child_refs + (4 * n_rows) + 16);
}

/// The model without presolve. Its node ids, node order and reservation are the
/// ones every published table was measured on; keep them unchanged.
void build_plain(const MpsProblem& prob, const RowMatrix& mat, const MpsToModelOptions& opts,
                 MpsToModelResult& result) {
    Model& m = result.model;

    const int n_cols = static_cast<int>(prob.vars.size());
    const int n_rows = static_cast<int>(prob.rows.size());

    // Sized from the matrix rather than grown into, so the node array is never
    // copied to grow it -- see Model::reserve for why that copy is the thing
    // worth avoiding rather than the allocation count.
    //
    // Counted, not guessed at. `build_lin_expr` spends nodes per coefficient at
    // three different rates (a 1.0 costs none, a -1.0 one Neg, anything else a
    // Const and a Prod), and the mix is a property of the instance: across the
    // MIPfeas roster the ratio runs from 0.88 nodes per nonzero to 1.87. A flat
    // "two per nonzero" would therefore over-reserve by up to 2.3x, and
    // over-reserving is not free here -- the benchmark driver caps ADDRESS
    // SPACE (`--mem-limit-gb` is `ulimit -v`), which an untouched reservation
    // counts against just as much as a used one.
    //
    // The DAG's edges are counted in the same pass: a coefficient's slot in its
    // row's Sum, plus the Neg's one child or the Prod's two.
    reserve_model(m, static_cast<std::size_t>(n_cols), static_cast<std::size_t>(n_rows), mat.coefs,
                  [&](const auto& count) {
                      for (const int k : mat.obj_nz) {
                          count(prob.nonzeros[static_cast<std::size_t>(k)].value);
                      }
                  });

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
}

/// The MPS problem as presolve's input. Takes the matrix by value and moves its
/// CSR arrays in: the matrix is not needed again on success, and on large
/// instances it is the largest allocation the build makes.
LinearProblem to_linear_problem(const MpsProblem& prob, RowMatrix mat) {
    LinearProblem lp;
    const std::size_t n_cols = prob.vars.size();
    lp.col_lb.resize(n_cols);
    lp.col_ub.resize(n_cols);
    lp.integral.resize(n_cols);
    lp.frozen.assign(n_cols, 0);
    lp.obj.assign(n_cols, 0.0);
    for (std::size_t j = 0; j < n_cols; ++j) {
        // NaN is "no bound", as in `column_boxes`; presolve rejects a NaN.
        lp.col_lb[j] = std::isnan(prob.vars[j].lb) ? -kMpsInf : prob.vars[j].lb;
        lp.col_ub[j] = std::isnan(prob.vars[j].ub) ? kMpsInf : prob.vars[j].ub;
        if (prob.vars[j].kind == MpsVarKind::Binary) {
            // The reader keeps `BV` then `UP 5` as Binary with ub 5; `add_column`
            // builds it in {0,1}, so presolve must not reason from the 5.
            lp.col_lb[j] = std::max(lp.col_lb[j], 0.0);
            lp.col_ub[j] = std::min(lp.col_ub[j], 1.0);
        }
        lp.integral[j] = prob.vars[j].kind == MpsVarKind::Continuous ? 0 : 1;
    }
    // A column listed twice in the objective row: its coefficients add up, as
    // the two terms do in the plain build's Sum. Summed with Neumaier's
    // compensation, as presolve's `load_row` merges a row's repeats: a plain
    // `1e16 + 1 - 1e16` gives 0, which would drop the column from the objective.
    // One extra double per column, O(n) beside the O(nnz) matrix.
    std::vector<double> compensation(n_cols, 0.0);
    for (const int k : mat.obj_nz) {
        const MpsNonzero& nz = prob.nonzeros[static_cast<std::size_t>(k)];
        const auto j = static_cast<std::size_t>(nz.col_idx);
        const double sum = lp.obj[j];
        const double t = sum + nz.value;
        compensation[j] +=
            std::abs(sum) >= std::abs(nz.value) ? (sum - t) + nz.value : (nz.value - t) + sum;
        lp.obj[j] = t;
    }
    for (std::size_t j = 0; j < n_cols; ++j) {
        lp.obj[j] += compensation[j];
    }
    lp.obj_offset = prob.objective_offset;
    lp.row_start.assign(mat.row_start.begin(), mat.row_start.end());
    lp.cols = std::move(mat.cols);
    lp.coefs = std::move(mat.coefs);
    lp.row_lo.resize(prob.rows.size());
    lp.row_hi.resize(prob.rows.size());
    for (std::size_t i = 0; i < prob.rows.size(); ++i) {
        const RowBounds rb = row_bounds(prob.rows[i]);
        lp.row_lo[i] = rb.lo;
        lp.row_hi[i] = rb.hi;
    }
    return lp;
}

/// The one or two constraints `lo <= lhs <= hi` stands for, and the primary
/// node: `eq_expr` when the sides meet, one comparison per finite side
/// otherwise. For a range the primary is the side the plain build makes primary
/// (`add_row_constraints`): `>=` for a G row, `<=` for L and E. -1 when neither
/// side is finite.
int32_t add_bounded_row(Model& m, double lo, double hi, int32_t lhs, MpsRowSense sense) {
    const bool has_lo = !is_unbounded_below(lo);
    const bool has_hi = !is_unbounded_above(hi);
    if (has_lo && has_hi && lo == hi) {
        const int32_t cn = m.eq_expr(lhs, m.constant(lo));
        m.add_constraint(cn);
        return cn;
    }
    int32_t lo_cn = -1;
    int32_t hi_cn = -1;
    if (has_lo) {
        lo_cn = m.geq(lhs, m.constant(lo));
        m.add_constraint(lo_cn);
    }
    if (has_hi) {
        hi_cn = m.leq(lhs, m.constant(hi));
        m.add_constraint(hi_cn);
    }
    if (lo_cn < 0 || (hi_cn >= 0 && sense != MpsRowSense::G)) {
        return hi_cn;
    }
    return lo_cn;
}

/// The objective of the reduced problem: its linear terms plus the offset,
/// which now carries the fixed columns' contribution. Built whenever the file
/// declared an objective, even one presolve reduced to a constant, so the node
/// the runner reads stays the objective the search minimises. -1 otherwise.
int32_t add_reduced_objective(Model& m, const MpsProblem& prob, const RowMatrix& mat,
                              const LinearProblem& red, const std::vector<int32_t>& var_handles) {
    if (mat.obj_nz.empty() && prob.objective_offset == 0.0) {
        return -1;
    }
    std::vector<int32_t> cols;
    std::vector<double> coefs;
    for (std::size_t j = 0; j < red.obj.size(); ++j) {
        if (red.obj[j] != 0.0) {
            cols.push_back(static_cast<int32_t>(j));
            coefs.push_back(red.obj[j]);
        }
    }
    int32_t obj_node = build_lin_expr(m, var_handles, cols.data(), coefs.data(),
                                      static_cast<int32_t>(cols.size()));
    if (red.obj_offset != 0.0) {
        obj_node = m.sum({obj_node, m.constant(red.obj_offset)});
    }
    // `Model::minimize` rejects a raw variable handle; wrap it, as
    // `add_objective` does.
    if (obj_node < 0) {
        obj_node = m.sum({obj_node});
    }
    m.minimize(obj_node);
    return obj_node;
}

/// The model built from the presolved problem. Returns false, with only
/// `result.presolve_stats` set, when presolve proved the problem infeasible:
/// the caller then builds the plain model, as `column_boxes` hands the search
/// the declared box when propagation proves it empty. Also false, with
/// `result.presolve_rejected` set, when presolve refuses the input (an infinite
/// or NaN side, a non-finite coefficient), which the plain build accepts:
/// `--presolve` must not turn a file the plain arm searches into a build error.
bool build_presolved(const MpsProblem& prob, RowMatrix mat, const MpsToModelOptions& opts,
                     MpsToModelResult& result) {
    PresolveOptions popts = opts.presolve;
    popts.propagate_bounds = opts.propagate_bounds;
    popts.propagation.max_passes = opts.max_propagation_passes;
    // Only the objective's index list is needed after this; the CSR moves into
    // the problem and is freed with it, before the model is built.
    RowMatrix obj_only;
    obj_only.obj_nz = mat.obj_nz;
    PresolveResult pr;
    try {
        pr = presolve(to_linear_problem(prob, std::move(mat)), popts);
    } catch (const std::invalid_argument& e) {  // validate() is presolve's only throw
        result.presolve_rejected = e.what();
        return false;
    }
    result.presolve_stats = pr.stats;
    if (pr.stats.status == PresolveStatus::Infeasible) {
        return false;
    }
    const LinearProblem& red = pr.reduced;
    Model& m = result.model;
    reserve_model(m, red.n_cols(), red.n_rows(), red.coefs, [&](const auto& count) {
        for (const double c : red.obj) {
            if (c != 0.0) {
                count(c);
            }
        }
    });

    result.var_handles.reserve(red.n_cols());
    for (std::size_t i = 0; i < red.n_cols(); ++i) {
        const auto col = static_cast<std::size_t>(pr.reduced_to_col[i]);
        const ColumnVar cv = add_column(m, prob.vars[col], red.col_lb[i], red.col_ub[i], opts);
        if (cv.clamped) {
            ++result.n_clamped_columns;
        }
        result.var_handles.push_back(cv.handle);
    }

    result.constraint_node_ids.assign(prob.rows.size(), -1);
    for (std::size_t r = 0; r < prob.rows.size(); ++r) {
        const int32_t k = pr.row_to_reduced[r];
        if (k < 0) {
            continue;
        }
        const auto row = static_cast<std::size_t>(k);
        const std::size_t begin = red.row_start[row];
        const int32_t lhs =
            build_lin_expr(m, result.var_handles, red.cols.data() + begin, red.coefs.data() + begin,
                           static_cast<int32_t>(red.row_start[row + 1] - begin));
        result.constraint_node_ids[r] =
            add_bounded_row(m, red.row_lo[row], red.row_hi[row], lhs, prob.rows[r].sense);
    }

    result.objective_node_id = add_reduced_objective(m, prob, obj_only, red, result.var_handles);
    result.postsolve = std::move(pr.postsolve);
    result.reduced_to_original = std::move(pr.reduced_to_col);
    return true;
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
    const int n_rows = static_cast<int>(prob.rows.size());
    if (opts.presolve.enabled &&
        build_presolved(prob, build_row_matrix(prob, n_rows), opts, result)) {
        result.model.close();
        return result;
    }
    // Without presolve, or after it proved the problem infeasible or refused
    // the input (the matrix moved into it is rebuilt here; O(nnz), and only on
    // that path).
    build_plain(prob, build_row_matrix(prob, n_rows), opts, result);
    result.model.close();
    return result;
}

std::vector<double> MpsToModelResult::original_values(const Model& model) const {
    const auto& vars = model.variables();
    std::vector<double> x;
    x.reserve(var_handles.size());
    for (const int32_t h : var_handles) {
        const int32_t id = handle_to_var_id(h);
        if (id < 0 || static_cast<std::size_t>(id) >= vars.size()) {
            throw std::invalid_argument("original_values: model does not match this build");
        }
        x.push_back(vars[static_cast<std::size_t>(id)].value);
    }
    if (postsolve.n_original() == 0) {
        return x;
    }
    return postsolve.apply(x);
}

}  // namespace cbls
