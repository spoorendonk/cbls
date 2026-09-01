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

/// The row's body bounds `lo <= body <= hi`, matching exactly the constraints
/// the adapter goes on to build from the same sense/rhs/range triple.
void row_bounds(const MpsRow& r, double& lo, double& hi) {
    const double rng = std::abs(r.range);
    switch (r.sense) {
        case MpsRowSense::L:
            hi = r.rhs;
            lo = r.range != 0.0 ? r.rhs - rng : -kMpsInf;
            break;
        case MpsRowSense::G:
            lo = r.rhs;
            hi = r.range != 0.0 ? r.rhs + rng : kMpsInf;
            break;
        case MpsRowSense::E:
            lo = r.range < 0.0 ? r.rhs + r.range : r.rhs;
            hi = r.range > 0.0 ? r.rhs + r.range : r.rhs;
            break;
    }
}

/// Derive implied column bounds from the rows. The rows are handed to
/// propagation as *views* into the caller's CSR arrays: the constraint matrix of
/// a large MIP runs to hundreds of megabytes and must not be duplicated here.
BoundPropagationStats tighten_column_bounds(const MpsProblem& prob,
                                            const std::vector<int32_t>& row_start,
                                            const std::vector<int32_t>& csr_cols,
                                            const std::vector<double>& csr_coefs,
                                            const MpsToModelOptions& opts, std::vector<double>& lb,
                                            std::vector<double>& ub,
                                            const std::vector<uint8_t>& integral) {
    std::vector<LinearRow> rows(prob.rows.size());
    for (std::size_t i = 0; i < prob.rows.size(); ++i) {
        LinearRow& row = rows[i];
        row_bounds(prob.rows[i], row.lo, row.hi);
        row.nnz = row_start[i + 1] - row_start[i];
        row.cols = csr_cols.data() + row_start[i];
        row.coefs = csr_coefs.data() + row_start[i];
    }
    BoundPropagationOptions popts;
    popts.max_passes = opts.max_propagation_passes;
    return propagate_bounds(rows, integral, lb, ub, popts);
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

    // ---------- Group nonzeros by row (objective row = -1) ----------
    // One flat CSR rather than a vector-per-row: it is the single structure both
    // the expression builder and bound propagation read, so the matrix is laid
    // out once instead of being grouped once and copied again.
    std::vector<int32_t> row_start(static_cast<std::size_t>(n_rows) + 1, 0);
    std::vector<int> obj_nz;
    for (const auto& nz : prob.nonzeros) {
        if (nz.row_idx >= 0 && nz.row_idx < n_rows) {
            ++row_start[static_cast<std::size_t>(nz.row_idx) + 1];
        }
    }
    for (int i = 0; i < n_rows; ++i) {
        row_start[static_cast<std::size_t>(i) + 1] += row_start[static_cast<std::size_t>(i)];
    }
    std::vector<int32_t> csr_cols(static_cast<std::size_t>(row_start[n_rows]));
    std::vector<double> csr_coefs(static_cast<std::size_t>(row_start[n_rows]));
    {
        std::vector<int32_t> fill(row_start.begin(), row_start.end() - 1);
        for (int k = 0; k < static_cast<int>(prob.nonzeros.size()); ++k) {
            const auto& nz = prob.nonzeros[k];
            if (nz.row_idx == -1) {
                obj_nz.push_back(k);
            } else if (nz.row_idx >= 0 && nz.row_idx < n_rows) {
                const std::size_t at = static_cast<std::size_t>(fill[nz.row_idx]++);
                csr_cols[at] = nz.col_idx;
                csr_coefs[at] = nz.value;
            }
        }
    }

    // ---------- Implied bounds ----------
    // Run before variable creation so the derived box is what the engine sees.
    // The objective row is excluded: it carries no bounds and implies nothing.
    std::vector<double> col_lb(n_cols);
    std::vector<double> col_ub(n_cols);
    std::vector<uint8_t> integral(n_cols);
    for (int j = 0; j < n_cols; ++j) {
        // A NaN bound is "no bound", which is how clamp_lo/clamp_hi below already
        // read it. Sanitising here keeps those guards reachable and keeps the two
        // adapters symmetric — without it, `propagate_bounds` rejects the NaN and
        // the same file would build under --no-propagate-bounds but not by
        // default, a divergence that has nothing to do with propagation.
        col_lb[j] = std::isnan(prob.vars[j].lb) ? -kMpsInf : prob.vars[j].lb;
        col_ub[j] = std::isnan(prob.vars[j].ub) ? kMpsInf : prob.vars[j].ub;
        integral[j] = prob.vars[j].kind == MpsVarKind::Continuous ? 0 : 1;
    }
    if (opts.propagate_bounds) {
        const std::vector<double> raw_lb = col_lb;
        const std::vector<double> raw_ub = col_ub;
        result.bound_stats = tighten_column_bounds(prob, row_start, csr_cols, csr_coefs, opts,
                                                   col_lb, col_ub, integral);
        if (result.bound_stats.infeasible) {
            // Propagation proved the linear relaxation empty. That is either a
            // genuinely infeasible instance or numerical trouble; either way the
            // honest thing is to hand the search the box the file declared and
            // let it report what it finds, rather than a derived empty one.
            col_lb = raw_lb;
            col_ub = raw_ub;
            // Nothing was applied, so the counts must not say otherwise; only
            // the verdict survives.
            result.bound_stats = BoundPropagationStats{};
            result.bound_stats.infeasible = true;
        }
    }

    // ---------- Variables ----------
    result.var_handles.reserve(n_cols);
    for (int j = 0; j < n_cols; ++j) {
        const MpsVar& v = prob.vars[j];
        double lb = clamp_lo(col_lb[j], opts.inf_clamp);
        double ub = clamp_hi(col_ub[j], opts.inf_clamp);
        // One column, one count. The int32 clip below can narrow the *same*
        // column again, and this is a count of columns, not of narrowings.
        bool clamped = lb != col_lb[j] || ub != col_ub[j];
        if (lb > ub) {
            throw std::runtime_error("MPS column " + v.name + " has lb > ub after clamping");
        }
        // Note: bool_var / int_var / float_var return already-encoded
        // negative variable handles, *not* raw var ids.
        int32_t handle;
        if (v.kind == MpsVarKind::Binary) {
            // Binary: enforce {0,1}. CBLS' bool_var has fixed [0,1] bounds;
            // use it whenever the MPS bounds align with {0,1}, otherwise
            // fall back to int_var with the explicit bounds.
            int ilb = static_cast<int>(std::lround(std::min(1.0, std::max(0.0, lb))));
            int iub = static_cast<int>(std::lround(std::max(0.0, std::min(1.0, ub))));
            if (ilb > iub) {
                throw std::runtime_error("MPS binary column " + v.name +
                                         " has empty integer domain after rounding");
            }
            if (ilb == 0 && iub == 1) {
                handle = m.bool_var(v.name);
            } else {
                handle = m.int_var(ilb, iub, v.name);
            }
        } else if (v.kind == MpsVarKind::Integer) {
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
            constexpr double kIntLo = static_cast<double>(std::numeric_limits<int>::min());
            constexpr double kIntHi = static_cast<double>(std::numeric_limits<int>::max());
            clamped = clamped || dlb < kIntLo || dub > kIntHi;
            const double ilb = std::min(std::max(dlb, kIntLo), kIntHi);
            const double iub = std::min(std::max(dub, kIntLo), kIntHi);
            handle = m.int_var(static_cast<int>(ilb), static_cast<int>(iub), v.name);
        } else {
            handle = m.float_var(lb, ub, v.name);
        }
        if (clamped) {
            ++result.n_clamped_columns;
        }
        result.var_handles.push_back(handle);
    }

    // Build sum_j coef_j * x_j as a CBLS sum node, from parallel (col, coef)
    // arrays. Constraint rows pass a slice of the CSR above; the objective row,
    // which the CSR excludes, passes arrays gathered from its own index list.
    auto build_lin_expr = [&](const int32_t* cols, const double* coefs, int32_t nnz) -> int32_t {
        std::vector<int32_t> terms;
        terms.reserve(static_cast<std::size_t>(nnz));
        for (int32_t t = 0; t < nnz; ++t) {
            int32_t var_handle = result.var_handles[cols[t]];
            const double value = coefs[t];
            if (value == 1.0) {
                terms.push_back(var_handle);
            } else if (value == -1.0) {
                terms.push_back(m.neg(var_handle));
            } else {
                int32_t c = m.constant(value);
                terms.push_back(m.prod(c, var_handle));
            }
        }
        if (terms.empty()) {
            return m.constant(0.0);
        }
        if (terms.size() == 1) {
            return terms[0];
        }
        return m.sum(terms);
    };

    // ---------- Constraints ----------
    result.constraint_node_ids.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const MpsRow& r = prob.rows[i];
        int32_t lhs =
            build_lin_expr(csr_cols.data() + row_start[i], csr_coefs.data() + row_start[i],
                           row_start[i + 1] - row_start[i]);
        int32_t rhs_node = m.constant(r.rhs);

        // Translate sense (with optional range) into CBLS constraints.
        // L: lhs <= rhs;            range -> lhs >= rhs - |range|
        // G: lhs >= rhs;            range -> lhs <= rhs + |range|
        // E: lhs == rhs (range>0 -> [rhs,rhs+r], range<0 -> [rhs+r,rhs])
        const double rng = r.range;
        int32_t cn = -1;
        switch (r.sense) {
            case MpsRowSense::L: {
                cn = m.leq(lhs, rhs_node);
                m.add_constraint(cn);
                if (rng != 0.0) {
                    int32_t lo = m.constant(r.rhs - std::abs(rng));
                    m.add_constraint(m.geq(lhs, lo));
                }
                break;
            }
            case MpsRowSense::G: {
                cn = m.geq(lhs, rhs_node);
                m.add_constraint(cn);
                if (rng != 0.0) {
                    int32_t hi = m.constant(r.rhs + std::abs(rng));
                    m.add_constraint(m.leq(lhs, hi));
                }
                break;
            }
            case MpsRowSense::E: {
                if (rng > 0.0) {
                    int32_t hi = m.constant(r.rhs + rng);
                    m.add_constraint(m.geq(lhs, rhs_node));
                    cn = m.leq(lhs, hi);
                    m.add_constraint(cn);
                } else if (rng < 0.0) {
                    int32_t lo = m.constant(r.rhs + rng);  // rng < 0
                    m.add_constraint(m.geq(lhs, lo));
                    cn = m.leq(lhs, rhs_node);
                    m.add_constraint(cn);
                } else {
                    cn = m.eq_expr(lhs, rhs_node);
                    m.add_constraint(cn);
                }
                break;
            }
        }
        result.constraint_node_ids.push_back(cn);
    }

    // ---------- Objective ----------
    if (!obj_nz.empty() || prob.objective_offset != 0.0) {
        std::vector<int32_t> obj_cols;
        std::vector<double> obj_coefs;
        obj_cols.reserve(obj_nz.size());
        obj_coefs.reserve(obj_nz.size());
        for (int k : obj_nz) {
            obj_cols.push_back(prob.nonzeros[k].col_idx);
            obj_coefs.push_back(prob.nonzeros[k].value);
        }
        int32_t obj_lin = obj_nz.empty() ? m.constant(0.0)
                                         : build_lin_expr(obj_cols.data(), obj_coefs.data(),
                                                          static_cast<int32_t>(obj_cols.size()));
        int32_t obj_node = obj_lin;
        if (prob.objective_offset != 0.0) {
            int32_t off = m.constant(prob.objective_offset);
            obj_node = m.sum({obj_lin, off});
        }
        // `Model::minimize` rejects raw variable handles. If the linear
        // objective collapsed to a single var, wrap it in a sum node.
        if (obj_node < 0) {
            obj_node = m.sum({obj_node});
        }
        m.minimize(obj_node);
        result.objective_node_id = obj_node;
    }

    m.close();
    return result;
}

}  // namespace cbls
