// Adapter: MpsProblem -> closed CBLS Model.

#include "cbls/expr.h"
#include "cbls/io_mps.h"
#include "cbls/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cbls {

namespace {

// Cap an infinite or very large bound to a finite value the CBLS engine
// can use in moves. We don't try to be clever — coordinates that would
// drift to infinity are typically not the ones we need to explore.
double clamp_lo(double lb, double inf_clamp) {
    if (!std::isfinite(lb) || lb < -inf_clamp) {
        return -inf_clamp;
    }
    return lb;
}
double clamp_hi(double ub, double inf_clamp) {
    if (!std::isfinite(ub) || ub > inf_clamp) {
        return inf_clamp;
    }
    return ub;
}

}  // namespace

MpsToModelResult mps_to_model(const MpsProblem& prob, const MpsToModelOptions& opts) {
    MpsToModelResult result;
    Model& m = result.model;

    const int n_cols = static_cast<int>(prob.vars.size());
    const int n_rows = static_cast<int>(prob.rows.size());

    // ---------- Variables ----------
    result.var_handles.reserve(n_cols);
    for (int j = 0; j < n_cols; ++j) {
        const MpsVar& v = prob.vars[j];
        double lb = clamp_lo(v.lb, opts.inf_clamp);
        double ub = clamp_hi(v.ub, opts.inf_clamp);
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
            int ilb = static_cast<int>(std::lround(std::max(0.0, lb)));
            int iub = static_cast<int>(std::lround(std::min(1.0, ub)));
            if (ilb == 0 && iub == 1) {
                handle = m.bool_var(v.name);
            } else {
                if (ilb > iub) {
                    ilb = 0;
                    iub = 1;
                }
                handle = m.int_var(ilb, iub, v.name);
            }
        } else if (v.kind == MpsVarKind::Integer) {
            // Integer: round bounds inward to nearest integers.
            long long ilb = static_cast<long long>(std::ceil(lb));
            long long iub = static_cast<long long>(std::floor(ub));
            if (ilb > iub) {
                throw std::runtime_error("MPS integer column " + v.name +
                                         " has empty integer domain after rounding");
            }
            // CBLS int_var takes int — clamp to int32 range.
            constexpr long long kMin = std::numeric_limits<int>::min();
            constexpr long long kMax = std::numeric_limits<int>::max();
            ilb = std::max<long long>(kMin, ilb);
            iub = std::min<long long>(kMax, iub);
            handle = m.int_var(static_cast<int>(ilb), static_cast<int>(iub), v.name);
        } else {
            handle = m.float_var(lb, ub, v.name);
        }
        result.var_handles.push_back(handle);
    }

    // ---------- Group nonzeros by row (objective row = -1) ----------
    std::vector<std::vector<int>> row_nz(n_rows);  // indices into prob.nonzeros
    std::vector<int> obj_nz;
    for (int k = 0; k < static_cast<int>(prob.nonzeros.size()); ++k) {
        const auto& nz = prob.nonzeros[k];
        if (nz.row_idx == -1) {
            obj_nz.push_back(k);
        } else if (nz.row_idx >= 0 && nz.row_idx < n_rows) {
            row_nz[nz.row_idx].push_back(k);
        }
    }

    auto build_lin_expr = [&](const std::vector<int>& nzlist) -> int32_t {
        // Build sum_j coef_j * x_j as a CBLS sum node.
        std::vector<int32_t> terms;
        terms.reserve(nzlist.size());
        for (int k : nzlist) {
            const auto& nz = prob.nonzeros[k];
            int32_t var_handle = result.var_handles[nz.col_idx];
            if (nz.value == 1.0) {
                terms.push_back(var_handle);
            } else if (nz.value == -1.0) {
                terms.push_back(m.neg(var_handle));
            } else {
                int32_t c = m.constant(nz.value);
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
        int32_t lhs = build_lin_expr(row_nz[i]);
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
        int32_t obj_lin = obj_nz.empty() ? m.constant(0.0) : build_lin_expr(obj_nz);
        int32_t obj_node = obj_lin;
        if (prob.objective_offset != 0.0) {
            int32_t off = m.constant(prob.objective_offset);
            obj_node = m.sum({obj_lin, off});
        }
        m.minimize(obj_node);
        result.objective_node_id = obj_node;
    }

    m.close();
    return result;
}

}  // namespace cbls
