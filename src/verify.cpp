#include <cbls/verify.h>
#include <cmath>
#include <string>

namespace cbls {

namespace {

std::string var_label(const Variable& v, size_t i) {
    return v.name.empty() ? "var[" + std::to_string(i) + "]"
                          : "var[" + std::to_string(i) + "] '" + v.name + "'";
}

// 1. Variable bounds and type checks.
void check_variables(const Model& model, double tol, VerifyResult& result) {
    for (size_t i = 0; i < model.num_vars(); ++i) {
        const auto& v = model.var(static_cast<int32_t>(i));
        std::string name = var_label(v, i);

        // Bounds check (for scalar types)
        if (v.type == VarType::Bool || v.type == VarType::Int || v.type == VarType::Float) {
            if (v.value < v.lb - tol) {
                result.add_error(
                    {VerifyError::Kind::VarBounds, name, v.lb, v.value, "value below lower bound"});
            }
            if (v.value > v.ub + tol) {
                result.add_error(
                    {VerifyError::Kind::VarBounds, name, v.ub, v.value, "value above upper bound"});
            }
        }

        // Bool must be 0 or 1
        if (v.type == VarType::Bool) {
            if (std::abs(v.value) > tol && std::abs(v.value - 1.0) > tol) {
                result.add_error(
                    {VerifyError::Kind::VarIntegrality, name, 0.0, v.value, "Bool var not 0 or 1"});
            }
        }

        // Int must be integral
        if (v.type == VarType::Int) {
            double rounded = std::round(v.value);
            if (std::abs(v.value - rounded) > tol) {
                result.add_error({VerifyError::Kind::VarIntegrality, name, rounded, v.value,
                                  "Int var not integral"});
            }
        }
    }
}

// 2. Constraint feasibility: each constraint node value <= tol.
void check_constraints(const Model& model, double tol, VerifyResult& result) {
    for (int32_t cid : model.constraint_ids()) {
        double val = model.node(cid).value;
        if (val > tol) {
            std::string name = "constraint[" + std::to_string(cid) + "]";
            result.add_error(
                {VerifyError::Kind::ConstraintViolation, name, 0.0, val, "constraint violated"});
        }
    }
}

// 3. DAG consistency: re-evaluate each node and compare against stored value.
void check_dag_consistency(const Model& model, double tol, VerifyResult& result) {
    for (const auto& node : model.nodes()) {
        double recomputed = evaluate(node, model);
        if (std::abs(recomputed - node.value) > tol) {
            std::string name = "node[" + std::to_string(node.id) + "]";
            result.add_error({VerifyError::Kind::DagConsistency, name, recomputed, node.value,
                              "DAG node value inconsistent with recomputation"});
        }
    }
}

}  // namespace

VerifyResult verify_model(const Model& model, double tol) {
    VerifyResult result;
    check_variables(model, tol, result);
    check_constraints(model, tol, result);
    check_dag_consistency(model, tol, result);
    return result;
}

}  // namespace cbls
