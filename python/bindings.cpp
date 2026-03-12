#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <cbls/cbls.h>

namespace nb = nanobind;
using namespace cbls;

NB_MODULE(_cbls_core, m) {
    m.doc() = "CBLS: Constraint-Based Local Search engine (C++ core)";

    // VarType enum
    nb::enum_<VarType>(m, "VarType")
        .value("Bool", VarType::Bool)
        .value("Int", VarType::Int)
        .value("Float", VarType::Float)
        .value("List", VarType::List)
        .value("Set", VarType::Set);

    // NodeOp enum
    nb::enum_<NodeOp>(m, "NodeOp")
        .value("Const", NodeOp::Const)
        .value("Neg", NodeOp::Neg)
        .value("Sum", NodeOp::Sum)
        .value("Prod", NodeOp::Prod)
        .value("Div", NodeOp::Div)
        .value("Pow", NodeOp::Pow)
        .value("Min", NodeOp::Min)
        .value("Max", NodeOp::Max)
        .value("Abs", NodeOp::Abs)
        .value("Sin", NodeOp::Sin)
        .value("Cos", NodeOp::Cos)
        .value("If", NodeOp::If)
        .value("At", NodeOp::At)
        .value("Count", NodeOp::Count)
        .value("Lambda", NodeOp::Lambda)
        .value("Leq", NodeOp::Leq)
        .value("Eq", NodeOp::Eq);

    // Variable (read-only access)
    nb::class_<Variable>(m, "Variable")
        .def_ro("id", &Variable::id)
        .def_ro("type", &Variable::type)
        .def_rw("value", &Variable::value)
        .def_ro("lb", &Variable::lb)
        .def_ro("ub", &Variable::ub)
        .def_ro("name", &Variable::name)
        .def_rw("elements", &Variable::elements);

    // ExprNode (read-only access)
    nb::class_<ExprNode>(m, "ExprNode")
        .def_ro("id", &ExprNode::id)
        .def_ro("op", &ExprNode::op)
        .def_ro("value", &ExprNode::value);

    // SearchResult
    nb::class_<SearchResult>(m, "SearchResult")
        .def_ro("objective", &SearchResult::objective)
        .def_ro("feasible", &SearchResult::feasible)
        .def_ro("iterations", &SearchResult::iterations)
        .def_ro("time_seconds", &SearchResult::time_seconds);

    // AdaptiveLambda
    nb::class_<AdaptiveLambda>(m, "AdaptiveLambda")
        .def(nb::init<double>(), nb::arg("initial_lambda") = 1.0)
        .def_rw("lambda_", &AdaptiveLambda::lambda_)
        .def("update", &AdaptiveLambda::update);

    // Model
    nb::class_<Model>(m, "Model")
        .def(nb::init<>())
        // Variable creation
        .def("bool_var", &Model::bool_var, nb::arg("name") = "")
        .def("int_var", &Model::int_var, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("float_var", &Model::float_var, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("list_var", &Model::list_var, nb::arg("n"), nb::arg("name") = "")
        .def("set_var", &Model::set_var, nb::arg("n"), nb::arg("min_size") = 0,
             nb::arg("max_size") = -1, nb::arg("name") = "")
        // Expression creation
        .def("constant", &Model::constant)
        .def("neg", &Model::neg)
        .def("sum", &Model::sum)
        .def("prod", &Model::prod)
        .def("div_expr", &Model::div_expr)
        .def("pow_expr", &Model::pow_expr)
        .def("min_expr", &Model::min_expr)
        .def("max_expr", &Model::max_expr)
        .def("abs_expr", &Model::abs_expr)
        .def("sin_expr", &Model::sin_expr)
        .def("cos_expr", &Model::cos_expr)
        .def("if_then_else", &Model::if_then_else)
        .def("at", &Model::at)
        .def("count", &Model::count)
        .def("leq", &Model::leq)
        .def("eq_expr", &Model::eq_expr)
        .def("lambda_sum", &Model::lambda_sum)
        // Constraint and objective
        .def("add_constraint", &Model::add_constraint)
        .def("minimize", &Model::minimize)
        .def("maximize", &Model::maximize)
        .def("close", &Model::close)
        // Accessors
        .def("var", &Model::var, nb::rv_policy::reference_internal)
        .def("var_mut", &Model::var_mut, nb::rv_policy::reference_internal)
        .def("node", &Model::node, nb::rv_policy::reference_internal)
        .def("objective_id", &Model::objective_id)
        .def("constraint_ids", &Model::constraint_ids)
        .def("num_vars", &Model::num_vars)
        .def("num_nodes", &Model::num_nodes);

    // ViolationManager
    nb::class_<ViolationManager>(m, "ViolationManager")
        .def(nb::init<Model&>())
        .def("constraint_violation", &ViolationManager::constraint_violation)
        .def("total_violation", &ViolationManager::total_violation)
        .def("augmented_objective", &ViolationManager::augmented_objective)
        .def("is_feasible", &ViolationManager::is_feasible, nb::arg("tol") = 1e-9)
        .def("violated_constraints", &ViolationManager::violated_constraints, nb::arg("tol") = 1e-9)
        .def("bump_weights", &ViolationManager::bump_weights, nb::arg("factor") = 1.0)
        .def_rw("adaptive_lambda", &ViolationManager::adaptive_lambda);

    // Free functions
    m.def("full_evaluate", &full_evaluate);
    m.def("delta_evaluate", &delta_evaluate);
    m.def("compute_partial", &compute_partial);
    m.def("solve", &cbls::solve,
          nb::arg("model"), nb::arg("time_limit") = 10.0,
          nb::arg("seed") = 42, nb::arg("use_fj") = true);
}
