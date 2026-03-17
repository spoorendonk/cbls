#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <cbls/cbls.h>

namespace nb = nanobind;
using namespace cbls;

struct PySolveCallback : SolveCallback {
    NB_TRAMPOLINE(SolveCallback, 1);
    void on_progress(const SolveProgress& p) override {
        NB_OVERRIDE_PURE(on_progress, p);
    }
};

NB_MODULE(_cbls_core, m) {
    m.doc() = "CBLS: Constraint-Based Local Search engine (C++ core)";

    // Exception translators
    nb::register_exception_translator([](const std::exception_ptr &p, void *) {
        try {
            std::rethrow_exception(p);
        } catch (const std::out_of_range &e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::invalid_argument &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

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
        .value("Eq", NodeOp::Eq)
        .value("Tan", NodeOp::Tan)
        .value("Exp", NodeOp::Exp)
        .value("Log", NodeOp::Log)
        .value("Sqrt", NodeOp::Sqrt)
        .value("Geq", NodeOp::Geq)
        .value("Neq", NodeOp::Neq)
        .value("Lt", NodeOp::Lt)
        .value("Gt", NodeOp::Gt);

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
        .def("tan_expr", &Model::tan_expr)
        .def("exp_expr", &Model::exp_expr)
        .def("log_expr", &Model::log_expr)
        .def("sqrt_expr", &Model::sqrt_expr)
        .def("if_then_else", &Model::if_then_else)
        .def("at", &Model::at)
        .def("count", &Model::count)
        .def("leq", &Model::leq)
        .def("eq_expr", &Model::eq_expr)
        .def("geq", &Model::geq)
        .def("neq", &Model::neq)
        .def("lt", &Model::lt)
        .def("gt", &Model::gt)
        .def("lambda_sum", &Model::lambda_sum)
        // Constraint and objective (int32_t overloads)
        .def("add_constraint", static_cast<void(Model::*)(int32_t)>(&Model::add_constraint))
        .def("minimize", static_cast<void(Model::*)(int32_t)>(&Model::minimize))
        .def("maximize", static_cast<void(Model::*)(int32_t)>(&Model::maximize))
        // Constraint and objective (Expr overloads)
        .def("add_constraint", static_cast<void(Model::*)(const Expr&)>(&Model::add_constraint))
        .def("minimize", static_cast<void(Model::*)(const Expr&)>(&Model::minimize))
        .def("maximize", static_cast<void(Model::*)(const Expr&)>(&Model::maximize))
        .def("close", &Model::close)
        // Accessors
        .def("var", &Model::var, nb::rv_policy::reference_internal)
        .def("var_mut", &Model::var_mut, nb::rv_policy::reference_internal)
        .def("node", &Model::node, nb::rv_policy::reference_internal)
        .def("objective_id", &Model::objective_id)
        .def("constraint_ids", &Model::constraint_ids)
        .def("num_vars", &Model::num_vars)
        .def("num_nodes", &Model::num_nodes)
        // State snapshot/restore
        .def("copy_state", &Model::copy_state)
        .def("restore_state", &Model::restore_state)
        // Expr-returning variable creation
        .def("Bool", &Model::Bool, nb::arg("name") = "")
        .def("Int", &Model::Int, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("Float", &Model::Float, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("List", &Model::List, nb::arg("n"), nb::arg("name") = "")
        .def("Set", &Model::Set, nb::arg("n"), nb::arg("min_size") = 0,
             nb::arg("max_size") = -1, nb::arg("name") = "")
        .def("Constant", &Model::Constant);

    // Expr
    nb::class_<Expr>(m, "Expr")
        .def_ro("model", &Expr::model)
        .def_ro("handle", &Expr::handle)
        .def("var_id", &Expr::var_id)
        .def("__add__", [](const Expr& a, const Expr& b) { return a + b; })
        .def("__add__", [](const Expr& a, double b) { return a + b; })
        .def("__radd__", [](const Expr& a, double b) { return b + a; })
        .def("__mul__", [](const Expr& a, const Expr& b) { return a * b; })
        .def("__mul__", [](const Expr& a, double b) { return a * b; })
        .def("__rmul__", [](const Expr& a, double b) { return b * a; })
        .def("__sub__", [](const Expr& a, const Expr& b) { return a - b; })
        .def("__sub__", [](const Expr& a, double b) { return a - b; })
        .def("__rsub__", [](const Expr& a, double b) { return b - a; })
        .def("__truediv__", [](const Expr& a, const Expr& b) { return a / b; })
        .def("__truediv__", [](const Expr& a, double b) { return a / b; })
        .def("__rtruediv__", [](const Expr& a, double b) { return b / a; })
        .def("__neg__", [](const Expr& a) { return -a; })
        .def("__pow__", [](const Expr& a, const Expr& b) { return a.pow(b); })
        .def("__pow__", [](const Expr& a, double b) { return a.pow(Expr{a.model, a.model->constant(b)}); })
        .def("__pow__", [](const Expr& a, int b) { return a.pow(Expr{a.model, a.model->constant(static_cast<double>(b))}); })
        .def("__rpow__", [](const Expr& a, double b) {
            return Expr{a.model, a.model->pow_expr(a.model->constant(b), a.handle)};
        })
        .def("__le__", [](const Expr& a, const Expr& b) { return a <= b; })
        .def("__le__", [](const Expr& a, double b) { return a <= b; })
        .def("__ge__", [](const Expr& a, const Expr& b) { return a >= b; })
        .def("__ge__", [](const Expr& a, double b) { return a >= b; })
        .def("__lt__", [](const Expr& a, const Expr& b) { return a < b; })
        .def("__lt__", [](const Expr& a, double b) { return a < b; })
        .def("__gt__", [](const Expr& a, const Expr& b) { return a > b; })
        .def("__gt__", [](const Expr& a, double b) { return a > b; })
        .def("__abs__", [](const Expr& a) { return cbls::abs(a); })
        .def("is_var", &Expr::is_var)
        .def("eq", &Expr::eq)
        .def("neq", &Expr::neq)
        .def("pow", &Expr::pow);

    // Expr free functions
    m.def("sin", [](const Expr& x) { return cbls::sin(x); });
    m.def("cos", [](const Expr& x) { return cbls::cos(x); });
    m.def("tan", [](const Expr& x) { return cbls::tan(x); });
    m.def("exp", [](const Expr& x) { return cbls::exp(x); });
    m.def("log", [](const Expr& x) { return cbls::log(x); });
    m.def("sqrt", [](const Expr& x) { return cbls::sqrt(x); });
    m.def("abs", [](const Expr& x) { return cbls::abs(x); });
    m.def("pow", [](const Expr& base, const Expr& exp) { return cbls::pow(base, exp); });
    m.def("min", [](const std::vector<Expr>& args) { return cbls::min(args); });
    m.def("max", [](const std::vector<Expr>& args) { return cbls::max(args); });
    m.def("if_then_else", [](const Expr& cond, const Expr& then_, const Expr& else_) {
        return cbls::if_then_else(cond, then_, else_);
    });

    // Model::State
    nb::class_<Model::State>(m, "ModelState")
        .def(nb::init<>())
        .def_rw("values", &Model::State::values)
        .def_rw("elements", &Model::State::elements);

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

    // RNG
    nb::class_<RNG>(m, "RNG")
        .def(nb::init<uint64_t>(), nb::arg("seed") = 42)
        .def("uniform", &RNG::uniform)
        .def("integers", &RNG::integers)
        .def("normal", &RNG::normal)
        .def("random", &RNG::random)
        .def("seed", &RNG::seed);

    // Move::Change
    nb::class_<Move::Change>(m, "MoveChange")
        .def(nb::init<>())
        .def_rw("var_id", &Move::Change::var_id)
        .def_rw("new_value", &Move::Change::new_value)
        .def_rw("new_elements", &Move::Change::new_elements);

    // Move
    nb::class_<Move>(m, "Move")
        .def(nb::init<>())
        .def_rw("changes", &Move::changes)
        .def_rw("move_type", &Move::move_type)
        .def_rw("delta_F", &Move::delta_F);

    // SavedValues
    nb::class_<SavedValues>(m, "SavedValues")
        .def(nb::init<>())
        .def_rw("values", &SavedValues::values)
        .def_rw("elements", &SavedValues::elements);

    // MoveProbabilities
    nb::class_<MoveProbabilities>(m, "MoveProbabilities")
        .def(nb::init<const std::vector<std::string>&>())
        .def("select", &MoveProbabilities::select)
        .def("update", &MoveProbabilities::update)
        .def("probabilities", &MoveProbabilities::probabilities);

    // LNS
    nb::class_<LNS>(m, "LNS")
        .def(nb::init<double>(), nb::arg("destroy_fraction") = 0.3)
        .def("destroy_repair", &LNS::destroy_repair)
        .def("destroy_repair_cycle", &LNS::destroy_repair_cycle,
             nb::arg("model"), nb::arg("vm"), nb::arg("rng"), nb::arg("n_rounds") = 10);

    // SolutionPool
    nb::class_<Solution>(m, "Solution")
        .def(nb::init<>())
        .def_rw("state", &Solution::state)
        .def_rw("objective", &Solution::objective)
        .def_rw("feasible", &Solution::feasible);

    nb::class_<SolutionPool>(m, "SolutionPool")
        .def(nb::init<int>(), nb::arg("capacity") = 10)
        .def("submit", &SolutionPool::submit)
        .def("best", &SolutionPool::best)
        .def("size", &SolutionPool::size);

    // Free functions
    m.def("full_evaluate", &full_evaluate);
    m.def("delta_evaluate", &delta_evaluate);
    m.def("compute_partial", &compute_partial);
    m.def("generate_standard_moves", &generate_standard_moves);
    m.def("newton_tight_move", &newton_tight_move);
    m.def("gradient_lift_move", &gradient_lift_move,
          nb::arg("var_id"), nb::arg("model"), nb::arg("step_size") = 0.1);
    m.def("apply_move", &apply_move);
    m.def("save_move_values", &save_move_values);
    m.def("undo_move", &undo_move);
    // InnerSolverHook + FloatIntensifyHook
    nb::class_<InnerSolverHook>(m, "InnerSolverHook");

    nb::class_<FloatIntensifyHook, InnerSolverHook>(m, "FloatIntensifyHook")
        .def(nb::init<>())
        .def_rw("max_sweeps", &FloatIntensifyHook::max_sweeps)
        .def_rw("initial_step_size", &FloatIntensifyHook::initial_step_size)
        .def_rw("max_line_search_steps", &FloatIntensifyHook::max_line_search_steps)
        .def_rw("max_multi_var_constraints", &FloatIntensifyHook::max_multi_var_constraints);

    // SolveProgress
    nb::class_<SolveProgress>(m, "SolveProgress")
        .def(nb::init<>())
        .def_ro("iteration", &SolveProgress::iteration)
        .def_ro("time_seconds", &SolveProgress::time_seconds)
        .def_ro("objective", &SolveProgress::objective)
        .def_ro("total_violation", &SolveProgress::total_violation)
        .def_ro("temperature", &SolveProgress::temperature)
        .def_ro("feasible", &SolveProgress::feasible)
        .def_ro("new_best", &SolveProgress::new_best)
        .def_ro("reheat_count", &SolveProgress::reheat_count);

    // SolveCallback with trampoline for Python subclassing
    nb::class_<SolveCallback, PySolveCallback>(m, "SolveCallback")
        .def(nb::init<>())
        .def("on_progress", [](SolveCallback& self, const SolveProgress& p) {
            self.on_progress(p);
        });

    m.def("solve", &cbls::solve,
          nb::arg("model"), nb::arg("time_limit") = 10.0,
          nb::arg("seed") = 42, nb::arg("use_fj") = true,
          nb::arg("hook") = nullptr,
          nb::arg("lns") = nullptr,
          nb::arg("lns_interval") = 3,
          nb::arg("callback") = nullptr);
    m.def("initialize_random", &initialize_random);
    m.def("fj_nl_initialize", &fj_nl_initialize,
          nb::arg("model"), nb::arg("vm"), nb::arg("max_iterations") = 10000,
          nb::arg("rng") = nullptr, nb::arg("time_limit") = 2.0);
}
