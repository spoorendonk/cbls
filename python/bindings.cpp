#include <cbls/cbls.h>
#include <cbls/model_extension.h>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <unordered_set>

namespace nb = nanobind;
using namespace cbls;

// Shared by both ParallelSearch.solve overloads: the concurrency contract is
// the same for either, and it is not obvious from the signature. Releasing the
// GIL is what makes the methods callable at all (see the call guards below),
// and the direct consequence is that the factory runs on several threads at
// once.
constexpr const char* kParallelSolveDoc =
    "Run the parallel search and return the best result the workers found.\n"
    "\n"
    "The GIL is released for the duration of the C++ call, which is what makes\n"
    "this callable at all: the workers have to acquire the GIL to invoke a\n"
    "Python callable, and a caller holding it across the join deadlocks them.\n"
    "\n"
    "model_factory is invoked ONCE PER WORKER -- not once per restart -- each\n"
    "from that worker's own thread, so several calls are in flight at once. A\n"
    "progress callback is called from EVERY worker, serialized on one mutex\n"
    "inside the portfolio: rows arrive one at a time and in order, but on\n"
    "whichever worker thread produced them, and a slow callback throttles\n"
    "every worker rather than only its own. nanobind re-acquires\n"
    "the GIL around every\n"
    "invocation, so the interpreter itself is safe; what that does not give\n"
    "you is atomicity across bytecodes, so a callable that mutates\n"
    "Python-side state must lock it.\n"
    "\n"
    "`solve_parallel` accepts a Python hook_factory and lns_factory. Both are\n"
    "invoked once per worker, on that worker's own thread, immediately after\n"
    "model_factory. Both return a shared_ptr, so\n"
    "the C++ side shares ownership with the interpreter instead of adopting a\n"
    "pointer Python still owns (issue #129). The last reference is dropped\n"
    "before solve_parallel returns, under the GIL, at the end of the worker\n"
    "that built it.\n"
    "\n"
    "Return a FRESH object from every call. Each worker searches its own model\n"
    "on its own thread and the search never locks the hook or the LNS, so one\n"
    "shared instance that carries state is a data race. The old raw-pointer\n"
    "signature caught that as a double free; a shared_ptr accepts it silently.\n"
    "\n"
    "A FROZEN Model is the one sanctioned exception, and the way to avoid\n"
    "duplicating a large DAG per worker (#157). model_factory is declared to\n"
    "return a Model by value, so nanobind COPIES whatever object it hands back --\n"
    "and copying a frozen model shares its immutable structure while giving the\n"
    "worker its own variables, node values and objective bound. So\n"
    "`m.freeze()` once and `lambda: m` is correct here, where returning a shared\n"
    "MUTABLE model would not be: `freeze()` is what makes the shared half\n"
    "unwritable, and every structural call on it raises instead.\n"
    "\n"
    "What a Python factory can usefully return is NARROW. Neither\n"
    "InnerSolverHook nor LNS has a nanobind trampoline, so C++ dispatches\n"
    "through the base vtable: a Python subclass is constructed and released\n"
    "correctly, but an override of `solve` or `destroy_repair` is never called.\n"
    "The two fail differently, and the second is the trap. `solve` is not bound\n"
    "at all, so a Python `solve` is merely a new attribute nothing reads.\n"
    "`LNS.destroy_repair` IS bound, so an override does shadow it when called\n"
    "from Python -- and silently does not when the search calls it from C++.\n"
    "In practice these arguments buy a separately configured FloatIntensifyHook\n"
    "or LNS per worker, not a Python implementation of either. InnerSolverHook\n"
    "itself is abstract and exposes no constructor.\n"
    "\n"
    "The Model handed to hook_factory is a COPY of the worker's model, not a\n"
    "handle on it: nanobind casts an lvalue reference by copying. Mutating it\n"
    "changes nothing the worker will search. What that copy costs depends on\n"
    "the model: a FROZEN one shares its structure, so the copy is the\n"
    "variables and the node values; an open one is deep-copied whole, once per\n"
    "worker (#157).\n"
    "\n"
    "An exception raised by the factory in every worker is re-raised here with\n"
    "its original type and message, while one that fails in only some workers\n"
    "is absorbed and the survivors' result is returned.\n"
    "\n"
    "An exception raised by the progress callback's on_progress ends the solve\n"
    "of the worker that made the call -- NOT the portfolio. That worker is\n"
    "restarted, and stops after three consecutive failed attempts, or earlier\n"
    "when the time limit runs out or a peer settles a model that has no\n"
    "objective. A worker that stops that way without ever completing an\n"
    "attempt has FAILED. An exception is re-raised here -- the original\n"
    "object, with its type, message and traceback; the lowest-numbered failed\n"
    "worker's last one -- only if every worker failed, or if some worker failed\n"
    "and no worker returned or shared any result, feasible or not. Otherwise\n"
    "it is DISCARDED, with nothing reported, and the best result found is\n"
    "returned. That includes a failed worker's incumbents shared before it\n"
    "died, and a worker that raised after completing an attempt\n"
    "(SearchConfig.max_iterations makes that possible), which does not count\n"
    "as failed.\n"
    "\n"
    "Which workers call the callback at all depends on timing: a worker other\n"
    "than the first reports only a new portfolio-wide best, so a callback that\n"
    "raises on every call may kill the first worker alone and let the run\n"
    "return normally -- after which the periodic no-improvement rows, which\n"
    "only the first worker sends, stop. Raising is therefore not a way to stop\n"
    "a run; catch inside on_progress if a failure there must be seen. (The\n"
    "module-level, single-threaded cbls.solve differs: there a raising callback\n"
    "ends the search and the exception propagates at once.)";

// A Python on_progress that raises leaves this override as an nb::python_error,
// which owns a strong reference to the Python exception object. ParallelSearch
// parks such exceptions in std::exception_ptr slots (src/pool.cpp,
// solve_portfolio and run_worker) and releases them wherever the last copy dies:
// on a worker thread, or on the calling thread while the call guard below has
// the GIL released. That is safe because python_error's destructor and copy
// constructor acquire the GIL themselves (gil_scoped_acquire, i.e.
// PyGILState_Ensure, which is legal on a thread that released the GIL and on a
// thread Python has never seen). They have since nanobind v0.1.0 -- v0.0.1's
// held nb::object members released WITHOUT the GIL -- so the whole
// `nanobind>=1.8` range pyproject.toml admits is covered (src/error.cpp at tags
// v0.1.0, v1.8.0 and v2.13.0). Converting the exception at this boundary would
// therefore buy no safety and would cost the caller the original exception
// object (#159).

struct PySolveCallback : SolveCallback {
    NB_TRAMPOLINE(SolveCallback, 1);
    void on_progress(const SolveProgress& p) override { NB_OVERRIDE_PURE(on_progress, p); }
};

constexpr const char* kModelExtendDoc =
    "Apply a ModelExtension to this closed model and return an ExtensionResult.\n"
    "\n"
    "Raises RuntimeError on a model that is frozen or not closed, when called\n"
    "from inside an evaluation (a lambda_sum callable extending the model it is\n"
    "being evaluated in), or while cbls.solve is running on this model (from a\n"
    "SolveCallback, or from another thread): extend is a between-solves\n"
    "operation. ValueError if the extension was built against a different model,\n"
    "or against this one before it last changed in any way -- which includes\n"
    "applying this same extension, and applying another one recorded against the\n"
    "same base. An extension is therefore single-use: record a new one against\n"
    "the current model for each extend. The first solve() of a model with an\n"
    "objective adds the objective row, so an extension recorded before that\n"
    "solve is stale.\n"
    "\n"
    "Afterwards, in this order: ViolationManager.on_extended(result) on every\n"
    "manager of this model, once per extend and before the next one (its\n"
    "weight-indexed reads raise until then, and a manager that missed an extend\n"
    "must be rebuilt), and pad_state(state, result) on every ModelState captured\n"
    "before the call, which restore_state otherwise rejects. Variable handles\n"
    "from var()/var_mut() stay valid and read the model's current state.\n"
    "\n"
    "There is no rollback: an exception from the growth itself (as opposed to\n"
    "the refusals above, which fire before anything changes) leaves the model\n"
    "corrupt. That takes running out of memory part-way through; the model then\n"
    "raises RuntimeError from extend, solve, freeze, ViolationManager,\n"
    "full_evaluate, delta_evaluate and ModelExtension rather than being searched.\n"
    "Objects built before the failed call (a ViolationManager, say) are not\n"
    "re-checked: discard them with the model.";

constexpr const char* kModelVarDoc =
    "A handle to variable `id` of this model. It holds the model and the id, not\n"
    "a pointer into the model's storage, so it stays valid across builders and\n"
    "Model.extend and always reads the variable's current state. Keeps the model\n"
    "alive. var() and var_mut() return the same kind of handle.";

constexpr const char* kModelNodeDoc =
    "A COPY of node `id`'s id and op -- neither ever changes once the node\n"
    "exists, so the copy cannot go stale. Its current value is\n"
    "Model.node_value(id).";

// The models a bound `cbls.solve` is currently running on (#167). `solve`
// releases the GIL and calls a SolveCallback on the search thread, so Python can
// reach `Model.extend` mid-search -- from the callback or from another thread.
// An extension that changes the variable or row count is then refused by the
// engine's own table checks, but one that only appends terms over existing
// variables to existing Sum rows changes neither count: the search carried on
// with stale FeasibilityJump tables and returned feasible=True on a model it had
// left infeasible. The engine's only in-search growth point is #168's
// ColumnGenerator, which Python cannot reach, so the binding refuses the call
// outright. A multiset, because nothing stops two
// solves on one model from two threads (a data race of its own, but not this
// check's to refuse). Only the bound single-model `solve` registers:
// ParallelSearch is bound in factory form only, whose workers solve copies.
class SolvingModels {
public:
    static SolvingModels& instance() {
        static SolvingModels registry;
        return registry;
    }
    void add(const Model* m) {
        const std::scoped_lock lock(mutex_);
        models_.insert(m);
    }
    void remove(const Model* m) {
        const std::scoped_lock lock(mutex_);
        models_.erase(models_.find(m));
    }
    bool contains(const Model* m) {
        const std::scoped_lock lock(mutex_);
        return models_.count(m) > 0;
    }

private:
    std::mutex mutex_;
    std::unordered_multiset<const Model*> models_;
};

// Model.extend as bound: refused while a bound solve runs on the model. A free
// function rather than a lambda in the module body, whose cognitive-complexity
// score counts every lambda's branches as its own.
ExtensionResult extend_unless_solving(Model& self, const ModelExtension& ext) {
    if (SolvingModels::instance().contains(&self)) {
        throw std::logic_error(
            "Model.extend: cbls.solve is running on this model (from a SolveCallback or "
            "another thread). extend is a between-solves operation: the running search's "
            "tables cannot grow with it");
    }
    return self.extend(ext);
}

// What `Model.var()` / `Model.var_mut()` return: a (model, id) pair resolved
// through the model on EVERY attribute access, never a pointer into `vars_`.
//
// They used to return `reference_internal` into that vector, and anything that
// appends a variable -- a builder before close(), `Model.extend` after it --
// can reallocate it. Writing `.value` through one held across such a call was a
// write into freed heap (#167's review), which is the segfault class CLAUDE.md
// says to close at the hand-over rather than document. Resolving per access
// costs one bounds-checked index per attribute read, on a path that crosses the
// Python boundary anyway.
//
// OWNERSHIP: returned by value, with keep_alive<0, 1> tying the model's
// lifetime to the handle's, so `model` cannot dangle. The Python class keeps the
// name `Variable` and every attribute the direct binding had, so callers do not
// change -- with one visible difference, and it is the fix: a handle now always
// reads the model's CURRENT variable rather than whatever the storage it pointed
// at happens to hold.
struct VariableRef {
    Model* model;
    int32_t id;
    [[nodiscard]] Variable& get() const { return model->var_mut(id); }
};

// Checked once here so that `m.var(999)` still raises at the call, as it did.
// A variable is never removed, so an id valid now stays valid.
VariableRef variable_ref(Model& model, int32_t id) {
    static_cast<void>(model.var(id));
    return VariableRef{&model, id};
}

class SolvingScope {
public:
    explicit SolvingScope(const Model& m) : model_(&m) { SolvingModels::instance().add(model_); }
    SolvingScope(const SolvingScope&) = delete;
    SolvingScope& operator=(const SolvingScope&) = delete;
    SolvingScope(SolvingScope&&) = delete;
    SolvingScope& operator=(SolvingScope&&) = delete;
    ~SolvingScope() { SolvingModels::instance().remove(model_); }

private:
    const Model* model_;
};

// ---------------------------------------------------------------------------
// Table-backed lambda_sum / pair_lambda_sum (#163).
//
// A Python functor handed to lambda_sum is called once per element per node
// evaluation, each call re-acquiring the GIL -- which serialises every
// portfolio worker on the interpreter. For a distance matrix, which is the
// common case, the table is copied into C++ once at node creation and the
// search then never calls back into Python at all.
//
// The copy is deliberate. Holding a reference to the caller's array would let
// Python resize or free it under a running search, and `nb::ndarray` conversion
// may hand back a temporary anyway.

namespace {

// A float64, C-contiguous, CPU array of the given rank. Anything else nanobind
// either converts (via the array library's own routines) or rejects with a
// TypeError -- neither of which can reach the engine as a wrong-typed read.
using Table1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Table2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// The universe a table must cover, and the check that the handle names a
// structured variable at all. Both structured types draw their elements from
// [0, universe_size) (#164) -- for a permutation List that is the same number
// `max_size` gave, which is why this used to read the latter.
int32_t table_universe(const Model& model, int32_t list_var_id, const char* what) {
    if (list_var_id >= 0) {
        throw std::invalid_argument(std::string(what) +
                                    ": expected a variable handle (negative), got a node handle");
    }
    const Variable& var = model.var(handle_to_var_id(list_var_id));  // throws on a bogus id
    if (!is_structured(var.type)) {
        throw std::invalid_argument(std::string(what) + ": expected a List or Set variable handle");
    }
    return var.universe_size;
}

std::vector<double> copy_vector(const Table1D& a, int32_t n, const char* what) {
    if (a.shape(0) != static_cast<size_t>(n)) {
        throw std::invalid_argument(std::string(what) + ": expected length " + std::to_string(n) +
                                    ", got " + std::to_string(a.shape(0)));
    }
    return {a.data(), a.data() + n};
}

std::vector<double> copy_matrix(const Table2D& a, int32_t n, const char* what) {
    if (a.shape(0) != static_cast<size_t>(n) || a.shape(1) != static_cast<size_t>(n)) {
        throw std::invalid_argument(std::string(what) + ": expected shape (" + std::to_string(n) +
                                    ", " + std::to_string(n) + "), got (" +
                                    std::to_string(a.shape(0)) + ", " + std::to_string(a.shape(1)) +
                                    ")");
    }
    return {a.data(), a.data() + (static_cast<size_t>(n) * static_cast<size_t>(n))};
}

// The element index is range-checked even though the table was sized against
// the variable's universe at creation. `Variable.elements` is writable from
// Python and `Model.restore_state` copies element vectors in wholesale, so an
// out-of-universe element can reach here without passing any creation-time
// check -- and an unchecked `tbl[e]` on that path is a heap read past the
// vector, i.e. a segfault rather than an exception. One predictable compare
// against an indirect call through std::function is not what this path costs.
std::function<double(int)> table_lookup(std::vector<double> tbl, int32_t n, std::string what) {
    return [tbl = std::move(tbl), n, what = std::move(what)](int e) -> double {
        if (e < 0 || e >= n) {
            throw std::out_of_range(what + ": element " + std::to_string(e) +
                                    " outside the tabulated universe [0, " + std::to_string(n) +
                                    ")");
        }
        return tbl[static_cast<size_t>(e)];
    };
}

std::function<double(int, int)> matrix_lookup(std::vector<double> tbl, int32_t n,
                                              std::string what) {
    return [tbl = std::move(tbl), n, what = std::move(what)](int a, int b) -> double {
        if (a < 0 || a >= n || b < 0 || b >= n) {
            throw std::out_of_range(what + ": element pair (" + std::to_string(a) + ", " +
                                    std::to_string(b) + ") outside the tabulated universe [0, " +
                                    std::to_string(n) + ")");
        }
        return tbl[(static_cast<size_t>(a) * static_cast<size_t>(n)) + static_cast<size_t>(b)];
    };
}

// `None` detaches; a token attaches a NON-OWNING view of it. The keep_alive on
// each setter is what keeps the token alive for as long as the config naming it,
// so this cannot hand the engine a dangling view (the #156 hazard class).
StopRef stop_ref_or_none(StopToken* token) {
    return token != nullptr ? StopRef(*token) : StopRef();
}

// Refuse a ViolationManager that is out of step with the model BEFORE LNS runs.
// The engine refuses it too -- the FeasibilityJump the repair builds checks the
// weight count -- but LNS destroys (moves a random share of the variables) before
// it builds one, so the engine's refusal arrives after the assignment has
// changed. fj_nl_initialize needs no such guard: it builds its FeasibilityJump
// before it moves anything. The reachable case is a Model.extend
// with no ViolationManager.on_extended after it (#167). The weights setter keeps
// `weights` the size of the manager's own cache, so this one compare is the
// whole of what the engine's check would find.
void require_vm_in_step(const Model& model, const ViolationManager& vm, const char* what) {
    if (vm.weights.size() != model.constraint_ids().size()) {
        throw std::logic_error(std::string(what) +
                               ": the ViolationManager does not have one weight per constraint of "
                               "this model. After Model.extend, call vm.on_extended(result) first");
    }
}

}  // namespace

constexpr const char* kPairLambdaSumDoc =
    "Sum `func(e_k, e_{k+1})` over the consecutive pairs of a List or Set\n"
    "variable's elements.\n"
    "\n"
    "cyclic=True adds the closing pair `func(e_{n-1}, e_0)` when n >= 2, which\n"
    "is a tour cost. head and tail, if given, add `head(e_0)` and\n"
    "`tail(e_{n-1})` -- the depot legs of a route, which a cyclic sum over the\n"
    "customers alone would get wrong.\n"
    "\n"
    "n == 0 is 0.0 for every variant; n == 1 is `head(e_0) + tail(e_0)`.\n"
    "\n"
    "A Set's elements have no modelled order, so a pair sum over one reads\n"
    "whatever order they are currently stored in.\n"
    "\n"
    "Every call re-acquires the GIL, so a Python func is a serialisation point\n"
    "for a portfolio. Use pair_table_sum where the function is a matrix.";

constexpr const char* kPairTableSumDoc =
    "pair_lambda_sum with the function given as a distance matrix.\n"
    "\n"
    "dist is an (n, n) float64 array over the variable's universe_size, and\n"
    "head/tail are length-n arrays.\n"
    "All are COPIED into the engine at node creation, so the search makes no\n"
    "Python call at all and resizing or freeing the caller's array afterwards\n"
    "is harmless. A wrong shape raises here rather than being read past.\n"
    "\n"
    "Copied once per MODEL, not once per process. ParallelSearch's Python entry\n"
    "points take a model factory, so a portfolio builds one Model -- and one\n"
    "copy of this matrix -- per worker: an (n, n) float64 table costs\n"
    "8 * n * n bytes per thread.";

NB_MODULE(_cbls_core, m) {
    m.doc() = "CBLS: Constraint-Based Local Search engine (C++ core)";

    // Exception translators
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::invalid_argument& e) {
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
        .value("SignPower", NodeOp::SignPower)
        .value("Tanh", NodeOp::Tanh)
        .value("Geq", NodeOp::Geq)
        .value("Neq", NodeOp::Neq)
        .value("Lt", NodeOp::Lt)
        .value("Gt", NodeOp::Gt);

    // Variable (read-only access)
    //
    // Bound over `VariableRef`, not `Variable`: see the note there. Same
    // attributes, same writability, each resolved through the model per access.
    nb::class_<VariableRef>(m, "Variable")
        .def_prop_ro("id", [](const VariableRef& v) { return v.get().id; })
        .def_prop_ro("type", [](const VariableRef& v) { return v.get().type; })
        .def_prop_rw(
            "value", [](const VariableRef& v) { return v.get().value; },
            [](const VariableRef& v, double value) { v.get().value = value; })
        .def_prop_ro("lb", [](const VariableRef& v) { return v.get().lb; })
        .def_prop_ro("ub", [](const VariableRef& v) { return v.get().ub; })
        .def_prop_ro("name", [](const VariableRef& v) { return v.get().name; })
        .def_prop_rw(
            "elements", [](const VariableRef& v) { return v.get().elements; },
            [](const VariableRef& v, std::vector<int32_t> elements) {
                v.get().elements = std::move(elements);
            })
        .def_prop_ro("universe_size", [](const VariableRef& v) { return v.get().universe_size; })
        .def_prop_ro("min_size", [](const VariableRef& v) { return v.get().min_size; })
        .def_prop_ro("max_size", [](const VariableRef& v) { return v.get().max_size; })
        .def_prop_ro("list_init", [](const VariableRef& v) { return v.get().list_init; })
        .def_prop_ro("partitioned", [](const VariableRef& v) { return v.get().partitioned; });

    // ListInit — how initialisation fills a List (#164). Identity is the
    // permutation `list_var(n)` builds and is the only one a fixed-length List
    // may carry.
    nb::enum_<ListInit>(m, "ListInit")
        .value("Identity", ListInit::Identity)
        .value("Empty", ListInit::Empty)
        .value("Random", ListInit::Random);

    // Cover — how completely a ListPartition covers its universe (#164).
    nb::enum_<Cover>(m, "Cover")
        .value("Exact", Cover::Exact)
        .value("AtMostOnce", Cover::AtMostOnce);

    // ListPartition — read-only. Built by Model.add_list_partition, which is
    // where every invariant is checked; handing Python a writable `list_ids`
    // would let it name a variable that is not a List, or one already in another
    // partition, with the engine indexing on it unchecked afterwards (#156).
    nb::class_<ListPartition>(m, "ListPartition")
        .def_ro("list_ids", &ListPartition::list_ids)
        .def_ro("cover", &ListPartition::cover)
        .def_ro("universe_size", &ListPartition::universe_size);

    // ExprNode (read-only access)
    //
    // No `value`: a node's current value is per-model state that lives in the
    // model, not in the node (#157). Read it with `Model.node_value(id)`, which
    // is range-checked; the unchecked writer is deliberately not exposed.
    nb::class_<ExprNode>(m, "ExprNode").def_ro("id", &ExprNode::id).def_ro("op", &ExprNode::op);

    // TerminationReason — which budget ended the run.
    nb::enum_<TerminationReason>(m, "TerminationReason")
        .value("TimeLimit", TerminationReason::TimeLimit)
        .value("IterationLimit", TerminationReason::IterationLimit)
        .value("Feasible", TerminationReason::Feasible)
        .value("NoBudget", TerminationReason::NoBudget)
        .value("Stopped", TerminationReason::Stopped)
        // The HOST cancelled through a StopToken, as against Stopped, which is a
        // peer worker ending the run from inside the portfolio (#169).
        .value("Cancelled", TerminationReason::Cancelled);

    // StopToken -- cancellation from another Python thread.
    //
    // Bound before SearchConfig, which takes one. Held by the Python caller:
    // SearchConfig.stop is a NON-OWNING view, so the binding below keeps the
    // token alive for as long as the config that names it (nb::keep_alive), and
    // a token that outlives neither is a dangling read rather than a Python
    // error -- the #156 hazard class, which is why the lifetime is enforced here
    // rather than documented and hoped for.
    nb::class_<StopToken>(m, "StopToken")
        .def(nb::init<>())
        .def("request", &StopToken::request,
             "Ask the solve to stop. Safe to call from any thread, including while\n"
             "a solve is running -- which is the point: cbls.solve releases the GIL,\n"
             "so another Python thread can reach this. The run ends at its next\n"
             "batch boundary with termination == TerminationReason.Cancelled.")
        .def("reset", &StopToken::reset,
             "Clear the flag so the token can be reused. A solve does NOT reset it:\n"
             "handing the same raised token to the next solve cancels that one too.")
        .def("requested", &StopToken::requested);

    // StructuralSelection — how the structural batch turns a generator's
    // candidates into a commit (#165). FirstImprovingSample is the default and
    // is bit-for-bit the pre-#165 rule; the other two are opt-in.
    nb::enum_<StructuralSelection>(m, "StructuralSelection")
        .value("FirstImprovingSample", StructuralSelection::FirstImprovingSample)
        .value("BestOfSample", StructuralSelection::BestOfSample)
        .value("ViolationGuided", StructuralSelection::ViolationGuided);

    // NeighbourList — a granular neighbourhood (Toth & Vigo 2003) for the
    // built-in structural generators.
    //
    // READ-ONLY from Python, deliberately. The engine indexes this list on the
    // move-generation hot path without re-validating it, so a writable `offsets`
    // or `ids` would be exactly the unguarded-index segfault class of #156. It
    // is built once, validated once in the constructor, and then immutable.
    // `MoveGenerator` itself is not bound at all: a Python-subclassable
    // generator needs the trampoline/GIL machinery of #132 and would sit on the
    // hot path, so it is deliberately a follow-up.
    nb::class_<NeighbourList>(m, "NeighbourList")
        .def(nb::init<>())
        .def(nb::init<const std::vector<std::vector<int32_t>>&>(), nb::arg("rows"),
             "rows[e] are element e's neighbours, nearest first. Raises ValueError on an "
             "id outside [0, len(rows)).")
        .def("universe", &NeighbourList::universe)
        .def(
            "neighbours_of",
            [](const NeighbourList& nl, int32_t element) {
                const ConstSpan<int32_t> span = nl.of(element);
                return std::vector<int32_t>(span.begin(), span.end());
            },
            nb::arg("element"),
            "Element's neighbours, nearest first. Empty for an out-of-range element.")
        .def_prop_ro("offsets", [](const NeighbourList& nl) { return nl.offsets(); })
        .def_prop_ro("ids", [](const NeighbourList& nl) { return nl.ids(); })
        .def("__len__", [](const NeighbourList& nl) { return static_cast<size_t>(nl.universe()); });

    m.def("nearest_neighbours", &nearest_neighbours, nb::arg("universe"), nb::arg("k"),
          nb::arg("cost"),
          "Each element's k nearest others under cost(a, b), nearest first, ties broken by "
          "ascending id.\n"
          "\n"
          "O(universe^2) cost calls, and `cost` is a Python callable invoked from C++, so "
          "this is a setup-time convenience for a universe of a few thousand. Build the rows "
          "yourself and use NeighbourList(rows) for anything larger.");

    // BatchKind -- which kind of batch the outer loop ran. Bound so a Python
    // reader of SearchCounters can name the buckets it is reading.
    nb::enum_<BatchKind>(m, "BatchKind")
        .value("FeasibilityJump", BatchKind::FeasibilityJump)
        .value("NoveltyJump", BatchKind::NoveltyJump)
        .value("Structural", BatchKind::Structural);

    // SearchCounters and its per-generator rows (#169). Read-only throughout:
    // these are a report about a finished run, and a writable field would be a
    // way to falsify it rather than a feature. Registered before SearchResult,
    // which exposes one.
    nb::class_<GeneratorCounters>(m, "GeneratorCounters")
        .def_ro("name", &GeneratorCounters::name)
        .def_ro("moves_tried", &GeneratorCounters::moves_tried)
        .def_ro("moves_accepted", &GeneratorCounters::moves_accepted);

    nb::class_<SearchCounters>(m, "SearchCounters")
        .def_ro("batches", &SearchCounters::batches)
        .def_ro("fj_batches", &SearchCounters::fj_batches)
        .def_ro("novelty_batches", &SearchCounters::novelty_batches)
        .def_ro("structural_batches", &SearchCounters::structural_batches)
        .def_ro("structural_moves_tried", &SearchCounters::structural_moves_tried)
        .def_ro("structural_moves_accepted", &SearchCounters::structural_moves_accepted)
        .def_ro("by_generator", &SearchCounters::by_generator)
        .def_ro("inner_solver_calls", &SearchCounters::inner_solver_calls)
        // 0.0 on a run with no wall-clock budget, by design -- see
        // include/cbls/counters.h. The call count above is always filled.
        .def_ro("inner_solver_seconds", &SearchCounters::inner_solver_seconds)
        .def_ro("portfolio_restarts", &SearchCounters::portfolio_restarts);

    // SearchResult
    nb::class_<SearchResult>(m, "SearchResult")
        .def_ro("objective", &SearchResult::objective)
        .def_ro("feasible", &SearchResult::feasible)
        .def_ro("iterations", &SearchResult::iterations)
        .def_ro("time_seconds", &SearchResult::time_seconds)
        .def_ro("termination", &SearchResult::termination)
        // By reference to the result that owns it: a SearchCounters is a plain
        // aggregate with a vector in it, and copying it per attribute read would
        // be a surprise on a field a caller reads several times.
        .def_ro("counters", &SearchResult::counters, nb::rv_policy::reference_internal);

    // Model
    nb::class_<Model>(m, "Model")
        .def(nb::init<>())
        // Variable creation
        .def("bool_var", &Model::bool_var, nb::arg("name") = "")
        .def("int_var", &Model::int_var, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("float_var", &Model::float_var, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        // Two overloads, tried in order: the permutation form first, so
        // `list_var(n)` and `list_var(n, "name")` keep resolving to it exactly as
        // they did before #164.
        .def("list_var", nb::overload_cast<int, const std::string&>(&Model::list_var), nb::arg("n"),
             nb::arg("name") = "",
             "A fixed-length permutation of {0..n-1}: universe == min_len == max_len.")
        .def("list_var",
             nb::overload_cast<int, int, int, ListInit, const std::string&>(&Model::list_var),
             nb::arg("universe"), nb::arg("min_len"), nb::arg("max_len"),
             nb::arg("init") = ListInit::Empty, nb::arg("name") = "",
             "An ordered sequence of distinct elements of {0..universe-1} whose\n"
             "length stays within [min_len, max_len].")
        .def("set_var", &Model::set_var, nb::arg("n"), nb::arg("min_size") = 0,
             nb::arg("max_size") = -1, nb::arg("name") = "")
        .def("add_list_partition", &Model::add_list_partition, nb::arg("lists"),
             nb::arg("cover") = Cover::Exact,
             "Declare that `lists` partition their shared universe, maintained by\n"
             "the moves rather than by a constraint row. Returns the partition index.\n"
             "`cover` accepts a Cover value or the strings 'exact' / 'at_most_once'.")
        .def(
            "add_list_partition",
            [](Model& model, const std::vector<int32_t>& lists, const std::string& cover) {
                if (cover == "exact") {
                    return model.add_list_partition(lists, Cover::Exact);
                }
                if (cover == "at_most_once") {
                    return model.add_list_partition(lists, Cover::AtMostOnce);
                }
                throw std::invalid_argument("cover must be 'exact' or 'at_most_once'");
            },
            nb::arg("lists"), nb::arg("cover"))
        .def("partition_of_list", &Model::partition_of_list, nb::arg("var_id"),
             "Index into list_partitions() of the partition this variable id belongs\n"
             "to, or -1. Takes a var ID, not a handle.")
        .def_prop_ro("list_partitions", [](const Model& model) { return model.list_partitions(); })
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
        .def("signpower_expr", &Model::signpower_expr)
        .def("tanh_expr", &Model::tanh_expr)
        .def("if_then_else", &Model::if_then_else)
        .def("at", &Model::at)
        .def("count", &Model::count)
        .def("leq", &Model::leq)
        .def("eq_expr", &Model::eq_expr)
        .def("geq", &Model::geq)
        .def("neq", &Model::neq)
        .def("lt", &Model::lt)
        .def("gt", &Model::gt)
        .def(
            "lambda_sum",
            [](Model& model, int32_t list_var, std::function<double(int)> func) {
                // Held to the same handle rule as lambda_table_sum and the pair
                // forms: `wrap()` alone accepts a node handle or a scalar
                // variable and builds a node that evaluates to 0.0 for ever,
                // which from Python -- where the handle is a bare int -- reads
                // as the model silently ignoring the term.
                (void)table_universe(model, list_var, "lambda_sum");
                return model.lambda_sum(list_var, std::move(func));
            },
            nb::arg("list_var"), nb::arg("func"))
        .def(
            "lambda_table_sum",
            [](Model& model, int32_t list_var, const Table1D& table) {
                const int32_t n = table_universe(model, list_var, "lambda_table_sum");
                return model.lambda_sum(
                    list_var, table_lookup(copy_vector(table, n, "lambda_table_sum table"), n,
                                           "lambda_table_sum"));
            },
            nb::arg("list_var"), nb::arg("table"),
            "lambda_sum with the function given as a length-n float64 array over the\n"
            "variable's universe. Copied into the engine at node creation, so the\n"
            "search makes no Python call.")
        .def(
            "pair_lambda_sum",
            [](Model& model, int32_t list_var, std::function<double(int, int)> func, bool cyclic,
               std::optional<std::function<double(int)>> head,
               std::optional<std::function<double(int)>> tail) {
                // Held to the same handle rule as pair_table_sum. `wrap()`
                // alone accepts a node handle or a scalar variable and builds
                // a node that then evaluates to 0.0 for ever, which from
                // Python -- where the handle is a bare int -- reads as the
                // model silently ignoring the term.
                (void)table_universe(model, list_var, "pair_lambda_sum");
                return model.pair_lambda_sum(
                    list_var, std::move(func), head ? std::move(*head) : nullptr,
                    tail ? std::move(*tail) : nullptr, cyclic ? PairMode::Cyclic : PairMode::Open);
            },
            nb::arg("list_var"), nb::arg("func"), nb::arg("cyclic") = false,
            nb::arg("head") = nb::none(), nb::arg("tail") = nb::none(), kPairLambdaSumDoc)
        .def(
            "pair_table_sum",
            [](Model& model, int32_t list_var, const Table2D& dist, bool cyclic,
               const std::optional<Table1D>& head, const std::optional<Table1D>& tail) {
                const int32_t n = table_universe(model, list_var, "pair_table_sum");
                auto endpoint = [n](const std::optional<Table1D>& t,
                                    const char* what) -> std::function<double(int)> {
                    if (!t) {
                        return nullptr;
                    }
                    return table_lookup(copy_vector(*t, n, what), n, what);
                };
                // Both endpoints are validated BEFORE the node is made, so a
                // bad `tail` cannot leave a half-registered node behind.
                auto head_func = endpoint(head, "pair_table_sum head");
                auto tail_func = endpoint(tail, "pair_table_sum tail");
                return model.pair_lambda_sum(
                    list_var,
                    matrix_lookup(copy_matrix(dist, n, "pair_table_sum dist"), n, "pair_table_sum"),
                    std::move(head_func), std::move(tail_func),
                    cyclic ? PairMode::Cyclic : PairMode::Open);
            },
            nb::arg("list_var"), nb::arg("dist"), nb::arg("cyclic") = false,
            nb::arg("head") = nb::none(), nb::arg("tail") = nb::none(), kPairTableSumDoc)
        // Constraint and objective, both overload sets. nb::overload_cast picks
        // the member by parameter list; a static_cast to the member-pointer type
        // does the same job but reads to readability-redundant-casting as a cast
        // to the type the expression already has -- the check resolves the
        // overload *using* the cast and then calls it redundant. Say which
        // overload is wanted instead of casting to say it.
        .def("add_constraint", nb::overload_cast<int32_t>(&Model::add_constraint))
        .def("minimize", nb::overload_cast<int32_t>(&Model::minimize))
        .def("maximize", nb::overload_cast<int32_t>(&Model::maximize))
        .def("add_constraint", nb::overload_cast<const Expr&>(&Model::add_constraint))
        .def("minimize", nb::overload_cast<const Expr&>(&Model::minimize))
        .def("maximize", nb::overload_cast<const Expr&>(&Model::maximize))
        .def("add_var_sequence", &Model::add_var_sequence, nb::arg("var_ids"),
             nb::arg("min_block_on") = 1, nb::arg("min_block_off") = 1)
        .def("var_sequence_for", &Model::var_sequence_for)
        .def("close", &Model::close)
        // Growth of a closed model (#167). Returns an ExtensionResult BY VALUE:
        // Python owns the copy and it holds no pointer into the model. Nothing
        // else Python can hold does either: var()/var_mut() return a (model, id)
        // handle and node() a copy, precisely because `extend` reallocates the
        // arrays they used to point into.
        .def("extend", &extend_unless_solving, nb::arg("ext"), kModelExtendDoc)
        // Freezing makes the structure immutable and shareable. It is what lets a
        // model_factory hand the SAME model to every worker without duplicating
        // the DAG: nanobind copies the returned object, and copying a frozen model
        // shares its structure (#157). A structural call on a frozen model raises
        // RuntimeError rather than corrupting a peer: the refusal is a
        // std::logic_error, which nanobind has no mapping for and so translates to
        // RuntimeError -- see tests/python/test_model_freeze.py.
        .def("freeze", &Model::freeze)
        .def("is_frozen", &Model::is_frozen)
        // True after an extend that threw part-way through its growth; the model
        // is then refused by extend, solve, full_evaluate and friends (#167).
        .def("extend_interrupted", &Model::extend_interrupted)
        // Accessors
        // NOT by reference into the model's arrays, which a builder before
        // close() and Model.extend after it reallocate (#167's review found a
        // held var_mut() writing into freed heap). var()/var_mut() return a
        // VariableRef -- see there -- and node() returns a copy of an ExprNode,
        // whose two exposed fields are immutable once the node exists.
        .def("var", &variable_ref, nb::keep_alive<0, 1>(), kModelVarDoc)
        .def("var_mut", &variable_ref, nb::keep_alive<0, 1>(), kModelVarDoc)
        .def("node", &Model::node, nb::rv_policy::copy, kModelNodeDoc)
        .def("node_value", &Model::node_value, nb::arg("id"))
        .def("objective_id", &Model::objective_id)
        .def("constraint_ids", &Model::constraint_ids)
        // A view into the model's flat G_v array; copied out to a list, so the
        // Python side holds nothing that a later rebuild could invalidate.
        .def(
            "constraints_of_var",
            [](const Model& m, int32_t var_id) {
                const ConstSpan<int32_t> cs = m.constraints_of_var(var_id);
                return std::vector<int32_t>(cs.begin(), cs.end());
            },
            nb::arg("var_id"))
        .def(
            "per_constraint_violation_delta",
            [](Model& m, int32_t var_id, double j) {
                m.require_intact("per_constraint_violation_delta");
                return m.per_constraint_violation_delta(var_id, j);
            },
            nb::arg("var_id"), nb::arg("j"))
        .def("num_vars", &Model::num_vars)
        .def("num_nodes", &Model::num_nodes)
        // State snapshot/restore
        .def("copy_state", &Model::copy_state)
        .def("restore_state", &Model::restore_state)
        // Expr-returning variable creation
        .def("Bool", &Model::Bool, nb::arg("name") = "")
        .def("Int", &Model::Int, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("Float", &Model::Float, nb::arg("lb"), nb::arg("ub"), nb::arg("name") = "")
        .def("List", nb::overload_cast<int, const std::string&>(&Model::List), nb::arg("n"),
             nb::arg("name") = "")
        .def("List", nb::overload_cast<int, int, int, ListInit, const std::string&>(&Model::List),
             nb::arg("universe"), nb::arg("min_len"), nb::arg("max_len"),
             nb::arg("init") = ListInit::Empty, nb::arg("name") = "")
        .def("Set", &Model::Set, nb::arg("n"), nb::arg("min_size") = 0, nb::arg("max_size") = -1,
             nb::arg("name") = "")
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
        .def("__pow__",
             [](const Expr& a, double b) { return a.pow(Expr{a.model, a.model->constant(b)}); })
        .def("__pow__",
             [](const Expr& a, int b) {
                 return a.pow(Expr{a.model, a.model->constant(static_cast<double>(b))});
             })
        .def("__rpow__",
             [](const Expr& a, double b) {
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

    // ---------------------------------------------------------------------------
    // Growing a closed model (#167).
    //
    // ExtensionResult has NO Python constructor and every field is read-only. The
    // engine indexes its search state by the ids it carries (on_extended,
    // pad_state), and a hand-built one with `touched_constraints = [999]` was a
    // SIGSEGV on the C++ side before that path gained its own checks. From Python
    // the only ExtensionResult that exists is one Model.extend returned, and the
    // two consumers bound here re-check it against the model/state they are given.
    nb::class_<ExtensionResult>(m, "ExtensionResult")
        .def_ro("first_new_var", &ExtensionResult::first_new_var)
        .def_ro("num_new_vars", &ExtensionResult::num_new_vars)
        .def_ro("first_new_node", &ExtensionResult::first_new_node)
        .def_ro("num_new_nodes", &ExtensionResult::num_new_nodes)
        .def_ro("first_new_constraint", &ExtensionResult::first_new_constraint)
        .def_ro("num_new_constraints", &ExtensionResult::num_new_constraints)
        .def_ro("new_var_initial", &ExtensionResult::new_var_initial)
        .def_ro("touched_constraints", &ExtensionResult::touched_constraints)
        .def_ro("new_incidences", &ExtensionResult::new_incidences)
        .def_ro("topo_order_rebuilt", &ExtensionResult::topo_order_rebuilt)
        .def("end_var", &ExtensionResult::end_var)
        .def("end_node", &ExtensionResult::end_node)
        .def("end_constraint", &ExtensionResult::end_constraint);

    // Every handle the builder takes is validated by ModelExtension itself at
    // record time, against the base model plus what has been recorded so far:
    // an out-of-range id raises IndexError (std::out_of_range), a variable handle
    // where a node is required and a cyclic append raise ValueError
    // (std::invalid_argument). Nothing here reaches the model until
    // Model.extend, which re-checks that the recording still matches it.
    //
    // OWNERSHIP: the extension keeps a raw `const Model*` to its base, so the
    // constructor carries keep_alive<1, 2> -- the Python ModelExtension holds a
    // reference to the Python Model, and dropping every other reference to the
    // model cannot leave the extension pointing at a freed one. The pointer is
    // only read (node ops and children, for the Sum check and the cycle walk);
    // nothing is ever written through it.
    nb::class_<ModelExtension>(m, "ModelExtension")
        .def(nb::init<const Model&>(), nb::arg("model"), nb::keep_alive<1, 2>(),
             "Record additions to a CLOSED model: new scalar variables, expression\n"
             "nodes over new or existing handles, new constraints, and terms appended\n"
             "to existing Sum rows. Nothing touches the model until Model.extend.\n"
             "\n"
             "Handles are absolute: the ones returned here are the ids the entities\n"
             "will have after extend, and existing handles may be used freely. Raises\n"
             "RuntimeError if the model is not closed, or if a failed extend left it\n"
             "corrupt (Model.extend_interrupted()). An extension is single-use; see\n"
             "Model.extend. Keeps the model alive.")
        .def("bool_var", &ModelExtension::bool_var, nb::arg("name") = "")
        .def("int_var", &ModelExtension::int_var, nb::arg("lb"), nb::arg("ub"),
             nb::arg("name") = "")
        .def("float_var", &ModelExtension::float_var, nb::arg("lb"), nb::arg("ub"),
             nb::arg("name") = "")
        .def("set_initial", &ModelExtension::set_initial, nb::arg("var"), nb::arg("value"),
             "Starting value of a variable THIS extension created, and the value\n"
             "pad_state writes for it. Defaults to the lower bound. Raises ValueError\n"
             "for any other handle or for a value outside the bounds.")
        .def("constant", &ModelExtension::constant)
        .def("neg", &ModelExtension::neg)
        .def("sum", &ModelExtension::sum)
        .def("prod", &ModelExtension::prod)
        .def("div_expr", &ModelExtension::div_expr)
        .def("pow_expr", &ModelExtension::pow_expr)
        .def("min_expr", &ModelExtension::min_expr)
        .def("max_expr", &ModelExtension::max_expr)
        .def("abs_expr", &ModelExtension::abs_expr)
        .def("sin_expr", &ModelExtension::sin_expr)
        .def("cos_expr", &ModelExtension::cos_expr)
        .def("tan_expr", &ModelExtension::tan_expr)
        .def("exp_expr", &ModelExtension::exp_expr)
        .def("log_expr", &ModelExtension::log_expr)
        .def("sqrt_expr", &ModelExtension::sqrt_expr)
        .def("signpower_expr", &ModelExtension::signpower_expr)
        .def("tanh_expr", &ModelExtension::tanh_expr)
        .def("if_then_else", &ModelExtension::if_then_else)
        .def("at", &ModelExtension::at)
        .def("count", &ModelExtension::count)
        .def("leq", &ModelExtension::leq)
        .def("eq_expr", &ModelExtension::eq_expr)
        .def("geq", &ModelExtension::geq)
        .def("neq", &ModelExtension::neq)
        .def("lt", &ModelExtension::lt)
        .def("gt", &ModelExtension::gt)
        .def("add_constraint", &ModelExtension::add_constraint, nb::arg("expr"))
        .def("append_to_sum", &ModelExtension::append_to_sum, nb::arg("sum_node"), nb::arg("term"),
             "Append `term` to an EXISTING Sum node of the base model: a new column\n"
             "entering an old row. Raises ValueError if the target is not a Sum of the\n"
             "base model, if `term` already reads the target (the append would make\n"
             "the DAG cyclic), or if the model has changed since this extension was\n"
             "started (another extend was applied, or a structural write after close).")
        .def("empty", &ModelExtension::empty)
        .def("num_new_vars", &ModelExtension::num_new_vars)
        .def("num_new_nodes", &ModelExtension::num_new_nodes);

    // In place, on a state the caller owns: nothing is retained.
    m.def("pad_state", &pad_state, nb::arg("state"), nb::arg("ext"),
          "Grow a ModelState captured BEFORE `ext` was applied so that restore_state\n"
          "accepts it again, giving each new variable its declared initial value.\n"
          "Raises ValueError unless the state is exactly the pre-extension size.");

    // ViolationManager
    nb::class_<ViolationManager>(m, "ViolationManager")
        .def(nb::init<Model&>())
        .def("constraint_violation", &ViolationManager::constraint_violation)
        .def("total_violation", &ViolationManager::total_violation)
        .def("augmented_objective", &ViolationManager::augmented_objective)
        .def("is_feasible", &ViolationManager::is_feasible,
             nb::arg("tol") = kDefaultFeasibilityTolerance)
        .def("violated_constraints", &ViolationManager::violated_constraints,
             nb::arg("tol") = kDefaultFeasibilityTolerance)
        .def("bump_weights", &ViolationManager::bump_weights, nb::arg("factor") = 1.0)
        .def("weighted_violation_delta", &ViolationManager::weighted_violation_delta,
             nb::arg("var_id"), nb::arg("j"))
        .def("invalidate_cache", &ViolationManager::invalidate_cache)
        // Validated in C++: `ext` must describe exactly this manager's model's
        // constraint growth, and new_weight must be finite and >= 0 (ValueError).
        // Until it runs after a Model.extend, every read that indexes the weights
        // (total_violation, augmented_objective, weighted_violation_delta,
        // bump_weights) raises RuntimeError rather than indexing the short vector;
        // is_feasible, violated_constraints and constraint_violation read node
        // values only and already answer for the grown model.
        .def("on_extended", &ViolationManager::on_extended, nb::arg("ext"),
             nb::arg("new_weight") = 1.0,
             "Grow with a model Model.extend just grew. Existing rows keep their GLS\n"
             "weights; new rows start at new_weight (0 = masked). Call it as soon as\n"
             "extend returns: until then total_violation, augmented_objective,\n"
             "weighted_violation_delta and bump_weights raise RuntimeError. Once per\n"
             "extend, in order: a manager that missed one must be rebuilt.")
        // `weights` is indexed by constraint index with no bounds check on the
        // hot path (weighted_violation_delta, total_violation), so a short list
        // assigned from Python read past its end. The length rule is enforced here
        // rather than per read, against the manager's OWN size: Model.extend is the
        // one legitimate desync, and on_extended (not this setter) is how it closes.
        .def_prop_rw(
            "weights", [](ViolationManager& self) -> std::vector<double>& { return self.weights; },
            [](ViolationManager& self, std::vector<double> w) {
                if (w.size() != self.weights.size()) {
                    throw std::invalid_argument("weights must have one entry per constraint (" +
                                                std::to_string(self.weights.size()) + ")");
                }
                self.weights = std::move(w);
            },
            nb::rv_policy::reference_internal);

    // RNG
    nb::class_<RNG>(m, "RNG")
        .def(nb::init<uint64_t>(), nb::arg("seed") = 42)
        .def("uniform", &RNG::uniform)
        .def("integers", &RNG::integers)
        .def("normal", &RNG::normal)
        .def("random", &RNG::random)
        .def("seed", &RNG::seed);

    // ElementEdit — a structured change as POSITIONS rather than as the whole
    // element vector (#164). Read-only: an edit is applied to a variable's
    // elements unchecked-by-any-type-system from here, and a hand-built one with
    // a nonsense position would be the #156 hazard again. Build a change with
    // the `Move` the generators hand out, or with `replacement`, which is
    // length-checked nowhere but also indexes nothing.
    nb::enum_<EditKind>(m, "EditKind")
        .value("None_", EditKind::None)
        .value("Replace", EditKind::Replace)
        .value("Swap", EditKind::Swap)
        .value("Reverse", EditKind::Reverse)
        .value("MoveSegment", EditKind::MoveSegment)
        .value("Insert", EditKind::Insert)
        .value("Erase", EditKind::Erase)
        .value("Assign", EditKind::Assign);

    nb::class_<ElementEdit>(m, "ElementEdit")
        .def_ro("kind", &ElementEdit::kind)
        .def_ro("from_pos", &ElementEdit::from)
        .def_ro("to_pos", &ElementEdit::to)
        .def_ro("length", &ElementEdit::length)
        .def_ro("element", &ElementEdit::element);

    // Move::Change
    // A structured change is READ-ONLY from Python, by design (#164). `main`
    // bound `new_elements` as `def_rw`, which let Python hand the engine an
    // arbitrary element vector -- the #156 class, since the hot paths index by
    // element id without bounds tests. The positional form is exposed for
    // reading and built by the engine; a scalar change is still constructible,
    // because `var_id` and `new_value` index nothing. Nothing in the tree
    // constructs a structured change from Python, and the way to propose one is
    // a C++ `MoveGenerator`, not a hand-built edit.
    nb::class_<Move::Change>(m, "MoveChange")
        .def(nb::init<>())
        .def_rw("var_id", &Move::Change::var_id)
        .def_rw("new_value", &Move::Change::new_value)
        .def_prop_ro("edits",
                     [](const Move::Change& change) {
                         return std::vector<ElementEdit>(change.edits.begin(), change.edits.end());
                     })
        .def_ro("replacement", &Move::Change::replacement)
        .def(
            "elements_after",
            [](const Move::Change& change, const std::vector<int32_t>& elements) {
                return elements_after(change, elements);
            },
            nb::arg("elements"),
            "The elements this change produces from `elements` -- the absolute\n"
            "vector a change carried before #164. The engine edits in place.")
        .def(
            "is_noop",
            [](const Move::Change& change, const std::vector<int32_t>& elements) {
                return change_is_noop(change, elements);
            },
            nb::arg("elements"));

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

    // LNS
    nb::class_<LNS>(m, "LNS")
        .def(nb::init<double>(), nb::arg("destroy_fraction") = 0.3)
        .def(
            "destroy_repair",
            [](LNS& self, Model& model, ViolationManager& vm, RNG& rng, double repair_time_limit) {
                require_vm_in_step(model, vm, "LNS.destroy_repair");
                return self.destroy_repair(model, vm, rng, repair_time_limit);
            },
            nb::arg("model"), nb::arg("vm"), nb::arg("rng"), nb::arg("repair_time_limit") = 2.0)
        .def(
            "destroy_repair_cycle",
            [](LNS& self, Model& model, ViolationManager& vm, RNG& rng, int n_rounds,
               double repair_time_limit) {
                require_vm_in_step(model, vm, "LNS.destroy_repair_cycle");
                return self.destroy_repair_cycle(model, vm, rng, n_rounds, repair_time_limit);
            },
            nb::arg("model"), nb::arg("vm"), nb::arg("rng"), nb::arg("n_rounds") = 10,
            nb::arg("repair_time_limit") = 2.0);

    // SolutionPool
    nb::class_<Solution>(m, "Solution")
        .def(nb::init<>())
        .def_rw("state", &Solution::state)
        .def_rw("objective", &Solution::objective)
        .def_rw("feasible", &Solution::feasible)
        .def_rw("violation", &Solution::violation);

    nb::class_<SolutionPool>(m, "SolutionPool")
        .def(nb::init<int>(), nb::arg("capacity") = 10)
        .def("submit", &SolutionPool::submit)
        .def("best", &SolutionPool::best)
        .def("top_k", &SolutionPool::top_k)
        .def("size", &SolutionPool::size);

    // ParallelConfig
    nb::class_<ParallelConfig>(m, "ParallelConfig")
        .def(nb::init<>())
        .def_rw("n_threads", &ParallelConfig::n_threads)
        // 0 = auto (max(10, 2 * workers that run); see include/cbls/pool.h.
        .def_rw("pool_capacity", &ParallelConfig::pool_capacity)
        // Same non-owning-view rule, and the same keep_alive, as
        // SearchConfig.stop below. OR-ed with that one rather than replacing it.
        .def_prop_rw(
            "stop", [](const ParallelConfig& pc) { return pc.stop.attached(); },
            [](ParallelConfig& pc, StopToken* token) { pc.stop = stop_ref_or_none(token); },
            nb::for_setter(nb::arg("token").none()), nb::for_setter(nb::keep_alive<1, 2>()),
            "A cbls.StopToken whose request() cancels every worker, or None. Reads\n"
            "back as a bool (whether one is attached), not as the token: the C++\n"
            "side holds a view, not the object. Same accumulating keep-alive as\n"
            "SearchConfig.stop -- see its docstring.");

    // SearchConfig — must be registered before ParallelSearch / solve, which
    // use SearchConfig{} as a default argument (nanobind casts defaults to
    // Python eagerly at .def() time; an unregistered type throws std::bad_cast).
    nb::class_<SearchConfig>(m, "SearchConfig")
        .def(nb::init<>())
        .def_rw("skip_init", &SearchConfig::skip_init)
        .def_rw("max_iterations", &SearchConfig::max_iterations)
        .def_rw("use_fj", &SearchConfig::use_fj)
        .def_rw("lns_interval", &SearchConfig::lns_interval)
        .def_rw("structural_batch_probability", &SearchConfig::structural_batch_probability)
        .def_rw("structural_selection", &SearchConfig::structural_selection)
        .def_rw("structural_sample_size", &SearchConfig::structural_sample_size)
        // Copies in and out rather than sharing the C++ shared_ptr: a
        // NeighbourList is immutable once built, so a copy is the same list, and
        // handing Python a live handle to the object every worker reads is the
        // aliasing this type's read-only binding exists to avoid. `None` clears
        // it, which is the default and the pre-#165 uniform draw.
        .def_prop_rw(
            "structural_neighbours",
            [](const SearchConfig& c) -> std::optional<NeighbourList> {
                if (c.structural_neighbours == nullptr) {
                    return std::nullopt;
                }
                return *c.structural_neighbours;
            },
            [](SearchConfig& c, std::optional<NeighbourList> value) {
                c.structural_neighbours =
                    value.has_value() ? std::make_shared<const NeighbourList>(*value) : nullptr;
            })
        .def_rw("feasibility_tolerance", &SearchConfig::feasibility_tolerance)
        // A NON-OWNING view of a StopToken the Python caller holds (#169). The
        // keep_alive is what makes that safe: it ties the token's lifetime to
        // this config, so `cbls.solve(m, cfg)` cannot be reading a token Python
        // already collected. Reads back as a bool for the same reason
        // ParallelConfig.stop does -- there is no object on the C++ side to hand
        // back, only a pointer and a thunk.
        //
        // Tracer is deliberately NOT bound: it is a C++ extension point whose
        // events arrive per batch on a worker thread, so a Python subclass would
        // need the trampoline-plus-GIL machinery of #132 and would serialise
        // every portfolio worker on the interpreter. #169 scopes it to C++.
        .def_prop_rw(
            "stop", [](const SearchConfig& c) { return c.stop.attached(); },
            [](SearchConfig& c, StopToken* token) { c.stop = stop_ref_or_none(token); },
            nb::for_setter(nb::arg("token").none()), nb::for_setter(nb::keep_alive<1, 2>()),
            "A cbls.StopToken whose request() ends this solve at its next batch\n"
            "boundary, or None. Reads back as a bool (whether one is attached).\n"
            "\n"
            "The token is kept alive by this config for as long as the config lives,\n"
            "which is what stops the C++ side reading a collected object. That tie\n"
            "only ACCUMULATES: a long-lived config assigned several tokens retains\n"
            "every one of them, and `config.stop = None` detaches the view but does\n"
            "not release the tie. Harmless -- it errs toward keeping objects alive --\n"
            "but build a fresh SearchConfig per solve if that matters.");

    // ParallelSearch
    nb::class_<ParallelSearch>(m, "ParallelSearch")
        .def(nb::init<int>(), nb::arg("n_threads") = 0)
        .def(
            "solve",
            static_cast<SearchResult (ParallelSearch::*)(std::function<Model()>, double, uint64_t)>(
                &ParallelSearch::solve),
            nb::arg("model_factory"), nb::arg("time_limit") = 10.0, nb::arg("seed") = 42,
            // Without this the caller keeps the GIL for the whole C++ call,
            // including the worker join, while nanobind's std::function caster
            // acquires the GIL inside every worker to invoke the factory --
            // a deadlock that made this method uncallable from Python (#128).
            nb::call_guard<nb::gil_scoped_release>(), kParallelSolveDoc)
        .def("solve_parallel",
             static_cast<SearchResult (ParallelSearch::*)(
                 std::function<Model()>, double, uint64_t, const SearchConfig&,
                 std::function<std::shared_ptr<InnerSolverHook>(Model&)>,
                 std::function<std::shared_ptr<LNS>()>, SolveCallback*, const ParallelConfig&)>(
                 &ParallelSearch::solve),
             nb::arg("model_factory"), nb::arg("time_limit") = 10.0, nb::arg("seed") = 42,
             nb::arg("config") = SearchConfig{}, nb::arg("hook_factory") = nb::none(),
             nb::arg("lns_factory") = nb::none(), nb::arg("callback") = nullptr,
             nb::arg("par_config") = ParallelConfig{},
             // Same reason as `solve` above, plus two more callables that
             // acquire the GIL from a worker thread: the hook and LNS
             // factories. A None default reaches the std::function caster as an
             // empty function, which src/pool.cpp skips.
             nb::call_guard<nb::gil_scoped_release>(), kParallelSolveDoc);

    // Free functions
    // Exposed so the "adjacent base seeds do not share worker streams" property
    // can be checked directly rather than inferred from two search trajectories.
    m.def("portfolio_worker_seed", &portfolio_worker_seed, nb::arg("base_seed"), nb::arg("worker"),
          nb::arg("restart"),
          "RNG seed for one portfolio worker's one run: a splitmix64 finalizer over "
          "(base_seed, worker, restart). Mixed rather than added so that adjacent base "
          "seeds give genuinely different portfolios.");

    // Both refuse a model a failed Model.extend left half-grown (#167), which the
    // engine's own evaluators do not check: evaluating one reads index arrays
    // that are shorter than the node array.
    m.def("full_evaluate", [](Model& model) {
        model.require_intact("full_evaluate");
        return full_evaluate(model);
    });
    m.def("delta_evaluate", [](Model& model, const std::set<int32_t>& changed) {
        model.require_intact("delta_evaluate");
        return delta_evaluate(model, changed);
    });
    m.def("compute_partial", [](const Model& model, int32_t expr_id, int32_t var_id) {
        model.require_intact("compute_partial");
        return compute_partial(model, expr_id, var_id);
    });
    m.def("compute_all_partials", [](const Model& model, int32_t expr_id) {
        model.require_intact("compute_all_partials");
        return compute_all_partials(model, expr_id);
    });
    // The vector-returning overload, explicitly: #165 added an appending one
    // that also takes a NeighbourList, and an unqualified address-of is then
    // ambiguous. Python keeps the simple form.
    m.def("generate_standard_moves",
          [](const VariableRef& var, RNG& rng) { return generate_standard_moves(var.get(), rng); });
    m.def("apply_move", &apply_move);
    m.def("save_move_values", &save_move_values);
    m.def("undo_move", &undo_move);
    // InnerSolverHook + FloatIntensifyHook
    // Named rather than a discarded temporary: nb::class_ registers the type
    // in its constructor, so the object exists only for that side effect and
    // an unnamed one reads as a mistake (bugprone-unused-raii). The base has
    // no members or methods to expose -- FloatIntensifyHook below is what
    // Python instantiates; this exists so nanobind knows the inheritance.
    nb::class_<InnerSolverHook> inner_solver_hook(m, "InnerSolverHook");

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
        .def_ro("feasible", &SolveProgress::feasible)
        .def_ro("new_best", &SolveProgress::new_best)
        .def_ro("perturbations", &SolveProgress::perturbations);

    // SolveCallback with trampoline for Python subclassing
    nb::class_<SolveCallback, PySolveCallback>(m, "SolveCallback")
        .def(nb::init<>())
        .def("on_progress",
             [](SolveCallback& self, const SolveProgress& p) { self.on_progress(p); });

    // `hook` and `lns` take the same narrowing as solve_parallel's factories:
    // neither InnerSolverHook nor LNS has a trampoline, so a Python subclass's
    // override of `solve` or `destroy_repair` is never dispatched to from the
    // search. Passing a configured FloatIntensifyHook or LNS works; passing a
    // Python implementation of either silently runs the base. Tracked as #132 --
    // this docstring is the note a caller of `solve` sees, since the fuller
    // explanation lives on solve_parallel.
    // Wrapped rather than bound directly: cbls::solve takes a trailing
    // SearchCoordination*, which is ParallelSearch's private cross-worker
    // channel and has no meaning to a Python caller. Dropping it here keeps the
    // Python signature what it was.
    m.def(
        "solve",
        [](Model& model, double time_limit, uint64_t seed, bool use_fj, InnerSolverHook* hook,
           LNS* lns, int lns_interval, SolveCallback* callback, const SearchConfig& config) {
            // Registered WHILE THE GIL IS HELD, and only then released. Released
            // first (as a call_guard did), there was a window in which this solve
            // was already running on the model but not yet registered, and a
            // Model.extend from another thread could pass the check inside it.
            // `extend_unless_solving` runs holding the GIL, so with registration
            // under it too the check and the registration cannot interleave.
            const SolvingScope solving(model);  // refuses Model.extend until this returns
            const nb::gil_scoped_release release;
            return cbls::solve(model, time_limit, seed, use_fj, hook, lns, lns_interval, callback,
                               config);
        },
        nb::arg("model"), nb::arg("time_limit") = 10.0, nb::arg("seed") = 42,
        nb::arg("use_fj") = true, nb::arg("hook") = nullptr, nb::arg("lns") = nullptr,
        nb::arg("lns_interval") = 3, nb::arg("callback") = nullptr,
        nb::arg("config") = SearchConfig{},
        // THE GIL IS RELEASED FOR THE WHOLE SEARCH (#128's fix, extended to this
        // entry point by #169) -- by the gil_scoped_release in the body, just
        // after the SolvingScope registers, not by a call_guard (see there). Without it
        // `config.stop` is unusable: a Python thread that wants to call StopToken.request()
        // mid-solve cannot run at all while this one holds the interpreter, so the only reachable
        // stop would be one raised before the call.
        //
        // Everything this call can reach back into Python re-acquires the GIL
        // for itself: a SolveCallback subclass through nanobind's trampoline,
        // and a lambda_sum/pair_lambda_sum functor through the std::function
        // caster. Both did so already -- they had to, being callable from
        // portfolio worker threads -- so releasing here adds no new requirement.
        // A raising on_progress still propagates out of this call unchanged;
        // nb::python_error re-acquires the GIL in its own destructor (see the
        // note above PySolveCallback).
        //
        // TWO CONSEQUENCES, both stated in the docstring below rather than left to
        // be rediscovered:
        //
        //  - another Python thread can now reach the `model` WHILE the search is
        //    running. It writes variables and node values throughout and locks
        //    nothing, so reading `m.var(i).value` or `m.copy_state()` from another
        //    thread is a data race -- the same hazard solve_parallel's docstring
        //    already carries for its master model.
        //  - a raw Python callable handed to lambda_sum/pair_lambda_sum now pays a
        //    real gil_scoped_acquire per evaluation, where it used to re-enter a
        //    GIL the caller already held. tests/python/test_pair_lambda.py already
        //    says to use the *_table_sum forms for anything hot; this makes that
        //    advice cost more to ignore.
        "Single-threaded solve.\n"
        "\n"
        "The GIL is RELEASED for the whole C++ call (#169). That is what lets another "
        "Python thread raise a cbls.StopToken mid-solve -- and it makes touching "
        "`model` from another thread while this runs a DATA RACE: the search writes "
        "variables and node values throughout and nothing locks them. Read the model "
        "only after this returns. A raw Python function passed to lambda_sum or "
        "pair_lambda_sum also re-acquires the GIL on every evaluation now; use the "
        "*_table_sum forms where the function is a table.\n"
        "\n"
        "An exception raised by callback.on_progress ends the "
        "search and propagates out of this call unchanged -- no result is returned, "
        "and the model is left at the assignment the search had reached -- with its "
        "internal objective bound still tightened if a feasible point had been found "
        "(the next solve resets it). "
        "(ParallelSearch.solve_parallel absorbs one instead unless every worker fails, "
        "or a worker fails and no worker returned any result; see its docstring.) "
        "NOTE: a Python subclass of InnerSolverHook or LNS is "
        "constructed and destroyed correctly, but its overrides are never called -- "
        "neither class has a nanobind trampoline, so C++ dispatches through the base "
        "vtable. These arguments configure the built-in classes; they do not let you "
        "implement one in Python.");
    // Randomises every variable. solve() does not call this (FJ owns the scalar
    // start); pair it with SearchConfig.skip_init = True for a random scalar start.
    m.def("initialize_random", &initialize_random, nb::arg("model"), nb::arg("rng"));
    // What solve() calls: List/Set only, scalars untouched (#108).
    m.def("initialize_structured_random", &initialize_structured_random, nb::arg("model"),
          nb::arg("rng"));
    m.def("fj_nl_initialize", &fj_nl_initialize, nb::arg("model"), nb::arg("vm"),
          nb::arg("max_iterations") = 10000, nb::arg("rng") = nullptr, nb::arg("time_limit") = 2.0);
}
