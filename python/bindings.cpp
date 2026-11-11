#include <cbls/cbls.h>
#include <cmath>
#include <memory>
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
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace nb = nanobind;
using namespace cbls;

// Shared by both ParallelSearch.solve overloads: the GIL release that makes
// them callable (call guards below) runs the factory on several threads at once.
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

constexpr const char* kSolveMasterDoc =
    "Run the parallel search on ONE model and return the best result.\n"
    "\n"
    "The C++ `Model& master` entry point, which the factory forms cannot reach.\n"
    "`model` is FROZEN on this thread first and each worker searches a copy that\n"
    "shares its immutable structure (#157), so the model comes back frozen: build\n"
    "any further structure before calling this. hook_factory, lns_factory,\n"
    "callback and par_config behave as in solve_parallel (see there, including\n"
    "that hook_factory is handed a COPY of its worker's model, the GIL release\n"
    "and the exception contract).\n"
    "\n"
    "While this runs, structural writes to `model` from Python -- builders,\n"
    "close, freeze -- raise RuntimeError, as under cbls.solve.";

// A raising on_progress leaves as an nb::python_error, which owns a reference
// to the Python exception. ParallelSearch parks it in std::exception_ptr slots
// (src/pool.cpp) and its last copy may die on a worker thread or with the GIL
// released. Safe: python_error's destructor and copy constructor take the GIL
// themselves (PyGILState_Ensure) in every nanobind >= 0.1.0, so all of
// pyproject.toml's `nanobind>=2.0`. Hence not converted here, which would lose
// the original exception object (#159).

struct PySolveCallback : SolveCallback {
    // nanobind 3 deprecates the slot count (clang warns here even with -isystem,
    // #355); the 2.x floor in pyproject.toml requires it.
#if NB_VERSION_MAJOR >= 3
    NB_TRAMPOLINE(SolveCallback);
#else
    NB_TRAMPOLINE(SolveCallback, 1);
#endif
    void on_progress(const SolveProgress& p) override { NB_OVERRIDE_PURE(on_progress, p); }
};

constexpr const char* kModelVarDoc =
    "A handle to variable `id` of this model. It holds the model and the id, not\n"
    "a pointer into the model's storage, so it stays valid across builders and\n"
    "always reads the variable's current state. Keeps the model alive. var() and\n"
    "var_mut() return the same kind of handle.";

constexpr const char* kModelNodeDoc =
    "A COPY of node `id`'s id and op -- neither ever changes once the node\n"
    "exists, so the copy cannot go stale. Its current value is\n"
    "Model.node_value(id).";

// The models a bound solve is running on, consulted by refuse_if_solving:
// Python can reach a model mid-search (GIL released, SolveCallback) while the
// search reads its structure without a lock. Registered by `solve` and
// `ParallelSearch.solve_master`; the factory forms solve frozen copies. A
// multiset so remove() always pairs with add(); callers refuse a duplicate first.
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

// What `Model.var()` / `Model.var_mut()` return: a (model, id) pair resolved
// through the model on EVERY attribute access. Must never be a
// `reference_internal` into `vars_`, which a builder before close() reallocates,
// leaving a held handle writing freed heap (#167). Costs one bounds-checked
// index per access, on a path that crosses the Python boundary anyway.
//
// OWNERSHIP: by value, keep_alive<0, 1> keeps the model alive, so `model`
// cannot dangle. Bound as `Variable`, with the direct binding's attributes.
struct VariableRef {
    Model* model;
    int32_t id;
    [[nodiscard]] Variable& get() const { return model->var_mut(id); }
};

// Checked here so `m.var(999)` raises at the call; a variable is never
// removed, so an id valid now stays valid.
VariableRef variable_ref(Model& model, int32_t id) {
    static_cast<void>(model.var(id));
    return VariableRef{&model, id};
}

// The `Variable.value` setter's check (#211): components rely on every scalar
// lying in its box, and `SearchConfig.skip_init` hands it to the search as is.
// So NaN/inf, a value outside [lb, ub], or a non-integer on a Bool/Int raises
// ValueError; never clamped, which would silently change the assignment.
void set_checked_value(Variable& var, double value) {
    if (!std::isfinite(value)) {
        throw nb::value_error(
            ("Variable.value: " + var.name + " cannot take a non-finite value").c_str());
    }
    if (value < var.lb || value > var.ub) {
        throw nb::value_error(("Variable.value: " + std::to_string(value) +
                               " lies outside the domain [" + std::to_string(var.lb) + ", " +
                               std::to_string(var.ub) + "] of " + var.name)
                                  .c_str());
    }
    if ((var.type == VarType::Bool || var.type == VarType::Int) && value != std::trunc(value)) {
        throw nb::value_error(
            ("Variable.value: " + var.name + " is integral; got " + std::to_string(value)).c_str());
    }
    var.value = value;
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
// Structural writes from Python while a bound solve runs on the model.
//
// A bound solve, with the GIL released, freezes the model on first solve
// (close, objective row, cone cache #271), and the search reads the node array,
// CSR indices and topological order without a lock. A structural write from a
// SolveCallback or another thread would race it.
//
// So every structural write Python can reach consults the registry: Model
// builders, close/freeze, and the Expr operators and free functions. Checks and
// registration both run holding the GIL, so they cannot interleave. The
// engine's own freeze never passes through here.
//
// Value writes (Variable.value, restore_state) are not refused: they are the
// data race the `solve` docstring warns about.
// ---------------------------------------------------------------------------

void refuse_if_solving(const Model& m, const char* what) {
    if (SolvingModels::instance().contains(&m)) {
        throw std::logic_error(
            std::string(what) +
            ": cbls.solve is running on this model (from a SolveCallback or another "
            "thread). Structural changes are between-solves operations: the running "
            "search reads the structure without a lock");
    }
}

// A Model builder bound through the registry check. `A...` is the member's own
// parameter list, not a forwarding pack, so nanobind sees its types and nb::arg
// names line up.
template <typename R, typename... A>
auto guarded(R (Model::*method)(A...), const char* what) {
    return [method, what](Model& self, A... args) -> R {
        refuse_if_solving(self, what);
        return (self.*method)(std::forward<A>(args)...);
    };
}

// An Expr as a Python object that keeps its model alive: `Expr` holds a raw
// `Model*`, else `x = cbls.Model().Float(0, 1)` reads freed heap. The patient is
// the MODEL's Python object, found by pointer, not the operand a
// `keep_alive<0, 1>` would name, which would chain every intermediate of
// `s = s + x` alive. Called directly because the policy names only call
// arguments: `nb::detail::keep_alive` up to 2.x, `nb::keep_alive_obj` from 3.1;
// 3.0.x has neither, so `pyproject.toml` excludes it.
//
// A model with no Python object (none is reachable from Python) stays untied.
nb::object expr_object(const Expr& e) {
    const nb::object owner = nb::find(*e.model);
    nb::object out = nb::cast(e, nb::rv_policy::copy);
    if (owner.is_valid()) {
#if NB_VERSION_MAJOR > 3 || (NB_VERSION_MAJOR == 3 && NB_VERSION_MINOR >= 1)
        nb::keep_alive_obj(out, owner);
#else
        nb::detail::keep_alive(out.ptr(), owner.ptr());
#endif
    }
    return out;
}

template <typename... A>
auto guarded_expr(Expr (Model::*method)(A...), const char* what) {
    return [method, what](Model& self, A... args) -> nb::object {
        refuse_if_solving(self, what);
        return expr_object((self.*method)(std::forward<A>(args)...));
    };
}

// Build an Expr through `anchor`'s model: refused while it is being solved, and
// tied to it on the way out.
template <typename F>
nb::object build_expr(const Expr& anchor, const char* what, F&& build) {
    refuse_if_solving(*anchor.model, what);
    return expr_object(std::forward<F>(build)());
}

// The same for an Expr list. An empty list reaches cbls::min/max, which reject it.
template <typename F>
nb::object build_expr_list(const std::vector<Expr>& args, const char* what, F&& build) {
    for (const Expr& a : args) {
        refuse_if_solving(*a.model, what);
    }
    return expr_object(std::forward<F>(build)());
}

SearchResult solve_master_bound(
    ParallelSearch& self, Model& master, double time_limit, uint64_t seed,
    const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
    std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
    const ParallelConfig& par_config) {
    refuse_if_solving(master, "ParallelSearch.solve_master");
    const SolvingScope solving(master);
    const nb::gil_scoped_release release;
    return self.solve(master, time_limit, seed, config, std::move(hook_factory),
                      std::move(lns_factory), callback, par_config);
}

// ---------------------------------------------------------------------------
// Table-backed lambda_sum / pair_lambda_sum (#163).
//
// A Python functor costs a GIL acquire per element per evaluation, serialising
// every portfolio worker. A table is copied into C++ once at node creation, so
// the search never calls into Python. Must be a copy: Python could resize or
// free a referenced array under a running search, and `nb::ndarray` conversion
// may hand back a temporary.

namespace {

// A float64, C-contiguous, CPU array of the given rank; nanobind converts or
// rejects (TypeError) anything else, so no wrong-typed read reaches the engine.
using Table1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Table2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// The universe a table must cover, after checking the handle names a
// structured variable: List and Set elements lie in [0, universe_size) (#164).
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

// The element must be range-checked despite the creation-time sizing:
// `Variable.elements` is writable from Python and `Model.restore_state` copies
// element vectors wholesale, so an unchecked `tbl[e]` can segfault. One
// predictable compare is negligible beside the std::function call.
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

// The extra-reading lambdas (#186) take their extras as a Python list, copied
// from the engine's call-scoped view: a few doubles, beside a GIL acquire.
using ExtraCallable1 = std::function<double(int, const std::vector<double>&)>;
using ExtraCallable2 = std::function<double(int, int, const std::vector<double>&)>;

LambdaExtraFunc adapt_extra(ExtraCallable1 f) {
    return [f = std::move(f)](int e, ConstSpan<double> x) {
        return f(e, std::vector<double>(x.begin(), x.end()));
    };
}

PairLambdaExtraFunc adapt_extra(ExtraCallable2 f) {
    return [f = std::move(f)](int a, int b, ConstSpan<double> x) {
        return f(a, b, std::vector<double>(x.begin(), x.end()));
    };
}

// The two extra-lambda bindings. The null test must be here: `adapt_extra`
// wraps an empty callable into a non-empty one, hiding it from the builder's
// test. Covers whatever nanobind passes as empty (it refuses `None` itself).
int32_t lambda_sum_extra(Model& model, int32_t list_var, ExtraCallable1 func,
                         const std::vector<int32_t>& extra) {
    refuse_if_solving(model, "Model.lambda_sum");
    if (!func) {
        throw std::invalid_argument("Model.lambda_sum: func must not be None");
    }
    return model.lambda_sum(list_var, adapt_extra(std::move(func)), extra);
}

int32_t pair_lambda_sum_extra(Model& model, int32_t list_var, ExtraCallable2 func,
                              const std::vector<int32_t>& extra, bool cyclic) {
    refuse_if_solving(model, "Model.pair_lambda_sum");
    if (!func) {
        throw std::invalid_argument("Model.pair_lambda_sum: func must not be None");
    }
    const PairMode mode = cyclic ? PairMode::Cyclic : PairMode::Open;
    return model.pair_lambda_sum(list_var, adapt_extra(std::move(func)), mode, extra);
}

// `round(x)`. `round(x, n)` (decimal places, which the op lacks) is refused,
// not silently rounded to an integer.
nb::object round_dunder(const Expr& a, std::optional<int> ndigits) {
    if (ndigits.has_value()) {
        throw std::invalid_argument(
            "Expr.__round__: ndigits is not supported; round(x) rounds to an integer");
    }
    return build_expr(a, "Expr.__round__", [&] { return cbls::round(a); });
}

// `None` detaches; a token attaches a NON-OWNING view. Each setter's
// keep_alive must keep the token alive as long as the config, else the engine
// reads a dangling view (#156).
StopRef stop_ref_or_none(StopToken* token) {
    return token != nullptr ? StopRef(*token) : StopRef();
}

// Refuse a ViolationManager out of step with the model BEFORE LNS runs: the
// engine's check (in the repair's FeasibilityJump) comes after LNS has already
// moved variables. fj_nl_initialize needs none (it builds its FJ first). The
// reachable case is a manager built before the objective row (freeze() or
// first solve). The weights setter keeps `weights` the cache's size, so this
// compare is all the engine's check would find.
void require_vm_in_step(const Model& model, const ViolationManager& vm, const char* what) {
    if (vm.weights.size() != model.constraint_ids().size()) {
        throw std::logic_error(std::string(what) +
                               ": the ViolationManager does not have one weight per constraint of "
                               "this model. Build the ViolationManager after the model's last row "
                               "(freeze() and solve() add the objective row)");
    }
}

}  // namespace

constexpr const char* kLambdaSumDoc =
    "Sum `func(e)` over the elements of a List or Set variable.\n"
    "\n"
    "A callable that reaches an Expr (or the Model) pins the model for the\n"
    "process lifetime: the model holds the callable where the cycle collector\n"
    "cannot see it. Capture handles (`x.handle`) or plain data instead.\n";

constexpr const char* kPairLambdaSumDoc =
    "Sum `func(e_k, e_{k+1})` over the consecutive pairs of a List or Set\n"
    "variable's elements.\n"
    "\n"
    "A callable that reaches an Expr (or the Model) pins the model for the\n"
    "process lifetime: the model holds the callable where the cycle collector\n"
    "cannot see it. Capture handles (`x.handle`) or plain data instead.\n"
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

constexpr const char* kLambdaExtraDoc =
    "Sum `func(e, x)` over the elements of a List or Set variable, where `x`\n"
    "is the list of the CURRENT values of the `extra` handles, in order -- a\n"
    "per-element term that depends on other decisions, e.g. a stop cost by the\n"
    "route's vehicle type: `func=lambda i, x: c[i][int(x[0])]`.\n"
    "\n"
    "A change to any extra, or any edit to the list, re-sums the list. The\n"
    "node cannot be written to a .cbls file: the C++ save_model (not bound in\n"
    "Python) refuses the model.\n"
    "Every call re-acquires the GIL, as a plain lambda_sum's does.\n"
    "\n"
    "`extra` is keyword-only, here and in pair_lambda_sum, whose third\n"
    "positional parameter is already `cyclic`.";

constexpr const char* kPairLambdaExtraDoc =
    "Sum `func(e_k, e_{k+1}, x)` over consecutive pairs, `x` as in the extra\n"
    "form of lambda_sum. cyclic=True adds the closing pair when n >= 2. There\n"
    "are no head/tail terms in this form. `extra` is keyword-only.";

constexpr const char* kElementDoc =
    "table[index]: a table looked up by a decision. The index may be\n"
    "any scalar; its value is truncated toward zero, as `at` reads its index,\n"
    "and an index outside [0, len(table)) reads 0.0. The table is copied into\n"
    "the model, and its entries must be finite.";

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

namespace {

// An expression handle the engine will index unchecked: must name a node of
// `model`. std::out_of_range becomes IndexError through the translator below.
void require_node_id(const Model& model, int32_t expr_id, const char* entry) {
    if (expr_id < 0 || static_cast<size_t>(expr_id) >= model.num_nodes()) {
        throw std::out_of_range(std::string(entry) + ": expr_id " + std::to_string(expr_id) +
                                " is not a node of this model (" +
                                std::to_string(model.num_nodes()) + " nodes)");
    }
}

// `Move::changes` is an InlineVector (#274); Python sees a list of copies.
std::vector<Move::Change> move_changes(const Move& move) {
    return {move.changes.begin(), move.changes.end()};
}
void set_move_changes(Move& move, const std::vector<Move::Change>& changes) {
    move.changes.clear();
    for (const Move::Change& change : changes) {
        move.changes.push_back(change);
    }
}

}  // namespace

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
        .value("PairLambda", NodeOp::PairLambda)
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
        .value("Gt", NodeOp::Gt)
        .value("Element", NodeOp::Element)
        .value("Ceil", NodeOp::Ceil)
        .value("Floor", NodeOp::Floor)
        .value("Round", NodeOp::Round)
        .value("LambdaExtra", NodeOp::LambdaExtra)
        .value("PairLambdaExtra", NodeOp::PairLambdaExtra);

    // Variable: bound over `VariableRef` (see there), each attribute resolved
    // through the model per access; `value` and `elements` are writable.
    nb::class_<VariableRef>(m, "Variable")
        .def_prop_ro("id", [](const VariableRef& v) { return v.get().id; })
        .def_prop_ro("type", [](const VariableRef& v) { return v.get().type; })
        .def_prop_rw(
            "value", [](const VariableRef& v) { return v.get().value; },
            [](const VariableRef& v, double value) { set_checked_value(v.get(), value); })
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

    // ListInit — how initialisation fills a List (#164). Identity is what
    // `list_var(n)` builds, the only one a fixed-length List may carry.
    nb::enum_<ListInit>(m, "ListInit")
        .value("Identity", ListInit::Identity)
        .value("Empty", ListInit::Empty)
        .value("Random", ListInit::Random);

    // Cover — how completely a ListPartition covers its universe (#164).
    nb::enum_<Cover>(m, "Cover")
        .value("Exact", Cover::Exact)
        .value("AtMostOnce", Cover::AtMostOnce);

    // ListPartition — must stay read-only: Model.add_list_partition checks every
    // invariant, and a writable `list_ids` could name a non-List or an already
    // partitioned one, which the engine then indexes unchecked (#156).
    nb::class_<ListPartition>(m, "ListPartition")
        .def_ro("list_ids", &ListPartition::list_ids)
        .def_ro("cover", &ListPartition::cover)
        .def_ro("universe_size", &ListPartition::universe_size);

    // ExprNode (read-only). No `value`: it is per-model state (#157), read with
    // the range-checked `Model.node_value(id)`; the unchecked writer is not bound.
    nb::class_<ExprNode>(m, "ExprNode").def_ro("id", &ExprNode::id).def_ro("op", &ExprNode::op);

    // TerminationReason — which budget ended the run.
    nb::enum_<TerminationReason>(m, "TerminationReason")
        .value("TimeLimit", TerminationReason::TimeLimit)
        .value("IterationLimit", TerminationReason::IterationLimit)
        .value("Feasible", TerminationReason::Feasible)
        .value("NoBudget", TerminationReason::NoBudget)
        .value("Stopped", TerminationReason::Stopped)
        // The host cancelled via a StopToken; Stopped is a peer worker (#169).
        .value("Cancelled", TerminationReason::Cancelled);

    // StopToken -- cancellation from another Python thread. Bound before
    // SearchConfig, which takes one. SearchConfig.stop is a NON-OWNING view, so
    // its binding must keep the token alive with the config (nb::keep_alive),
    // else the engine reads a dangling pointer (#156).
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

    // StructuralSelection — how the structural batch turns candidates into a
    // commit (#165). Auto (default): GranularBest with a NeighbourList on the
    // config, else FirstImprovingSample, the pre-#165 rule (#203). GranularBest
    // needs a NeighbourList to differ from ViolationGuided.
    nb::enum_<StructuralSelection>(m, "StructuralSelection")
        .value("FirstImprovingSample", StructuralSelection::FirstImprovingSample)
        .value("BestOfSample", StructuralSelection::BestOfSample)
        .value("ViolationGuided", StructuralSelection::ViolationGuided)
        .value("GranularBest", StructuralSelection::GranularBest)
        .value("Auto", StructuralSelection::Auto);

    // NeighbourList — a granular neighbourhood (Toth & Vigo 2003) for the
    // built-in structural generators.
    //
    // Must stay READ-ONLY: validated once in the constructor, then indexed
    // unchecked on the hot path, so a writable `offsets`/`ids` is a segfault
    // (#156). `MoveGenerator` is not bound: a Python subclass needs #132's
    // trampoline/GIL machinery and would sit on the hot path.
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

    // BatchKind -- which kind of batch the outer loop ran; names
    // SearchCounters' buckets.
    nb::enum_<BatchKind>(m, "BatchKind")
        .value("FeasibilityJump", BatchKind::FeasibilityJump)
        .value("NoveltyJump", BatchKind::NoveltyJump)
        .value("Structural", BatchKind::Structural);

    // SearchCounters and its per-generator rows (#169): read-only, a report on
    // a finished run. Registered before SearchResult, which exposes one.
    nb::class_<GeneratorCounters>(m, "GeneratorCounters")
        .def_ro("name", &GeneratorCounters::name)
        .def_ro("moves_tried", &GeneratorCounters::moves_tried)
        .def_ro("moves_accepted", &GeneratorCounters::moves_accepted);

    nb::class_<SearchCounters>(m, "SearchCounters")
        .def_ro("batches", &SearchCounters::batches)
        .def_ro("fj_batches", &SearchCounters::fj_batches)
        .def_ro("novelty_batches", &SearchCounters::novelty_batches)
        .def_ro("novelty_moves", &SearchCounters::novelty_moves)
        .def_ro("novelty_commits", &SearchCounters::novelty_commits)
        .def_ro("novelty_weight_bumps", &SearchCounters::novelty_weight_bumps)
        .def_ro("structural_batches", &SearchCounters::structural_batches)
        .def_ro("structural_moves_tried", &SearchCounters::structural_moves_tried)
        .def_ro("structural_moves_accepted", &SearchCounters::structural_moves_accepted)
        .def_ro("by_generator", &SearchCounters::by_generator)
        .def_ro("inner_solver_calls", &SearchCounters::inner_solver_calls)
        // 0.0 without a wall-clock budget (include/cbls/counters.h); the call
        // count above is always filled.
        .def_ro("inner_solver_seconds", &SearchCounters::inner_solver_seconds)
        .def_ro("portfolio_restarts", &SearchCounters::portfolio_restarts)
        .def_ro("cone_cache_walks", &SearchCounters::cone_cache_walks)
        .def_ro("cone_fallback_walks", &SearchCounters::cone_fallback_walks)
        // The shared objective bound's engagement (#179); 0 on a single solve.
        .def_ro("shared_bound_tightenings", &SearchCounters::shared_bound_tightenings)
        .def_ro("own_best_behind_global", &SearchCounters::own_best_behind_global)
        .def_ro("bound_behind_global_batches", &SearchCounters::bound_behind_global_batches)
        .def_ro("bound_behind_global_seconds", &SearchCounters::bound_behind_global_seconds);

    // One portfolio worker that did not complete (#170). Registered before
    // SearchResult, whose `worker_failures` converts to a list of these.
    nb::class_<WorkerFailure>(m, "WorkerFailure")
        .def_ro("worker", &WorkerFailure::worker)
        .def_ro("produced_result", &WorkerFailure::produced_result)
        .def_ro("reason", &WorkerFailure::reason);

    // SearchResult
    nb::class_<SearchResult>(m, "SearchResult")
        .def_ro("objective", &SearchResult::objective)
        .def_ro("feasible", &SearchResult::feasible)
        .def_ro("iterations", &SearchResult::iterations)
        .def_ro("time_seconds", &SearchResult::time_seconds)
        .def_ro("termination", &SearchResult::termination)
        // Portfolio worker accounting (#170); see SearchResult::workers_completed.
        // `worker_failures` elements are reference_internal into this result
        // (`def_ro`): safe only because nothing reachable from Python grows it.
        .def_ro("workers_launched", &SearchResult::workers_launched)
        .def_ro("workers_completed", &SearchResult::workers_completed)
        .def_ro("worker_failures", &SearchResult::worker_failures)
        // By reference, not copied per read: an aggregate holding a vector.
        .def_ro("counters", &SearchResult::counters, nb::rv_policy::reference_internal);

    // Model
    nb::class_<Model>(m, "Model")
        .def(nb::init<>())
        // Variable creation
        .def("bool_var", guarded(&Model::bool_var, "Model.bool_var"), nb::arg("name") = "")
        .def("int_var", guarded(&Model::int_var, "Model.int_var"), nb::arg("lb"), nb::arg("ub"),
             nb::arg("name") = "")
        .def("float_var", guarded(&Model::float_var, "Model.float_var"), nb::arg("lb"),
             nb::arg("ub"), nb::arg("name") = "")
        // Overloads are tried in order: the permutation form must come first so
        // `list_var(n)` and `list_var(n, "name")` resolve to it (#164).
        .def(
            "list_var",
            guarded(nb::overload_cast<int, const std::string&>(&Model::list_var), "Model.list_var"),
            nb::arg("n"), nb::arg("name") = "",
            "A fixed-length permutation of {0..n-1}: universe == min_len == max_len.")
        .def("list_var",
             guarded(
                 nb::overload_cast<int, int, int, ListInit, const std::string&>(&Model::list_var),
                 "Model.list_var"),
             nb::arg("universe"), nb::arg("min_len"), nb::arg("max_len"),
             nb::arg("init") = ListInit::Empty, nb::arg("name") = "",
             "An ordered sequence of distinct elements of {0..universe-1} whose\n"
             "length stays within [min_len, max_len].")
        .def("set_var", guarded(&Model::set_var, "Model.set_var"), nb::arg("n"),
             nb::arg("min_size") = 0, nb::arg("max_size") = -1, nb::arg("name") = "")
        .def("add_list_partition", guarded(&Model::add_list_partition, "Model.add_list_partition"),
             nb::arg("lists"), nb::arg("cover") = Cover::Exact,
             "Declare that `lists` partition their shared universe, maintained by\n"
             "the moves rather than by a constraint row. Returns the partition index.\n"
             "`cover` accepts a Cover value or the strings 'exact' / 'at_most_once'.")
        .def(
            "add_list_partition",
            [](Model& model, const std::vector<int32_t>& lists, const std::string& cover) {
                refuse_if_solving(model, "Model.add_list_partition");
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
        .def("constant", guarded(&Model::constant, "Model.constant"))
        .def("neg", guarded(&Model::neg, "Model.neg"))
        .def("sum", guarded(&Model::sum, "Model.sum"))
        .def("prod", guarded(&Model::prod, "Model.prod"))
        .def("div_expr", guarded(&Model::div_expr, "Model.div_expr"))
        .def("pow_expr", guarded(&Model::pow_expr, "Model.pow_expr"))
        .def("min_expr", guarded(&Model::min_expr, "Model.min_expr"))
        .def("max_expr", guarded(&Model::max_expr, "Model.max_expr"))
        .def("abs_expr", guarded(&Model::abs_expr, "Model.abs_expr"))
        .def("sin_expr", guarded(&Model::sin_expr, "Model.sin_expr"))
        .def("cos_expr", guarded(&Model::cos_expr, "Model.cos_expr"))
        .def("tan_expr", guarded(&Model::tan_expr, "Model.tan_expr"))
        .def("exp_expr", guarded(&Model::exp_expr, "Model.exp_expr"))
        .def("log_expr", guarded(&Model::log_expr, "Model.log_expr"))
        .def("sqrt_expr", guarded(&Model::sqrt_expr, "Model.sqrt_expr"))
        .def("signpower_expr", guarded(&Model::signpower_expr, "Model.signpower_expr"))
        .def("tanh_expr", guarded(&Model::tanh_expr, "Model.tanh_expr"))
        .def("if_then_else", guarded(&Model::if_then_else, "Model.if_then_else"))
        .def("at", guarded(&Model::at, "Model.at"))
        .def("count", guarded(&Model::count, "Model.count"))
        // #186. The C++ builders refuse an empty or ragged table, a bogus handle
        // or a List/Set index before a node exists, so the engine never indexes
        // past. The table is copied into the model.
        .def("element",
             guarded(nb::overload_cast<const std::vector<double>&, int32_t>(&Model::element),
                     "Model.element"),
             nb::arg("table"), nb::arg("index"), kElementDoc)
        .def("element",
             guarded(nb::overload_cast<const std::vector<std::vector<double>>&, int32_t, int32_t>(
                         &Model::element),
                     "Model.element"),
             nb::arg("table"), nb::arg("row"), nb::arg("col"),
             "table[row][col] over a rectangular table; 0.0 when either index is out of range.")
        .def("ceil_expr", guarded(&Model::ceil_expr, "Model.ceil_expr"), nb::arg("x"))
        .def("floor_expr", guarded(&Model::floor_expr, "Model.floor_expr"), nb::arg("x"))
        .def("round_expr", guarded(&Model::round_expr, "Model.round_expr"), nb::arg("x"),
             "round half away from zero, as C's round().")
        .def("leq", guarded(&Model::leq, "Model.leq"))
        .def("eq_expr", guarded(&Model::eq_expr, "Model.eq_expr"))
        .def("geq", guarded(&Model::geq, "Model.geq"))
        .def("neq", guarded(&Model::neq, "Model.neq"))
        .def("lt", guarded(&Model::lt, "Model.lt"))
        .def("gt", guarded(&Model::gt, "Model.gt"))
        .def(
            "lambda_sum",
            [](Model& model, int32_t list_var, std::function<double(int)> func) {
                refuse_if_solving(model, "Model.lambda_sum");
                // Must check the handle like lambda_table_sum: `wrap()` accepts
                // a node or scalar handle and builds a node stuck at 0.0, which
                // silently drops the term.
                (void)table_universe(model, list_var, "lambda_sum");
                return model.lambda_sum(list_var, std::move(func));
            },
            nb::arg("list_var"), nb::arg("func"), kLambdaSumDoc)
        .def("lambda_sum", &lambda_sum_extra, nb::arg("list_var"), nb::arg("func"),
             nb::kw_only(), nb::arg("extra"), kLambdaExtraDoc)
        .def(
            "lambda_table_sum",
            [](Model& model, int32_t list_var, const Table1D& table) {
                refuse_if_solving(model, "Model.lambda_table_sum");
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
                refuse_if_solving(model, "Model.pair_lambda_sum");
                // Same handle check as lambda_sum, for the same reason.
                (void)table_universe(model, list_var, "pair_lambda_sum");
                return model.pair_lambda_sum(
                    list_var, std::move(func), head ? std::move(*head) : nullptr,
                    tail ? std::move(*tail) : nullptr, cyclic ? PairMode::Cyclic : PairMode::Open);
            },
            nb::arg("list_var"), nb::arg("func"), nb::arg("cyclic") = false,
            nb::arg("head") = nb::none(), nb::arg("tail") = nb::none(), kPairLambdaSumDoc)
        .def("pair_lambda_sum", &pair_lambda_sum_extra, nb::arg("list_var"), nb::arg("func"),
             nb::kw_only(), nb::arg("extra"), nb::arg("cyclic") = false, kPairLambdaExtraDoc)
        .def(
            "pair_table_sum",
            [](Model& model, int32_t list_var, const Table2D& dist, bool cyclic,
               const std::optional<Table1D>& head, const std::optional<Table1D>& tail) {
                refuse_if_solving(model, "Model.pair_table_sum");
                const int32_t n = table_universe(model, list_var, "pair_table_sum");
                auto endpoint = [n](const std::optional<Table1D>& t,
                                    const char* what) -> std::function<double(int)> {
                    if (!t) {
                        return nullptr;
                    }
                    return table_lookup(copy_vector(*t, n, what), n, what);
                };
                // Both endpoints validated BEFORE the node is made, so a bad
                // `tail` cannot leave a half-registered node.
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
        // Constraint and objective, both overload sets: nb::overload_cast, since
        // a static_cast to the member-pointer type trips
        // readability-redundant-casting.
        .def("add_constraint",
             guarded(nb::overload_cast<int32_t>(&Model::add_constraint), "Model.add_constraint"))
        .def("minimize", guarded(nb::overload_cast<int32_t>(&Model::minimize), "Model.minimize"))
        .def("maximize", guarded(nb::overload_cast<int32_t>(&Model::maximize), "Model.maximize"))
        .def("add_constraint", guarded(nb::overload_cast<const Expr&>(&Model::add_constraint),
                                       "Model.add_constraint"))
        .def("minimize",
             guarded(nb::overload_cast<const Expr&>(&Model::minimize), "Model.minimize"))
        .def("maximize",
             guarded(nb::overload_cast<const Expr&>(&Model::maximize), "Model.maximize"))
        .def("add_var_sequence", guarded(&Model::add_var_sequence, "Model.add_var_sequence"),
             nb::arg("var_ids"), nb::arg("min_block_on") = 1, nb::arg("min_block_off") = 1)
        .def("var_sequence_for", &Model::var_sequence_for)
        .def("close", guarded(&Model::close, "Model.close"))
        // Freezing makes the structure immutable and shareable: a model_factory
        // can return the SAME model to every worker, since nanobind copies it and
        // a frozen copy shares structure (#157). A structural call on it then
        // raises RuntimeError (unmapped std::logic_error; test_model_freeze.py).
        .def("freeze", guarded(&Model::freeze, "Model.freeze"),
             nb::arg("cone_cache_bytes") = Model::kDefaultConeCacheBytes)
        .def("is_frozen", &Model::is_frozen)
        // The cone cache's size (#271): read-only counts, no handle into it.
        .def("num_cached_cones", &Model::num_cached_cones)
        .def("cone_cache_bytes", &Model::cone_cache_bytes)
        // Accessors: never by reference into the model's arrays, which a builder
        // before close() reallocates (#167). var()/var_mut() return a
        // VariableRef; node() a copy, whose two exposed fields never change.
        .def("var", &variable_ref, nb::keep_alive<0, 1>(), kModelVarDoc)
        .def("var_mut", &variable_ref, nb::keep_alive<0, 1>(), kModelVarDoc)
        .def("node", &Model::node, nb::rv_policy::copy, kModelNodeDoc)
        .def("node_value", &Model::node_value, nb::arg("id"))
        .def("objective_id", &Model::objective_id)
        .def("constraint_ids", &Model::constraint_ids)
        // The modeller's rows are constraint_ids()[:num_user_constraints()];
        // close() appends the hidden domain rows after them (#248).
        .def("num_user_constraints", &Model::num_user_constraints)
        .def("num_domain_rows", &Model::num_domain_rows)
        .def("is_domain_node", &Model::is_domain_node, nb::arg("id"))
        // Copied out of the flat G_v view, so a later rebuild cannot invalidate it.
        .def(
            "constraints_of_var",
            [](const Model& m, int32_t var_id) {
                const ConstSpan<int32_t> cs = m.constraints_of_var(var_id);
                return std::vector<int32_t>(cs.begin(), cs.end());
            },
            nb::arg("var_id"))
        .def("per_constraint_violation_delta", &Model::per_constraint_violation_delta,
             nb::arg("var_id"), nb::arg("j"))
        .def("num_vars", &Model::num_vars)
        .def("num_nodes", &Model::num_nodes)
        // State snapshot/restore
        .def("copy_state", &Model::copy_state)
        .def("restore_state", &Model::restore_state)
        // Expr-returning variable creation
        .def("Bool", guarded_expr(&Model::Bool, "Model.Bool"), nb::arg("name") = "")
        .def("Int", guarded_expr(&Model::Int, "Model.Int"), nb::arg("lb"), nb::arg("ub"),
             nb::arg("name") = "")
        .def("Float", guarded_expr(&Model::Float, "Model.Float"), nb::arg("lb"), nb::arg("ub"),
             nb::arg("name") = "")
        .def("List",
             guarded_expr(nb::overload_cast<int, const std::string&>(&Model::List), "Model.List"),
             nb::arg("n"), nb::arg("name") = "")
        .def("List",
             guarded_expr(
                 nb::overload_cast<int, int, int, ListInit, const std::string&>(&Model::List),
                 "Model.List"),
             nb::arg("universe"), nb::arg("min_len"), nb::arg("max_len"),
             nb::arg("init") = ListInit::Empty, nb::arg("name") = "")
        .def("Set", guarded_expr(&Model::Set, "Model.Set"), nb::arg("n"), nb::arg("min_size") = 0,
             nb::arg("max_size") = -1, nb::arg("name") = "")
        .def("Constant", guarded_expr(&Model::Constant, "Model.Constant"));

    // Expr. Every Expr-returning entry point goes through build_expr: refused
    // while its model is solved, and tied to the model (see expr_object).
    nb::class_<Expr>(m, "Expr")
        .def_ro("model", &Expr::model)
        .def_ro("handle", &Expr::handle)
        .def("var_id", &Expr::var_id)
        .def("__add__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__add__", [&] { return a + b; });
             })
        .def("__add__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__add__", [&] { return a + b; });
             })
        .def("__radd__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__radd__", [&] { return b + a; });
             })
        .def("__mul__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__mul__", [&] { return a * b; });
             })
        .def("__mul__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__mul__", [&] { return a * b; });
             })
        .def("__rmul__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__rmul__", [&] { return b * a; });
             })
        .def("__sub__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__sub__", [&] { return a - b; });
             })
        .def("__sub__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__sub__", [&] { return a - b; });
             })
        .def("__rsub__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__rsub__", [&] { return b - a; });
             })
        .def("__truediv__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__truediv__", [&] { return a / b; });
             })
        .def("__truediv__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__truediv__", [&] { return a / b; });
             })
        .def("__rtruediv__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__rtruediv__", [&] { return b / a; });
             })
        .def("__neg__",
             [](const Expr& a) { return build_expr(a, "Expr.__neg__", [&] { return -a; }); })
        .def("__pow__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__pow__", [&] { return a.pow(b); });
             })
        .def("__pow__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__pow__",
                                   [&] { return a.pow(Expr{a.model, a.model->constant(b)}); });
             })
        .def("__pow__",
             [](const Expr& a, int b) {
                 return build_expr(a, "Expr.__pow__", [&] {
                     return a.pow(Expr{a.model, a.model->constant(static_cast<double>(b))});
                 });
             })
        .def("__rpow__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__rpow__", [&] {
                     return Expr{a.model, a.model->pow_expr(a.model->constant(b), a.handle)};
                 });
             })
        .def("__le__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__le__", [&] { return a <= b; });
             })
        .def("__le__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__le__", [&] { return a <= b; });
             })
        .def("__ge__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__ge__", [&] { return a >= b; });
             })
        .def("__ge__",
             [](const Expr& a, double b) {
                 return build_expr(a, "Expr.__ge__", [&] { return a >= b; });
             })
        .def("__lt__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__lt__", [&] { return a < b; });
             })
        .def("__lt__", [](const Expr& a,
                          double b) { return build_expr(a, "Expr.__lt__", [&] { return a < b; }); })
        .def("__gt__",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.__gt__", [&] { return a > b; });
             })
        .def("__gt__", [](const Expr& a,
                          double b) { return build_expr(a, "Expr.__gt__", [&] { return a > b; }); })
        .def("__ceil__",
             [](const Expr& a) {
                 return build_expr(a, "Expr.__ceil__", [&] { return cbls::ceil(a); });
             })
        .def("__floor__",
             [](const Expr& a) {
                 return build_expr(a, "Expr.__floor__", [&] { return cbls::floor(a); });
             })
        .def("__round__", &round_dunder, nb::arg("ndigits") = nb::none())
        .def("__abs__",
             [](const Expr& a) {
                 return build_expr(a, "Expr.__abs__", [&] { return cbls::abs(a); });
             })
        .def("is_var", &Expr::is_var)
        .def("eq", [](const Expr& a,
                      const Expr& b) { return build_expr(a, "Expr.eq", [&] { return a.eq(b); }); })
        .def("neq",
             [](const Expr& a, const Expr& b) {
                 return build_expr(a, "Expr.neq", [&] { return a.neq(b); });
             })
        .def("pow", [](const Expr& a, const Expr& b) {
            return build_expr(a, "Expr.pow", [&] { return a.pow(b); });
        });

    // Expr free functions
    m.def("sin",
          [](const Expr& x) { return build_expr(x, "cbls.sin", [&] { return cbls::sin(x); }); });
    m.def("cos",
          [](const Expr& x) { return build_expr(x, "cbls.cos", [&] { return cbls::cos(x); }); });
    m.def("tan",
          [](const Expr& x) { return build_expr(x, "cbls.tan", [&] { return cbls::tan(x); }); });
    m.def("exp",
          [](const Expr& x) { return build_expr(x, "cbls.exp", [&] { return cbls::exp(x); }); });
    m.def("log",
          [](const Expr& x) { return build_expr(x, "cbls.log", [&] { return cbls::log(x); }); });
    m.def("sqrt",
          [](const Expr& x) { return build_expr(x, "cbls.sqrt", [&] { return cbls::sqrt(x); }); });
    m.def("abs",
          [](const Expr& x) { return build_expr(x, "cbls.abs", [&] { return cbls::abs(x); }); });
    m.def("pow", [](const Expr& base, const Expr& exp) {
        return build_expr(base, "cbls.pow", [&] { return cbls::pow(base, exp); });
    });
    m.def("min", [](const std::vector<Expr>& args) {
        return build_expr_list(args, "cbls.min", [&] { return cbls::min(args); });
    });
    m.def("max", [](const std::vector<Expr>& args) {
        return build_expr_list(args, "cbls.max", [&] { return cbls::max(args); });
    });
    m.def("if_then_else", [](const Expr& cond, const Expr& then_, const Expr& else_) {
        return build_expr(cond, "cbls.if_then_else",
                          [&] { return cbls::if_then_else(cond, then_, else_); });
    });
    m.def("ceil",
          [](const Expr& x) { return build_expr(x, "cbls.ceil", [&] { return cbls::ceil(x); }); });
    m.def("floor", [](const Expr& x) {
        return build_expr(x, "cbls.floor", [&] { return cbls::floor(x); });
    });
    m.def("round", [](const Expr& x) {
        return build_expr(x, "cbls.round", [&] { return cbls::round(x); });
    });
    m.def(
        "element",
        [](const std::vector<double>& table, const Expr& index) {
            return build_expr(index, "cbls.element", [&] { return cbls::element(table, index); });
        },
        nb::arg("table"), nb::arg("index"), kElementDoc);
    m.def(
        "element",
        [](const std::vector<std::vector<double>>& table, const Expr& row, const Expr& col) {
            refuse_if_solving(*col.model, "cbls.element");
            return build_expr(row, "cbls.element", [&] { return cbls::element(table, row, col); });
        },
        nb::arg("table"), nb::arg("row"), nb::arg("col"));

    // Model::State
    nb::class_<Model::State>(m, "ModelState")
        .def(nb::init<>())
        .def_rw("values", &Model::State::values)
        .def_rw("elements", &Model::State::elements);

    // ViolationManager
    nb::class_<ViolationManager>(m, "ViolationManager")
        // keep_alive<1, 2>: the manager holds a Model& and reads it on every
        // call, so it must not outlive the Python Model.
        .def(nb::init<Model&>(), nb::keep_alive<1, 2>())
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
        // `weights` is indexed unchecked on the hot path (weighted_violation_delta,
        // total_violation), so the setter must enforce the manager's OWN length,
        // else a short list reads past its end. A manager out of step with its
        // MODEL is refused by its own reads instead.
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

    // ElementEdit — a structured change as POSITIONS, not the whole element
    // vector (#164). Must stay read-only: edits are applied unchecked, so a
    // hand-built bad position is a segfault (#156). Edits come from the engine.
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

    // Move::Change. A structured change must stay READ-ONLY from Python (#164):
    // the hot paths index by element id unchecked, so a writable element vector
    // is a segfault (#156). A scalar change stays constructible (`var_id`,
    // `new_value` index nothing); structured ones come from a C++ MoveGenerator.
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
        // `move_type` is a view onto a static name, so must stay READ-ONLY: a
        // string set from Python would not outlive the call.
        .def_prop_rw("changes", &move_changes, &set_move_changes)
        .def_prop_ro("move_type", [](const Move& move) { return std::string(move.move_type); })
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
        .def("size", &SolutionPool::size)
        // #179's global best: finite feasible submissions only, +inf when none.
        .def("best_feasible_objective", &SolutionPool::best_feasible_objective);

    // ParallelConfig
    nb::class_<ParallelConfig>(m, "ParallelConfig")
        .def(nb::init<>())
        .def_rw("n_threads", &ParallelConfig::n_threads)
        // 0 = auto (max(10, 2 * workers that run); see include/cbls/pool.h.
        .def_rw("pool_capacity", &ParallelConfig::pool_capacity)
        // #179: tighten every worker's objective bound to the portfolio's best at
        // each batch boundary. False is the A/B control arm.
        .def_rw("share_objective_bound", &ParallelConfig::share_objective_bound)
        // Same non-owning view and keep_alive as SearchConfig.stop; OR-ed with
        // it, not replacing it.
        .def_prop_rw(
            "stop", [](const ParallelConfig& pc) { return pc.stop.attached(); },
            [](ParallelConfig& pc, StopToken* token) { pc.stop = stop_ref_or_none(token); },
            nb::for_setter(nb::arg("token").none()), nb::for_setter(nb::keep_alive<1, 2>()),
            "A cbls.StopToken whose request() cancels every worker, or None. Reads\n"
            "back as a bool (whether one is attached), not as the token: the C++\n"
            "side holds a view, not the object. Same accumulating keep-alive as\n"
            "SearchConfig.stop -- see its docstring.");

    // SearchConfig — must be registered before ParallelSearch / solve, whose
    // SearchConfig{} default nanobind casts at .def() time (else std::bad_cast).
    nb::class_<SearchConfig>(m, "SearchConfig")
        .def(nb::init<>())
        .def_rw("skip_init", &SearchConfig::skip_init)
        .def_rw("max_iterations", &SearchConfig::max_iterations)
        .def_rw("use_fj", &SearchConfig::use_fj)
        .def_rw("lns_interval", &SearchConfig::lns_interval)
        .def_rw("structural_batch_probability", &SearchConfig::structural_batch_probability)
        // Novelty Jump batches (#209): plain scalars, read once per batch-kind
        // draw, so a bad value cannot reach an unchecked index.
        .def_rw("use_compound_moves", &SearchConfig::use_compound_moves)
        .def_rw("novelty_jump_probability", &SearchConfig::novelty_jump_probability)
        .def_rw("structural_selection", &SearchConfig::structural_selection)
        .def_rw("structural_sample_size", &SearchConfig::structural_sample_size)
        // Copied in and out, never the shared_ptr: Python must not alias the
        // list every worker reads, and an immutable copy is the same list.
        // `None` (default) clears it: the pre-#165 uniform draw.
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
        .def_rw("cone_cache_bytes", &SearchConfig::cone_cache_bytes)
        .def_rw("fj_objective_tie_break", &SearchConfig::fj_objective_tie_break)
        // A NON-OWNING view of the caller's StopToken (#169); the keep_alive ties
        // the token to this config so a solve never reads a collected one. Reads
        // back as a bool: C++ holds only a pointer and a thunk.
        //
        // Tracer is not bound (#169): its per-batch worker-thread events would
        // need #132's trampoline-plus-GIL and serialise every worker.
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
            // Must release the GIL: the std::function caster acquires it in
            // every worker to call the factory, so holding it through the join
            // deadlocks (#128).
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
             // As `solve` above; the hook and LNS factories also take the GIL
             // on workers. A None default arrives as an empty function, which
             // src/pool.cpp skips.
             nb::call_guard<nb::gil_scoped_release>(), kParallelSolveDoc)
        // The `Model& master` overload (#157), wrapped like `solve`: the master
        // must be registered WHILE THE GIL IS HELD, then released, so no
        // structural write slips in between.
        .def(
            "solve_master",
            [](ParallelSearch& self, Model& master, double time_limit, uint64_t seed,
               const SearchConfig& config,
               std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
               std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
               const ParallelConfig& par_config) {
                return solve_master_bound(self, master, time_limit, seed, config,
                                          std::move(hook_factory), std::move(lns_factory), callback,
                                          par_config);
            },
            nb::arg("model"), nb::arg("time_limit") = 10.0, nb::arg("seed") = 42,
            nb::arg("config") = SearchConfig{}, nb::arg("hook_factory") = nb::none(),
            nb::arg("lns_factory") = nb::none(), nb::arg("callback") = nullptr,
            nb::arg("par_config") = ParallelConfig{}, kSolveMasterDoc);

    // Free functions
    // Exposed so "adjacent base seeds do not share worker streams" is testable
    // directly.
    m.def("portfolio_worker_seed", &portfolio_worker_seed, nb::arg("base_seed"), nb::arg("worker"),
          nb::arg("restart"),
          "RNG seed for one portfolio worker's one run: a splitmix64 finalizer over "
          "(base_seed, worker, restart). Mixed rather than added so that adjacent base "
          "seeds give genuinely different portfolios.");

    m.def("full_evaluate", [](Model& model) { return full_evaluate(model); });
    m.def("delta_evaluate", [](Model& model, const std::set<int32_t>& changed) {
        return delta_evaluate(model, changed);
    });
    // The engine indexes its adjoint and cone buffers by `expr_id` unchecked, so
    // it must be checked at the hand-over, else a bad id writes out of bounds
    // (#156).
    m.def("compute_partial", [](const Model& model, int32_t expr_id, int32_t var_id) {
        require_node_id(model, expr_id, "compute_partial");
        return compute_partial(model, expr_id, var_id);
    });
    m.def("compute_all_partials", [](const Model& model, int32_t expr_id) {
        require_node_id(model, expr_id, "compute_all_partials");
        return compute_all_partials(model, expr_id);
    });
    // A lambda selecting the vector-returning overload: #165's appending one
    // makes `&generate_standard_moves` ambiguous.
    m.def("generate_standard_moves",
          [](const VariableRef& var, RNG& rng) { return generate_standard_moves(var.get(), rng); });
    m.def("apply_move", &apply_move);
    m.def("save_move_values", &save_move_values);
    m.def("undo_move", &undo_move);
    // InnerSolverHook + FloatIntensifyHook. The base is registered only so
    // nanobind knows the inheritance; named, not a discarded temporary, which
    // bugprone-unused-raii flags.
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

    // `hook` and `lns` have no trampoline (#132): a Python subclass's `solve` /
    // `destroy_repair` override is never dispatched; the base runs. Wrapped to
    // drop cbls::solve's trailing SearchCoordination*, ParallelSearch's private
    // cross-worker channel.
    m.def(
        "solve",
        [](Model& model, double time_limit, uint64_t seed, bool use_fj, InnerSolverHook* hook,
           LNS* lns, int lns_interval, SolveCallback* callback, const SearchConfig& config) {
            // Must register WHILE THE GIL IS HELD, then release (not a
            // call_guard), else a builder on another thread can pass the check
            // before registration. A solve is a structural writer (objective
            // row), so a second, nested or concurrent, is refused too.
            refuse_if_solving(model, "cbls.solve");
            const SolvingScope solving(model);  // refuses structural writes until this returns
            const nb::gil_scoped_release release;
            return cbls::solve(model, time_limit, seed, use_fj, hook, lns, lns_interval, callback,
                               config);
        },
        nb::arg("model"), nb::arg("time_limit") = 10.0, nb::arg("seed") = 42,
        nb::arg("use_fj") = true, nb::arg("hook") = nullptr, nb::arg("lns") = nullptr,
        nb::arg("lns_interval") = 3, nb::arg("callback") = nullptr,
        nb::arg("config") = SearchConfig{},
        // THE GIL IS RELEASED FOR THE WHOLE SEARCH (#128, #169), in the body after
        // registration; else no thread could call StopToken.request() mid-solve.
        // Every path back into Python re-acquires it (SolveCallback trampoline,
        // the lambda functors' std::function caster, python_error's destructor;
        // see PySolveCallback). Consequences, in the docstring: reading the model
        // from another thread mid-search is a data race, and a raw Python lambda
        // pays a GIL acquire per evaluation.
        "Single-threaded solve.\n"
        "\n"
        "The GIL is RELEASED for the whole C++ call (#169). That is what lets another "
        "Python thread raise a cbls.StopToken mid-solve -- and it makes touching "
        "`model` from another thread while this runs a DATA RACE: the search writes "
        "variables and node values throughout and nothing locks them. Read the model "
        "only after this returns. STRUCTURAL writes to `model` -- its builders, "
        "close, freeze and Expr operators over it -- raise RuntimeError while this "
        "runs, from any thread, a SolveCallback included. A raw Python function passed to "
        "lambda_sum or "
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
          nb::arg("max_iterations") = 10000, nb::arg("rng") = nullptr, nb::arg("time_limit") = 2.0,
          nb::arg("tolerance") = kDefaultFeasibilityTolerance);
}
