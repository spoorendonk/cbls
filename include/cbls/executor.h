#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#include <concepts>
#endif

namespace cbls {

/// A NON-OWNING view of a callable, for a parameter that is invoked before the
/// call returns.
///
/// `std::function` would allocate and copy; a template parameter would make
/// `ExecutorRef` a template, and the whole point of that class is to be a
/// concrete type on a concrete API compiled into `cbls_lib`. This is the usual
/// `function_ref` (as proposed for C++26): two pointers, no allocation, no
/// ownership.
///
/// THE REFERENT MUST OUTLIVE THE CALL. Every use in this library passes a lambda
/// as an argument to a function that invokes it synchronously, which is the only
/// shape this is for. Storing one is a dangling reference waiting to happen.
///
/// Binds a non-const lvalue or a temporary; a `const` callable does not compile,
/// deliberately, since the thunk has to be able to invoke a `mutable` lambda.
template <typename Signature>
class FunctionRef;

template <typename R, typename... Args>
class FunctionRef<R(Args...)> {
public:
    /// The `enable_if` keeps this from hijacking the copy constructor, which a
    /// forwarding-reference constructor otherwise matches better than the
    /// implicit one.
    template <typename F,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, FunctionRef>>>
    // Implicit by design: this is a parameter type, and the call site writes a
    // lambda.
    FunctionRef(F&& callable) noexcept
        : obj_(std::addressof(callable)), fn_(&thunk<std::remove_reference_t<F>>) {}

    R operator()(Args... args) const { return fn_(obj_, std::forward<Args>(args)...); }

private:
    template <typename F>
    static R thunk(void* obj, Args... args) {
        return (*static_cast<F*>(obj))(std::forward<Args>(args)...);
    }

    void* obj_;
    R (*fn_)(void*, Args...);
};

/// A NON-OWNING, type-erased view of a caller-owned thread pool (#169).
///
/// WHY THIS EXISTS. `ParallelSearch` creates its own `std::thread`s. A host that
/// already owns a pool -- another solver, a pipeline stage, a batch job -- then
/// oversubscribes its cores the moment it runs cbls next to anything else, and
/// has no way to say "use mine". Other optimisation libraries are converging on
/// this same small executor shape for exactly that reason, so taking it makes
/// cbls drop in beside them without an adapter.
///
/// THE SHAPE. Anything with these four members, however spelled internally:
///
/// ```
///   e.parallel_for(begin, end, f);            // f(i) for i in [begin, end)
///   e.parallel_for_chunked(begin, end, f);    // f(chunk_begin, chunk_end, chunk_idx)
///   e.parallel_invoke(f, g);                  // both, possibly concurrently
///   int n = e.n_threads();
/// ```
///
/// `chunk_idx` is unique in `[0, n_threads())` -- a CHUNK index, not a thread id
/// -- so a caller can index per-chunk storage without a lock. Empty and inverted
/// ranges are no-ops, short-circuited here so an adapted executor never sees one.
///
/// A C++20 caller can check its type against the concept below at compile time
/// (`static_assert(cbls::Executor<MyPool>)`). Under C++17 -- which this library
/// is -- a mismatch is an ordinary template error inside the constructor instead.
/// Note what that means for the concept: NO translation unit in this repository is
/// C++20, so nothing here compiles it and no gate would catch a syntax error in
/// it. It is checked by hand against a C++20 compiler when it changes.
///
/// OWNERSHIP AND LIFETIME. The CALLER owns the pool. This is a pointer plus a
/// vtable pointer, copied by value; nothing in cbls owns, copies or destroys the
/// pool, and the pool must outlive the solve. Same rule, same reason, as
/// `StopRef`.
///
/// CONCURRENCY IS REQUIRED, not merely permitted. A portfolio worker is
/// long-running and COOPERATIVE: it shares incumbents through the pool as it
/// finds them and restarts from a peer's, so the workers have to run at the same
/// time to be a portfolio at all. An executor that runs chunks SEQUENTIALLY is
/// not an error, and what it degenerates to depends on the budget: under a WALL
/// CLOCK it is a one-worker portfolio, because the first worker takes the whole
/// shared deadline and every later one finds it already past and returns without
/// searching. With `time_limit <= 0` there is no shared deadline to take, so each
/// worker instead runs its full `max_iterations` budget one after another -- N
/// solves in series, N times the wall time. Nothing detects either; both are
/// properties of the executor the caller supplied.
class ExecutorRef {
public:
    /// Non-explicit on purpose: `par_config.executor = my_pool;` is the call
    /// shape this exists for. The `enable_if` keeps the copy constructor from
    /// being hijacked.
    template <typename E,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<E>, ExecutorRef>>>
    // Implicit by design; see above.
    ExecutorRef(E& executor) : obj_(std::addressof(executor)), table_(&kTableFor<E>) {}

    /// `f(i)` for each `i` in `[begin, end)`, possibly concurrently.
    void parallel_for(int begin, int end, FunctionRef<void(int)> f) const {
        if (end <= begin) {
            return;
        }
        table_->parallel_for(obj_, begin, end, f);
    }

    /// `f(chunk_begin, chunk_end, chunk_idx)` over a partition of
    /// `[begin, end)`, possibly concurrently. This is the one `ParallelSearch`
    /// uses.
    void parallel_for_chunked(int begin, int end, FunctionRef<void(int, int, int)> f) const {
        if (end <= begin) {
            return;
        }
        table_->parallel_for_chunked(obj_, begin, end, f);
    }

    /// Run both, possibly concurrently. Returns once both have finished.
    void parallel_invoke(FunctionRef<void()> f, FunctionRef<void()> g) const {
        table_->parallel_invoke(obj_, f, g);
    }

    /// How many chunks may run at once -- the bound on the portfolio's worker
    /// count. The adapted `n_threads()` should return `int`; anything merely
    /// convertible is converted in the return statement, which is a narrowing
    /// conversion if it is wider.
    ///
    /// A value below 1 is clamped to 1 by `src/pool.cpp`, which keeps the
    /// per-worker vectors non-empty. That is all the clamp buys: it cannot make a
    /// zero-width pool actually RUN a chunk, and a pool that runs nothing leaves
    /// the portfolio with no result to return -- reported as `NoBudget`, exactly
    /// as a run handed no budget is. Must be >= 1: supply a pool that runs what it
    /// is given.
    [[nodiscard]] int n_threads() const { return table_->n_threads(obj_); }

private:
    struct VTable {
        void (*parallel_for)(void*, int, int, FunctionRef<void(int)>);
        void (*parallel_for_chunked)(void*, int, int, FunctionRef<void(int, int, int)>);
        void (*parallel_invoke)(void*, FunctionRef<void()>, FunctionRef<void()>);
        int (*n_threads)(void*);
    };

    // One table per adapted type, with static storage duration: a `static
    // constexpr` data member of a template is implicitly inline in C++17, so
    // this costs one 32-byte object per executor type and no initialisation
    // order to reason about.
    template <typename E>
    static constexpr VTable kTableFor{
        [](void* obj, int begin, int end, FunctionRef<void(int)> f) {
            static_cast<E*>(obj)->parallel_for(begin, end, f);
        },
        [](void* obj, int begin, int end, FunctionRef<void(int, int, int)> f) {
            static_cast<E*>(obj)->parallel_for_chunked(begin, end, f);
        },
        [](void* obj, FunctionRef<void()> f, FunctionRef<void()> g) {
            static_cast<E*>(obj)->parallel_invoke(f, g);
        },
        // The return type is spelled on the lambda rather than cast inside it: an
        // `n_threads()` that already returns `int` -- which every executor in
        // this tree has -- would make the cast redundant, and a redundant cast is
        // itself a lint finding.
        [](void* obj) -> int { return static_cast<E*>(obj)->n_threads(); },
    };

    void* obj_;
    const VTable* table_;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
/// The shape `ExecutorRef` adapts, as a concept, for a C++20 caller that wants
/// the error at its own declaration rather than inside our constructor.
///
/// This library is C++17 (`CMAKE_CXX_STANDARD 17`), so nothing here uses it --
/// it is compiled only when the INCLUDING translation unit is C++20 or later,
/// which is why it is an `#if` rather than a hard requirement. `ExecutorRef`
/// itself deliberately does not constrain on it: doing so would make the
/// constructor's availability depend on the caller's language mode.
template <typename E>
concept Executor = requires(E e, int n) {
    e.parallel_for(0, n, [](int) {});
    e.parallel_for_chunked(0, n, [](int, int, int) {});
    e.parallel_invoke([] {}, [] {});
    { e.n_threads() } -> std::convertible_to<int>;
};

/// `cbls::static_assert_executor<MyPool>();` -- a one-line compile-time check
/// with a readable failure, for callers who would rather not write the
/// `static_assert` themselves.
template <typename E>
constexpr void static_assert_executor() {
    static_assert(Executor<E>,
                  "type does not satisfy cbls::Executor: it needs parallel_for, "
                  "parallel_for_chunked, parallel_invoke and n_threads");
}
#endif

}  // namespace cbls
