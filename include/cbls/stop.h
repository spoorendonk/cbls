#pragma once

#include <atomic>
#include <type_traits>

namespace cbls {

/// A host's cancellation channel, type-erased and NON-OWNING.
///
/// cbls runs as a component: inside another solver's primal heuristic, as one
/// stage of a pipeline, as a step in a batch job. Such a host cancels for
/// reasons the engine cannot see -- a peer component finished, the user pressed
/// stop, the enclosing deadline is a different clock -- and before #169 there
/// was no supported way to say so. `SearchCoordination::stop` existed but is
/// `ParallelSearch`'s private cross-worker channel, not an API.
///
/// Adapts ANYTHING with `bool requested() const`: `cbls::StopToken` below, a
/// `std::stop_token` (C++20), a host's own flag wrapper. Eight bytes of pointer
/// plus eight of thunk; nothing here owns, copies or destroys the source, so the
/// SOURCE MUST OUTLIVE THE SOLVE. A default-constructed `StopRef` is never
/// requested and reads one null pointer compare, which is what keeps the
/// unconfigured search on exactly the trajectory it had.
///
/// `requested()` is polled at the search's existing safe points -- the same
/// places the wall clock is read -- so a raised stop ends the run within ONE
/// batch, not within one iteration. See `SearchConfig::stop`.
///
/// The adapted object's `requested()` is called from the search thread, and from
/// EVERY worker thread under `ParallelSearch`. It must therefore be safe to call
/// concurrently; `StopToken` below is, through a relaxed atomic.
class StopRef {
public:
    /// Never requested. The default on `SearchConfig` and `ParallelConfig`.
    StopRef() = default;

    /// Non-explicit on purpose: `config.stop = token;` is the call shape this
    /// exists for. The `enable_if` keeps the copy constructor from being
    /// hijacked by this template when the argument is another `StopRef`.
    template <typename S, typename = std::enable_if_t<!std::is_same_v<std::decay_t<S>, StopRef>>>
    // Implicit by design; see above.
    StopRef(const S& source) : obj_(&source), fn_(&thunk<S>) {}

    /// False when no source is attached. Never throws: a source whose
    /// `requested()` throws is a contract violation, not a supported shape.
    [[nodiscard]] bool requested() const { return fn_ != nullptr && fn_(obj_); }

    /// Whether a source is attached at all. `!attached()` is what a caller
    /// checks to know the run is bounded by its budgets alone.
    [[nodiscard]] bool attached() const noexcept { return fn_ != nullptr; }

private:
    template <typename S>
    static bool thunk(const void* obj) {
        return static_cast<const S*>(obj)->requested();
    }

    const void* obj_ = nullptr;
    bool (*fn_)(const void*) = nullptr;
};

/// The built-in stop source: a thread-safe flag a host raises from anywhere.
///
/// Relaxed ordering on both ends, for the same reason `SearchCoordination::stop`
/// uses it: the flag guards no data, and the only cost of observing it a batch
/// late is that batch.
///
/// Non-copyable (it holds an atomic) and non-movable, which is also what makes
/// a `StopRef` to it safe to hold: the object cannot be relocated under the
/// search. Own it for at least as long as the solve.
class StopToken {
public:
    StopToken() = default;
    StopToken(const StopToken&) = delete;
    StopToken& operator=(const StopToken&) = delete;
    StopToken(StopToken&&) = delete;
    StopToken& operator=(StopToken&&) = delete;
    ~StopToken() = default;

    void request() noexcept { flag_.store(true, std::memory_order_relaxed); }
    /// Clear the flag so the token can be reused for a second solve. A token is
    /// NOT reset by a solve: a run that was cancelled leaves it raised, and
    /// handing the same raised token to the next solve cancels that one before
    /// it starts.
    void reset() noexcept { flag_.store(false, std::memory_order_relaxed); }
    [[nodiscard]] bool requested() const noexcept { return flag_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> flag_{false};
};

}  // namespace cbls
