#pragma once

#include <atomic>
#include <type_traits>

namespace cbls {

/// A host's cancellation channel, type-erased and NON-OWNING (#169), for a
/// host that cancels for reasons the engine cannot see (a peer finished, a
/// user stop, another clock). `SearchCoordination::stop` is not this API.
///
/// Adapts anything with `bool requested() const`. Nothing here owns, copies or
/// destroys the source, so the SOURCE MUST OUTLIVE THE SOLVE. A default
/// `StopRef` is never requested and costs one null compare, so the
/// unconfigured search keeps exactly its trajectory.
///
/// `std::stop_token` does not fit (its member is `stop_requested()`; it fails
/// to compile inside `thunk`). A C++20 host writes the adaptor, held as a NAMED
/// lvalue because the deleted rvalue constructor refuses a temporary:
///
/// ```
///   struct CancelOn {
///       std::stop_token token;
///       bool requested() const { return token.stop_requested(); }
///   };
///   CancelOn cancel{source.get_token()};
///   config.stop = cancel;
/// ```
///
/// `requested()` is polled at the search's safe points (where the wall clock
/// is read), so a raised stop ends the run within ONE batch, not one
/// iteration; see `SearchConfig::stop`. It is called from the search thread
/// and from EVERY `ParallelSearch` worker, so it must be safe to call
/// concurrently.
class StopRef {
public:
    /// Never requested. The default on `SearchConfig` and `ParallelConfig`.
    StopRef() = default;

    /// Non-explicit so `config.stop = token;` works. The `enable_if` keeps this
    /// template from hijacking the copy constructor.
    template <typename S, typename = std::enable_if_t<!std::is_same_v<std::decay_t<S>, StopRef>>>
    // Implicit by design; see above.
    StopRef(const S& source) : obj_(&source), fn_(&thunk<S>) {}

    /// Rejects a TEMPORARY at compile time: `const S&` above would bind
    /// `config.stop = HostFlag{};` and read freed memory at every poll. Mirrors
    /// `ExecutorRef`, which takes `E&`. `const S&&` binds rvalues only.
    template <typename S, typename = std::enable_if_t<!std::is_same_v<std::decay_t<S>, StopRef>>>
    StopRef(const S&&) = delete;

    /// False when no source is attached. A source whose `requested()` throws
    /// is a contract violation.
    [[nodiscard]] bool requested() const { return fn_ != nullptr && fn_(obj_); }

    /// Whether a source is attached (for reporting and the bindings).
    ///
    /// AN ATTACHED STOP IS NOT A BUDGET: a run with no wall clock and no
    /// iteration limit returns at once with `TerminationReason::NoBudget`, so a
    /// host that forgets to cancel cannot hang it. Give such a run a generous
    /// `time_limit` and cancel inside it.
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
/// Relaxed ordering: the flag guards no data, and seeing it late costs one
/// batch. Non-copyable and non-movable, so a `StopRef` to it cannot dangle by
/// relocation; own it for at least as long as the solve.
class StopToken {
public:
    StopToken() = default;
    StopToken(const StopToken&) = delete;
    StopToken& operator=(const StopToken&) = delete;
    StopToken(StopToken&&) = delete;
    StopToken& operator=(StopToken&&) = delete;
    ~StopToken() = default;

    void request() noexcept { flag_.store(true, std::memory_order_relaxed); }
    /// Clear the flag for reuse. A solve does NOT reset it: a raised token
    /// handed to the next solve cancels that one before it starts.
    void reset() noexcept { flag_.store(false, std::memory_order_relaxed); }
    [[nodiscard]] bool requested() const noexcept { return flag_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> flag_{false};
};

}  // namespace cbls
