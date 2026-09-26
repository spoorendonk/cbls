// A caller-owned executor (#169): `ExecutorRef`, `FunctionRef`, and
// `ParallelConfig::executor`.
//
// The load-bearing assertion is negative and is made by COUNTING: with an
// executor set, `ParallelSearch` must create no `std::thread` of its own. A test
// that only checked "it still returns a result" would pass on an implementation
// that ignored the executor entirely, which is exactly the failure worth
// guarding. So the pool below counts the threads it creates, and the test
// compares that against the pool's own width -- and separately asserts that
// every worker index the portfolio asked for was actually run.

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace cbls;

namespace {

Model quadratic_model() {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));  // x + y >= 1
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
    return m;
}

// A minimal thread pool with the `Executor` shape, which COUNTS the threads it
// creates. It spawns per call rather than keeping a persistent set -- a real host
// pool would not, but that is the host's business, and spawning per call is what
// makes "how many threads exist because of this run" a number this test can
// read.
//
// One chunk per index, which is what `ParallelSearch` relies on to get a
// concurrent portfolio (`ParallelConfig::executor` says so, and says what a
// sequential executor degenerates to).
class CountingPool {
public:
    explicit CountingPool(int width) : width_(std::max(1, width)) {}

    [[nodiscard]] int n_threads() const { return width_; }
    [[nodiscard]] int threads_created() const { return created_.load(std::memory_order_relaxed); }

    void parallel_for(int begin, int end, const std::function<void(int)>& f) {
        parallel_for_chunked(begin, end, [&f](int chunk_begin, int chunk_end, int /*chunk*/) {
            for (int i = chunk_begin; i < chunk_end; ++i) {
                f(i);
            }
        });
    }

    void parallel_for_chunked(int begin, int end, const std::function<void(int, int, int)>& f) {
        if (end <= begin) {
            return;
        }
        const int n = end - begin;
        const int chunks = std::min(n, width_);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(chunks));
        for (int c = 0; c < chunks; ++c) {
            // Even split, remainder to the first chunks.
            const int base = n / chunks;
            const int extra = n % chunks;
            const int chunk_begin = begin + (c * base) + std::min(c, extra);
            const int chunk_end = chunk_begin + base + (c < extra ? 1 : 0);
            created_.fetch_add(1, std::memory_order_relaxed);
            threads.emplace_back(
                [&f, chunk_begin, chunk_end, c]() { f(chunk_begin, chunk_end, c); });
        }
        for (std::thread& t : threads) {
            t.join();
        }
    }

    void parallel_invoke(const std::function<void()>& f, const std::function<void()>& g) {
        created_.fetch_add(1, std::memory_order_relaxed);
        std::thread other(g);
        f();
        other.join();
    }

private:
    int width_;
    std::atomic<int> created_{0};
};

// The degenerate shape the field documents: everything on the calling thread, no
// thread created at all.
class SequentialPool {
public:
    explicit SequentialPool(int width) : width_(std::max(1, width)) {}
    [[nodiscard]] int n_threads() const { return width_; }
    /// Chunks handed to `f`, so a test can show that the whole range arrived as
    /// ONE chunk -- which is what makes this a one-worker portfolio.
    [[nodiscard]] int chunks_run() const { return chunks_; }

    void parallel_for(int begin, int end, const std::function<void(int)>& f) {
        ++chunks_;
        for (int i = begin; i < end; ++i) {
            f(i);
        }
    }
    void parallel_for_chunked(int begin, int end, const std::function<void(int, int, int)>& f) {
        if (end <= begin) {
            return;
        }
        ++chunks_;
        f(begin, end, 0);
    }
    void parallel_invoke(const std::function<void()>& f, const std::function<void()>& g) {
        ++chunks_;
        f();
        g();
    }

private:
    int width_;
    int chunks_ = 0;
};

}  // namespace

TEST_CASE("FunctionRef calls through to the referent", "[executor]") {
    int seen = 0;
    auto add = [&seen](int i) { seen += i; };
    const FunctionRef<void(int)> ref(add);
    ref(3);
    ref(4);
    REQUIRE(seen == 7);

    // A mutable lambda: the thunk invokes a non-const callable, which is why
    // FunctionRef stores a non-const pointer.
    auto counter = [n = 0]() mutable { return ++n; };
    const FunctionRef<int()> counting(counter);
    REQUIRE(counting() == 1);
    REQUIRE(counting() == 2);

    // Arguments are forwarded, not copied into a std::function.
    auto by_ref = [](int& out, int value) { out = value; };
    const FunctionRef<void(int&, int)> writer(by_ref);
    int target = 0;
    writer(target, 9);
    REQUIRE(target == 9);
}

TEST_CASE("ExecutorRef adapts a pool by shape", "[executor]") {
    CountingPool pool(4);
    const ExecutorRef exec(pool);
    REQUIRE(exec.n_threads() == 4);

    std::mutex mutex;
    std::vector<int> visited;
    exec.parallel_for(0, 7, [&mutex, &visited](int i) {
        const std::scoped_lock lock(mutex);
        visited.push_back(i);
    });
    std::sort(visited.begin(), visited.end());
    const std::vector<int> expected{0, 1, 2, 3, 4, 5, 6};
    REQUIRE(visited == expected);

    // chunk_idx is unique in [0, n_threads()), which is what lets a caller index
    // per-chunk storage without a lock.
    std::mutex chunk_mutex;
    std::vector<int> chunk_ids;
    int covered = 0;
    exec.parallel_for_chunked(0, 7,
                              [&chunk_mutex, &chunk_ids, &covered](int begin, int end, int chunk) {
                                  const std::scoped_lock lock(chunk_mutex);
                                  chunk_ids.push_back(chunk);
                                  covered += end - begin;
                              });
    REQUIRE(covered == 7);
    REQUIRE(std::set<int>(chunk_ids.begin(), chunk_ids.end()).size() == chunk_ids.size());
    for (int id : chunk_ids) {
        REQUIRE(id >= 0);
        REQUIRE(id < exec.n_threads());
    }

    bool first = false;
    bool second = false;
    exec.parallel_invoke([&first]() { first = true; }, [&second]() { second = true; });
    REQUIRE(first);
    REQUIRE(second);
}

TEST_CASE("an empty or inverted range is a no-op", "[executor]") {
    // Short-circuited inside ExecutorRef, so an adapted executor never has to
    // handle one. Counted rather than merely "did not crash": a pool that was
    // entered would have created a thread.
    CountingPool pool(4);
    const ExecutorRef exec(pool);
    int calls = 0;

    exec.parallel_for(3, 3, [&calls](int) { ++calls; });
    exec.parallel_for(5, 2, [&calls](int) { ++calls; });
    exec.parallel_for_chunked(3, 3, [&calls](int, int, int) { ++calls; });
    exec.parallel_for_chunked(5, 2, [&calls](int, int, int) { ++calls; });

    REQUIRE(calls == 0);
    REQUIRE(pool.threads_created() == 0);
}

TEST_CASE("a portfolio on a caller's executor creates no threads of its own",
          "[executor][parallel]") {
    // THE acceptance criterion. `CountingPool` counts every thread it creates,
    // and it creates exactly one per chunk, so with one chunk per worker the
    // count IS the worker count -- and any thread `ParallelSearch` created for
    // itself would be a thread this number does not include. The second
    // assertion is what makes the first mean something: every worker index was
    // actually run, so the portfolio did not simply do the work on the calling
    // thread and ignore the pool.
    CountingPool pool(3);

    std::mutex mutex;
    std::set<int> worker_indices;

    ParallelConfig par_config;
    par_config.n_threads = 3;
    par_config.executor = pool;
    par_config.tracer_factory = [&mutex, &worker_indices](int worker) -> std::unique_ptr<Tracer> {
        const std::scoped_lock lock(mutex);
        worker_indices.insert(worker);
        return nullptr;  // the index is all this test wants
    };

    ParallelSearch ps(3);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/31, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(r.feasible);
    // Three workers, three chunks, three threads -- all of them the pool's.
    REQUIRE(pool.threads_created() == 3);
    const std::set<int> expected{0, 1, 2};
    REQUIRE(worker_indices == expected);
}

TEST_CASE("the worker count is capped by the executor's width", "[executor][parallel]") {
    // `min(requested, n_threads())`. Asking for more workers than the pool can
    // run at once would queue workers behind others that hold their chunk for
    // the whole shared deadline, so the late ones would search on no budget at
    // all.
    CountingPool pool(2);

    std::mutex mutex;
    std::set<int> worker_indices;

    ParallelConfig par_config;
    par_config.n_threads = 8;  // far more than the pool's width
    par_config.executor = pool;
    par_config.tracer_factory = [&mutex, &worker_indices](int worker) -> std::unique_ptr<Tracer> {
        const std::scoped_lock lock(mutex);
        worker_indices.insert(worker);
        return nullptr;
    };

    ParallelSearch ps(8);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/33, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(r.feasible);
    REQUIRE(pool.threads_created() == 2);
    const std::set<int> expected{0, 1};
    REQUIRE(worker_indices == expected);
}

TEST_CASE("a sequential executor yields a working one-worker portfolio", "[executor][parallel]") {
    // Not an error, and documented as not being one: the first worker takes the
    // whole shared deadline, so the rest find it already past. What must still
    // hold is that the call returns a usable result rather than hanging or
    // reporting NoBudget.
    SequentialPool pool(4);

    ParallelConfig par_config;
    par_config.n_threads = 4;
    par_config.executor = pool;

    ParallelSearch ps(4);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/35, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(r.feasible);
    REQUIRE(r.iterations > 0);
    REQUIRE(r.termination == TerminationReason::TimeLimit);
    // One chunk, on the calling thread: no concurrency was available and none
    // was faked.
    REQUIRE(pool.chunks_run() == 1);
}

TEST_CASE("without an executor the portfolio still owns its threads", "[executor][parallel]") {
    // The control. `ParallelConfig::executor` unset is the shape every caller in
    // the tree uses, and it must keep creating its own workers -- a regression
    // that made the executor path unconditional would show up here as a
    // portfolio that ran one worker.
    std::mutex mutex;
    std::set<int> worker_indices;

    ParallelConfig par_config;
    par_config.n_threads = 3;
    REQUIRE_FALSE(par_config.executor.has_value());
    par_config.tracer_factory = [&mutex, &worker_indices](int worker) -> std::unique_ptr<Tracer> {
        const std::scoped_lock lock(mutex);
        worker_indices.insert(worker);
        return nullptr;
    };

    ParallelSearch ps(3);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.3, /*seed=*/37, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(r.feasible);
    const std::set<int> expected{0, 1, 2};
    REQUIRE(worker_indices == expected);
}

TEST_CASE("a worker that throws on an executor does not escape the executor",
          "[executor][parallel]") {
    // `run_one` swallows every worker exception, which is what makes an executor
    // usable at all: an executor is free to do anything with an escaping
    // exception, including abort. A model factory that always throws is the
    // strongest form -- every worker fails -- and the portfolio's own contract
    // then says the failure is rethrown from `solve`, on the CALLING thread.
    CountingPool pool(2);

    ParallelConfig par_config;
    par_config.n_threads = 2;
    par_config.executor = pool;

    ParallelSearch ps(2);
    REQUIRE_THROWS_AS(ps.solve([]() -> Model { throw std::runtime_error("no model"); },
                               /*time_limit=*/0.2, /*seed=*/39, SearchConfig{},
                               /*hook_factory=*/nullptr, /*lns_factory=*/nullptr,
                               /*callback=*/nullptr, par_config),
                      std::runtime_error);
    // The pool was used, and it survived: two chunks ran and joined normally.
    REQUIRE(pool.threads_created() == 2);
}

TEST_CASE("an executor and a stop token compose", "[executor][parallel][stop]") {
    // The two host-integration features are independent, and a host embedding
    // cbls uses both at once. A pre-raised token on an executor-run portfolio
    // must report Cancelled, not the clock.
    CountingPool pool(2);
    StopToken token;
    token.request();

    ParallelConfig par_config;
    par_config.n_threads = 2;
    par_config.executor = pool;
    par_config.stop = token;

    ParallelSearch ps(2);
    const SearchResult r =
        ps.solve([] { return quadratic_model(); }, /*time_limit=*/30.0, /*seed=*/41, SearchConfig{},
                 /*hook_factory=*/nullptr, /*lns_factory=*/nullptr,
                 /*callback=*/nullptr, par_config);

    REQUIRE(r.termination == TerminationReason::Cancelled);
}
