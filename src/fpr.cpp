#include "cbls/fpr.h"

#include "cbls/bound_propagation.h"
#include "cbls/dag.h"
#include "cbls/dag_ops.h"
#include "cbls/domain_propagation.h"
#include "cbls/model.h"
#include "cbls/rng.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>

namespace cbls {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
/// Paper Sect. 6: an infinite bound is boxed to [-1e5, 1e5], anchored at the
/// finite side when there is one (mip-heuristics `boxed_domain`).
constexpr double kBox = 1e5;
/// mip-heuristics' zero-cost test for GoodObj/BadObj.
constexpr double kZeroCost = 1e-15;
/// `compute_alt`'s "is the preferred value the lower bound" tolerance.
constexpr double kFeasTol = 1e-6;
/// The Random order's seed: fixed, as the reference precomputes its orders.
constexpr uint64_t kOrderSeed = 0x5EED;

struct Box {
    double lo;
    double hi;
};

bool finite_bound(double b, double inf) {
    return std::abs(b) < inf;
}

Box boxed(double lo, double hi, double inf) {
    const bool has_lo = finite_bound(lo, inf);
    const bool has_hi = finite_bound(hi, inf);
    if (has_lo && has_hi) {
        return {lo, hi};
    }
    if (has_lo) {
        return {lo, lo + kBox};
    }
    if (has_hi) {
        return {hi - kBox, hi};
    }
    return {-kBox, kBox};
}

double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

/// Static locks: per variable, the rows a move up (down) could violate.
void count_locks(const LinearRowStore& store, size_t n, double inf, std::vector<int32_t>& up,
                 std::vector<int32_t>& down) {
    up.assign(n, 0);
    down.assign(n, 0);
    for (int32_t r = 0; r < store.num_rows(); ++r) {
        const bool has_lo = finite_bound(store.row_lo(r), inf);
        const bool has_hi = finite_bound(store.row_hi(r), inf);
        for (size_t k = store.row_start(r); k < store.row_start(r + 1); ++k) {
            const auto v = static_cast<size_t>(store.cols()[k]);
            const double a = store.coefs()[k];
            up[v] += static_cast<int32_t>(a > 0 && has_hi) + static_cast<int32_t>(a < 0 && has_lo);
            down[v] +=
                static_cast<int32_t>(a < 0 && has_hi) + static_cast<int32_t>(a > 0 && has_lo);
        }
    }
}

/// The model rows of `var`, the objective row not counted.
size_t rows_of(const Model& model, int32_t var) {
    const auto rows = model.constraints_of_var(var);
    const int32_t obj_row = model.objective_constraint_idx();
    return static_cast<size_t>(
        std::count_if(rows.begin(), rows.end(), [&](int32_t c) { return c != obj_row; }));
}

struct LockDelta {
    int32_t up;
    int32_t down;
};

/// loosedyn's locks from one row (paper Sect. 4.2): none from a row already
/// infeasible or already redundant at the current activities.
LockDelta dynamic_locks(double a, double row_lo, double row_hi, double min_act, double max_act,
                        double inf) {
    const bool has_lo = finite_bound(row_lo, inf);
    const bool has_hi = finite_bound(row_hi, inf);
    if ((has_hi && min_act > row_hi) || (has_lo && max_act < row_lo)) {
        return {0, 0};
    }
    if ((!has_lo || min_act >= row_lo) && (!has_hi || max_act <= row_hi)) {
        return {0, 0};
    }
    return {static_cast<int32_t>(has_hi && a > 0) + static_cast<int32_t>(has_lo && a < 0),
            static_cast<int32_t>(has_hi && a < 0) + static_cast<int32_t>(has_lo && a > 0)};
}

}  // namespace

FprConfig fpr_portfolio_config(int64_t worker, int64_t attempt) {
    const auto n = static_cast<int64_t>(kFprPortfolio.size());
    const int64_t idx = (((worker + attempt) % n) + n) % n;
    return kFprPortfolio[static_cast<size_t>(idx)];
}

FprSearch::FprSearch(const Model& model, const LinearRowStore& store, DomainPropagator& prop)
    : model_(model), store_(store), prop_(prop), inf_(prop.options().bounds.infinity) {
    const size_t n = model.num_vars();
    integral_.assign(n, 0);
    binary_.assign(n, 0);
    linear_only_.assign(n, 0);
    cost_.assign(n, 0.0);
    forced_.assign(n, 0);
    forced_value_.assign(n, 0.0);
    for (size_t v = 0; v < n; ++v) {
        const Variable& var = model.var(static_cast<int32_t>(v));
        if (is_structured(var.type)) {
            continue;
        }
        scalars_.push_back(static_cast<int32_t>(v));
        const bool integral = var.type == VarType::Bool || var.type == VarType::Int;
        integral_[v] = static_cast<uint8_t>(integral);
        binary_[v] = static_cast<uint8_t>(integral && var.lb == 0.0 && var.ub == 1.0);
    }
    node_cap_ = static_cast<int64_t>(n) + 1;
    count_locks(store, n, inf_, up_locks_, down_locks_);
    for (const int32_t v : scalars_) {
        const size_t stored = store.col_start(v + 1) - store.col_start(v);
        linear_only_[static_cast<size_t>(v)] = static_cast<uint8_t>(rows_of(model, v) == stored);
    }
    if (model.objective_id() >= 0) {
        std::vector<std::pair<int32_t, double>> partials;
        compute_partials_sparse(model, model.objective_id(), partials);
        for (const auto& [v, d] : partials) {
            cost_[static_cast<size_t>(v)] = d;
        }
    }
    RNG order_rng(kOrderSeed);
    for (const FprVarOrder o :
         {FprVarOrder::LR, FprVarOrder::Type, FprVarOrder::Random, FprVarOrder::Locks}) {
        static_order(o, order_rng, orders_[static_cast<size_t>(o)]);
    }
}

bool FprSearch::propagate_root() {
    // propagate_all checks rows only; an empty model domain (lb > ub) on a
    // scalar in no row is a refuted root too.
    root_ok_ = prop_.propagate_all() &&
               std::none_of(scalars_.begin(), scalars_.end(),
                            [this](int32_t v) { return prop_.lo(v) > prop_.hi(v); });
    return root_ok_;
}

void FprSearch::static_order(FprVarOrder order, RNG& rng, std::vector<int32_t>& out) const {
    out.clear();
    if (order == FprVarOrder::LR) {
        out = scalars_;
        return;
    }
    std::vector<int32_t> bin;
    std::vector<int32_t> gen;
    std::vector<int32_t> cont;
    for (const int32_t v : scalars_) {
        const auto i = static_cast<size_t>(v);
        if (binary_[i] != 0) {
            bin.push_back(v);
        } else if (integral_[i] != 0) {
            gen.push_back(v);
        } else {
            cont.push_back(v);
        }
    }
    if (order == FprVarOrder::Random) {
        rng.shuffle(bin);
        rng.shuffle(gen);
        rng.shuffle(cont);
    } else if (order == FprVarOrder::Locks) {
        // Stable: formulation order breaks ties.
        const auto by_locks = [&](int32_t a, int32_t b) {
            const auto ia = static_cast<size_t>(a);
            const auto ib = static_cast<size_t>(b);
            return std::max(up_locks_[ia], down_locks_[ia]) >
                   std::max(up_locks_[ib], down_locks_[ib]);
        };
        std::stable_sort(bin.begin(), bin.end(), by_locks);
        std::stable_sort(gen.begin(), gen.end(), by_locks);
        std::stable_sort(cont.begin(), cont.end(), by_locks);
    }
    out.insert(out.end(), bin.begin(), bin.end());
    out.insert(out.end(), gen.begin(), gen.end());
    out.insert(out.end(), cont.begin(), cont.end());
}

double FprSearch::loose_dyn(int32_t var, double lo, double hi) const {
    // Paper Sect. 4.2: the direction fewer still-active rows lock.
    int32_t up = 0;
    int32_t down = 0;
    const size_t begin = store_.col_start(var);
    const size_t end = store_.col_start(var + 1);
    for (size_t e = begin; e < end; ++e) {
        const int32_t r = store_.col_rows()[e];
        const double min_act =
            prop_.min_activity_infinite(r) > 0 ? -kInf : prop_.min_activity_sum(r);
        const double max_act =
            prop_.max_activity_infinite(r) > 0 ? kInf : prop_.max_activity_sum(r);
        const LockDelta d = dynamic_locks(store_.coefs()[store_.col_entries()[e]], store_.row_lo(r),
                                          store_.row_hi(r), min_act, max_act, inf_);
        up += d.up;
        down += d.down;
        // Early exit, as the reference: one side already wins outright.
        const auto remaining = static_cast<int32_t>(end - e - 1);
        if (up > down + remaining) {
            return lo;
        }
        if (down > up + remaining) {
            return hi;
        }
    }
    if (up != down) {
        return up < down ? hi : lo;
    }
    const double c = cost_[static_cast<size_t>(var)];
    return std::abs(c) < kZeroCost || c > 0 ? lo : hi;
}

double FprSearch::choose_value(int32_t var, double lo, double hi, FprValueRule rule,
                               RNG& rng) const {
    const Box b = boxed(lo, hi, inf_);
    const double c = cost_[static_cast<size_t>(var)];
    const bool integral = integral_[static_cast<size_t>(var)] != 0;
    double v = b.lo;
    switch (rule) {
        case FprValueRule::Up:
            v = b.hi;
            break;
        case FprValueRule::Random:
            v = rng.uniform(b.lo, b.hi);
            break;
        case FprValueRule::GoodObj:
            v = std::abs(c) < kZeroCost || c > 0 ? b.lo : b.hi;
            break;
        case FprValueRule::BadObj:
            v = std::abs(c) >= kZeroCost && c > 0 ? b.hi : b.lo;
            break;
        case FprValueRule::LooseDyn:
            v = loose_dyn(var, b.lo, b.hi);
            break;
        case FprValueRule::Incumbent:
            v = model_.var(var).value;
            break;
    }
    if (integral) {
        v = std::round(v);
    }
    return clamp(v, b.lo, b.hi);
}

double FprSearch::alternative(int32_t var, double preferred) const {
    const auto i = static_cast<size_t>(var);
    if (binary_[i] != 0) {  // unfixed, so still [0, 1]
        return preferred < 0.5 ? 1.0 : 0.0;
    }
    const Box b = boxed(prop_.lo(var), prop_.hi(var), inf_);
    const double alt = std::abs(preferred - b.lo) < kFeasTol ? b.hi : b.lo;
    return integral_[i] != 0 ? std::round(alt) : alt;
}

bool FprSearch::fixed(int32_t var) const {
    return forced_[static_cast<size_t>(var)] != 0 || prop_.lo(var) >= prop_.hi(var);
}

void FprSearch::trivial_fixings() {
    // Paper Sect. 6: a variable with locks on one side only goes to its other
    // bound. Only where every row of it is linear (the store's): a nonlinear
    // row could lock it unseen. Every value is read off the root box first, as
    // the reference fixes them all before one propagation; a refuted one is
    // skipped (a deviation, see the header; #427).
    trivial_.clear();
    for (const int32_t v : scalars_) {
        const auto i = static_cast<size_t>(v);
        if (integral_[i] == 0 || linear_only_[i] == 0 || fixed(v)) {
            continue;
        }
        const bool up_free = up_locks_[i] == 0 && down_locks_[i] != 0;
        const bool down_free = down_locks_[i] == 0 && up_locks_[i] != 0;
        if (!up_free && !down_free) {
            continue;
        }
        const Box b = boxed(prop_.lo(v), prop_.hi(v), inf_);
        trivial_.emplace_back(v, up_free ? b.hi : b.lo);
    }
    for (const auto& [v, value] : trivial_) {
        const int32_t level = prop_.checkpoint();
        if (prop_.fix(v, value)) {
            continue;  // the checkpoint stays held: `run` unwinds it at exit
        }
        prop_.backtrack(level);
    }
}

void FprSearch::heap_push(int32_t var) {
    if (integral_[static_cast<size_t>(var)] == 0 || fixed(var)) {
        return;
    }
    heap_.emplace_back(prop_.hi(var) - prop_.lo(var), var);
    std::push_heap(heap_.begin(), heap_.end(), std::greater<>());
}

void FprSearch::heap_push_changed(int32_t level) {
    prop_.changed_since(level, changed_);
    for (const int32_t v : changed_) {
        heap_push(v);
    }
}

int32_t FprSearch::next_var(bool dynamic, size_t& cursor) {
    if (!dynamic) {
        for (; cursor < order_.size(); ++cursor) {
            const int32_t v = order_[cursor];
            if (integral_[static_cast<size_t>(v)] != 0 && !fixed(v)) {
                return v;
            }
        }
        return -1;
    }
    // Lazy deletion: an entry is live while its variable is unfixed with that
    // domain size; every size change pushed a fresh entry (`heap_push_changed`).
    // Ties go to the smaller id, as the (size, var) pair orders them.
    while (!heap_.empty()) {
        const auto [size, v] = heap_.front();
        if (!fixed(v) && size == prop_.hi(v) - prop_.lo(v)) {
            return v;
        }
        std::pop_heap(heap_.begin(), heap_.end(), std::greater<>());
        heap_.pop_back();
    }
    return -1;
}

void FprSearch::push_children(int32_t var, size_t cursor, bool backtracks, const FprConfig& cfg,
                              RNG& rng) {
    const double pref = choose_value(var, prop_.lo(var), prop_.hi(var), cfg.value, rng);
    const double alt = alternative(var, pref);
    const int32_t level = prop_.checkpoint();
    if (backtracks) {
        stack_.push_back(Node{var, alt, level, cursor});
    }
    stack_.push_back(Node{var, pref, level, cursor});
}

void FprSearch::fill(const FprConfig& cfg, RNG& rng, FprResult& out) const {
    out.values.assign(model_.num_vars(), 0.0);
    for (const int32_t v : scalars_) {
        const auto i = static_cast<size_t>(v);
        const Variable& var = model_.var(v);
        double lo = prop_.lo(v);
        double hi = prop_.hi(v);
        if (lo > hi) {  // a model domain with lb > ub: fall back to the model's box
            lo = var.lb;
            hi = var.ub;
        }
        const double c = cost_[i];
        double value = 0.0;
        if (forced_[i] != 0) {
            value = forced_value_[i];
        } else if (integral_[i] == 0) {
            // The cost-preferred bound, else the reference's cont_fallback (the
            // current value); either clamped to the boxed domain.
            const Box b = boxed(lo, hi, inf_);
            double want = var.value;
            if (std::abs(c) > kZeroCost) {
                want = c > 0 ? lo : hi;
            }
            value = clamp(want, b.lo, b.hi);
        } else if (lo == hi) {
            value = lo;
        } else {
            value = std::round(clamp(choose_value(v, lo, hi, cfg.value, rng), lo, hi));
        }
        out.values[i] = clamp(value, var.lb, var.ub);
    }
}

void FprSearch::run(const FprConfig& cfg, int64_t attempt, RNG& rng, FprResult& out) {
    out = FprResult{};
    out.wants_repair = cfg.mode != FprMode::Dfs;
    const bool backtracks = cfg.mode != FprMode::Diveprop;
    const bool dynamic = cfg.order == FprVarOrder::DomainSize;
    const int32_t entry = prop_.checkpoint();
    std::fill(forced_.begin(), forced_.end(), 0);
    stack_.clear();
    heap_.clear();

    if (!root_ok_) {  // no leaf: nothing is branched on a propagator in conflict
        fill(cfg, rng, out);
        prop_.backtrack(entry);
        return;
    }

    order_ = orders_[static_cast<size_t>(dynamic ? FprVarOrder::Type : cfg.order)];
    if (attempt > 0 && !order_.empty()) {
        const size_t len = std::max<size_t>(1, order_.size() * 3 / 10);
        std::shuffle(order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(len),
                     rng.engine());
    }
    trivial_fixings();
    if (dynamic) {
        for (const int32_t v : order_) {
            heap_push(v);
        }
    }

    size_t cursor = 0;
    const int32_t first = next_var(dynamic, cursor);
    if (first < 0) {
        out.complete = true;
    } else {
        push_children(first, cursor + 1, backtracks, cfg, rng);
        dive(cfg, rng, out);
    }
    fill(cfg, rng, out);
    prop_.backtrack(entry);
}

void FprSearch::dive(const FprConfig& cfg, RNG& rng, FprResult& out) {
    // Fig. 1 lines 4-16. In-tree repair (lines 7-8) is deferred (#419).
    const bool backtracks = cfg.mode != FprMode::Diveprop;
    const bool dynamic = cfg.order == FprVarOrder::DomainSize;
    while (!stack_.empty() && out.nodes < node_cap_) {
        // Every 16 nodes: a clock read is cheap next to 16 fixings (#429).
        if (has_deadline_ && (out.nodes & 15) == 15 &&
            std::chrono::steady_clock::now() >= deadline_) {
            return;
        }
        const Node node = stack_.back();
        stack_.pop_back();
        ++out.nodes;
        if (dynamic) {
            prop_.changed_since(node.level, changed_);
        }
        prop_.backtrack(node.level);
        prop_.checkpoint();  // == node.level again: this node's changes
        if (dynamic) {
            for (const int32_t v : changed_) {
                heap_push(v);  // their domains grew back
            }
        }
        size_t cursor = node.cursor;
        bool infeas = !prop_.fix(node.var, node.value);
        if (infeas) {
            ++out.conflicts;
            if (backtracks) {
                // Lines 9-10: the sibling is next on the stack. Undone now, so
                // a stack this empties fills from the parent's consistent state.
                prop_.backtrack(node.level);
                prop_.checkpoint();  // the sibling's pop backtracks to it
                continue;
            }
            // Diveprop does not stop (the paper's single big dive); the fixing
            // is kept outside the propagator (a deviation, see the header; #427).
            prop_.backtrack(node.level);
            prop_.checkpoint();
            forced_[static_cast<size_t>(node.var)] = 1;
            forced_value_[static_cast<size_t>(node.var)] = node.value;
        }
        if (dynamic && !infeas) {
            heap_push_changed(node.level);
        }
        const int32_t next = next_var(dynamic, cursor);
        if (next < 0) {
            out.complete = true;  // lines 12-16: a leaf
            return;
        }
        push_children(next, cursor + 1, backtracks, cfg, rng);
    }
}

}  // namespace cbls
