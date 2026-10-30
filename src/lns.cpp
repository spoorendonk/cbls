#include "cbls/lns.h"

#include "cbls/dag_ops.h"
#include "cbls/move_generator.h"
#include "cbls/randomize.h"
#include "cbls/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace cbls {

LNS::LNS(double destroy_fraction) : destroy_fraction_(destroy_fraction) {}

// Out of line so the vtable has a single translation unit to live in.
LNS::~LNS() = default;

// Lexicographic search-state key (real violation, objective): feasibility first,
// then objective — mirroring solve()'s record_best. Real violation excludes the
// artificial `obj <= bound` soft constraint (when present) so the objective is
// not double-counted, the way the dead augmented_objective() accept rule did.
//
// A row's violation counts only ABOVE `tol`: at or below it the row is
// satisfied, as the search reads it, and its residual is rounding noise that
// must not decide between two feasible states -- the objective does.
std::pair<double, double> lns_acceptance_key(const Model& model, double tol) {
    const int32_t obj_ci = model.has_objective_constraint() ? model.objective_constraint_idx() : -1;
    const auto& cids = model.constraint_ids();
    double real_violation = 0.0;
    for (size_t i = 0; i < cids.size(); ++i) {
        if (static_cast<int32_t>(i) == obj_ci) {
            continue;
        }
        const double v = model.node_value(cids[i]);
        // NaN must be handled before the clamp: `NaN > tol` is false, which
        // would score a NaN-poisoned repair as perfectly feasible and accept it
        // over a genuinely better incumbent. Same guard as
        // clamped_node_violation and search.cpp's max_real_violation.
        if (std::isnan(v)) {
            real_violation = std::numeric_limits<double>::infinity();
            break;
        }
        // A domain row is violated at any v > 0 (#258), as is_feasible reads it.
        real_violation += v > (model.is_domain_node(cids[i]) ? 0.0 : tol) ? v : 0.0;
    }
    double obj = model.objective_id() >= 0 ? model.node_value(model.objective_id()) : 0.0;
    // A NaN objective (a domain error, inf - inf) ranks LAST, as +inf: compared
    // raw, `x < NaN` is false for every x, so no repair could displace a
    // NaN-objective incumbent at equal violation (#205). Same reading as
    // record_best's, which treats a non-finite objective as the worst witness.
    if (std::isnan(obj)) {
        obj = std::numeric_limits<double>::infinity();
    }
    return {real_violation, obj};
}

namespace {

// ASB-RR's ruin and recreate parameters (Christiaens & Vanden Berghe, the KU
// Leuven technical report of 7.11.2016, Table 4 and Section 6.2; the same
// values the SISR paper, Transportation Science 54(2) 2020, publishes).
constexpr double kMeanRemoved = 10.0;      // c-bar: elements removed on average
constexpr double kMaxStringLength = 10.0;  // L_max (L_min = K_min = 1)
constexpr double kSplitStringRate = 0.5;   // 'string' and 'split string' equally often
constexpr double kSplitDepth = 0.01;       // alpha: the preserved-run length's stop rate
constexpr double kBlinkRate = 0.01;        // beta: a position is skipped with this rate

using Clock = std::chrono::steady_clock;

// The acceptance key's real violation restricted to `rows` (indices into
// constraint_ids()), with the same clamp at `tol` and the same NaN rule.
double real_violation_over(const Model& model, ConstSpan<int32_t> rows, int32_t obj_ci,
                           double tol) {
    const auto& cids = model.constraint_ids();
    double total = 0.0;
    for (const int32_t ci : rows) {
        if (ci == obj_ci) {
            continue;
        }
        const double v = model.node_value(cids[static_cast<size_t>(ci)]);
        if (std::isnan(v)) {
            return std::numeric_limits<double>::infinity();
        }
        total += v > tol ? v : 0.0;
    }
    return total;
}

double objective_value(const Model& model) {
    if (model.objective_id() < 0) {
        return 0.0;
    }
    const double obj = model.node_value(model.objective_id());
    return std::isnan(obj) ? std::numeric_limits<double>::infinity() : obj;
}

// Ruin and recreate one `ListPartition` (#202) is ASB-RR's adjacent string
// removal (Algorithm 1) and greedy insertion with blinks (Algorithm 2), on any
// partition rather than on routes. The pieces below are its steps: who holds
// what, which elements one string takes, the ruin, the best insertion, and the
// recreate that drives it.
//
// It departs from the reference in two places, both because the engine knows
// nothing about what an element MEANS:
//
//  - adj(seed) is "every customer by increasing distance from the seed". The
//    engine has only `neighbours` -- the k nearest, when the caller supplied a
//    list at all -- so adj(seed) is the seed, then its neighbour row, then the
//    remaining assigned elements in random order.
//  - Recreate sorts the removed set by Random, Demand, Far or Close with
//    weights 4/4/2/1. Demand and depot distance are routing quantities, so the
//    order here is always Random.

// Which list holds each element of one partition.
struct PartitionView {
    std::vector<int32_t> owner;     // per element: index into list_ids, or -1
    std::vector<int32_t> assigned;  // every element some list holds
    int32_t non_empty = 0;          // lists holding at least one element
};

PartitionView view_partition(const Model& model, const ListPartition& part) {
    PartitionView v;
    v.owner.assign(static_cast<size_t>(part.universe_size), -1);
    for (size_t li = 0; li < part.list_ids.size(); ++li) {
        const auto& el = model.var(part.list_ids[li]).elements;
        v.non_empty += el.empty() ? 0 : 1;
        for (const int32_t e : el) {
            v.owner[static_cast<size_t>(e)] = static_cast<int32_t>(li);
            v.assigned.push_back(e);
        }
    }
    return v;
}

// Which positions of a list of `len` one string takes: `l` of them, in a
// window that holds position `pos`. 'string' (or whenever the whole list goes)
// takes l consecutive positions; 'split string' takes a window of l + m and
// keeps a run of m inside it, m starting at 1 and kept when U < alpha or at its
// maximum len - l, otherwise grown by one. Where the kept run sits inside the
// window is not pinned by the reference's text; it is uniform over the l + 1
// placements.
std::vector<bool> string_positions(int32_t len, int32_t pos, int32_t l, RNG& rng) {
    // A window of `width` positions holding `pos`, uniformly among those that
    // fit inside the list.
    auto window_start = [&](int32_t width) {
        const int32_t lo = std::max(0, pos - width + 1);
        const int32_t hi = std::min(pos, len - width);
        return static_cast<int32_t>(rng.integers(lo, static_cast<int64_t>(hi) + 1));
    };
    std::vector<bool> take(static_cast<size_t>(len), false);
    if (l == len || rng.random() < 1.0 - kSplitStringRate) {
        const int32_t s = window_start(l);
        std::fill(take.begin() + s, take.begin() + s + l, true);
        return take;
    }
    const int32_t m_max = len - l;
    int32_t m = 1;
    while (m < m_max && rng.random() >= kSplitDepth) {
        ++m;
    }
    const int32_t s = window_start(l + m);
    const auto keep_at = static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(l) + 1));
    for (int32_t i = 0; i < l + m; ++i) {
        take[static_cast<size_t>(s) + static_cast<size_t>(i)] = i < keep_at || i >= keep_at + m;
    }
    return take;
}

// Algorithm 1: strings near a random seed, from at most k lists, one string per
// list, never below a list's `min_size`. Returns the removed elements; the
// lists they left are appended to `ruined_vars`, and `view` is kept current.
std::vector<int32_t> ruin(Model& model, const ListPartition& part, PartitionView& view, RNG& rng,
                          const NeighbourList* neighbours, std::vector<int32_t>& ruined_vars) {
    // Eq. 3-5, with K_min = L_min = 1: l_max over the mean length of the lists
    // in use (the reference's tours), k_max from it, k uniform in [1, k_max].
    const double mean_len =
        static_cast<double>(view.assigned.size()) / static_cast<double>(view.non_empty);
    const double l_max = std::min(mean_len, kMaxStringLength);
    const double k_max = (4.0 * kMeanRemoved / (1.0 + l_max)) - 1.0;
    const auto k = static_cast<size_t>(std::floor(rng.uniform(1.0, k_max + 1.0)));

    const int32_t seed = view.assigned[static_cast<size_t>(
        rng.integers(0, static_cast<int64_t>(view.assigned.size())))];
    std::vector<int32_t> adj{seed};
    if (neighbours != nullptr) {
        for (const int32_t e : neighbours->of(seed)) {
            if (e >= 0 && e < part.universe_size) {
                adj.push_back(e);
            }
        }
    }
    std::vector<int32_t> rest = view.assigned;
    rng.shuffle(rest);
    adj.insert(adj.end(), rest.begin(), rest.end());

    std::vector<bool> ruined(part.list_ids.size(), false);
    std::vector<int32_t> removed;
    for (const int32_t c : adj) {
        if (ruined_vars.size() >= k) {
            break;
        }
        const int32_t li = view.owner[static_cast<size_t>(c)];
        if (li < 0 || ruined[static_cast<size_t>(li)]) {
            continue;  // already removed, unassigned, or its list already ruined
        }
        const int32_t vid = part.list_ids[static_cast<size_t>(li)];
        Variable& var = model.var_mut(vid);
        const auto len = static_cast<int32_t>(var.elements.size());
        const int32_t removable = len - var.min_size;
        if (removable <= 0) {
            continue;  // at its minimum length: nothing may leave it
        }
        // Eq. 6-7: l uniform in [1, min(|t|, l_max)].
        const double l_max_t = std::min(static_cast<double>(len), l_max);
        const int32_t l =
            std::min(static_cast<int32_t>(std::floor(rng.uniform(1.0, l_max_t + 1.0))), removable);
        const auto pos = static_cast<int32_t>(
            std::find(var.elements.begin(), var.elements.end(), c) - var.elements.begin());
        const std::vector<bool> take = string_positions(len, pos, l, rng);
        std::vector<int32_t> kept;
        kept.reserve(var.elements.size());
        for (size_t i = 0; i < var.elements.size(); ++i) {
            const int32_t e = var.elements[i];
            if (take[i]) {
                removed.push_back(e);
                view.owner[static_cast<size_t>(e)] = -1;
            } else {
                kept.push_back(e);
            }
        }
        var.elements = std::move(kept);
        ruined[static_cast<size_t>(li)] = true;
        ruined_vars.push_back(vid);
    }
    return removed;
}

using InsertKey = std::pair<double, double>;  // (real violation added, objective)

struct Insertion {
    InsertKey key{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    int32_t list = -1;  // index into list_ids; -1 for none
    int32_t pos = -1;
};

// Algorithm 2's inner loops for one element: the lists in random order, every
// position of each with room, each evaluated with probability 1 - beta. "A tour
// that can serve c" is read through the cost rather than as a filter: the key
// is (real violation added on the list's rows, objective), lexicographic, the
// acceptance key's own rule, so a position that breaks no row beats every one
// that does, and an empty list is the reference's new tour. Each position is
// probed through the DAG and rolled back. Improves on `best` only strictly.
Insertion best_insertion(Model& model, const ListPartition& part, int32_t e, Insertion best,
                         std::vector<int32_t>& order, RNG& rng, double tol) {
    const int32_t obj_ci = model.has_objective_constraint() ? model.objective_constraint_idx() : -1;
    rng.shuffle(order);
    for (const int32_t li : order) {
        const int32_t vid = part.list_ids[static_cast<size_t>(li)];
        if (static_cast<int32_t>(model.var(vid).elements.size()) >= model.var(vid).max_size) {
            continue;
        }
        auto& el = model.var_mut(vid).elements;
        const ConstSpan<int32_t> rows = model.constraints_of_var(vid);
        const double before = real_violation_over(model, rows, obj_ci, tol);
        const auto len = static_cast<int32_t>(el.size());
        for (int32_t p = 0; p <= len; ++p) {
            if (rng.random() >= 1.0 - kBlinkRate) {
                continue;  // blink
            }
            el.insert(el.begin() + p, e);
            delta_evaluate(model, &vid, 1, DeltaMode::Probe);
            const InsertKey key{real_violation_over(model, rows, obj_ci, tol) - before,
                                objective_value(model)};
            el.erase(el.begin() + p);
            delta_evaluate(model, &vid, 1, DeltaMode::Rollback);
            if (key < best.key) {
                best = {key, li, p};
            }
        }
    }
    return best;
}

// Algorithm 2: the removed elements back in, in random order, each at its best
// insertion. Under `AtMostOnce`, leaving an element out is a candidate too, at
// the key the partition has without it. Where nothing was evaluated -- every
// position blinked, or the deadline reached -- an `Exact` partition still has
// to place the element, and takes the first list with room, at its end.
void recreate(Model& model, const ListPartition& part, std::vector<int32_t>& removed, RNG& rng,
              double tol, const Clock::time_point* deadline) {
    rng.shuffle(removed);  // the 'Random' order
    std::vector<int32_t> order(part.list_ids.size());
    std::iota(order.begin(), order.end(), 0);
    for (const int32_t e : removed) {
        Insertion best;
        if (part.cover == Cover::AtMostOnce) {
            best.key = {0.0, objective_value(model)};  // leave it out
        }
        if (deadline == nullptr || Clock::now() < *deadline) {
            best = best_insertion(model, part, e, best, order, rng, tol);
        }
        for (size_t li = 0;
             best.list < 0 && part.cover == Cover::Exact && li < part.list_ids.size(); ++li) {
            const Variable& var = model.var(part.list_ids[li]);
            if (static_cast<int32_t>(var.elements.size()) < var.max_size) {
                best.list = static_cast<int32_t>(li);
                best.pos = static_cast<int32_t>(var.elements.size());
            }
        }
        if (best.list < 0) {
            continue;  // AtMostOnce: left out
        }
        const int32_t vid = part.list_ids[static_cast<size_t>(best.list)];
        auto& el = model.var_mut(vid).elements;
        el.insert(el.begin() + best.pos, e);
        delta_evaluate(model, &vid, 1, DeltaMode::Commit);
    }
}

void ruin_recreate_partition(Model& model, const ListPartition& part, RNG& rng,
                             const NeighbourList* neighbours, double tol,
                             const Clock::time_point* deadline) {
    if (part.universe_size <= 0 || part.list_ids.empty()) {
        return;
    }
    PartitionView view = view_partition(model, part);
    if (view.assigned.empty()) {
        return;
    }
    std::vector<int32_t> ruined_vars;
    std::vector<int32_t> removed = ruin(model, part, view, rng, neighbours, ruined_vars);
    if (removed.empty()) {
        return;
    }
    delta_evaluate(model, ruined_vars, DeltaMode::Commit);
    recreate(model, part, removed, rng, tol, deadline);
}

// The variables the original destroy randomises: every variable outside a
// partition (`candidates`). With no partition that is every variable, and the
// draws are the ones this loop always made: k whole sequences plus a fraction
// of the rest when `VarSequence`s are registered, a uniform fraction otherwise.
std::vector<int32_t> scalar_destroy_set(const Model& model, std::vector<int32_t> candidates,
                                        RNG& rng, double fraction) {
    if (candidates.empty()) {
        return {};
    }
    const auto& seqs = model.var_sequences();
    if (seqs.empty()) {
        const int n_destroy =
            std::max(1, static_cast<int>(static_cast<double>(candidates.size()) * fraction));
        rng.shuffle(candidates);
        candidates.resize(n_destroy);
        return candidates;
    }
    const int n_seqs = static_cast<int>(seqs.size());
    const int k = std::max(1, static_cast<int>(std::ceil(n_seqs * fraction)));
    std::vector<int32_t> seq_indices(n_seqs);
    std::iota(seq_indices.begin(), seq_indices.end(), 0);
    rng.shuffle(seq_indices);
    seq_indices.resize(k);

    std::vector<bool> in_seq(model.num_vars(), false);
    for (const auto& s : seqs) {
        for (const int32_t v : s.var_ids) {
            in_seq[v] = true;
        }
    }
    std::vector<int32_t> out;
    for (const int32_t si : seq_indices) {
        out.insert(out.end(), seqs[si].var_ids.begin(), seqs[si].var_ids.end());
    }
    std::vector<int32_t> non_seq;
    for (const int32_t i : candidates) {
        if (!in_seq[i]) {
            non_seq.push_back(i);
        }
    }
    const int n_non_seq_destroy =
        std::max(0, static_cast<int>(static_cast<double>(non_seq.size()) * fraction));
    rng.shuffle(non_seq);
    non_seq.resize(n_non_seq_destroy);
    out.insert(out.end(), non_seq.begin(), non_seq.end());
    return out;
}

}  // namespace

bool LNS::destroy_repair(Model& model, ViolationManager& vm, RNG& rng, double repair_time_limit) {
    auto old_key = lns_acceptance_key(model, feasibility_tolerance_);
    auto saved_state = model.copy_state();

    // Every ListPartition is ruined and recreated (#202); its members are then
    // left out of the scalar destroy below, which FJ could not repair anyway.
    // One deadline for the whole repair: the FJ pass below gets what the
    // partitions left of it, so a kick on a model with both stays inside
    // `repair_time_limit` rather than taking it twice.
    const auto& partitions = model.list_partitions();
    const Clock::time_point deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(std::max(0.0, repair_time_limit)));
    if (!partitions.empty()) {
        for (const ListPartition& part : partitions) {
            ruin_recreate_partition(model, part, rng, neighbours_.get(), feasibility_tolerance_,
                                    repair_time_limit > 0.0 ? &deadline : nullptr);
        }
    }

    // The original pair, on every variable outside a partition.
    std::vector<int32_t> candidates;
    candidates.reserve(model.num_vars());
    for (int32_t i = 0; i < static_cast<int32_t>(model.num_vars()); ++i) {
        if (!model.var(i).partitioned) {
            candidates.push_back(i);
        }
    }
    const std::vector<int32_t> var_indices =
        scalar_destroy_set(model, std::move(candidates), rng, destroy_fraction_);

    if (!var_indices.empty()) {
        // Randomize destroyed variables. ListOrder::Perturb because
        // destroy/repair works on an incumbent: a List outside a partition
        // keeps its elements and gets a new order, which is what this loop did
        // before the three copies were merged.
        for (int32_t idx : var_indices) {
            randomize_var(model.var_mut(idx), rng, ListOrder::Perturb);
        }

        full_evaluate(model);

        // Repair via FJ-NL, on what the partitions left of the limit (with no
        // partition, the limit as it always was). Floored at a tiny positive
        // value while a limit exists: 0 means "no wall-clock limit" to
        // fj_nl_initialize, the opposite of a deadline already used up.
        const double fj_limit =
            repair_time_limit > 0.0 && !partitions.empty()
                ? std::max(1e-9, std::chrono::duration<double>(deadline - Clock::now()).count())
                : repair_time_limit;
        fj_nl_initialize(model, vm, 2000, &rng, fj_limit, feasibility_tolerance_, exact_int_jump_);
    } else {
        full_evaluate(model);
    }

    auto new_key = lns_acceptance_key(model, feasibility_tolerance_);

    if (new_key < old_key) {
        return true;
    }
    model.restore_state(saved_state);
    full_evaluate(model);
    return false;
}

int LNS::destroy_repair_cycle(Model& model, ViolationManager& vm, RNG& rng, int n_rounds,
                              double repair_time_limit) {
    int improvements = 0;
    for (int i = 0; i < n_rounds; ++i) {
        if (destroy_repair(model, vm, rng, repair_time_limit)) {
            improvements++;
        }
    }
    return improvements;
}

}  // namespace cbls
