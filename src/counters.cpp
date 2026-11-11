#include "cbls/counters.h"

#include <algorithm>
#include <cstddef>

namespace cbls {

const char* batch_kind_name(BatchKind kind) {
    switch (kind) {
        case BatchKind::FeasibilityJump:
            return "feasibility_jump";
        case BatchKind::NoveltyJump:
            return "novelty_jump";
        case BatchKind::Structural:
            return "structural";
    }
    // Unreachable for any value of the enum; keeps the function total so a caller
    // can print the kind unconditionally.
    return "feasibility_jump";
}

void SearchCounters::merge(const SearchCounters& other) {
    batches += other.batches;
    fj_batches += other.fj_batches;
    novelty_batches += other.novelty_batches;
    novelty_moves += other.novelty_moves;
    novelty_commits += other.novelty_commits;
    novelty_weight_bumps += other.novelty_weight_bumps;
    structural_batches += other.structural_batches;
    structural_moves_tried += other.structural_moves_tried;
    structural_moves_accepted += other.structural_moves_accepted;
    inner_solver_calls += other.inner_solver_calls;
    inner_solver_seconds += other.inner_solver_seconds;
    portfolio_restarts += other.portfolio_restarts;
    cone_cache_walks += other.cone_cache_walks;
    cone_fallback_walks += other.cone_fallback_walks;
    shared_bound_tightenings += other.shared_bound_tightenings;
    own_best_behind_global += other.own_best_behind_global;
    bound_behind_global_batches += other.bound_behind_global_batches;
    bound_behind_global_seconds += other.bound_behind_global_seconds;

    // By name is the rule: a worker that threw before building its batch contributes an EMPTY
    // vector, so a blind merge by index would misattribute rows. The two fast paths keep it.
    //
    // Cost: `by_generator` has one entry per List/Set variable, so the by-name scan is O(G^2)
    // string compares (~1.1M at G = 1500) and `merge` runs per restart per worker. An empty
    // target (each site's first merge) is a copy, and identical names at every position (the
    // normal case: unique names, same build order) are an O(G) walk. A mismatch costs one O(G)
    // check before the by-name scan, which exists for exactly that case. Without the empty-target
    // path the first merge per worker takes the by-name scan and the total stays O(W x G^2).
    if (by_generator.empty()) {
        by_generator = other.by_generator;
        return;
    }
    if (by_generator.size() == other.by_generator.size() &&
        std::equal(by_generator.begin(), by_generator.end(), other.by_generator.begin(),
                   [](const GeneratorCounters& mine, const GeneratorCounters& theirs) {
                       return mine.name == theirs.name;
                   })) {
        for (size_t i = 0; i < by_generator.size(); ++i) {
            by_generator[i].moves_tried += other.by_generator[i].moves_tried;
            by_generator[i].moves_accepted += other.by_generator[i].moves_accepted;
        }
        return;
    }
    for (const GeneratorCounters& g : other.by_generator) {
        auto it = std::find_if(by_generator.begin(), by_generator.end(),
                               [&g](const GeneratorCounters& mine) { return mine.name == g.name; });
        if (it == by_generator.end()) {
            by_generator.push_back(g);
            continue;
        }
        it->moves_tried += g.moves_tried;
        it->moves_accepted += g.moves_accepted;
    }
}

}  // namespace cbls
