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
    structural_batches += other.structural_batches;
    structural_moves_tried += other.structural_moves_tried;
    structural_moves_accepted += other.structural_moves_accepted;
    inner_solver_calls += other.inner_solver_calls;
    inner_solver_seconds += other.inner_solver_seconds;
    portfolio_restarts += other.portfolio_restarts;

    // Positionally when the two describe the SAME generator set, by name
    // otherwise.
    //
    // By name is the correct rule and the fast path does not weaken it: a worker
    // that threw before building its batch contributes an EMPTY vector, and a
    // blind merge by index would then attribute the survivors' rows to the wrong
    // generator. `StructuralBatch` makes the names unique within one batch (see
    // its constructor), and every worker builds the same generators in the same
    // order, so in the normal case the two vectors agree name-for-name at every
    // position -- checked in O(G) -- and the merge is a walk.
    //
    // The fast path is a complexity fix, not a micro-optimisation, and it wins
    // and loses in stated regimes. `by_generator` carries ONE ENTRY PER
    // List/Set VARIABLE, since the built-ins register per variable, and
    // `structural_batch.h` reasons about a 1500-List model: the by-name scan is
    // O(G^2) string comparisons, ~1.1M at G = 1500, and `merge` runs once per
    // restart per worker as well as once per worker. The fast path is O(G). It
    // loses nothing when it does not apply -- one O(G) name comparison before
    // falling through -- and it does not apply exactly when the vectors differ,
    // which is the case the by-name scan exists for.
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
