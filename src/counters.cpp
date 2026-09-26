#include "cbls/counters.h"

#include "cbls/tracer.h"

#include <algorithm>

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
    return "feasibility_jump";
}

const char* kick_kind_name(KickKind kind) {
    switch (kind) {
        case KickKind::Perturb:
            return "perturb";
        case KickKind::LNS:
            return "lns";
        case KickKind::Adopt:
            return "adopt";
    }
    return "perturb";
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

    // By NAME, not by position. Two workers build their own clones of the same
    // registered generators, so their vectors agree position-for-position
    // today -- but a worker that threw before building its batch contributes an
    // EMPTY vector, and a linear merge by index would then silently attribute
    // the survivors' rows to the wrong generator. The name is the only identity
    // a generator has (`MoveGenerator::name()`), and the vector is a handful of
    // entries, so the quadratic scan is not worth an index.
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

// Key function: emits this class's vtable here rather than in every translation
// unit that includes tracer.h. See the note on the declaration.
Tracer::~Tracer() = default;

void Tracer::batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool /*improved*/) {}
void Tracer::new_best(double /*objective*/, double /*seconds*/) {}
void Tracer::kick(KickKind /*kind*/) {}
void Tracer::lns(bool /*accepted*/) {}
void Tracer::hook(double /*seconds*/) {}

}  // namespace cbls
