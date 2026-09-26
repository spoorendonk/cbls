#include "cbls/tracer.h"

#include <cstdint>

namespace cbls {

const char* kick_kind_name(KickKind kind) {
    switch (kind) {
        case KickKind::Perturb:
            return "perturb";
        case KickKind::LNS:
            return "lns";
        case KickKind::Adopt:
            return "adopt";
    }
    // Unreachable for any value of the enum; keeps the function total so a caller
    // can print the kind unconditionally, exactly as batch_kind_name does.
    return "perturb";
}

// Key function: emits this class's vtable here rather than in every translation
// unit that includes tracer.h, the way SolveCallback's destructor does. Every
// event method below is a no-op, so a subclass overrides only what it wants.
Tracer::~Tracer() = default;

void Tracer::batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool /*improved*/) {}
void Tracer::new_best(double /*objective*/, double /*seconds*/) {}
void Tracer::kick(KickKind /*kind*/) {}
void Tracer::lns(bool /*accepted*/) {}
void Tracer::hook(double /*seconds*/) {}

}  // namespace cbls
