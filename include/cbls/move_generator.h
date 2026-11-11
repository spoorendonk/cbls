#pragma once

#include "dag.h"
#include "moves.h"
#include "rng.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace cbls {

class Model;
class ViolationManager;

/// Granular neighbourhood (Toth & Vigo, INFORMS J. Computing 15(4), 2003): for
/// each element `e` of `{0..n-1}`, the others a move involving `e` may reach,
/// nearest first, supplied by the model author since the engine knows no cost.
///
/// CSR: `of(e)` is `ids[offsets[e] .. offsets[e+1])`. Invariants (offsets
/// non-decreasing from 0 to `ids.size()`, ids in the universe) are checked
/// ONCE at construction, since `of()` is on the hot path. `of()` is still
/// total: an out-of-range `e` gives an empty span, not a heap read (#156; this
/// type is reachable from Python).
class NeighbourList {
public:
    NeighbourList() = default;

    /// From a per-element row list: `rows[e]` are `e`'s neighbours, nearest
    /// first. Throws `std::invalid_argument` on an id outside `[0, rows.size())`.
    explicit NeighbourList(const std::vector<std::vector<int32_t>>& rows);

    /// From CSR arrays. Throws `std::invalid_argument` unless `offsets` is
    /// non-decreasing, starts at 0, ends at `ids.size()`, and every id lies in
    /// `[0, offsets.size() - 1)`.
    NeighbourList(std::vector<int32_t> offsets, std::vector<int32_t> ids);

    /// Elements the list is defined over; 0 for a default-constructed list.
    [[nodiscard]] int32_t universe() const noexcept {
        return offsets_.empty() ? 0 : static_cast<int32_t>(offsets_.size()) - 1;
    }
    [[nodiscard]] bool empty() const noexcept { return ids_.empty(); }
    /// `e`'s neighbours, nearest first. Empty for an out-of-range `e`.
    [[nodiscard]] ConstSpan<int32_t> of(int32_t e) const noexcept {
        // Widened before +1: `e + 1` overflows at INT32_MAX, reachable from
        // Python, in a function that must be total.
        if (e < 0 || static_cast<size_t>(e) + 1 >= offsets_.size()) {
            return {};
        }
        const int32_t begin = offsets_[static_cast<size_t>(e)];
        const int32_t end = offsets_[static_cast<size_t>(e) + 1];
        return {ids_.data() + begin, static_cast<size_t>(end - begin)};
    }
    [[nodiscard]] const std::vector<int32_t>& offsets() const noexcept { return offsets_; }
    [[nodiscard]] const std::vector<int32_t>& ids() const noexcept { return ids_; }

private:
    void validate() const;

    std::vector<int32_t> offsets_;
    std::vector<int32_t> ids_;
};

/// Each element's `k` nearest others under `cost`, nearest first, ties by
/// ascending id (a function of the cost, not the sort's stability).
/// O(universe^2) cost calls, O(universe^2 log k) work: a setup convenience for
/// a few thousand elements, not a spatial index; larger or geometric callers
/// build rows themselves. `k <= 0` or `universe <= 1` yields no neighbours,
/// which every built-in generator reads as "use the uniform draw".
NeighbourList nearest_neighbours(int universe, int k, const std::function<double(int, int)>& cost);

/// How the structural batch turns a generator's candidates into a commit.
/// `FirstImprovingSample` is bit-for-bit the pre-#165 batch; the default
/// `Auto` picks it or `GranularBest`. The others change the trajectory.
enum class StructuralSelection : std::uint8_t {
    /// Take every improving candidate from one sample, in order.
    FirstImprovingSample,
    /// Call the generator until it offered at least
    /// `SearchConfig::structural_sample_size` candidates (at least one call),
    /// score all, commit the best improving one. A floor on effort, not a cap
    /// on count: the built-in List generator yields 5 even at size 1.
    BestOfSample,
    /// `BestOfSample`, restricted to generators whose scope can change a
    /// VIOLATED row. Exact, not heuristic: if no row in the scope's G_v union
    /// is violated, every move there leaves the weighted violation equal or
    /// higher. Variable-level, not #165's element-level: incidence is per
    /// variable, and a `Set` read through one `Lambda` puts every element in
    /// every row; element-level needs a generator-supplied scorer, beyond what
    /// a `NeighbourList` gives.
    ViolationGuided,
    /// `ViolationGuided`, with each `ListPartition` searched by the GRANULAR
    /// COMPLETE neighbourhood of one random element per `generate` call (#203;
    /// Toth & Vigo 2003): every relocate / swap / 2-opt / 2-opt* with each of
    /// its k nearest neighbours, plus a move into each empty list
    /// (`generate_granular_partition_moves` states how it differs from the
    /// paper). Commits the best improving candidate; short calls are topped up
    /// with another element's neighbourhood. Needs
    /// `SearchConfig::structural_neighbours`, else it is `ViolationGuided`.
    /// With a list, partition members' intra-list generators propose nothing
    /// (the neighbour pairs cover them); Lists/Sets outside a partition are
    /// sampled as under `ViolationGuided`.
    GranularBest,
    /// The default (#203): `GranularBest` with a
    /// `SearchConfig::structural_neighbours` list (won #203's CVRP smoke, 15/15
    /// vs `ViolationGuided`), else `FirstImprovingSample`, so a run without one
    /// keeps the pre-#165 trajectory. Resolved once when the batch is built.
    Auto,
};

/// The policy a search actually runs: `Auto` resolved against whether a
/// neighbour list is present, every other value unchanged.
StructuralSelection resolve_structural_selection(StructuralSelection selection,
                                                 bool has_neighbours);

/// Stable snake_case token for a `StructuralSelection`, and its inverse; the
/// one spelling used by the benchmark flags and the Python bindings.
const char* structural_selection_name(StructuralSelection selection);
bool try_parse_structural_selection(std::string_view text, StructuralSelection& out);

/// What a generator may look at while proposing. READ-ONLY except the RNG:
/// the batch applies, scores and rolls back each candidate itself, so a
/// generator moving a variable corrupts the baseline later candidates use.
struct MoveContext {
    const Model& model;
    /// GLS weights `W` and the per-constraint violations of the CURRENT
    /// assignment.
    const ViolationManager& vm;
    RNG& rng;
    /// The policy the batch is running, so a generator can spend more effort
    /// when its candidates are going to be ranked rather than taken in order.
    StructuralSelection selection = StructuralSelection::FirstImprovingSample;
    /// Per-constraint clamped violations of the last ACCEPTED assignment -- the
    /// baseline the batch scores candidates against. One entry per constraint,
    /// indexed as `ViolationManager::weights` is. Never null.
    const std::vector<double>* violations = nullptr;
};

/// A source of candidate structural moves. The built-in List/Set moves are
/// generators too (`default_move_generators`), so a domain move is a peer.
/// Register instances on `SearchConfig::move_generators`.
///
/// CONTRACTS the batch relies on:
///  - a structured change is a POSITIONAL EDIT on `elements` (`ElementEdit`)
///    or a whole `replacement` (`replace_change`, paying the copy), applied
///    to the assignment the sample was drawn against, which the batch
///    restores before each candidate.
///  - `generate` returns in bounded time AND appends a bounded number of
///    candidates: the deadline is checked only BETWEEN generators (#105), and
///    every candidate on offer is applied, evaluated, scored and rolled back,
///    so 100 000 instant candidates overrun `solve()`'s budget like a hang.
///    Built-ins append at most 5, except the partition generator under
///    `GranularBest` (bound in `generate_granular_partition_moves`). Only the
///    sampling policies cap the count (`SearchConfig::structural_sample_size`);
///    `FirstImprovingSample` takes all.
///  - `generate` APPENDS; it must not clear or reorder `out`.
///  - `generate` must not change the model's assignment (see `MoveContext`).
///  - `clone()` returns an independent object and is safe to call
///    concurrently on one prototype: each portfolio worker clones the single
///    registered instance on its own thread, so a `clone()` writing shared
///    state needs a lock (as `tests/test_parallel.cpp`'s `CountingGenerator`).
///    Clone-held caches, cursors and counters are per-worker; sharing them
///    races (#157). A clone dies with its worker's `solve()`; each restart
///    (`src/pool.cpp`) clones afresh, so no state survives a restart.
///  - `scope()` names every variable the moves can change and is constant for
///    the generator's lifetime. `ViolationGuided`/`GranularBest` skip
///    generators by it, so an incomplete scope can skip a helpful move.
///    Scoring ignores it: it uses the rows each move's variables reach (#273).
class MoveGenerator {
public:
    MoveGenerator() = default;
    MoveGenerator(const MoveGenerator&) = default;
    MoveGenerator& operator=(const MoveGenerator&) = default;
    MoveGenerator(MoveGenerator&&) = default;
    MoveGenerator& operator=(MoveGenerator&&) = default;
    virtual ~MoveGenerator();

    /// A stable name for logging and for per-generator counters. Must outlive
    /// the generator (a string literal, or a member it owns).
    [[nodiscard]] virtual std::string_view name() const = 0;

    /// The variables this generator's moves can change: one list, a partition of
    /// lists, a set. Ascending order is not required.
    [[nodiscard]] virtual ConstSpan<int32_t> scope() const = 0;

    /// Append candidate moves. Called once per batch under
    /// `FirstImprovingSample`, and repeatedly under the sampling policies until
    /// the sample is full or a call adds nothing.
    ///
    /// Every candidate of one call is built against, and applied to, the SAME
    /// assignment: the batch restores every variable any candidate of the
    /// sample names before applying each one (#164), so a positional
    /// `ElementEdit` lands where it was built and the state is always the
    /// sample's own plus one candidate. Under `FirstImprovingSample` the batch
    /// may commit several in turn, so committing k then k+1 leaves k's change
    /// gone (harmless for the built-ins: same as rejecting k). The built-in
    /// `ListPartition` still emits at most ONE candidate per call (except under
    /// take-best `GranularBest`), since a cover kept by construction is costly
    /// to get wrong. For a larger sample, rely on the baseline or run under
    /// `BestOfSample` / `ViolationGuided`, which commit at most one per call.
    virtual void generate(MoveContext& ctx, std::vector<Move>& out) = 0;

    /// One of THIS generator's moves was committed; the default does nothing.
    /// A private notification, NOT an assignment-change broadcast: peer
    /// commits, FJ/Novelty batches, the kick, LNS and pool restarts move the
    /// assignment silently. So anything DERIVED from the assignment (an
    /// element-to-list index, a position map) must be recomputed in `generate`
    /// from `ctx.model`; only the generator's own state (cursor, counter, tabu
    /// tenure) may carry across calls. Under `FirstImprovingSample` a notified
    /// commit can still be undone by a later one from the same sample (see
    /// `generate`); a generator whose bookkeeping must track the assignment
    /// should emit one candidate per call or run under a one-commit policy.
    virtual void on_commit(const Move& move);

    /// A per-worker copy. See the cloning contract above.
    [[nodiscard]] virtual std::unique_ptr<MoveGenerator> clone() const = 0;
};

/// The built-in structural generators for `model`: one per List/Set variable,
/// in variable-id order, each proposing the typed moves of
/// `generate_standard_moves` (List: swap / 2-opt / relocate / or-opt(2) /
/// or-opt(3) from one random position pair; Set: one add, remove and swap).
/// Empty with no structured variable. Non-null `neighbours` draws a move's
/// target from the moved element's list, changing the trajectory; null keeps
/// the pre-#165 path. Public so a custom set can extend the built-ins.
std::vector<std::shared_ptr<const MoveGenerator>> default_move_generators(
    const Model& model, const std::shared_ptr<const NeighbourList>& neighbours = nullptr);

}  // namespace cbls
