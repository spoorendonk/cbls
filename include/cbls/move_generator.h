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

/// Granular neighbourhood: for each element `e` of a universe `{0..n-1}`, the
/// other elements a move involving `e` is allowed to reach, nearest first.
///
/// This is Toth & Vigo's granular neighbourhood (INFORMS J. Computing 15(4),
/// 2003), the standard way to make a sequence neighbourhood affordable on a
/// large universe: relocating a customer next to one on the far side of the map
/// is never improving, so the uniform `(i, j)` draw spends almost all of its
/// candidates on moves that cannot win. The engine has no idea what a "cost"
/// between two elements means, so the list is supplied by the model author --
/// either built here from a cost callback, or handed over as arrays.
///
/// CSR, like the model's own adjacency: `of(e)` is
/// `ids[offsets[e] .. offsets[e+1])`. The invariants (offsets non-decreasing,
/// starting at 0, ending at `ids.size()`, every id inside the universe) are
/// checked ONCE at construction and never again, because `of()` is read on the
/// move-generation hot path. `of()` is nonetheless total: an out-of-range `e`
/// returns an empty span rather than indexing past the array, so a generator
/// holding a list built for a different variable degrades to "no neighbours"
/// instead of reading the heap (the #156 hazard class -- this type is reachable
/// from Python).
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
        if (e < 0 || e + 1 >= static_cast<int32_t>(offsets_.size())) {
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

/// Each element's `k` nearest others under `cost`, nearest first, ties broken by
/// ascending id so the list is a function of the cost alone and not of the
/// sort's stability.
///
/// O(universe^2) cost evaluations and O(universe^2 log k) work: this is a
/// setup-time convenience for a universe of a few thousand, not a spatial index.
/// A caller with a bigger instance, or with a cost it can index geometrically,
/// builds the rows itself and uses the `NeighbourList` constructors.
///
/// `k <= 0` or `universe <= 1` yields a list with no neighbours, which every
/// built-in generator reads as "fall back to the uniform draw".
NeighbourList nearest_neighbours(int universe, int k, const std::function<double(int, int)>& cost);

/// How the structural batch turns a generator's candidates into a commit.
///
/// `FirstImprovingSample` is the DEFAULT and is bit-for-bit what the batch did
/// before generators existed (#165): one `generate` call per generator, each
/// candidate applied in turn and kept if it strictly lowers weighted violation.
/// The other two are opt-in and change the trajectory.
enum class StructuralSelection : std::uint8_t {
    /// Today's rule: take every improving candidate from one sample, in order.
    FirstImprovingSample,
    /// Draw up to `SearchConfig::structural_sample_size` candidates from the
    /// generator, score them all, commit the best improving one.
    BestOfSample,
    /// `BestOfSample`, restricted to generators whose scope can still change a
    /// VIOLATED row.
    ///
    /// The restriction is exact rather than a heuristic: if no constraint in the
    /// union of the scope's G_v is violated at the accepted assignment, every
    /// one of those rows contributes 0 to the weighted violation, so any move
    /// over that scope can only leave the total alone or raise it. Skipping the
    /// generator therefore cannot skip an improving move.
    ///
    /// NOTE ON SCOPE. #165 describes this policy as picking the *element* to
    /// move from those appearing in violated constraints. That is not
    /// recoverable from the DAG: incidence is tracked per VARIABLE (G_v), and a
    /// `Set` read through a single `Lambda` node puts every element of the
    /// universe in every row that node feeds. So what ships here is the
    /// variable-level form -- which variable to spend candidates on -- plus
    /// whatever element-level granularity a `NeighbourList` supplies. The
    /// element-level form needs either a generator-supplied element scorer or
    /// the custom-invariant work of #166.
    ViolationGuided,
};

/// Stable snake_case token for a `StructuralSelection`, and its inverse.
/// Used by the benchmark runners' flags and by the Python bindings, so there is
/// one spelling of each name.
const char* structural_selection_name(StructuralSelection selection);
bool try_parse_structural_selection(std::string_view text, StructuralSelection& out);

/// What a generator is allowed to look at while proposing candidates.
///
/// Everything here is READ-ONLY apart from the RNG. A generator must not touch
/// the model's assignment: the batch applies, scores and rolls back each
/// candidate itself, and a generator that moved a variable behind its back
/// would corrupt the baseline every candidate after it is scored against.
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

/// A source of candidate structural moves.
///
/// The built-in List and Set moves are registered as generators like any other
/// (see `default_move_generators`), so a domain move -- an inter-route
/// exchange, a block move, an ejection chain -- is a peer rather than a special
/// case. Register instances on `SearchConfig::move_generators`.
///
/// CONTRACTS, all of which the batch relies on:
///
///  - `generate` must RETURN IN BOUNDED TIME. The structural batch checks its
///    wall-clock deadline BETWEEN generators, never inside one (#105), so an
///    unbounded `generate` is an unbounded overrun of `solve()`'s budget.
///  - `generate` APPENDS; it must not clear or reorder what is already in `out`.
///  - `generate` must not change the model's assignment (see `MoveContext`).
///  - `clone()` must return an independent object. Every portfolio worker gets
///    its own clone, so any cache, cursor or counter a generator holds is
///    per-worker state; a clone that shared it would be a data race across
///    worker threads (#157).
///  - `scope()` must name every variable the generator's moves can change, and
///    must stay constant for the generator's lifetime. The batch uses it to
///    restrict candidate scoring to the union of those variables' G_v, and
///    `ViolationGuided` uses it to skip generators that cannot help. A move
///    touching a variable outside the scope is scored WRONGLY, not just
///    inefficiently.
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
    /// lists, a set. Ascending order is not required; the batch sorts the union.
    [[nodiscard]] virtual ConstSpan<int32_t> scope() const = 0;

    /// Append candidate moves. Called once per batch under
    /// `FirstImprovingSample`, and repeatedly under the sampling policies until
    /// the sample is full or a call adds nothing.
    virtual void generate(MoveContext& ctx, std::vector<Move>& out) = 0;

    /// One of this generator's moves was committed. The default does nothing.
    virtual void on_commit(const Move& move);

    /// A per-worker copy. See the cloning contract above.
    [[nodiscard]] virtual std::unique_ptr<MoveGenerator> clone() const = 0;
};

/// The built-in structural generators for `model`: one per List/Set variable,
/// in variable-id order, each proposing exactly the typed moves
/// `generate_standard_moves` has always proposed for that type (List:
/// swap / 2-opt / relocate / or-opt(2) / or-opt(3) built from one random
/// position pair; Set: one add, one remove, one swap).
///
/// Empty for a model with no structured variable.
///
/// With `neighbours` non-null, the target of a move is drawn from the moved
/// element's neighbour list rather than uniformly. That changes the trajectory,
/// so the default (null) is what keeps an unconfigured run on the pre-#165 path.
///
/// Exposed rather than private so a caller building its own generator set --
/// #164's partition-level inter-list moves are the first -- can start from the
/// built-ins and add to them instead of re-deriving them.
std::vector<std::shared_ptr<const MoveGenerator>> default_move_generators(
    const Model& model, const std::shared_ptr<const NeighbourList>& neighbours = nullptr);

}  // namespace cbls
