#include "cbls/move_generator.h"

#include "cbls/model.h"
#include "cbls/moves.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cbls {

MoveGenerator::~MoveGenerator() = default;

void MoveGenerator::on_commit(const Move& /*move*/) {}

// ---------------------------------------------------------------------------
// NeighbourList
// ---------------------------------------------------------------------------

void NeighbourList::validate() const {
    if (offsets_.empty()) {
        if (!ids_.empty()) {
            throw std::invalid_argument("NeighbourList: ids without offsets");
        }
        return;
    }
    if (offsets_.front() != 0) {
        throw std::invalid_argument("NeighbourList: offsets must start at 0");
    }
    if (static_cast<size_t>(offsets_.back()) != ids_.size()) {
        throw std::invalid_argument("NeighbourList: last offset must equal ids.size()");
    }
    for (size_t i = 1; i < offsets_.size(); ++i) {
        if (offsets_[i] < offsets_[i - 1]) {
            throw std::invalid_argument("NeighbourList: offsets must be non-decreasing");
        }
    }
    const int32_t n = universe();
    for (int32_t id : ids_) {
        if (id < 0 || id >= n) {
            throw std::invalid_argument("NeighbourList: neighbour id outside the universe");
        }
    }
}

NeighbourList::NeighbourList(std::vector<int32_t> offsets, std::vector<int32_t> ids)
    : offsets_(std::move(offsets)), ids_(std::move(ids)) {
    validate();
}

NeighbourList::NeighbourList(const std::vector<std::vector<int32_t>>& rows) {
    offsets_.reserve(rows.size() + 1);
    offsets_.push_back(0);
    size_t total = 0;
    for (const std::vector<int32_t>& row : rows) {
        total += row.size();
    }
    ids_.reserve(total);
    for (const std::vector<int32_t>& row : rows) {
        for (int32_t id : row) {
            ids_.push_back(id);
        }
        offsets_.push_back(static_cast<int32_t>(ids_.size()));
    }
    validate();
}

NeighbourList nearest_neighbours(int universe, int k, const std::function<double(int, int)>& cost) {
    if (universe < 0) {
        throw std::invalid_argument("nearest_neighbours: negative universe");
    }
    std::vector<std::vector<int32_t>> rows(static_cast<size_t>(universe));
    if (universe <= 1 || k <= 0) {
        return NeighbourList(rows);
    }
    const size_t want = std::min(static_cast<size_t>(k), static_cast<size_t>(universe) - 1);
    std::vector<std::pair<double, int32_t>> scored;
    scored.reserve(static_cast<size_t>(universe) - 1);
    for (int e = 0; e < universe; ++e) {
        scored.clear();
        for (int f = 0; f < universe; ++f) {
            if (f != e) {
                scored.emplace_back(cost(e, f), static_cast<int32_t>(f));
            }
        }
        // Ties broken by ascending id, so the list depends only on the cost and
        // the search is reproducible across machines.
        std::partial_sort(
            scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(want), scored.end(),
            [](const std::pair<double, int32_t>& a, const std::pair<double, int32_t>& b) {
                // NaN sorts last, and two NaNs must TIE (broken by id). Comparing
                // values first fails: NaN != NaN, so comp(x, y) and comp(y, x)
                // could both hold, not a strict weak ordering, and partial_sort
                // is then UB (libstdc++ runs off the range).
                const bool a_nan = std::isnan(a.first);
                const bool b_nan = std::isnan(b.first);
                if (a_nan != b_nan) {
                    return b_nan;
                }
                if (!a_nan && a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
            });
        std::vector<int32_t>& row = rows[static_cast<size_t>(e)];
        row.reserve(want);
        for (size_t i = 0; i < want; ++i) {
            row.push_back(scored[i].second);
        }
    }
    return NeighbourList(rows);
}

// ---------------------------------------------------------------------------
// StructuralSelection names
// ---------------------------------------------------------------------------

const char* structural_selection_name(StructuralSelection selection) {
    switch (selection) {
        case StructuralSelection::FirstImprovingSample:
            return "first_improving";
        case StructuralSelection::BestOfSample:
            return "best_of_sample";
        case StructuralSelection::ViolationGuided:
            return "violation_guided";
        case StructuralSelection::GranularBest:
            return "granular_best";
        case StructuralSelection::Auto:
            return "auto";
    }
    return "unknown";
}

bool try_parse_structural_selection(std::string_view text, StructuralSelection& out) {
    if (text == "first_improving") {
        out = StructuralSelection::FirstImprovingSample;
        return true;
    }
    if (text == "best_of_sample") {
        out = StructuralSelection::BestOfSample;
        return true;
    }
    if (text == "violation_guided") {
        out = StructuralSelection::ViolationGuided;
        return true;
    }
    if (text == "granular_best") {
        out = StructuralSelection::GranularBest;
        return true;
    }
    if (text == "auto") {
        out = StructuralSelection::Auto;
        return true;
    }
    return false;
}

StructuralSelection resolve_structural_selection(StructuralSelection selection,
                                                 bool has_neighbours) {
    if (selection != StructuralSelection::Auto) {
        return selection;
    }
    return has_neighbours ? StructuralSelection::GranularBest
                          : StructuralSelection::FirstImprovingSample;
}

// ---------------------------------------------------------------------------
// The built-in per-variable generator
// ---------------------------------------------------------------------------

namespace {

/// One List or Set variable's standard typed moves, wrapped as a generator.
///
/// One per variable: the structural batch checks its deadline between
/// generators (#105), and `ViolationGuided` skips a generator whose scope is
/// all in satisfied rows, both useful per variable.
class StandardStructuralGenerator final : public MoveGenerator {
public:
    StandardStructuralGenerator(int32_t var_id, VarType type,
                                std::shared_ptr<const NeighbourList> neighbours)
        : var_id_(var_id), type_(type), neighbours_(std::move(neighbours)) {}

    [[nodiscard]] std::string_view name() const override {
        return type_ == VarType::List ? "builtin_list" : "builtin_set";
    }

    [[nodiscard]] ConstSpan<int32_t> scope() const override { return {&var_id_, 1}; }

    void generate(MoveContext& ctx, std::vector<Move>& out) override {
        const Variable& var = ctx.model.var(var_id_);
        // Under GranularBest a partition member's intra-list moves come from
        // the partition generator's same-list neighbour pairs (#203).
        if (ctx.selection == StructuralSelection::GranularBest && var.partitioned &&
            neighbours_ != nullptr && !neighbours_->empty()) {
            return;
        }
        generate_standard_moves(var, ctx.rng, out, neighbours_.get());
    }

    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        // The immutable neighbour list is shared, not copied per worker (#157).
        return std::make_unique<StandardStructuralGenerator>(var_id_, type_, neighbours_);
    }

private:
    int32_t var_id_;
    VarType type_;
    std::shared_ptr<const NeighbourList> neighbours_;
};

/// One `ListPartition`'s inter-list moves (#164): relocate, swap, 2-opt*, and
/// the insert/remove pair under `Cover::AtMostOnce`.
///
/// Must hold NO derived state (`MoveGenerator::on_commit`'s contract): peer
/// commits, FJ batches, the kick, LNS and pool restarts move the assignment
/// unannounced, and Python can rewrite `Variable.elements` (#156), so a cache
/// would be a stale index, a crash risk. Recomputing costs
/// O(sum |lists| + universe) on the branch that needs it.
///
/// Scope is every member list, so `ViolationGuided` can skip a partition all
/// of whose rows are satisfied. A candidate is scored over the one or two
/// lists it moves, not the scope (#273).
class ListPartitionGenerator final : public MoveGenerator {
public:
    ListPartitionGenerator(int partition, std::vector<int32_t> list_ids,
                           std::shared_ptr<const NeighbourList> neighbours)
        : partition_(partition),
          list_ids_(std::move(list_ids)),
          neighbours_(std::move(neighbours)) {}

    [[nodiscard]] std::string_view name() const override { return "builtin_list_partition"; }

    [[nodiscard]] ConstSpan<int32_t> scope() const override {
        return {list_ids_.data(), list_ids_.size()};
    }

    void generate(MoveContext& ctx, std::vector<Move>& out) override {
        if (ctx.selection == StructuralSelection::GranularBest && neighbours_ != nullptr &&
            !neighbours_->empty()) {
            generate_granular_partition_moves(ctx.model, partition_, ctx.rng, *neighbours_, out,
                                              slots_);
            return;
        }
        // anchor = -1: both lists are drawn. The kick is the only caller that
        // names one.
        generate_partition_moves(ctx.model, partition_, /*anchor=*/-1, ctx.rng, out,
                                 neighbours_.get());
    }

    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        // Const reads only, so every worker may clone the one prototype
        // concurrently without a lock. The neighbour list stays shared.
        return std::make_unique<ListPartitionGenerator>(partition_, list_ids_, neighbours_);
    }

private:
    int partition_;
    std::vector<int32_t> list_ids_;
    std::shared_ptr<const NeighbourList> neighbours_;
    /// GranularBest's element -> slot index, rebuilt on every call: a reused
    /// buffer, not state. Per clone, so per worker.
    std::vector<int64_t> slots_;
};

}  // namespace

std::vector<std::shared_ptr<const MoveGenerator>> default_move_generators(
    const Model& model, const std::shared_ptr<const NeighbourList>& neighbours) {
    std::vector<std::shared_ptr<const MoveGenerator>> generators;
    for (const Variable& var : model.variables()) {
        if (!is_structured(var.type)) {
            continue;
        }
        generators.push_back(
            std::make_shared<const StandardStructuralGenerator>(var.id, var.type, neighbours));
    }
    // Partition generators come after every per-variable one, so a model
    // without a partition keeps its pre-#164 list and trajectory. A member
    // list keeps its per-variable generator for the intra-list moves, except
    // under GranularBest with neighbours, where the partition supplies them
    // (#203).
    for (size_t i = 0; i < model.list_partitions().size(); ++i) {
        generators.push_back(std::make_shared<const ListPartitionGenerator>(
            static_cast<int>(i), model.list_partitions()[i].list_ids, neighbours));
    }
    return generators;
}

}  // namespace cbls
