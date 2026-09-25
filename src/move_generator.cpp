#include "cbls/move_generator.h"

#include "cbls/model.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
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
        // Ties broken by ascending id, so the list is a function of the cost and
        // not of the sort's stability -- two builds of the same instance on two
        // machines must give the same neighbourhood or the search is not
        // reproducible across them.
        std::partial_sort(
            scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(want), scored.end(),
            [](const std::pair<double, int32_t>& a, const std::pair<double, int32_t>& b) {
                if (a.first != b.first) {
                    // A NaN cost sorts last rather than making the
                    // comparator non-strict (which is UB in sort).
                    return a.first < b.first || std::isnan(b.first);
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
    return false;
}

// ---------------------------------------------------------------------------
// The built-in per-variable generator
// ---------------------------------------------------------------------------

namespace {

/// One List or Set variable's standard typed moves, wrapped as a generator.
///
/// One generator per variable rather than one covering them all, for two
/// reasons. The structural batch checks its wall-clock deadline BETWEEN
/// generators (#105), so per-variable generators keep the check per variable,
/// which is exactly where the pre-#165 sweep checked it. And `ViolationGuided`
/// skips a generator whose whole scope is in satisfied rows -- a decision that
/// is only useful at variable granularity.
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
        generate_standard_moves(ctx.model.var(var_id_), ctx.rng, out, neighbours_.get());
    }

    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        // The neighbour list is immutable and shared on purpose: it is the one
        // piece of a built-in generator that is read-only, and copying a
        // k-nearest list per worker would be the cost this sharing exists to
        // avoid (#157's argument, one level down).
        return std::make_unique<StandardStructuralGenerator>(var_id_, type_, neighbours_);
    }

private:
    int32_t var_id_;
    VarType type_;
    std::shared_ptr<const NeighbourList> neighbours_;
};

}  // namespace

std::vector<std::shared_ptr<const MoveGenerator>> default_move_generators(
    const Model& model, std::shared_ptr<const NeighbourList> neighbours) {
    std::vector<std::shared_ptr<const MoveGenerator>> generators;
    for (const Variable& var : model.variables()) {
        if (!is_structured(var.type)) {
            continue;
        }
        generators.push_back(
            std::make_shared<const StandardStructuralGenerator>(var.id, var.type, neighbours));
    }
    return generators;
}

}  // namespace cbls
