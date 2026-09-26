// Positional edits for a `CustomInvariant` (#172): `InvariantInputs::edits`.
//
// The fixture is an OPEN ROUTE COST -- the sum of distances between consecutive
// elements of each structured input -- which is #166's first motivating use case
// and the one #172 exists to make expressible. It keeps a MIRROR of each input
// as it last committed to, and prices every edit from the handful of edges the
// edit touches. So:
//
//  - a wrong or missing edit desynchronises the mirror, which the fixture
//    detects by comparing it with `elements(i)` (`mirror_mismatches`), and which
//    also shows up as a wrong value against `full_evaluate`;
//  - the operation counter is the number of DISTANCE EVALUATIONS, which is the
//    cost a real route or time-window invariant is about. The mirror itself is
//    kept up to date by `apply_positional_edit`, which is an insert/erase/rotate
//    on a vector and so costs what the engine's own apply of the same edit costs
//    -- it is not part of the claim, and the counter does not count it.

#include "cbls/custom_invariant.h"
#include "cbls/dag_ops.h"
#include "cbls/element_edit.h"
#include "cbls/model.h"
#include "cbls/move_generator.h"
#include "cbls/moves.h"
#include "cbls/rng.h"
#include "cbls/search.h"
#include "cbls/structural_batch.h"
#include "cbls/violation.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace cbls;

namespace {

const std::chrono::steady_clock::time_point kNoDeadline{};

// Single-threaded throughout this file, so plain counters.
struct RouteStats {
    int64_t dist_evals = 0;      // the operation counter
    int64_t evaluates = 0;       // from-scratch evaluate() calls
    int64_t incremental = 0;     // inputs priced from their edits
    int64_t rereads = 0;         // inputs that fell back to elements(i)
    int64_t edits_consumed = 0;  // PositionalEdits priced
    int64_t mirror_mismatches = 0;
    int64_t deltas = 0;
    int64_t reread_lengths = 0;  // sum of |elements| over the rereads

    void clear() { *this = RouteStats{}; }
};

// Symmetric, integer-valued, so incremental and from-scratch sums are EXACT and
// the comparisons below can be bitwise.
double coord(int32_t e) {
    return static_cast<double>((static_cast<int64_t>(e) * 37) % 101);
}

class RouteCost : public CustomInvariant {
public:
    // `use_edits = false` is the re-read baseline: identical in every other way,
    // it ignores `edits(i)` and re-reads every structured input that moved.
    RouteCost(std::shared_ptr<RouteStats> stats, bool use_edits, bool verify_mirror)
        : stats_(std::move(stats)), use_edits_(use_edits), verify_mirror_(verify_mirror) {}

    double evaluate(const InvariantInputs& in) override {
        ++stats_->evaluates;
        const auto n = static_cast<size_t>(in.size());
        mirror_.assign(n, {});
        cost_.assign(n, 0.0);
        for (int32_t i = 0; i < in.size(); ++i) {
            if (!in.is_structured_input(i)) {
                continue;
            }
            const ConstSpan<int32_t> el = in.elements(i);
            auto& mirror = mirror_[static_cast<size_t>(i)];
            mirror.assign(el.begin(), el.end());
            cost_[static_cast<size_t>(i)] = full_cost(mirror);
        }
        staged_cost_ = cost_;
        log_.clear();
        saved_.clear();
        return total(cost_);
    }

    double delta(const InvariantInputs& in, ConstSpan<int32_t> changed) override {
        ++stats_->deltas;
        staged_cost_ = cost_;
        log_.clear();
        saved_.clear();
        for (const int32_t i : changed) {
            if (!in.is_structured_input(i)) {
                continue;
            }
            const auto slot = static_cast<size_t>(i);
            const InputEdits edits = in.edits(i);
            if (use_edits_ && edits.available()) {
                ++stats_->incremental;
                for (const PositionalEdit& edit : edits.list()) {
                    staged_cost_[slot] += apply_and_price(mirror_[slot], edit);
                    log_.emplace_back(i, edit);
                    ++stats_->edits_consumed;
                }
            } else {
                ++stats_->rereads;
                saved_.emplace_back(i, mirror_[slot]);
                const ConstSpan<int32_t> el = in.elements(i);
                mirror_[slot].assign(el.begin(), el.end());
                stats_->reread_lengths += static_cast<int64_t>(el.size());
                staged_cost_[slot] = full_cost(mirror_[slot]);
            }
            if (verify_mirror_) {
                const ConstSpan<int32_t> el = in.elements(i);
                if (!std::equal(el.begin(), el.end(), mirror_[slot].begin(), mirror_[slot].end())) {
                    ++stats_->mirror_mismatches;
                }
            }
        }
        return total(staged_cost_);
    }

    void commit() override {
        cost_ = staged_cost_;
        log_.clear();
        saved_.clear();
    }

    void rollback() override {
        for (auto it = log_.rbegin(); it != log_.rend(); ++it) {
            apply_positional_edit(inverse_edit(it->second),
                                  mirror_[static_cast<size_t>(it->first)]);
        }
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
            mirror_[static_cast<size_t>(it->first)] = std::move(it->second);
        }
        staged_cost_ = cost_;
        log_.clear();
        saved_.clear();
    }

    [[nodiscard]] bool wants_positional_edits() const override { return use_edits_; }

    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        return std::make_unique<RouteCost>(*this);
    }

private:
    [[nodiscard]] double dist(int32_t a, int32_t b) const {
        ++stats_->dist_evals;
        return std::abs(coord(a) - coord(b));
    }

    [[nodiscard]] double full_cost(const std::vector<int32_t>& m) const {
        double c = 0.0;
        for (size_t j = 0; j + 1 < m.size(); ++j) {
            c += dist(m[j], m[j + 1]);
        }
        return c;
    }

    static double total(const std::vector<double>& costs) {
        double t = 0.0;
        for (const double c : costs) {
            t += c;
        }
        return t;
    }

    // Edge j joins positions j and j+1. Sums the listed edges that exist, each
    // once.
    [[nodiscard]] double edges(const std::vector<int32_t>& m, std::vector<int64_t> which) const {
        std::sort(which.begin(), which.end());
        which.erase(std::unique(which.begin(), which.end()), which.end());
        double c = 0.0;
        for (const int64_t j : which) {
            if (j >= 0 && static_cast<size_t>(j) + 1 < m.size()) {
                c += dist(m[static_cast<size_t>(j)], m[static_cast<size_t>(j) + 1]);
            }
        }
        return c;
    }

    // Price one edit from the edges it touches, then apply it to the mirror.
    // At most 8 distance evaluations, whatever the length of the route.
    double apply_and_price(std::vector<int32_t>& m, const PositionalEdit& e) const {
        const int64_t from = e.from;
        const int64_t to = e.to;
        double before = 0.0;
        double after = 0.0;
        switch (e.kind) {
            case EditKind::Swap:
                before = edges(m, {from - 1, from, to - 1, to});
                apply_positional_edit(e, m);
                after = edges(m, {from - 1, from, to - 1, to});
                break;
            case EditKind::Assign:
                before = edges(m, {from - 1, from});
                apply_positional_edit(e, m);
                after = edges(m, {from - 1, from});
                break;
            case EditKind::Reverse:
                // Symmetric distances: only the two boundary edges change.
                before = edges(m, {from - 1, to});
                apply_positional_edit(e, m);
                after = edges(m, {from - 1, to});
                break;
            case EditKind::Insert:
                before = edges(m, {from - 1});
                apply_positional_edit(e, m);
                after = edges(m, {from - 1, from});
                break;
            case EditKind::Erase:
                before = edges(m, {from - 1, from});
                apply_positional_edit(e, m);
                after = edges(m, {from - 1});
                break;
            case EditKind::MoveSegment:
                return price_segment(m, e);
            case EditKind::None:
            case EditKind::Replace:
                throw std::logic_error("RouteCost: a PositionalEdit is never None or Replace");
        }
        return after - before;
    }

    // Erase `length` at `from`, reinsert at `to` of the shortened vector: three
    // edges out, three in, with positions in the shortened vector mapped back.
    double price_segment(std::vector<int32_t>& m, const PositionalEdit& e) const {
        const auto n = static_cast<int64_t>(m.size());
        const int64_t from = e.from;
        const int64_t len = e.length;
        const int64_t to = e.to;
        const int32_t s0 = m[static_cast<size_t>(from)];
        const int32_t s1 = m[static_cast<size_t>(from + len - 1)];
        auto at = [&m](int64_t k) { return m[static_cast<size_t>(k)]; };
        auto shortened = [&](int64_t k) { return k < from ? at(k) : at(k + len); };
        const int64_t nb = n - len;
        double delta = 0.0;
        if (from > 0) {
            delta -= dist(at(from - 1), s0);
        }
        if (from + len < n) {
            delta -= dist(s1, at(from + len));
        }
        if (from > 0 && from + len < n) {
            delta += dist(at(from - 1), at(from + len));
        }
        if (to > 0 && to < nb) {
            delta -= dist(shortened(to - 1), shortened(to));
        }
        if (to > 0) {
            delta += dist(shortened(to - 1), s0);
        }
        if (to < nb) {
            delta += dist(s1, shortened(to));
        }
        apply_positional_edit(e, m);
        return delta;
    }

    std::shared_ptr<RouteStats> stats_;
    bool use_edits_;
    bool verify_mirror_;
    // Committed: per input, the elements it last committed to and their cost.
    std::vector<std::vector<int32_t>> mirror_;
    std::vector<double> cost_;
    // Staged by the last delta(): the edits applied to the mirror (undone on a
    // rollback), and whole mirrors replaced by a re-read.
    std::vector<double> staged_cost_;
    std::vector<std::pair<int32_t, PositionalEdit>> log_;
    std::vector<std::pair<int32_t, std::vector<int32_t>>> saved_;
};

double route_cost_of(const std::vector<int32_t>& m) {
    double c = 0.0;
    for (size_t j = 0; j + 1 < m.size(); ++j) {
        c += std::abs(coord(m[j]) - coord(m[j + 1]));
    }
    return c;
}

// A random admissible List assignment: `len` distinct elements of the universe.
std::vector<int32_t> random_route(RNG& rng, int universe, int len) {
    std::vector<int32_t> p = rng.permutation(universe);
    p.resize(static_cast<size_t>(len));
    return p;
}

struct RouteModel {
    Model model;
    std::shared_ptr<RouteStats> stats = std::make_shared<RouteStats>();
    int32_t k = 0;
    int32_t list = 0;
    int32_t set = 0;
    int32_t route = -1;
    int32_t row = -1;
};

// route = RouteCost(k, L, S) -- a scalar input first, so the input INDICES of
// the two structured inputs differ from their positions in any variable list,
// and two structured inputs, so the edits must be keyed per input.
std::unique_ptr<RouteModel> build_route_model(bool use_edits, int universe, int min_len,
                                              int max_len) {
    auto rm = std::make_unique<RouteModel>();
    Model& m = rm->model;
    rm->k = m.int_var(0, 9, "k");
    rm->list = m.list_var(universe, min_len, max_len, ListInit::Random, "L");
    rm->set = m.set_var(12, 0, 12, "S");
    rm->route = m.custom({rm->k, rm->list, rm->set},
                         std::make_unique<RouteCost>(rm->stats, use_edits, /*verify_mirror=*/true),
                         "route");
    // Always violated, so the batch has something to improve and the weighted
    // violation is the route cost itself.
    rm->row = m.leq(m.sum({rm->route, rm->k}), m.constant(0.0));
    m.add_constraint(rm->row);
    m.minimize(rm->route);
    m.close();
    return rm;
}

double expected_route(const RouteModel& rm) {
    return route_cost_of(rm.model.var(handle_to_var_id(rm.list)).elements) +
           route_cost_of(rm.model.var(handle_to_var_id(rm.set)).elements);
}

void set_route(RouteModel& rm, const std::vector<int32_t>& list, const std::vector<int32_t>& set) {
    rm.model.var_mut(handle_to_var_id(rm.list)).elements = list;
    rm.model.var_mut(handle_to_var_id(rm.set)).elements = set;
    full_evaluate(rm.model);
}

}  // namespace

TEST_CASE("an edit-consuming List invariant is O(edits) where a re-read is O(n)",
          "[custom][edits]") {
    // Criterion 1 of #172, through the path that supplies the edits: the
    // structural batch. Two models, identical but for whether the invariant
    // consumes `edits(i)`, run the same sweeps from the same assignment and the
    // same seed. Their values agree bit for bit at every step, so they accept the
    // same moves and see the same move sequence -- and the only thing that
    // differs is how many distances each evaluates.
    constexpr int kUniverse = 400;
    constexpr int kMinLen = 200;
    auto selection =
        GENERATE(StructuralSelection::FirstImprovingSample, StructuralSelection::BestOfSample);
    auto inc = build_route_model(/*use_edits=*/true, kUniverse, kMinLen, kUniverse);
    auto full = build_route_model(/*use_edits=*/false, kUniverse, kMinLen, kUniverse);
    RNG init(172);
    const std::vector<int32_t> start_list = random_route(init, kUniverse, 300);
    const std::vector<int32_t> start_set = random_route(init, 12, 6);
    set_route(*inc, start_list, start_set);
    set_route(*full, start_list, start_set);

    SearchConfig config;
    config.structural_selection = selection;
    StructuralBatch inc_batch(inc->model, config, /*enabled=*/true);
    StructuralBatch full_batch(full->model, config, /*enabled=*/true);
    ViolationManager inc_vm(inc->model);
    ViolationManager full_vm(full->model);
    inc->stats->clear();
    full->stats->clear();

    RNG inc_rng(99);
    RNG full_rng(99);
    int moved = 0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        const int64_t evals_before = inc->stats->dist_evals;
        const int64_t edits_before = inc->stats->edits_consumed;
        const bool a = inc_batch.run(inc->model, inc_vm, inc_rng, false, kNoDeadline);
        const bool b = full_batch.run(full->model, full_vm, full_rng, false, kNoDeadline);
        REQUIRE(a == b);
        moved += a ? 1 : 0;
        // The same move sequence: same elements, same value, and the value is
        // the true route cost.
        REQUIRE(inc->model.var(handle_to_var_id(inc->list)).elements ==
                full->model.var(handle_to_var_id(full->list)).elements);
        REQUIRE(inc->model.node_value(inc->route) == full->model.node_value(full->route));
        REQUIRE(inc->model.node_value(inc->route) == expected_route(*inc));
        // O(edits): at most 8 distance evaluations per edit, per sweep.
        REQUIRE(inc->stats->dist_evals - evals_before <=
                8 * (inc->stats->edits_consumed - edits_before));
    }
    REQUIRE(moved > 0);
    REQUIRE(inc->stats->mirror_mismatches == 0);
    REQUIRE(full->stats->mirror_mismatches == 0);
    REQUIRE(inc->stats->deltas == full->stats->deltas);
    REQUIRE(inc->stats->deltas > 100);

    // Every delta the batch drove carried edits: the incremental invariant never
    // re-read, and the baseline re-read every structured input that moved.
    REQUIRE(inc->stats->rereads == 0);
    REQUIRE(inc->stats->incremental > 0);
    REQUIRE(full->stats->incremental == 0);
    REQUIRE(full->stats->rereads == inc->stats->incremental);
    // The baseline's cost is the route length per re-read (|L| >= 200 here);
    // the incremental one's is bounded by the edits.
    REQUIRE(full->stats->dist_evals >= full->stats->reread_lengths - full->stats->rereads);
    REQUIRE(inc->stats->dist_evals <= 8 * inc->stats->edits_consumed);
    // The difference, stated as a ratio so the test says what it measured.
    // Observed ~28-30x (about 5.2k against 156k distance evaluations over
    // ~800-1100 deltas, with |L| shrinking from 300 towards its floor of 200);
    // asserting 10x leaves room for a different move mix without letting an
    // O(n) regression through.
    REQUIRE(inc->stats->dist_evals * 10 < full->stats->dist_evals);
}

namespace {

// The random apply/undo sequence below, one method per kind of step, so each
// reads on its own. Every step leaves `live` consistent with its assignment by
// its own route; the test then checks that against a from-scratch pass.
class ApplyUndoSequence {
public:
    explicit ApplyUndoSequence(uint64_t seed)
        : live_(build_route_model(/*use_edits=*/true, 40, 0, 40)), rng_(seed) {
        set_route(*live_, random_route(rng_, 40, 25), random_route(rng_, 12, 5));
        kid_ = handle_to_var_id(live_->k);
        lid_ = handle_to_var_id(live_->list);
        sid_ = handle_to_var_id(live_->set);
        vm_ = std::make_unique<ViolationManager>(live_->model);
        anchor_ = live_->model.copy_state();
    }

    [[nodiscard]] RouteModel& live() { return *live_; }
    [[nodiscard]] RNG& rng() { return rng_; }
    int undos = 0;
    int probes = 0;
    int two_var = 0;

    // A journaled move on L or S, kept on the undo stack.
    void journaled_move() {
        const int32_t var = rng_.integers(0, 2) == 0 ? lid_ : sid_;
        const std::vector<Move> moves = draw(var);
        if (moves.empty()) {
            return;
        }
        commit_recorded(pick(moves));
    }

    // ONE move over TWO structured inputs: the edits are keyed per input.
    void two_variable_move() {
        const std::vector<Move> lm = draw(lid_);
        const std::vector<Move> sm = draw(sid_);
        if (lm.empty() || sm.empty()) {
            return;
        }
        Move both = pick(lm);
        both.changes.push_back(pick(sm).changes.front());
        commit_recorded(both);
        ++two_var;
    }

    // Undo the most recent journaled move EXACTLY, by its inverse edits.
    void undo() {
        if (undo_stack_.empty()) {
            return;
        }
        Applied a = std::move(undo_stack_.back());
        undo_stack_.pop_back();
        EditJournal inverse;
        for (const int32_t var : a.vars) {
            inverse.begin(var);
            inverse.append_inverse(a.journal, var);
            ConstSpan<PositionalEdit> edits;
            REQUIRE(inverse.lookup(var, edits) == EditJournal::Status::Known);
            for (const PositionalEdit& e : edits) {
                apply_positional_edit(e, model().var_mut(var).elements);
            }
        }
        delta_evaluate(model(), a.vars, DeltaMode::Commit, &inverse);
        ++undos;
    }

    // A journaled PROBE, put back and rolled back: the next delta's edits must
    // be relative to the committed state, not to the probe.
    void probe_and_roll_back() {
        const std::vector<Move> moves = draw(lid_);
        if (moves.empty()) {
            return;
        }
        Model& m = model();
        const double before = m.node_value(live_->route);
        const std::vector<int32_t> saved = m.var(lid_).elements;
        EditJournal journal;
        const std::vector<int32_t> vars = apply_move_recorded(m, pick(moves), journal);
        delta_evaluate(m, vars, DeltaMode::Probe, &journal);
        REQUIRE(m.node_value(live_->route) == expected_route(*live_));
        m.var_mut(lid_).elements = saved;
        delta_evaluate(m, vars, DeltaMode::Rollback);
        REQUIRE(m.node_value(live_->route) == before);
        ++probes;
    }

    // A move with NO journal: the fallback.
    void unjournaled_move() {
        const std::vector<Move> moves = draw(lid_);
        if (moves.empty()) {
            return;
        }
        delta_evaluate(model(), apply_move(model(), pick(moves)));
        undo_stack_.clear();
    }

    // A Replace, journaled: the record is Unknown, so the fallback.
    void replace() {
        std::vector<int32_t> reversed = model().var(lid_).elements;
        std::reverse(reversed.begin(), reversed.end());
        Move move;
        move.changes.push_back(replace_change(lid_, reversed));
        EditJournal journal;
        const std::vector<int32_t> vars = apply_move_recorded(model(), move, journal);
        delta_evaluate(model(), vars, DeltaMode::Commit, &journal);
        undo_stack_.clear();
    }

    // Mostly a scalar move; one time in ten a restore, the other fallback.
    void scalar_or_restore() {
        if (rng_.integers(0, 10) == 0) {
            model().restore_state(anchor_);
            full_evaluate(model());
            undo_stack_.clear();
            return;
        }
        model().var_mut(kid_).value = static_cast<double>(rng_.integers(0, 10));
        delta_evaluate(model(), &kid_, 1);
    }

    // The bracketed scalar probe Feasibility Jump runs.
    void scalar_probe() {
        (void)vm_->weighted_violation_delta(kid_, static_cast<double>(rng_.integers(0, 10)));
    }

private:
    struct Applied {
        std::vector<int32_t> vars;
        EditJournal journal;
    };

    [[nodiscard]] Model& model() { return live_->model; }

    std::vector<Move> draw(int32_t var_id) {
        std::vector<Move> moves;
        generate_standard_moves(model().var(var_id), rng_, moves, nullptr);
        return moves;
    }
    const Move& pick(const std::vector<Move>& moves) {
        return moves[static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(moves.size())))];
    }
    void commit_recorded(const Move& move) {
        Applied a;
        a.vars = apply_move_recorded(model(), move, a.journal);
        delta_evaluate(model(), a.vars, DeltaMode::Commit, &a.journal);
        undo_stack_.push_back(std::move(a));
    }

    std::unique_ptr<RouteModel> live_;
    RNG rng_;
    int32_t kid_ = 0;
    int32_t lid_ = 0;
    int32_t sid_ = 0;
    std::unique_ptr<ViolationManager> vm_;
    Model::State anchor_;
    std::vector<Applied> undo_stack_;
};

}  // namespace

TEST_CASE("delta_evaluate and full_evaluate agree over a random apply/undo sequence with edits",
          "[custom][edits]") {
    // Criterion 2 of #172: #166's equivalence property, now with an invariant
    // that CONSUMES the positional edits, and with every way the engine can
    // reach it mixed in: journaled moves (one and two variables), their exact
    // undo, a journaled probe rolled back, a move with no journal, a Replace,
    // a restore, and the bracketed scalar probe. After every step the live
    // value must match a from-scratch pass on a different instance.
    ApplyUndoSequence seq(20260926);
    auto ref = build_route_model(/*use_edits=*/true, 40, 0, 40);
    RouteModel& live = seq.live();

    for (int step = 0; step < 3000; ++step) {
        switch (seq.rng().integers(0, 10)) {
            case 0:
            case 1:
            case 2:
                seq.journaled_move();
                break;
            case 3:
                seq.two_variable_move();
                break;
            case 4:
                seq.undo();
                break;
            case 5:
                seq.probe_and_roll_back();
                break;
            case 6:
                seq.unjournaled_move();
                break;
            case 7:
                seq.replace();
                break;
            case 8:
                seq.scalar_or_restore();
                break;
            default:
                seq.scalar_probe();
                break;
        }
        ref->model.restore_state(live.model.copy_state());
        full_evaluate(ref->model);
        REQUIRE(live.model.node_value(live.route) == ref->model.node_value(ref->route));
        REQUIRE(live.model.node_value(live.row) == ref->model.node_value(ref->row));
        REQUIRE(live.stats->mirror_mismatches == 0);
    }
    // Every path above must actually have been exercised.
    REQUIRE(live.stats->incremental > 500);
    REQUIRE(live.stats->rereads > 50);
    REQUIRE(seq.undos > 100);
    REQUIRE(seq.probes > 100);
    REQUIRE(seq.two_var > 100);
}

namespace {

// Records what `edits(i)` reported for input `watch`, and nothing else.
struct EditsSeen {
    int calls = 0;
    bool available = false;
    size_t count = 0;
    bool threw = false;
    bool scalar_available = true;
    int delta_calls_available = 0;  // across delta() calls only
};

class EditsProbe : public CustomInvariant {
public:
    EditsProbe(std::shared_ptr<EditsSeen> seen, int32_t watch, bool wants = true)
        : seen_(std::move(seen)), watch_(watch), wants_(wants) {}

    double evaluate(const InvariantInputs& in) override {
        record(in);
        return sum(in);
    }
    double delta(const InvariantInputs& in, ConstSpan<int32_t> changed) override {
        record(in);
        if (seen_->available && std::binary_search(changed.begin(), changed.end(), watch_)) {
            ++seen_->delta_calls_available;
        }
        return sum(in);
    }
    [[nodiscard]] bool wants_positional_edits() const override { return wants_; }
    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        return std::make_unique<EditsProbe>(*this);
    }

private:
    void record(const InvariantInputs& in) {
        ++seen_->calls;
        const InputEdits e = in.edits(watch_);
        seen_->available = e.available();
        seen_->count = e.available() ? e.list().size() : 0;
        seen_->threw = false;
        if (!e.available()) {
            try {
                (void)e.list();
            } catch (const std::logic_error&) {
                seen_->threw = true;
            }
        }
        seen_->scalar_available = in.edits(0).available();
    }
    static double sum(const InvariantInputs& in) {
        double t = in.value(0);
        for (int32_t i = 1; i < in.size(); ++i) {
            for (const int32_t e : in.elements(i)) {
                t += static_cast<double>(e);
            }
        }
        return t;
    }

    std::shared_ptr<EditsSeen> seen_;
    int32_t watch_;
    bool wants_;
};

}  // namespace

TEST_CASE("no positional information is never mistaken for no changes", "[custom][edits]") {
    auto seen = std::make_shared<EditsSeen>();
    Model m;
    const int32_t k = m.int_var(0, 9, "k");
    const int32_t l = m.list_var(10, "L");
    const int32_t node = m.custom({k, l}, std::make_unique<EditsProbe>(seen, 1), "probe");
    m.minimize(node);
    m.close();
    const int32_t kid = handle_to_var_id(k);
    const int32_t lid = handle_to_var_id(l);

    SECTION("evaluate has none, and list() throws rather than reading as empty") {
        full_evaluate(m);
        REQUIRE_FALSE(seen->available);
        REQUIRE(seen->threw);
    }
    SECTION("a scalar input never has any") {
        m.var_mut(kid).value = 3.0;
        delta_evaluate(m, &kid, 1);
        REQUIRE_FALSE(seen->scalar_available);
    }
    SECTION("an input that did not move has an available, empty list") {
        m.var_mut(kid).value = 4.0;
        delta_evaluate(m, &kid, 1);
        REQUIRE(seen->available);
        REQUIRE(seen->count == 0);
    }
    SECTION("a moved input with no journal has none") {
        apply_move(m, Move{{edit_change(lid, swap_edit(0, 1))}, "swap", 0.0});
        delta_evaluate(m, &lid, 1);
        REQUIRE_FALSE(seen->available);
        REQUIRE(seen->threw);
    }
    SECTION("a moved input with a journal has exactly its edits") {
        EditJournal j;
        apply_move_recorded(
            m, Move{{edit_change(lid, swap_edit(0, 1), reverse_edit(2, 5))}, "x", 0.0}, j);
        delta_evaluate(m, &lid, 1, DeltaMode::Commit, &j);
        REQUIRE(seen->available);
        REQUIRE(seen->count == 2);
    }
    SECTION("a journal with no record for a moved input has none") {
        EditJournal j;  // empty: the caller described nothing
        apply_move(m, Move{{edit_change(lid, swap_edit(0, 1))}, "swap", 0.0});
        delta_evaluate(m, &lid, 1, DeltaMode::Commit, &j);
        REQUIRE_FALSE(seen->available);
    }
    SECTION("a Replace has none") {
        EditJournal j;
        std::vector<int32_t> rev = m.var(lid).elements;
        std::reverse(rev.begin(), rev.end());
        apply_move_recorded(m, Move{{replace_change(lid, rev)}, "replace", 0.0}, j);
        delta_evaluate(m, &lid, 1, DeltaMode::Commit, &j);
        REQUIRE_FALSE(seen->available);
    }
}

TEST_CASE("the structural batch records edits only for an invariant that opts in",
          "[custom][edits]") {
    // The journal costs ~5-7% per candidate on a re-reading invariant, so it is
    // opt-in. Not opting in must still be CORRECT -- `available()` false -- and
    // opting in must actually switch it on.
    const bool wants = GENERATE(false, true);
    auto seen = std::make_shared<EditsSeen>();
    Model m;
    const int32_t k = m.int_var(0, 9, "k");
    const int32_t l = m.list_var(30, "L");
    const int32_t node = m.custom({k, l}, std::make_unique<EditsProbe>(seen, 1, wants), "probe");
    m.add_constraint(m.leq(node, m.constant(0.0)));
    m.minimize(node);
    m.close();

    SearchConfig config;
    StructuralBatch batch(m, config, /*enabled=*/true);
    ViolationManager vm(m);
    RNG rng(3);
    const int calls_before = seen->calls;
    for (int sweep = 0; sweep < 20; ++sweep) {
        (void)batch.run(m, vm, rng, false, kNoDeadline);
    }
    REQUIRE(seen->calls > calls_before);
    if (wants) {
        REQUIRE(seen->delta_calls_available > 0);
    } else {
        REQUIRE(seen->delta_calls_available == 0);
    }
}

namespace {

// A registered generator emitting the move shapes no built-in does, in turn:
// the SAME variable changed by two separate Changes, a two-variable move, a
// whole-vector Replace, and an empty Move. `describe_transition` has to get
// each of them right as both the current and the PREVIOUS candidate.
class ShapesGenerator final : public MoveGenerator {
public:
    ShapesGenerator(int32_t list, int32_t set) : scope_{list, set} {}
    [[nodiscard]] std::string_view name() const override { return "shapes"; }
    [[nodiscard]] ConstSpan<int32_t> scope() const override { return {scope_.data(), 2}; }

    void generate(MoveContext& ctx, std::vector<Move>& out) override {
        const std::vector<int32_t>& l = ctx.model.var(scope_[0]).elements;
        const std::vector<int32_t>& s = ctx.model.var(scope_[1]).elements;
        const auto n = static_cast<int64_t>(l.size());
        if (n < 4) {
            return;
        }
        auto pos = [&ctx, n]() { return static_cast<int32_t>(ctx.rng.integers(0, n)); };
        Move move;
        move.move_type = "shape";
        switch (next_++ % 4) {
            case 0:  // one variable, two Changes
                move.changes.push_back(edit_change(scope_[0], swap_edit(pos(), pos())));
                move.changes.push_back(edit_change(scope_[0], reverse_edit(1, 3)));
                break;
            case 1:  // two variables
                move.changes.push_back(edit_change(
                    scope_[0], segment_edit(pos() % static_cast<int32_t>(n - 2), 2, 0)));
                if (s.size() > 1) {
                    move.changes.push_back(edit_change(scope_[1], swap_edit(0, 1)));
                }
                break;
            case 2: {  // a Replace
                std::vector<int32_t> reversed = l;
                std::reverse(reversed.begin(), reversed.end());
                move.changes.push_back(replace_change(scope_[0], std::move(reversed)));
                break;
            }
            default:  // nothing at all
                break;
        }
        out.push_back(std::move(move));
    }

    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        return std::make_unique<ShapesGenerator>(*this);
    }

private:
    std::array<int32_t, 2> scope_;
    int next_ = 0;
};

}  // namespace

TEST_CASE("the structural batch describes every move shape exactly", "[custom][edits]") {
    // Repeated-variable, two-variable, Replace and empty candidates, through the
    // batch, under every selection policy -- including `take_best`'s no-winner
    // restore. A doubled or stale description shows up as a mirror mismatch or
    // a wrong value; a Replace must push the invariant into a re-read.
    const auto selection =
        GENERATE(StructuralSelection::FirstImprovingSample, StructuralSelection::BestOfSample,
                 StructuralSelection::ViolationGuided);
    auto rm = build_route_model(/*use_edits=*/true, 40, 0, 40);
    RNG init(11);
    set_route(*rm, random_route(init, 40, 20), random_route(init, 12, 5));

    SearchConfig config;
    config.structural_selection = selection;
    config.structural_sample_size = 3;
    config.default_structural_generators = false;
    config.move_generators.push_back(std::make_shared<const ShapesGenerator>(
        handle_to_var_id(rm->list), handle_to_var_id(rm->set)));
    StructuralBatch batch(rm->model, config, /*enabled=*/true);
    ViolationManager vm(rm->model);
    RNG rng(5);
    rm->stats->clear();
    int idle = 0;
    for (int sweep = 0; sweep < 200; ++sweep) {
        idle += batch.run(rm->model, vm, rng, false, kNoDeadline) ? 0 : 1;
        REQUIRE(rm->stats->mirror_mismatches == 0);
        REQUIRE(rm->model.node_value(rm->route) == expected_route(*rm));
    }
    REQUIRE(rm->stats->incremental > 100);
    REQUIRE(rm->stats->rereads > 10);
    REQUIRE(idle > 0);  // the no-winner restore ran
}

TEST_CASE("the fallback paths re-read and land on the incremental value", "[custom][edits]") {
    // Criterion 3 of #172: a full evaluation, a state restore and a move with no
    // edits each drive the invariant down the re-read path, and each produces
    // the value the incremental path produces for the same assignment.
    auto inc = build_route_model(/*use_edits=*/true, 60, 0, 60);
    auto other = build_route_model(/*use_edits=*/true, 60, 0, 60);
    RNG rng(4242);
    const std::vector<int32_t> l0 = random_route(rng, 60, 40);
    const std::vector<int32_t> s0 = random_route(rng, 12, 6);
    set_route(*inc, l0, s0);
    set_route(*other, l0, s0);
    const int32_t lid = handle_to_var_id(inc->list);
    const Move move{{edit_change(lid, segment_edit(3, 4, 20))}, "or_opt", 0.0};

    // The incremental reference: the move, journaled.
    EditJournal j;
    apply_move_recorded(inc->model, move, j);
    inc->stats->clear();
    delta_evaluate(inc->model, &lid, 1, DeltaMode::Commit, &j);
    REQUIRE(inc->stats->incremental == 1);
    REQUIRE(inc->stats->rereads == 0);
    const double incremental_value = inc->model.node_value(inc->route);
    REQUIRE(incremental_value == expected_route(*inc));

    SECTION("a move that supplies no edits") {
        apply_move(other->model, move);
        other->stats->clear();
        delta_evaluate(other->model, &lid, 1);
        REQUIRE(other->stats->rereads == 1);
        REQUIRE(other->stats->incremental == 0);
        REQUIRE(other->model.node_value(other->route) == incremental_value);
    }
    SECTION("a full evaluation") {
        other->model.var_mut(lid).elements = inc->model.var(lid).elements;
        other->stats->clear();
        full_evaluate(other->model);
        REQUIRE(other->stats->evaluates == 1);
        REQUIRE(other->stats->incremental == 0);
        REQUIRE(other->model.node_value(other->route) == incremental_value);
    }
    SECTION("a state restore, and the edits after it are relative to it") {
        other->model.restore_state(inc->model.copy_state());
        other->stats->clear();
        full_evaluate(other->model);
        REQUIRE(other->stats->evaluates == 1);
        REQUIRE(other->model.node_value(other->route) == incremental_value);
        // The incremental path picks up from the restored state.
        EditJournal j2;
        apply_move_recorded(other->model,
                            Move{{edit_change(lid, reverse_edit(1, 30))}, "2opt", 0.0}, j2);
        delta_evaluate(other->model, &lid, 1, DeltaMode::Commit, &j2);
        REQUIRE(other->stats->incremental == 1);
        REQUIRE(other->model.node_value(other->route) == expected_route(*other));
        REQUIRE(other->stats->mirror_mismatches == 0);
    }
}

TEST_CASE("a probe left open by an exception withholds the edits", "[custom][edits]") {
    // The one engine-side reason to withhold edits from a journaled call: a
    // custom node still owes a commit or rollback, so its committed state is not
    // the state the caller's edits start from.
    class ThrowWhenArmed : public CustomInvariant {
    public:
        explicit ThrowWhenArmed(std::shared_ptr<bool> armed) : armed_(std::move(armed)) {}
        double evaluate(const InvariantInputs& in) override { return in.value(0); }
        double delta(const InvariantInputs& in, ConstSpan<int32_t> /*changed*/) override {
            if (*armed_) {
                *armed_ = false;
                throw std::runtime_error("refused");
            }
            return evaluate(in);
        }
        [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
            return std::make_unique<ThrowWhenArmed>(*this);
        }

    private:
        std::shared_ptr<bool> armed_;
    };

    auto seen = std::make_shared<EditsSeen>();
    auto armed = std::make_shared<bool>(false);
    Model m;
    const int32_t k = m.int_var(0, 9, "k");
    const int32_t l = m.list_var(10, "L");
    const int32_t inner = m.custom({k, l}, std::make_unique<EditsProbe>(seen, 1), "inner");
    const int32_t outer = m.custom({inner}, std::make_unique<ThrowWhenArmed>(armed), "outer");
    m.minimize(outer);
    m.close();
    const int32_t lid = handle_to_var_id(l);

    EditJournal j;
    apply_move_recorded(m, Move{{edit_change(lid, swap_edit(0, 1))}, "swap", 0.0}, j);
    *armed = true;
    // `inner` opens its probe, then `outer` throws: `inner` is left pending.
    REQUIRE_THROWS_AS(delta_evaluate(m, &lid, 1, DeltaMode::Probe, &j), std::runtime_error);
    REQUIRE(seen->available);

    EditJournal j2;
    apply_move_recorded(m, Move{{edit_change(lid, swap_edit(2, 3))}, "swap", 0.0}, j2);
    delta_evaluate(m, &lid, 1, DeltaMode::Commit, &j2);
    REQUIRE_FALSE(seen->available);

    // And the next call, with nothing pending, has them again.
    EditJournal j3;
    apply_move_recorded(m, Move{{edit_change(lid, swap_edit(4, 5))}, "swap", 0.0}, j3);
    delta_evaluate(m, &lid, 1, DeltaMode::Commit, &j3);
    REQUIRE(seen->available);
    REQUIRE(seen->count == 1);
}

namespace {

/// Whether a recorded edit was a legal, effective edit on the vector it was
/// replayed against -- written out independently of `edit_takes_effect` so the
/// test catches a guard that is too permissive, not only one that is too
/// strict: an out-of-range edit replays and inverts as a no-op, so replay
/// equality alone cannot see it.
bool recorded_edit_in_range(const PositionalEdit& e, const std::vector<int32_t>& v) {
    const auto n = static_cast<int32_t>(v.size());
    auto idx = [n](int32_t i) { return i >= 0 && i < n; };
    switch (e.kind) {
        case EditKind::Swap:
            return idx(e.from) && idx(e.to);
        case EditKind::Reverse:
            return idx(e.from) && idx(e.to) && e.from <= e.to;
        case EditKind::MoveSegment:
            return e.length > 0 && e.from >= 0 && e.to >= 0 && e.from + e.length <= n &&
                   e.to + e.length <= n;
        case EditKind::Insert:
            return e.from >= 0 && e.from <= n;
        case EditKind::Erase:
        case EditKind::Assign:
            return idx(e.from);
        case EditKind::None:
        case EditKind::Replace:
            return false;  // never produced by an edit-carrying change
    }
    return false;
}

}  // namespace

TEST_CASE("a recorded apply matches the plain one, replays, and inverts exactly",
          "[custom][edits][moves]") {
    // The journal is only as good as its agreement with `apply_element_edits`'s
    // range guards: an edit the guard ignores must not be recorded, or its
    // inverse is applied to a vector it never touched.
    RNG rng(7);
    int recorded = 0;
    int ignored = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const auto n = static_cast<int32_t>(rng.integers(0, 8));
        std::vector<int32_t> base(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            base[static_cast<size_t>(i)] = 100 + i;
        }
        auto pos = [&rng, n]() { return static_cast<int32_t>(rng.integers(-1, n + 2)); };
        auto one = [&]() {
            switch (rng.integers(0, 6)) {
                case 0:
                    return swap_edit(pos(), pos());
                case 1:
                    return reverse_edit(pos(), pos());
                case 2:
                    return segment_edit(pos(), static_cast<int32_t>(rng.integers(0, 4)), pos());
                case 3:
                    return insert_edit(pos(), 500 + trial);
                case 4:
                    return erase_edit(pos());
                default:
                    return assign_edit(pos(), 900 + trial);
            }
        };
        const Move::Change change = edit_change(0, one(), one());

        std::vector<int32_t> plain = base;
        apply_element_edits(change, plain);
        std::vector<int32_t> rec = base;
        EditJournal j;
        apply_element_edits(change, rec, j);
        REQUIRE(rec == plain);

        ConstSpan<PositionalEdit> edits;
        REQUIRE(j.lookup(0, edits) == EditJournal::Status::Known);
        recorded += static_cast<int>(edits.size());
        ignored += 2 - static_cast<int>(edits.size());
        std::vector<int32_t> replay = base;
        for (const PositionalEdit& e : edits) {
            REQUIRE(recorded_edit_in_range(e, replay));
            apply_positional_edit(e, replay);
        }
        REQUIRE(replay == plain);

        EditJournal inverse;
        inverse.begin(0);
        inverse.append_inverse(j, 0);
        ConstSpan<PositionalEdit> back;
        REQUIRE(inverse.lookup(0, back) == EditJournal::Status::Known);
        for (const PositionalEdit& e : back) {
            apply_positional_edit(e, replay);
        }
        REQUIRE(replay == base);
    }
    // Both halves of the guard were exercised.
    REQUIRE(recorded > 1000);
    REQUIRE(ignored > 1000);
}

TEST_CASE("EditJournal keeps a variable's edits contiguous or says it cannot", "[custom][edits]") {
    EditJournal j;
    PositionalEdit e;
    e.kind = EditKind::Swap;
    ConstSpan<PositionalEdit> out;

    j.begin(3);
    j.push(e);
    j.begin(3);  // re-opening the most recent record continues it
    j.push(e);
    REQUIRE(j.lookup(3, out) == EditJournal::Status::Known);
    REQUIRE(out.size() == 2);

    j.begin(5);
    j.push(e);
    j.begin(3);  // an earlier record cannot stay contiguous
    j.push(e);
    REQUIRE(j.lookup(3, out) == EditJournal::Status::Unknown);
    REQUIRE(j.lookup(5, out) == EditJournal::Status::Known);
    REQUIRE(out.size() == 1);
    REQUIRE(j.lookup(9, out) == EditJournal::Status::Absent);

    j.clear();
    REQUIRE(j.empty());
    REQUIRE(j.lookup(3, out) == EditJournal::Status::Absent);
}
