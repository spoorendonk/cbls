// Variable-length List variables and the optional partition across them (#164).
//
// The property test below is the centre of this file. A partition's invariant is
// maintained BY THE MOVES rather than by a constraint row, so there is nothing
// in the engine that would notice it being broken: an element in two routes
// evaluates perfectly happily, scores well (it is served twice), and the search
// keeps it. The only guard available is to apply long random sequences of every
// move and check the invariant after each one.
//
// Every check here reads the lists THROUGH THE DAG as well -- `Count`, `At`,
// `Lambda` and `PairLambda` nodes over each list, compared against a
// `full_evaluate` of a fresh copy. A List that no node reads would make the
// incremental-vs-full comparison vacuous, which is the failure mode a structural
// probe falls into by default.

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/dag_ops.h>
#include <cbls/move_generator.h>
#include <cbls/moves.h>
#include <cbls/randomize.h>
#include <cbls/rng.h>
#include <cbls/search.h>
#include <cbls/violation.h>
#include <sstream>
#include <string>
#include <vector>

using namespace cbls;

namespace {

// A model with `routes` variable-length Lists over one universe, each read by a
// Count, an At, a Lambda and a PairLambda so that every incremental update has
// something to be wrong about.
struct PartitionModel {
    Model model;
    std::vector<int32_t> handles;  // var handles
    std::vector<int32_t> ids;      // var ids
    std::vector<int32_t> probes;   // node ids to compare against full_evaluate
};

double element_weight(int e) {
    return 1.0 + static_cast<double>((e * 5) % 7);
}

double pair_weight(int a, int b) {
    return 1.0 + static_cast<double>(((a * 7) + (b * 3)) % 11);
}

PartitionModel build(int universe, int routes, int min_len, int max_len, ListInit init,
                     bool with_partition, Cover cover) {
    PartitionModel pm;
    for (int r = 0; r < routes; ++r) {
        const int32_t h =
            pm.model.list_var(universe, min_len, max_len, init, "r" + std::to_string(r));
        pm.handles.push_back(h);
        pm.ids.push_back(handle_to_var_id(h));
    }
    if (with_partition) {
        pm.model.add_list_partition(pm.handles, cover);
    }
    std::vector<int32_t> terms;
    for (int32_t h : pm.handles) {
        const int32_t cnt = pm.model.count(h);
        const int32_t head = pm.model.at(h, pm.model.constant(0));
        const int32_t lam = pm.model.lambda_sum(h, element_weight);
        const int32_t pair = pm.model.pair_lambda_sum(h, pair_weight);
        pm.probes.insert(pm.probes.end(), {cnt, head, lam, pair});
        terms.insert(terms.end(), {cnt, head, lam, pair});
    }
    pm.model.minimize(pm.model.sum(terms));
    pm.model.close();
    full_evaluate(pm.model);
    return pm;
}

// Every node value the incremental path produced, against a fresh `full_evaluate`
// of a copy holding the same assignment. The copy is taken from the model itself,
// so it shares the structure and differs only in having been evaluated from
// scratch.
void require_incremental_matches_full(const Model& model, const std::vector<int32_t>& probes) {
    Model fresh = model;
    full_evaluate(fresh);
    for (int32_t nid : probes) {
        if (model.node_value(nid) != fresh.node_value(nid)) {
            FAIL("node " << nid << ": incremental " << model.node_value(nid) << " != full "
                         << fresh.node_value(nid));
            return;
        }
    }
}

// Distinct, inside the universe, and within the declared length bounds.
void require_list_well_formed(const Variable& var) {
    REQUIRE(static_cast<int32_t>(var.elements.size()) >= var.min_size);
    REQUIRE(static_cast<int32_t>(var.elements.size()) <= var.max_size);
    std::vector<int32_t> sorted = var.elements;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    for (int32_t e : var.elements) {
        REQUIRE(e >= 0);
        REQUIRE(e < var.universe_size);
    }
}

// The cover, checked over the whole partition: every element in at most one
// list, and for Exact in exactly one.
void require_cover(const Model& model, const ListPartition& part) {
    std::vector<int> count(static_cast<size_t>(part.universe_size), 0);
    for (int32_t vid : part.list_ids) {
        require_list_well_formed(model.var(vid));
        for (int32_t e : model.var(vid).elements) {
            count[static_cast<size_t>(e)]++;
        }
    }
    for (int e = 0; e < part.universe_size; ++e) {
        const int seen = count[static_cast<size_t>(e)];
        REQUIRE(seen <= 1);
        if (part.cover == Cover::Exact) {
            REQUIRE(seen == 1);
        }
    }
}

// Apply `rounds` random moves drawn from every generator the model has -- the
// per-variable intra-list ones and, where there is a partition, the inter-list
// ones -- checking the full set of invariants after each.
void hammer(PartitionModel& pm, int rounds, uint64_t seed, bool with_partition, Cover cover) {
    RNG rng(seed);
    for (int round = 0; round < rounds; ++round) {
        std::vector<Move> candidates;
        for (size_t k = 0; k < pm.ids.size(); ++k) {
            generate_standard_moves(pm.model.var(pm.ids[k]), rng, candidates, nullptr);
        }
        if (with_partition) {
            // Several calls, because each appends at most one candidate.
            for (size_t k = 0; k < pm.ids.size(); ++k) {
                generate_partition_moves(pm.model, 0, /*anchor=*/-1, rng, candidates, nullptr);
            }
        }
        if (candidates.empty()) {
            continue;
        }
        const auto pick =
            static_cast<size_t>(rng.integers(0, static_cast<int64_t>(candidates.size())));
        const std::vector<int32_t> touched = apply_move(pm.model, candidates[pick]);
        delta_evaluate(pm.model, touched);

        for (int32_t id : pm.ids) {
            require_list_well_formed(pm.model.var(id));
        }
        if (with_partition) {
            require_cover(pm.model, pm.model.list_partitions()[0]);
            REQUIRE(pm.model.list_partitions()[0].cover == cover);
        }
        require_incremental_matches_full(pm.model, pm.probes);
    }
}

}  // namespace

TEST_CASE("list_var(n) is still exactly a permutation", "[list][partition]") {
    Model m;
    const int32_t h = m.list_var(6, "perm");
    const Variable& v = m.var(handle_to_var_id(h));
    REQUIRE(v.universe_size == 6);
    REQUIRE(v.min_size == 6);
    REQUIRE(v.max_size == 6);
    REQUIRE(v.list_init == ListInit::Identity);
    REQUIRE(v.elements == std::vector<int32_t>{0, 1, 2, 3, 4, 5});
    REQUIRE_FALSE(v.partitioned);
}

TEST_CASE("a permutation List draws exactly one rng.permutation on init", "[list][partition]") {
    // The claim that permutation trajectories are bit-identical rests on this
    // one arm of randomize_structured_var, so pin it directly rather than only
    // through the whole-search digests.
    Model m;
    const int32_t h = m.list_var(7, "perm");
    RNG a(4242);
    RNG b(4242);
    initialize_structured_random(m, a);
    const std::vector<int32_t> expected = b.permutation(7);
    REQUIRE(m.var(handle_to_var_id(h)).elements == expected);
    // "exactly one" is half the claim, and the half that matters for a shared
    // RNG: a second draw here would shift every later draw in the search. The
    // elements check alone would pass with one.
    REQUIRE(a.integers(0, 1000) == b.integers(0, 1000));
}

TEST_CASE("list_var rejects an impossible length window", "[list][partition]") {
    Model m;
    REQUIRE_THROWS_AS(m.list_var(5, 3, 2, ListInit::Empty), std::invalid_argument);
    REQUIRE_THROWS_AS(m.list_var(5, 0, 6, ListInit::Empty), std::invalid_argument);
    REQUIRE_THROWS_AS(m.list_var(5, -1, 2, ListInit::Empty), std::invalid_argument);
    REQUIRE_THROWS_AS(m.list_var(-1, 0, 0, ListInit::Empty), std::invalid_argument);
    // Identity is only a legal assignment when the length is pinned at the
    // universe; anything else would be a list that starts outside its own bounds.
    REQUIRE_THROWS_AS(m.list_var(5, 0, 5, ListInit::Identity), std::invalid_argument);
    REQUIRE_NOTHROW(m.list_var(5, 5, 5, ListInit::Identity));
    // And Empty is only legal where zero is a legal length: otherwise the List
    // starts below its own min_len, and every later re-randomisation puts it
    // back there, with no constraint row and no move guard reporting it.
    REQUIRE_THROWS_AS(m.list_var(5, 1, 3, ListInit::Empty), std::invalid_argument);
    REQUIRE_NOTHROW(m.list_var(5, 1, 3, ListInit::Random));
}

TEST_CASE("add_list_partition validates its group", "[list][partition]") {
    SECTION("a non-List member is refused") {
        Model m;
        const int32_t list = m.list_var(4, 0, 4);
        const int32_t scalar = m.bool_var("b");
        REQUIRE_THROWS_AS(m.add_list_partition({list, scalar}, Cover::AtMostOnce),
                          std::invalid_argument);
    }
    SECTION("mismatched universes are refused") {
        Model m;
        const int32_t a = m.list_var(4, 0, 4);
        const int32_t b = m.list_var(5, 0, 5);
        REQUIRE_THROWS_AS(m.add_list_partition({a, b}, Cover::AtMostOnce), std::invalid_argument);
    }
    SECTION("a list may be in at most one partition, and named at most once") {
        Model m;
        const int32_t a = m.list_var(4, 0, 4);
        const int32_t b = m.list_var(4, 0, 4);
        REQUIRE_THROWS_AS(m.add_list_partition({a, a}, Cover::AtMostOnce), std::invalid_argument);
        m.add_list_partition({a, b}, Cover::AtMostOnce);
        const int32_t c = m.list_var(4, 0, 4);
        REQUIRE_THROWS_AS(m.add_list_partition({a, c}, Cover::AtMostOnce), std::invalid_argument);
    }
    SECTION("minimum lengths that overflow the universe are refused") {
        Model m;
        const int32_t a = m.list_var(4, 3, 4, ListInit::Random);
        const int32_t b = m.list_var(4, 3, 4, ListInit::Random);
        REQUIRE_THROWS_AS(m.add_list_partition({a, b}, Cover::AtMostOnce), std::invalid_argument);
    }
    SECTION("an Exact cover the maximum lengths cannot reach is refused") {
        Model m;
        const int32_t a = m.list_var(10, 0, 3);
        const int32_t b = m.list_var(10, 0, 3);
        REQUIRE_THROWS_AS(m.add_list_partition({a, b}, Cover::Exact), std::invalid_argument);
        REQUIRE_NOTHROW(m.add_list_partition({a, b}, Cover::AtMostOnce));
    }
    SECTION("Identity in a group of more than one is refused") {
        Model m;
        const int32_t a = m.list_var(4, "a");
        const int32_t b = m.list_var(4, "b");
        REQUIRE_THROWS_AS(m.add_list_partition({a, b}, Cover::Exact), std::invalid_argument);
    }
    SECTION("membership is recorded on both sides") {
        Model m;
        const int32_t a = m.list_var(6, 0, 6);
        const int32_t b = m.list_var(6, 0, 6);
        const int32_t p = m.add_list_partition({a, b}, Cover::Exact);
        REQUIRE(p == 0);
        REQUIRE(m.partition_of_list(handle_to_var_id(a)) == 0);
        REQUIRE(m.partition_of_list(handle_to_var_id(b)) == 0);
        REQUIRE(m.partition_of_list(99) == -1);
        REQUIRE(m.var(handle_to_var_id(a)).partitioned);
        REQUIRE(m.list_partitions().size() == 1);
        REQUIRE(m.list_partitions()[0].universe_size == 6);
    }
}

TEST_CASE("ListInit decides the initial assignment", "[list][partition]") {
    RNG rng(7);
    SECTION("Empty starts empty and draws nothing") {
        Model m;
        const int32_t h = m.list_var(8, 0, 8, ListInit::Empty);
        m.close();
        RNG probe(7);
        initialize_structured_random(m, rng);
        REQUIRE(m.var(handle_to_var_id(h)).elements.empty());
        REQUIRE(rng.integers(0, 1000) == probe.integers(0, 1000));
    }
    SECTION("Random lands inside the declared window") {
        Model m;
        const int32_t h = m.list_var(8, 2, 5, ListInit::Random);
        m.close();
        initialize_structured_random(m, rng);
        require_list_well_formed(m.var(handle_to_var_id(h)));
    }
}

TEST_CASE("initialize_random lays out a partition rather than its members", "[list][partition]") {
    // `initialize_random` is the documented recipe for a seed-varying start
    // (`initialize_random(model, rng)` then `solve(..., skip_init = true)`), and
    // it randomises EVERY variable. Run per member, that follows each list's own
    // `ListInit` with no account of its siblings -- which under `Cover::Exact`
    // is a cover no move can repair, because Exact admits no insert or remove
    // that is not half of an inter-list move.
    PartitionModel pm = build(/*universe=*/12, /*routes=*/3, /*min_len=*/0, /*max_len=*/12,
                              ListInit::Empty, /*with_partition=*/true, Cover::Exact);
    RNG rng(13);
    initialize_random(pm.model, rng);
    require_cover(pm.model, pm.model.list_partitions()[0]);
}

TEST_CASE("an Exact partition is initialised complete", "[list][partition]") {
    // It has to be: Exact admits no insert and no remove that is not half of an
    // inter-list move, so a partition that starts incomplete can never be
    // repaired. The per-list ListInit is deliberately ignored here.
    PartitionModel pm = build(/*universe=*/12, /*routes=*/3, /*min_len=*/0, /*max_len=*/12,
                              ListInit::Empty, /*with_partition=*/true, Cover::Exact);
    RNG rng(11);
    initialize_structured_random(pm.model, rng);
    require_cover(pm.model, pm.model.list_partitions()[0]);
}

TEST_CASE("an AtMostOnce partition honours each member's ListInit", "[list][partition]") {
    // Under `Exact` the cover has to hold at the first assignment, so the
    // per-list `ListInit` is ignored and everything is placed. Under
    // `AtMostOnce` there is no such obligation, and the init is what a
    // prize-collecting model uses to say where it wants to start: all `Empty`
    // means everything unassigned and the skip penalty at its worst, which the
    // search can improve from.
    Model m;
    std::vector<int32_t> handles;
    handles.reserve(3);
    for (int r = 0; r < 3; ++r) {
        handles.push_back(m.list_var(9, 0, 9, ListInit::Empty, "r" + std::to_string(r)));
    }
    m.add_list_partition(handles, Cover::AtMostOnce);
    m.close();
    RNG rng(5);
    initialize_structured_random(m, rng);
    require_cover(m, m.list_partitions()[0]);
    for (int32_t h : handles) {
        REQUIRE(m.var(handle_to_var_id(h)).elements.empty());
    }
}

TEST_CASE("an AtMostOnce partition respects every minimum length", "[list][partition]") {
    Model m;
    std::vector<int32_t> handles;
    handles.reserve(3);
    for (int r = 0; r < 3; ++r) {
        handles.push_back(m.list_var(9, 2, 4, ListInit::Random, "r" + std::to_string(r)));
    }
    m.add_list_partition(handles, Cover::AtMostOnce);
    m.close();
    RNG rng(5);
    initialize_structured_random(m, rng);
    require_cover(m, m.list_partitions()[0]);
    for (int32_t h : handles) {
        REQUIRE(m.var(handle_to_var_id(h)).elements.size() >= 2);
    }
}

TEST_CASE("random move sequences keep an Exact partition well formed", "[list][partition]") {
    PartitionModel pm = build(/*universe=*/14, /*routes=*/4, /*min_len=*/0, /*max_len=*/14,
                              ListInit::Empty, /*with_partition=*/true, Cover::Exact);
    RNG init(3);
    initialize_structured_random(pm.model, init);
    full_evaluate(pm.model);
    hammer(pm, /*rounds=*/600, /*seed=*/91, /*with_partition=*/true, Cover::Exact);
}

TEST_CASE("random move sequences keep an AtMostOnce partition well formed", "[list][partition]") {
    PartitionModel pm = build(/*universe=*/14, /*routes=*/3, /*min_len=*/1, /*max_len=*/6,
                              ListInit::Random, /*with_partition=*/true, Cover::AtMostOnce);
    RNG init(3);
    initialize_structured_random(pm.model, init);
    full_evaluate(pm.model);
    hammer(pm, /*rounds=*/600, /*seed=*/92, /*with_partition=*/true, Cover::AtMostOnce);
}

TEST_CASE("random move sequences keep an unpartitioned variable-length List well formed",
          "[list][partition]") {
    // No partition: `list_insert` and `list_remove` are live here, which is the
    // path a scheduling-with-rejection model takes.
    PartitionModel pm = build(/*universe=*/12, /*routes=*/2, /*min_len=*/2, /*max_len=*/9,
                              ListInit::Random, /*with_partition=*/false, Cover::Exact);
    RNG init(3);
    initialize_structured_random(pm.model, init);
    full_evaluate(pm.model);
    hammer(pm, /*rounds=*/600, /*seed=*/93, /*with_partition=*/false, Cover::Exact);
}

TEST_CASE("a fixed-length ordered subset can still change its membership", "[list][partition]") {
    // `min_len == max_len < universe` is a fixed-count selection model -- pick k
    // of n sites and sequence them. Insert is refused at max_len and remove at
    // min_len, so without `list_exchange` the membership drawn at initialisation
    // would be the membership for the whole run and the search would silently
    // explore one of C(universe, k) equivalence classes.
    Model m;
    const int32_t h = m.list_var(9, 3, 3, ListInit::Random, "pick3");
    m.minimize(m.lambda_sum(h, element_weight));
    m.close();
    auto& var = m.var_mut(handle_to_var_id(h));
    var.elements = {0, 1, 2};

    RNG rng(77);
    bool saw_exchange = false;
    for (int trial = 0; trial < 50 && !saw_exchange; ++trial) {
        std::vector<Move> out;
        generate_standard_moves(var, rng, out, nullptr);
        for (const Move& mv : out) {
            REQUIRE(mv.move_type != "list_insert");
            REQUIRE(mv.move_type != "list_remove");
            if (mv.move_type == "list_exchange") {
                saw_exchange = true;
                const std::vector<int32_t> after = elements_after(mv.changes.front(), var.elements);
                REQUIRE(after.size() == 3);
                std::vector<int32_t> sorted = after;
                std::sort(sorted.begin(), sorted.end());
                REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
                REQUIRE(sorted.back() >= 3);  // an element from outside the initial set
            }
        }
    }
    REQUIRE(saw_exchange);
}

TEST_CASE("a permutation List still offers no insert or remove", "[list][partition]") {
    Model m;
    const int32_t h = m.list_var(5, "perm");
    m.close();
    RNG rng(1);
    const std::vector<Move> moves = generate_standard_moves(m.var(handle_to_var_id(h)), rng);
    for (const Move& mv : moves) {
        REQUIRE(mv.move_type != "list_insert");
        REQUIRE(mv.move_type != "list_remove");
        REQUIRE(mv.move_type != "list_exchange");
    }
    // ... and the draws it takes are unchanged: a second generator on a fresh
    // RNG of the same seed lands on the same state.
    RNG a(1);
    RNG b(1);
    std::vector<Move> out;
    generate_standard_moves(m.var(handle_to_var_id(h)), a, out, nullptr);
    for (int k = 0; k < 2; ++k) {
        static_cast<void>(b.integers(0, 5));
    }
    REQUIRE(a.integers(0, 1000) == b.integers(0, 1000));
}

TEST_CASE("a partition member offers no intra-list insert or remove", "[list][partition]") {
    Model m;
    const int32_t a = m.list_var(8, 0, 8, ListInit::Empty, "a");
    const int32_t b = m.list_var(8, 0, 8, ListInit::Empty, "b");
    m.add_list_partition({a, b}, Cover::AtMostOnce);
    m.close();
    RNG rng(1);
    auto& va = m.var_mut(handle_to_var_id(a));
    va.elements = {0, 1, 2};
    const std::vector<Move> moves = generate_standard_moves(va, rng);
    for (const Move& mv : moves) {
        REQUIRE(mv.move_type != "list_insert");
        REQUIRE(mv.move_type != "list_remove");
        REQUIRE(mv.move_type != "list_exchange");
    }
}

TEST_CASE("the partition generator appends at most one candidate", "[list][partition]") {
    // Not a budget: every candidate from one call is built against, and applied
    // to, the same assignment, and FirstImprovingSample may commit several of
    // them in turn. The batch's per-sample baseline is what makes that safe in
    // general; one candidate per call is the belt-and-braces choice for an
    // invariant maintained by construction.
    PartitionModel pm = build(/*universe=*/10, /*routes=*/3, /*min_len=*/0, /*max_len=*/10,
                              ListInit::Empty, /*with_partition=*/true, Cover::Exact);
    RNG init(2);
    initialize_structured_random(pm.model, init);
    RNG rng(17);
    for (int k = 0; k < 200; ++k) {
        std::vector<Move> out;
        generate_partition_moves(pm.model, 0, -1, rng, out, nullptr);
        REQUIRE(out.size() <= 1);
        for (const Move& mv : out) {
            REQUIRE(mv.changes.size() <= 2);
        }
    }
}

TEST_CASE("the partition generator is registered next to the per-variable ones",
          "[list][partition]") {
    PartitionModel pm = build(/*universe=*/6, /*routes=*/2, /*min_len=*/0, /*max_len=*/6,
                              ListInit::Empty, /*with_partition=*/true, Cover::Exact);
    const auto generators = default_move_generators(pm.model, nullptr);
    REQUIRE(generators.size() == 3);
    REQUIRE(generators[0]->name() == "builtin_list");
    REQUIRE(generators[1]->name() == "builtin_list");
    REQUIRE(generators[2]->name() == "builtin_list_partition");
    REQUIRE(generators[2]->scope().size() == 2);
    // A model with no partition keeps exactly the generator list it had before.
    PartitionModel plain = build(/*universe=*/6, /*routes=*/2, /*min_len=*/0, /*max_len=*/6,
                                 ListInit::Empty, /*with_partition=*/false, Cover::Exact);
    REQUIRE(default_move_generators(plain.model, nullptr).size() == 2);
}

TEST_CASE("the diversification kick fills an empty unpartitioned List", "[list][partition]") {
    // The partition case below goes through the kick's partition branch. An
    // unpartitioned variable-length List at n = 0 takes a different route --
    // list_insert_move via list_complement -- which no other test reaches,
    // because the property tests all start from a complete cover or from
    // min_len >= 1.
    Model m;
    const int32_t h = m.list_var(8, 0, 8, ListInit::Empty, "free");
    m.minimize(m.lambda_sum(h, element_weight));
    m.close();
    full_evaluate(m);
    REQUIRE(m.var(handle_to_var_id(h)).elements.empty());

    ViolationManager vm(m);
    RNG rng(17);
    FeasibilityJump fj(m, vm, rng, GFJConfig{});
    fj.begin(/*set_initial_x=*/false);
    fj.perturb(0.1);
    REQUIRE_FALSE(m.var(handle_to_var_id(h)).elements.empty());
}

TEST_CASE("the diversification kick moves an all-empty partition", "[list][partition]") {
    // structural_kick_size asks for one move on a zero-length list; the five
    // intra-list moves need two elements and `list_insert` is suppressed for a
    // partition member, so before #164 the kick found nothing and silently did
    // nothing -- #109/#111's defect by a new route.
    Model m;
    std::vector<int32_t> handles;
    handles.reserve(3);
    for (int r = 0; r < 3; ++r) {
        handles.push_back(m.list_var(9, 0, 9, ListInit::Empty, "r" + std::to_string(r)));
    }
    m.add_list_partition(handles, Cover::AtMostOnce);
    std::vector<int32_t> terms;
    terms.reserve(handles.size());
    for (int32_t h : handles) {
        terms.push_back(m.lambda_sum(h, element_weight));
    }
    m.minimize(m.sum(terms));
    m.close();
    full_evaluate(m);
    for (int32_t h : handles) {
        REQUIRE(m.var(handle_to_var_id(h)).elements.empty());
    }

    ViolationManager vm(m);
    RNG rng(31);
    FeasibilityJump fj(m, vm, rng, GFJConfig{});
    fj.begin(/*set_initial_x=*/false);
    fj.perturb(0.1);

    size_t total = 0;
    for (int32_t h : handles) {
        total += m.var(handle_to_var_id(h)).elements.size();
    }
    REQUIRE(total > 0);
    require_cover(m, m.list_partitions()[0]);
}

TEST_CASE("copy_state and restore_state round-trip lists of differing lengths",
          "[list][partition]") {
    PartitionModel pm = build(/*universe=*/10, /*routes=*/3, /*min_len=*/0, /*max_len=*/10,
                              ListInit::Empty, /*with_partition=*/true, Cover::Exact);
    RNG init(8);
    initialize_structured_random(pm.model, init);
    full_evaluate(pm.model);
    const Model::State saved = pm.model.copy_state();
    std::vector<std::vector<int32_t>> before;
    before.reserve(pm.ids.size());
    for (int32_t id : pm.ids) {
        before.push_back(pm.model.var(id).elements);
    }
    // Deliberately land on a set of lengths the snapshot does not have.
    hammer(pm, /*rounds=*/50, /*seed=*/77, /*with_partition=*/true, Cover::Exact);
    pm.model.restore_state(saved);
    full_evaluate(pm.model);
    for (size_t k = 0; k < pm.ids.size(); ++k) {
        REQUIRE(pm.model.var(pm.ids[k]).elements == before[k]);
    }
    require_incremental_matches_full(pm.model, pm.probes);
}

TEST_CASE("a permutation model's .cbls bytes are unchanged", "[list][partition][io]") {
    Model m;
    const int32_t lv = m.list_var(5, "perm");
    m.minimize(m.lambda_sum(lv, [](int e) { return e * 1.1; }));
    m.close();
    std::ostringstream out;
    save_model(m, out);
    // The variable record is `n` alone -- exactly what it was before #164, which
    // is what save(load(save)) == save rests on for every file that predates it.
    REQUIRE(out.str().find(R"({"n":5,"type":"List","var":"perm"})") != std::string::npos);
    REQUIRE(out.str().find("min_len") == std::string::npos);

    std::istringstream in(out.str());
    Model m2 = load_model(in);
    std::ostringstream out2;
    save_model(m2, out2);
    REQUIRE(out.str() == out2.str());
}

TEST_CASE("a variable-length List and its partition round-trip through .cbls",
          "[list][partition][io]") {
    Model m;
    const int32_t a = m.list_var(6, 1, 4, ListInit::Random, "a");
    const int32_t b = m.list_var(6, 0, 6, ListInit::Empty, "b");
    m.add_list_partition({a, b}, Cover::AtMostOnce);
    m.minimize(m.sum({m.lambda_sum(a, element_weight), m.pair_lambda_sum(b, pair_weight)}));
    m.close();

    std::ostringstream out;
    save_model(m, out);
    std::istringstream in(out.str());
    Model m2 = load_model(in);

    const Variable& va = m2.var(0);
    REQUIRE(va.universe_size == 6);
    REQUIRE(va.min_size == 1);
    REQUIRE(va.max_size == 4);
    REQUIRE(va.list_init == ListInit::Random);
    REQUIRE(va.partitioned);
    REQUIRE(m2.list_partitions().size() == 1);
    REQUIRE(m2.list_partitions()[0].cover == Cover::AtMostOnce);
    REQUIRE(m2.list_partitions()[0].list_ids == std::vector<int32_t>{0, 1});

    // The PairLambda matrix is tabulated over the universe, not over max_size:
    // sized by the latter, a 6-element universe in a length-6 list would be a
    // 6x6 matrix only by accident, and `b`'s elements would index past `a`'s 4x4.
    std::ostringstream out2;
    save_model(m2, out2);
    REQUIRE(out.str() == out2.str());
}

TEST_CASE("a partitioned model searches its way to a feasible cover", "[list][partition]") {
    // End to end through solve(): initialisation, the structural batch's
    // registered generators, the diversification kick and LNS destroy-repair all
    // touch these lists, and the cover has to survive every one of them.
    //
    // FEASIBILITY HERE IS THE PARTITION GENERATOR'S OWN WORK, which is what
    // makes this more than a "did not crash" probe. The model has no scalar
    // variable, so Feasibility Jump has nothing to jump, and every intra-list
    // move is length-preserving -- so `count(route) <= 5` cannot be satisfied
    // except by moving elements BETWEEN routes. `randomize_list_partition`
    // routinely deals one route more than five of the twelve elements, so the
    // run starts infeasible and only the inter-list moves can fix it.
    Model m;
    std::vector<int32_t> handles;
    handles.reserve(3);
    for (int r = 0; r < 3; ++r) {
        handles.push_back(m.list_var(12, 0, 12, ListInit::Empty, "r" + std::to_string(r)));
    }
    m.add_list_partition(handles, Cover::Exact);
    std::vector<int32_t> terms;
    terms.reserve(handles.size());
    for (int32_t h : handles) {
        terms.push_back(m.pair_lambda_sum(h, pair_weight));
        m.add_constraint(m.leq(m.count(h), m.constant(5)));
    }
    m.minimize(m.sum(terms));
    m.close();

    const SearchResult r = solve_deterministic(m, /*max_iterations=*/20000);
    REQUIRE(r.feasible);
    m.restore_state(r.best_state);
    require_cover(m, m.list_partitions()[0]);
    for (int32_t h : handles) {
        REQUIRE(m.var(handle_to_var_id(h)).elements.size() <= 5);
    }
}

TEST_CASE("granular guidance survives a universe wider than the list", "[list][partition]") {
    // `pick_second_position` built its element -> position index sized by the
    // LIST LENGTH, which coincides with the universe on a permutation and on
    // nothing else. On a variable-length List every element id at or past the
    // length was dropped from the index, so the neighbour list named a position
    // the function could not find and it fell back to the uniform draw -- the
    // granular neighbourhood silently switched off for most of the universe.
    //
    // Here every element's only neighbour is 19, which sits at position 9, and
    // 19's is 10 at position 0. So every guided swap must involve position 9.
    // Under the uniform fallback it would involve it about one time in nine.
    constexpr int kUniverse = 30;
    Model m;
    const int32_t h = m.list_var(kUniverse, 0, kUniverse, ListInit::Empty, "seq");
    m.minimize(m.lambda_sum(h, element_weight));
    m.close();
    auto& var = m.var_mut(handle_to_var_id(h));
    var.elements = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    std::vector<std::vector<int32_t>> rows(kUniverse);
    for (int e = 0; e < kUniverse; ++e) {
        rows[static_cast<size_t>(e)] = {e == 19 ? 10 : 19};
    }
    const NeighbourList neighbours(rows);

    RNG rng(2024);
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<Move> out;
        generate_standard_moves(var, rng, out, &neighbours);
        REQUIRE_FALSE(out.empty());
        REQUIRE(out[0].move_type == "list_swap");
        const std::vector<int32_t> after = elements_after(out[0].changes.front(), var.elements);
        REQUIRE(after.size() == var.elements.size());
        REQUIRE(after[9] != var.elements[9]);
    }
}

TEST_CASE("a frozen model's replicas keep the partition and its cover", "[list][partition]") {
    // A portfolio worker gets a COPY of the model, which shares the frozen
    // `ModelStructure` (#157) and owns its own variables. The partition lives in
    // the shared half and `Variable::partitioned` in the per-worker half, so
    // this is the one place the two could drift apart.
    Model m;
    std::vector<int32_t> handles;
    handles.reserve(3);
    for (int r = 0; r < 3; ++r) {
        handles.push_back(m.list_var(12, 0, 12, ListInit::Empty, "r" + std::to_string(r)));
    }
    m.add_list_partition(handles, Cover::Exact);
    std::vector<int32_t> terms;
    terms.reserve(handles.size());
    for (int32_t h : handles) {
        terms.push_back(m.pair_lambda_sum(h, pair_weight));
        m.add_constraint(m.leq(m.count(h), m.constant(5)));
    }
    m.minimize(m.sum(terms));
    m.close();
    m.freeze();
    REQUIRE(m.is_frozen());

    Model replica = m;
    REQUIRE(replica.is_frozen());
    REQUIRE(replica.list_partitions().size() == 1);
    REQUIRE(replica.partition_of_list(handle_to_var_id(handles[1])) == 0);
    REQUIRE(replica.var(handle_to_var_id(handles[1])).partitioned);
    REQUIRE(default_move_generators(replica, nullptr).size() == 4);

    const SearchResult r = solve_deterministic(replica, /*max_iterations=*/20000);
    REQUIRE(r.feasible);
    replica.restore_state(r.best_state);
    require_cover(replica, replica.list_partitions()[0]);
}

TEST_CASE("a malformed .cbls partition record is refused", "[list][partition][io]") {
    // Node and variable names live in one map, and `add_list_partition` reads a
    // non-negative handle as a raw variable id -- so a record naming a node
    // would silently partition whichever variable carries that id.
    const std::string text =
        R"({"n":4,"type":"List","var":"a","min_len":0,"max_len":4,"init":"empty"})"
        "\n"
        R"({"node":"c0","op":"Const","value":1.0})"
        "\n"
        R"({"partition":["c0"],"cover":"exact"})"
        "\n";
    std::istringstream in(text);
    REQUIRE_THROWS_AS(load_model(in), std::invalid_argument);

    std::istringstream unknown_cover(
        R"({"n":4,"type":"List","var":"a","min_len":0,"max_len":4,"init":"empty"})"
        "\n"
        R"({"partition":["a"],"cover":"sometimes"})"
        "\n");
    REQUIRE_THROWS_AS(load_model(unknown_cover), std::invalid_argument);

    std::istringstream unknown_init(
        R"({"n":4,"type":"List","var":"a","min_len":0,"max_len":4,"init":"whatever"})"
        "\n");
    REQUIRE_THROWS_AS(load_model(unknown_init), std::invalid_argument);

    // A partition the model cannot satisfy is refused at load, not accepted and
    // then searched: `add_list_partition` is the one place the rule lives.
    std::istringstream impossible(
        R"({"n":9,"type":"List","var":"a","min_len":0,"max_len":2,"init":"empty"})"
        "\n"
        R"({"n":9,"type":"List","var":"b","min_len":0,"max_len":2,"init":"empty"})"
        "\n"
        R"({"partition":["a","b"],"cover":"exact"})"
        "\n");
    REQUIRE_THROWS_AS(load_model(impossible), std::invalid_argument);
}

TEST_CASE("the degenerate partitions are legal and inert", "[list][partition]") {
    SECTION("a single-list Exact partition offers no inter-list move") {
        // Legal -- one list that must hold everything -- and by construction
        // there is nowhere to move an element to. The generator is still
        // registered; it simply never yields a candidate.
        Model m;
        const int32_t only = m.list_var(5, 5, 5, ListInit::Random, "only");
        m.add_list_partition({only}, Cover::Exact);
        m.minimize(m.lambda_sum(only, element_weight));
        m.close();
        RNG init(3);
        initialize_structured_random(m, init);
        require_cover(m, m.list_partitions()[0]);

        RNG rng(9);
        std::vector<Move> out;
        for (int k = 0; k < 20; ++k) {
            generate_partition_moves(m, 0, /*anchor=*/-1, rng, out, nullptr);
        }
        REQUIRE(out.empty());
    }
    SECTION("a zero-size universe is a partition over nothing") {
        Model m;
        const int32_t a = m.list_var(0, 0, 0, ListInit::Empty, "a");
        const int32_t b = m.list_var(0, 0, 0, ListInit::Empty, "b");
        m.add_list_partition({a, b}, Cover::Exact);
        m.minimize(m.sum({m.count(a), m.count(b)}));
        m.close();
        RNG init(3);
        initialize_structured_random(m, init);
        REQUIRE(m.var(handle_to_var_id(a)).elements.empty());
        require_cover(m, m.list_partitions()[0]);

        RNG rng(9);
        std::vector<Move> out;
        for (int k = 0; k < 20; ++k) {
            generate_partition_moves(m, 0, /*anchor=*/-1, rng, out, nullptr);
        }
        REQUIRE(out.empty());
        // And a partition index nobody has is a no-op rather than a read past
        // the vector -- `Model.partition_of_list` is reachable from Python.
        generate_partition_moves(m, 7, -1, rng, out, nullptr);
        generate_partition_moves(m, -1, -1, rng, out, nullptr);
        REQUIRE(out.empty());
    }
}
