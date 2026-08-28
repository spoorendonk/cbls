#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

using namespace cbls;

TEST_CASE("Bool flip move", "[moves]") {
    Model m;
    auto x = m.bool_var();
    m.minimize(m.sum({x}));
    m.close();
    RNG rng(42);

    m.var_mut(vid(x)).value = 0.0;
    auto moves = generate_standard_moves(m.var(vid(x)), rng);
    REQUIRE(moves.size() == 1);
    REQUIRE(moves[0].changes[0].new_value == 1.0);

    m.var_mut(vid(x)).value = 1.0;
    moves = generate_standard_moves(m.var(vid(x)), rng);
    REQUIRE(moves[0].changes[0].new_value == 0.0);
}

TEST_CASE("Int moves", "[moves]") {
    Model m;
    auto x = m.int_var(0, 10);
    m.minimize(m.sum({x}));
    m.close();
    RNG rng(42);

    m.var_mut(vid(x)).value = 5.0;
    auto moves = generate_standard_moves(m.var(vid(x)), rng);
    REQUIRE(moves.size() == 3);  // dec, inc, random

    bool has_dec = false;
    bool has_inc = false;
    for (const auto& mv : moves) {
        if (mv.changes[0].new_value == 4.0) {
            has_dec = true;
        }
        if (mv.changes[0].new_value == 6.0) {
            has_inc = true;
        }
    }
    REQUIRE(has_dec);
    REQUIRE(has_inc);
}

TEST_CASE("Int at bounds", "[moves]") {
    Model m;
    auto x = m.int_var(0, 10);
    m.minimize(m.sum({x}));
    m.close();
    RNG rng(42);

    m.var_mut(vid(x)).value = 0.0;
    auto moves = generate_standard_moves(m.var(vid(x)), rng);
    for (const auto& mv : moves) {
        REQUIRE(mv.changes[0].new_value >= 0.0);
    }
}

TEST_CASE("Float perturb", "[moves]") {
    Model m;
    auto x = m.float_var(0, 10);
    m.minimize(m.sum({x}));
    m.close();
    RNG rng(42);

    m.var_mut(vid(x)).value = 5.0;
    auto moves = generate_standard_moves(m.var(vid(x)), rng);
    REQUIRE(moves.size() == 1);
    REQUIRE(moves[0].changes[0].new_value >= 0.0);
    REQUIRE(moves[0].changes[0].new_value <= 10.0);
}

TEST_CASE("List moves", "[moves]") {
    Model m;
    auto lv = m.list_var(5);
    m.minimize(m.lambda_sum(lv, [](int e) { return static_cast<double>(e); }));
    m.close();
    RNG rng(42);

    auto& v = m.var_mut(vid(lv));
    v.elements = {0, 1, 2, 3, 4};
    auto moves = generate_standard_moves(m.var(vid(lv)), rng);
    REQUIRE(moves.size() >= 2);  // swap + 2-opt + relocate + or_opt
    for (const auto& mv : moves) {
        auto new_elems = mv.changes[0].new_elements;
        auto sorted = new_elems;
        std::sort(sorted.begin(), sorted.end());
        REQUIRE(sorted == std::vector<int32_t>{0, 1, 2, 3, 4});
    }
}

TEST_CASE("Set moves", "[moves]") {
    Model m;
    auto sv = m.set_var(5, 1, 4);
    m.minimize(m.count(sv));
    m.close();
    RNG rng(42);

    auto& v = m.var_mut(vid(sv));
    v.elements = {0, 1, 2};
    auto moves = generate_standard_moves(m.var(vid(sv)), rng);
    REQUIRE(moves.size() >= 2);
}

TEST_CASE("Apply and undo move", "[moves]") {
    Model m;
    auto x = m.float_var(0, 10);
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 5.0;

    Move move;
    move.move_type = "test";
    move.changes.push_back({vid(x), 8.0, {}});

    auto saved = save_move_values(m, move);
    REQUIRE(saved.values[0] == 5.0);

    apply_move(m, move);
    REQUIRE(m.var(vid(x)).value == 8.0);

    undo_move(m, move, saved);
    REQUIRE(m.var(vid(x)).value == 5.0);
}
