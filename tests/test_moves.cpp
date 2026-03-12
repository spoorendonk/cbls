#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

using namespace cbls;

static int32_t vid(int32_t handle) { return -(handle + 1); }

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

    bool has_dec = false, has_inc = false;
    for (const auto& mv : moves) {
        if (mv.changes[0].new_value == 4.0) has_dec = true;
        if (mv.changes[0].new_value == 6.0) has_inc = true;
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
    REQUIRE(moves.size() == 2);  // swap + 2-opt
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

TEST_CASE("Newton tight move on linear constraint", "[moves]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    auto constraint = m.sum({x, neg5});  // x - 5 <= 0
    m.add_constraint(constraint);
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    full_evaluate(m);
    REQUIRE(m.node(constraint).value == 3.0);

    auto moves = newton_tight_move(vid(x), m, 0);
    REQUIRE(moves.size() == 1);
    REQUIRE(std::abs(moves[0].changes[0].new_value - 5.0) < 1e-10);
}

TEST_CASE("Gradient lift move", "[moves]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto two = m.constant(2);
    m.minimize(m.pow_expr(x, two));
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);

    auto moves = gradient_lift_move(vid(x), m, 0.1);
    REQUIRE(moves.size() == 1);
    REQUIRE(moves[0].changes[0].new_value < 3.0);
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

TEST_CASE("MoveProbabilities initial uniform", "[moves]") {
    MoveProbabilities mp({"a", "b", "c"});
    const auto& probs = mp.probabilities();
    REQUIRE(probs.size() == 3);
    for (double p : probs) {
        REQUIRE(std::abs(p - 1.0 / 3.0) < 1e-10);
    }
}

TEST_CASE("MoveProbabilities rebalance", "[moves]") {
    MoveProbabilities mp({"a", "b"});
    // Force faster rebalance
    for (int i = 0; i < 500; ++i) {
        mp.update("a", true);
    }
    for (int i = 0; i < 500; ++i) {
        mp.update("b", false);
    }
    // After 1000 updates, rebalance triggers
    const auto& probs = mp.probabilities();
    REQUIRE(probs[0] > probs[1]);
}

TEST_CASE("MoveProbabilities floor", "[moves]") {
    MoveProbabilities mp({"a", "b", "c"});
    for (int i = 0; i < 334; ++i) mp.update("a", true);
    for (int i = 0; i < 333; ++i) mp.update("b", false);
    for (int i = 0; i < 333; ++i) mp.update("c", false);
    const auto& probs = mp.probabilities();
    for (double p : probs) {
        REQUIRE(p >= 0.049);
    }
}
