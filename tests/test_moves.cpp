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
        auto new_elems = elements_after(mv.changes[0], v.elements);
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
    move.changes.push_back(scalar_change(vid(x), 8.0));

    auto saved = save_move_values(m, move);
    REQUIRE(saved.values[0] == 5.0);

    apply_move(m, move);
    REQUIRE(m.var(vid(x)).value == 8.0);

    undo_move(m, move, saved);
    REQUIRE(m.var(vid(x)).value == 5.0);
}

// ---------------------------------------------------------------------------
// The positional edit representation (#164)
// ---------------------------------------------------------------------------

TEST_CASE("every edit kind rewrites the elements it names", "[moves][edits]") {
    const std::vector<int32_t> base = {10, 11, 12, 13, 14};

    SECTION("Swap exchanges two positions") {
        REQUIRE(elements_after(edit_change(0, swap_edit(0, 3)), base) ==
                std::vector<int32_t>{13, 11, 12, 10, 14});
    }
    SECTION("Reverse is inclusive at both ends") {
        REQUIRE(elements_after(edit_change(0, reverse_edit(1, 3)), base) ==
                std::vector<int32_t>{10, 13, 12, 11, 14});
    }
    SECTION("MoveSegment moves a run to the left") {
        REQUIRE(elements_after(edit_change(0, segment_edit(3, 2, 1)), base) ==
                std::vector<int32_t>{10, 13, 14, 11, 12});
    }
    SECTION("MoveSegment moves a run to the right, to a position of the SHORTENED vector") {
        // Erase [11, 12] and reinsert at index 2 of {10, 13, 14}.
        REQUIRE(elements_after(edit_change(0, segment_edit(1, 2, 2)), base) ==
                std::vector<int32_t>{10, 13, 11, 12, 14});
    }
    SECTION("Insert and Erase change the length") {
        REQUIRE(elements_after(edit_change(0, insert_edit(2, 99)), base) ==
                std::vector<int32_t>{10, 11, 99, 12, 13, 14});
        REQUIRE(elements_after(edit_change(0, erase_edit(0)), base) ==
                std::vector<int32_t>{11, 12, 13, 14});
    }
    SECTION("Assign overwrites one position") {
        REQUIRE(elements_after(edit_change(0, assign_edit(4, 7)), base) ==
                std::vector<int32_t>{10, 11, 12, 13, 7});
    }
    SECTION("Two edits run in order -- the set_swap shape") {
        const Move::Change change = edit_change(0, erase_edit(1), insert_edit(4, 42));
        REQUIRE(elements_after(change, base) == std::vector<int32_t>{10, 12, 13, 14, 42});
    }
    SECTION("Replace carries the whole vector") {
        REQUIRE(elements_after(replace_change(0, {1, 2}), base) == std::vector<int32_t>{1, 2});
    }
}

TEST_CASE("an edit whose positions do not fit is ignored, not indexed", "[moves][edits]") {
    // `Variable.elements` is writable from Python, so an edit built against a
    // longer vector can be replayed against a shorter one with nothing in the
    // way (#156). The result is a move that does nothing, never a heap read.
    const std::vector<int32_t> tiny = {1, 2};
    REQUIRE(elements_after(edit_change(0, swap_edit(0, 9)), tiny) == tiny);
    REQUIRE(elements_after(edit_change(0, reverse_edit(0, 9)), tiny) == tiny);
    REQUIRE(elements_after(edit_change(0, segment_edit(1, 4, 0)), tiny) == tiny);
    REQUIRE(elements_after(edit_change(0, erase_edit(7)), tiny) == tiny);
    REQUIRE(elements_after(edit_change(0, assign_edit(7, 3)), tiny) == tiny);
    REQUIRE(elements_after(edit_change(0, insert_edit(9, 3)), tiny) == tiny);
    REQUIRE(elements_after(Move::Change{}, tiny) == tiny);
}

TEST_CASE("change_is_noop agrees with materialising the change", "[moves][edits]") {
    // The predicate exists so the no-op filter never builds the vector it is
    // asking about. It has to give the same answer as building it would.
    const std::vector<int32_t> base = {10, 11, 12, 13, 14};
    const std::vector<Move::Change> changes = {
        edit_change(0, swap_edit(2, 2)),
        edit_change(0, swap_edit(1, 4)),
        edit_change(0, reverse_edit(2, 2)),
        edit_change(0, reverse_edit(0, 4)),
        edit_change(0, segment_edit(1, 2, 1)),
        edit_change(0, segment_edit(1, 2, 3)),
        edit_change(0, assign_edit(3, 13)),
        edit_change(0, assign_edit(3, 99)),
        edit_change(0, insert_edit(0, 99)),
        edit_change(0, erase_edit(0)),
        edit_change(0, erase_edit(1), insert_edit(4, 42)),
        replace_change(0, {10, 11, 12, 13, 14}),
        replace_change(0, {1}),
        Move::Change{},
    };
    for (const Move::Change& change : changes) {
        REQUIRE(change_is_noop(change, base) == (elements_after(change, base) == base));
    }
}

TEST_CASE("apply_move edits the vector already in the variable", "[moves][edits]") {
    Model m;
    const auto lv = m.list_var(6, "perm");
    m.minimize(m.lambda_sum(lv, [](int e) { return static_cast<double>(e); }));
    m.close();
    auto& var = m.var_mut(vid(lv));
    var.elements = {0, 1, 2, 3, 4, 5};
    const int32_t* data_before = var.elements.data();

    Move move;
    move.move_type = "test";
    move.changes.push_back(edit_change(vid(lv), reverse_edit(1, 4)));
    const SavedValues saved = save_move_values(m, move);
    apply_move(m, move);
    REQUIRE(m.var(vid(lv)).elements == std::vector<int32_t>{0, 4, 3, 2, 1, 5});
    // The point of the representation: no reallocation for a length-preserving
    // edit, so the candidate cost is the edit rather than the vector.
    REQUIRE(m.var(vid(lv)).elements.data() == data_before);

    undo_move(m, move, saved);
    REQUIRE(m.var(vid(lv)).elements == std::vector<int32_t>{0, 1, 2, 3, 4, 5});
}
