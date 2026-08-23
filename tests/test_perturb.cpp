// Regression tests for #109: a diversification kick must always move
// something. perturb() used to randomise each variable independently, so at the
// default perturbation_probability = 0.1 a kick changed nothing at all with
// probability (1-p)^n — 90% on a one-variable model, 81% on two. The tests
// below run many seeds rather than one lucky draw, so a reintroduced
// no-op would fail deterministically rather than flake.
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

// The default from SearchConfig::perturbation_probability — the setting the
// issue is about.
constexpr double kDefaultP = 0.1;
constexpr uint64_t kSeeds = 200;

std::vector<double> assignment(const Model& m) {
    std::vector<double> values;
    values.reserve(m.num_vars());
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        values.push_back(m.var(v).value);
    }
    return values;
}

int num_changed(const std::vector<double>& before, const std::vector<double>& after) {
    int n = 0;
    for (size_t i = 0; i < before.size(); ++i) {
        n += (before[i] != after[i]) ? 1 : 0;
    }
    return n;
}

std::vector<std::vector<int32_t>> all_elements(const Model& m) {
    std::vector<std::vector<int32_t>> elements;
    elements.reserve(m.num_vars());
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        elements.push_back(m.var(v).elements);
    }
    return elements;
}

// Wraps the variables from `make_vars` in a constraint and an objective, so a
// kick exercises the full re-evaluation path (full_evaluate + reset_weights)
// the search sees, not just the assignment write.
template <class MakeVars>
void build(Model& m, MakeVars&& make_vars) {
    std::vector<int32_t> handles = make_vars(m);
    m.add_constraint(m.leq(m.sum(handles), m.constant(1.0)));
    m.minimize(m.sum(handles));
    m.close();
}

// Fewest variables a kick moved across `kicks` kicks on each of `kSeeds` seeds.
// 0 means some kick was a no-op — the #109 bug.
template <class MakeVars>
int min_vars_moved_per_kick(MakeVars&& make_vars, double probability, int kicks = 4) {
    int fewest = -1;
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        Model m;
        build(m, make_vars);
        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        for (int k = 0; k < kicks; ++k) {
            std::vector<double> before = assignment(m);
            fj.perturb(probability);
            int moved = num_changed(before, assignment(m));
            if (fewest < 0 || moved < fewest) {
                fewest = moved;
            }
        }
    }
    return fewest;
}

}  // namespace

TEST_CASE("perturb always moves a one-variable model", "[fj][perturb]") {
    // The headline case: at p = 0.1 the old code left a one-variable model
    // untouched 9 kicks in 10. Bool is the sharpest check — resampling a Bool
    // uniformly redraws its current value half the time, so "force one variable"
    // is only enough if the forced value is drawn to DIFFER.
    REQUIRE(min_vars_moved_per_kick([](Model& m) { return std::vector<int32_t>{m.bool_var()}; },
                                    kDefaultP) == 1);
    REQUIRE(min_vars_moved_per_kick([](Model& m) { return std::vector<int32_t>{m.int_var(0, 5)}; },
                                    kDefaultP) == 1);
    REQUIRE(min_vars_moved_per_kick(
                [](Model& m) { return std::vector<int32_t>{m.float_var(0.0, 10.0)}; },
                kDefaultP) == 1);
}

TEST_CASE("perturb always moves a two-variable model", "[fj][perturb]") {
    // P(no-op) was 81% here at the default probability.
    REQUIRE(min_vars_moved_per_kick(
                [](Model& m) {
                    return std::vector<int32_t>{m.bool_var(), m.bool_var()};
                },
                kDefaultP) >= 1);
    REQUIRE(min_vars_moved_per_kick(
                [](Model& m) {
                    return std::vector<int32_t>{m.int_var(0, 5), m.int_var(-3, 3)};
                },
                kDefaultP) >= 1);
    REQUIRE(min_vars_moved_per_kick(
                [](Model& m) {
                    return std::vector<int32_t>{m.float_var(0.0, 10.0), m.float_var(-1.0, 1.0)};
                },
                kDefaultP) >= 1);
    REQUIRE(min_vars_moved_per_kick(
                [](Model& m) {
                    return std::vector<int32_t>{m.bool_var(), m.float_var(0.0, 10.0)};
                },
                kDefaultP) >= 1);
}

TEST_CASE("perturb moves exactly one variable at probability zero", "[fj][perturb]") {
    // p = 0 isolates the guarantee from the per-variable draws: exactly one
    // variable moves, never zero (the bug) and never more (which would mean the
    // forced move had leaked into the configured density).
    auto make = [](Model& m) {
        std::vector<int32_t> h;
        for (int i = 0; i < 20; ++i) {
            h.push_back(m.float_var(0.0, 10.0));
        }
        return h;
    };
    Model m;
    build(m, make);
    ViolationManager vm(m);
    RNG rng(7);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(/*set_initial_x=*/true);
    for (int k = 0; k < 200; ++k) {
        std::vector<double> before = assignment(m);
        fj.perturb(0.0);
        REQUIRE(num_changed(before, assignment(m)) == 1);
    }
}

TEST_CASE("perturb keeps the configured density on a large model", "[fj][perturb]") {
    // The forced move must not change the regime on models big enough for the
    // per-variable probability to work: the moved count stays ~p*n, not 1 and
    // not n. Bounds are wide (p*n = 40, sd ~ 6 per kick, ~0.9 over 200 kicks) so
    // this cannot flake, while still catching "nothing moves"/"everything moves".
    const int n = 400;
    auto make = [n](Model& m) {
        std::vector<int32_t> h;
        for (int i = 0; i < n; ++i) {
            h.push_back(m.float_var(0.0, 10.0));
        }
        return h;
    };
    Model m;
    build(m, make);
    ViolationManager vm(m);
    RNG rng(11);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(/*set_initial_x=*/true);

    const int kicks = 200;
    long total_moved = 0;
    for (int k = 0; k < kicks; ++k) {
        std::vector<double> before = assignment(m);
        fj.perturb(kDefaultP);
        int moved = num_changed(before, assignment(m));
        REQUIRE(moved >= 1);  // the guarantee still holds
        total_moved += moved;
    }
    const double mean_fraction = static_cast<double>(total_moved) / (kicks * n);
    REQUIRE(mean_fraction > 0.07);
    REQUIRE(mean_fraction < 0.14);
}

TEST_CASE("perturb changes nothing when no variable can move", "[fj][perturb]") {
    SECTION("the only structures cannot move") {
        // A one-element List has no second position to permute, and a Set whose
        // min_size, max_size and universe all coincide has no legal add, remove
        // or swap. Neither is jumpable either, so the kick has nothing to move
        // anywhere: it must return quietly rather than spin or crash.
        Model m;
        m.list_var(1);
        int32_t sv = m.set_var(3, /*min_size=*/3, /*max_size=*/3);
        m.var_mut(handle_to_var_id(sv)).elements = {0, 1, 2};
        m.close();
        ViolationManager vm(m);
        RNG rng(3);
        FeasibilityJump fj(m, vm, rng);
        std::vector<std::vector<int32_t>> elements_before = all_elements(m);
        std::vector<double> before = assignment(m);
        for (int k = 0; k < 10; ++k) {
            fj.perturb(kDefaultP);
        }
        REQUIRE(assignment(m) == before);
        REQUIRE(all_elements(m) == elements_before);
    }

    SECTION("every jumpable variable is pinned") {
        // lb == ub: jumpable by type, but no other value exists to move to.
        Model m;
        build(m, [](Model& mm) {
            return std::vector<int32_t>{mm.int_var(3, 3), mm.float_var(1.5, 1.5)};
        });
        ViolationManager vm(m);
        RNG rng(3);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        std::vector<double> before = assignment(m);
        for (int k = 0; k < 50; ++k) {
            fj.perturb(1.0);  // even at p = 1 a pinned domain cannot move
            REQUIRE(assignment(m) == before);
        }
    }
}

TEST_CASE("perturb's forced move skips pinned variables", "[fj][perturb]") {
    // A pinned variable must not absorb the forced move and turn the kick back
    // into a no-op — the free variable has to be the one chosen.
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        Model m;
        build(m, [](Model& mm) {
            return std::vector<int32_t>{mm.int_var(3, 3), mm.float_var(0.0, 10.0),
                                        mm.int_var(-2, -2)};
        });
        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        std::vector<double> before = assignment(m);
        fj.perturb(0.0);
        std::vector<double> after = assignment(m);
        REQUIRE(after[0] == before[0]);  // pinned
        REQUIRE(after[2] == before[2]);  // pinned
        REQUIRE(after[1] != before[1]);  // the only variable that can move
    }
}

TEST_CASE("perturb's forced integer move covers the domain minus the current value",
          "[fj][perturb]") {
    // The forced integer draw samples the domain with the current value removed
    // and steps over the hole. A mis-stepped hole would either redraw the
    // current value or never reach an endpoint, so pin the value, kick, and look
    // at what comes out.
    Model m;
    build(m, [](Model& mm) { return std::vector<int32_t>{mm.int_var(0, 3)}; });
    ViolationManager vm(m);
    RNG rng(23);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(/*set_initial_x=*/true);

    for (int current = 0; current <= 3; ++current) {
        std::set<double> seen;
        for (int k = 0; k < 400; ++k) {
            m.var_mut(0).value = static_cast<double>(current);
            fj.perturb(0.0);
            const double drawn = m.var(0).value;
            REQUIRE(drawn != static_cast<double>(current));
            REQUIRE(drawn >= 0.0);
            REQUIRE(drawn <= 3.0);
            seen.insert(drawn);
        }
        REQUIRE(seen.size() == 3);  // all three other domain values are reachable
    }
}

// --- #111: the structural half of the kick ---------------------------------
// #109's guarantee reached only the variables Feasibility Jump can jump, and
// jumpability is scalar-only. On a model whose decisions live in List/Set
// variables — pharma-glsp's campaign scheduling is the benchmark that cares — a
// kick randomised nothing at all, burned the stagnation counter and let the
// search resume exactly where it was. The tests below run many seeds rather than
// one lucky draw, for the same reason the scalar ones above do.

namespace {

constexpr int kStructuralKicks = 4;

// Order-dependent constraint per List, so a kick runs through the same
// full_evaluate path the search does rather than just the element write. The
// constraint is unsatisfiable on purpose — nothing here should depend on the
// model being solvable.
void add_list_vars(Model& m, int num_lists, int n) {
    for (int i = 0; i < num_lists; ++i) {
        int32_t lv = m.list_var(n);
        auto len = m.pair_lambda_sum(lv, [](int a, int b) { return 1.0 + 0.5 * std::abs(a - b); });
        m.add_constraint(m.leq(len, m.constant(0.5)));
    }
}

void add_set_vars(Model& m, int num_sets, int universe, int min_size, int max_size) {
    for (int i = 0; i < num_sets; ++i) {
        int32_t sv = m.set_var(universe, min_size, max_size);
        m.add_constraint(m.leq(m.count(sv), m.constant(static_cast<double>(min_size))));
    }
}

bool is_permutation_of(std::vector<int32_t> a, std::vector<int32_t> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

// Cardinality and domain invariants a Set must satisfy after any move.
bool is_valid_set(const Variable& var) {
    const std::set<int32_t> distinct(var.elements.begin(), var.elements.end());
    if (distinct.size() != var.elements.size()) {
        return false;  // an element added twice
    }
    for (int32_t e : var.elements) {
        if (e < 0 || e >= var.universe_size) {
            return false;
        }
    }
    const int32_t size = static_cast<int32_t>(var.elements.size());
    return size >= var.min_size && size <= var.max_size;
}

// Fraction of the list's adjacent (unordered) element pairs a kick left intact.
// This is the metric in which the structural move generators are local — a 2-opt
// reversal rewrites a whole sub-range of POSITIONS but breaks only two adjacent
// pairs — so it is also the metric in which "a kick must not become a full
// restart" means something.
double kept_adjacency_fraction(const std::vector<int32_t>& before,
                               const std::vector<int32_t>& after) {
    std::set<std::pair<int32_t, int32_t>> pairs;
    for (size_t i = 0; i + 1 < before.size(); ++i) {
        pairs.emplace(std::min(before[i], before[i + 1]), std::max(before[i], before[i + 1]));
    }
    int kept = 0;
    for (size_t i = 0; i + 1 < after.size(); ++i) {
        kept += pairs.count({std::min(after[i], after[i + 1]), std::max(after[i], after[i + 1])});
    }
    if (after.size() < 2) {
        return 1.0;  // nothing to break
    }
    return static_cast<double>(kept) / static_cast<double>(after.size() - 1);
}

}  // namespace

TEST_CASE("perturb always moves a List-only model", "[fj][perturb][structural]") {
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        Model m;
        // n = 20 puts k = round(0.1 * 20) = 2 moves per kick, so the sweep
        // exercises a multi-move run rather than the trivial single move.
        add_list_vars(m, /*num_lists=*/2, /*n=*/20);
        m.close();
        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        const std::vector<std::vector<int32_t>> original = all_elements(m);
        for (int k = 0; k < kStructuralKicks; ++k) {
            const std::vector<std::vector<int32_t>> before = all_elements(m);
            fj.perturb(kDefaultP);
            const std::vector<std::vector<int32_t>> after = all_elements(m);
            REQUIRE(after != before);  // the kick a List-only model used to not get
            for (size_t v = 0; v < after.size(); ++v) {
                REQUIRE(is_permutation_of(after[v], original[v]));
            }
        }
    }
}

TEST_CASE("perturb always moves a Set-only model", "[fj][perturb][structural]") {
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        Model m;
        // Sized so k >= 2, as for the List sweep above.
        add_set_vars(m, /*num_sets=*/2, /*universe=*/60, /*min_size=*/15, /*max_size=*/40);
        m.close();
        RNG rng(seed);
        initialize_structured_random(m, rng);  // Sets start empty until initialised
        ViolationManager vm(m);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        for (int k = 0; k < kStructuralKicks; ++k) {
            const std::vector<std::vector<int32_t>> before = all_elements(m);
            fj.perturb(kDefaultP);
            REQUIRE(all_elements(m) != before);
            for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
                REQUIRE(is_valid_set(m.var(v)));  // min_size/max_size and the universe hold
            }
        }
    }
}

TEST_CASE("perturb moves scalars and structures on a mixed model", "[fj][perturb][structural]") {
    int scalar_kicks = 0;
    int structural_kicks = 0;
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        Model m;
        std::vector<int32_t> scalars{m.bool_var(), m.int_var(0, 9), m.float_var(0.0, 10.0)};
        m.add_constraint(m.leq(m.sum(scalars), m.constant(1.0)));
        m.minimize(m.sum(scalars));
        add_list_vars(m, /*num_lists=*/1, /*n=*/8);
        add_set_vars(m, /*num_sets=*/1, /*universe=*/12, /*min_size=*/3, /*max_size=*/7);
        m.close();
        RNG rng(seed);
        initialize_structured_random(m, rng);
        ViolationManager vm(m);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        for (int k = 0; k < kStructuralKicks; ++k) {
            const std::vector<double> values_before = assignment(m);
            const std::vector<std::vector<int32_t>> elements_before = all_elements(m);
            fj.perturb(kDefaultP);
            const bool scalar_moved = num_changed(values_before, assignment(m)) > 0;
            const bool structure_moved = all_elements(m) != elements_before;
            REQUIRE((scalar_moved || structure_moved));  // never a no-op
            scalar_kicks += scalar_moved ? 1 : 0;
            structural_kicks += structure_moved ? 1 : 0;
        }
    }
    // Both kinds move. The structures move on every kick (they are scaled, not
    // sampled); the scalars move at the configured rate, so only on some.
    REQUIRE(structural_kicks == static_cast<int>(kSeeds) * kStructuralKicks);
    REQUIRE(scalar_kicks > 0);
    REQUIRE(scalar_kicks < static_cast<int>(kSeeds) * kStructuralKicks);
}

TEST_CASE("perturb's structural kick size scales with the probability",
          "[fj][perturb][structural]") {
    // k = round(p * |elements|) moves per structure, so the configured
    // probability still governs how much of the model moves — a kick on a large
    // structure must not turn into a restart. Deterministic (one fixed seed per
    // measurement): these thresholds cannot flake, they either hold or they do
    // not.
    auto mean_kept = [](double probability) {
        Model m;
        add_list_vars(m, /*num_lists=*/1, /*n=*/200);
        m.close();
        ViolationManager vm(m);
        RNG rng(17);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        constexpr int kicks = 20;
        double total = 0.0;
        for (int k = 0; k < kicks; ++k) {
            const std::vector<int32_t> before = m.var(0).elements;
            fj.perturb(probability);
            total += kept_adjacency_fraction(before, m.var(0).elements);
        }
        return total / kicks;
    };

    const double small = mean_kept(0.02);          // k = 4 moves
    const double standard = mean_kept(kDefaultP);  // k = 20 moves
    const double large = mean_kept(0.5);           // k = 100 moves

    REQUIRE(small > 0.85);             // a small p barely disturbs the structure
    REQUIRE(standard > 0.5);           // the default kick keeps most of it: not a restart
    REQUIRE(standard < small - 0.05);  // ... and p is what decides how much moves
    REQUIRE(large < standard - 0.15);
}

TEST_CASE("perturb keeps structures legal at an over-large probability",
          "[fj][perturb][structural]") {
    // p > 1 would ask for more moves than the structure has slots. k is clamped
    // at one move per slot — already a full scramble — so a misconfigured
    // probability stays bounded work and the invariants still hold.
    Model m;
    add_list_vars(m, /*num_lists=*/1, /*n=*/50);
    add_set_vars(m, /*num_sets=*/1, /*universe=*/12, /*min_size=*/3, /*max_size=*/7);
    m.close();
    RNG rng(5);
    initialize_structured_random(m, rng);
    ViolationManager vm(m);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(/*set_initial_x=*/true);
    const std::vector<int32_t> original = m.var(0).elements;
    for (int k = 0; k < 20; ++k) {
        fj.perturb(3.0);
        REQUIRE(is_permutation_of(m.var(0).elements, original));
        REQUIRE(is_valid_set(m.var(1)));
    }
}

TEST_CASE("perturb never leaves a structure where it found it", "[fj][perturb][structural]") {
    // A run of k >= 2 moves can undo itself — add an element, then remove the
    // same one — and counting applied moves rather than the net effect would
    // report that as a kick. It is rare (~1.5% of kicks on this model) but it is
    // exactly the no-op #111 exists to remove, so the sweep compares elements
    // before and after and falls through to the guarantee when they match.
    //
    // The comparison has to be a SET comparison. `elements` is unordered
    // membership for a Set — Count and Lambda both read it order-insensitively —
    // so remove-then-re-add returns a permuted vector holding the identical set,
    // and asserting on the raw vector passes while the search sees a kick that
    // moved nothing. Measured at ~0.5% of kicks on this model.
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        Model m;
        add_set_vars(m, /*num_sets=*/1, /*universe=*/30, /*min_size=*/5, /*max_size=*/30);
        m.close();
        RNG rng(seed);
        initialize_structured_random(m, rng);
        ViolationManager vm(m);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        for (int k = 0; k < kStructuralKicks; ++k) {
            std::vector<int32_t> before = m.var(0).elements;
            fj.perturb(kDefaultP);
            std::vector<int32_t> after = m.var(0).elements;
            std::sort(before.begin(), before.end());
            std::sort(after.begin(), after.end());
            REQUIRE(after != before);
            REQUIRE(is_valid_set(m.var(0)));
        }
    }
}

TEST_CASE("an immovable structure costs a scalar model no randomness",
          "[fj][perturb][structural]") {
    // The structural pass must not disturb the scalar draw sequence: it tests
    // the variable type before it draws, and the move generators draw nothing
    // for a structure that cannot move. Pin that by running the same scalar
    // model with and without immovable structures — the assignments, and the
    // draws that follow each kick, must coincide. (The stronger property, that a
    // model with no structures at all matches the pre-#111 engine draw for draw,
    // was verified against that engine directly; it cannot be expressed here.)
    auto scalar_sequence = [](bool with_structures, uint64_t seed) {
        Model m;
        std::vector<int32_t> scalars{m.bool_var(), m.int_var(0, 9), m.float_var(0.0, 10.0)};
        m.add_constraint(m.leq(m.sum(scalars), m.constant(1.0)));
        m.minimize(m.sum(scalars));
        if (with_structures) {
            m.list_var(1);  // no second position to permute
            int32_t sv = m.set_var(3, /*min_size=*/3, /*max_size=*/3);
            m.var_mut(handle_to_var_id(sv)).elements = {0, 1, 2};  // pinned at the full universe
        }
        m.close();
        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        std::vector<double> sequence;
        for (int k = 0; k < 20; ++k) {
            fj.perturb(kDefaultP);
            for (int32_t v = 0; v < static_cast<int32_t>(scalars.size()); ++v) {
                sequence.push_back(m.var(v).value);
            }
            sequence.push_back(rng.random());  // catches a shift that spared the assignment
        }
        return sequence;
    };
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        REQUIRE(scalar_sequence(/*with_structures=*/false, seed) ==
                scalar_sequence(/*with_structures=*/true, seed));
    }
}

// ---------------------------------------------------------------------------
// #115: the structural kick's deadline bound.
//
// The pass used to check the wall clock only BETWEEN structural variables, so a
// model whose structure lives in one large List or Set had no effective bound:
// one variable costs k = round(p * |elements|) move-set generations, quadratic
// in that variable's size, and the check ran only on the way to a variable that
// did not exist. It now checks between MOVES, on a stride counted in moves and
// capped at FeasibilityJump::kMaxDeadlineStride.
//
// The bound is observed directly — as the moves the pass applied, which is the
// quantity the guarantee is about — rather than as elapsed time. The deadline is
// armed at 1 ns so it is already in the past when the pass starts; that is a
// setup detail, not the assertion, and it is the same shape as the live deadline
// tests in test_feasibility_jump.cpp.
namespace {

constexpr double kBigP = 0.5;  // k = |elements| / 2 moves: a run worth bounding

// A deadline already spent by the time the kick runs.
GFJConfig expired_deadline_config() {
    GFJConfig cfg;
    cfg.two_phase = false;
    cfg.time_limit = 1e-9;
    return cfg;
}

}  // namespace

TEST_CASE("a kick on one large List is bounded by the deadline, not by the List",
          "[fj][perturb][structural][deadline]") {
    Model m;
    // Consumed by a pair_lambda_sum constraint: an unconsumed List would let the
    // engine leave it alone and the probe would silently prove nothing. n = 20000
    // at p = 0.5 puts k = 10000 moves in this ONE variable, each of them O(n)
    // element copies — the quadratic run with no clock read inside it.
    add_list_vars(m, /*num_lists=*/1, /*n=*/20000);
    m.close();
    ViolationManager vm(m);
    RNG rng(42);
    GFJConfig cfg = expired_deadline_config();
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);

    const std::vector<int32_t> before = m.var(0).elements;
    fj.perturb(kBigP);

    // EXACTLY one move, not merely "within the bound". The general guarantee is
    // one capped stride (64), but the observed value here is deterministic: the
    // stride is re-armed to 1 at the top of every kick, so the first check lands
    // after a single move and finds the deadline already gone. Asserting the
    // loose bound instead would accept 65 and let the re-arm be deleted in
    // silence — a kick would then inherit a grown stride and run 64 moves inside
    // the first large variable (~35 ms rather than ~0.55 ms on the 41k Set of
    // #115). That exact stride-persistence bug already shipped once in
    // structural_pass, where it went inert on 160 of 170 pharma-glsp instances.
    // Unbounded, this is 10000 — what the pass ran before, whatever the budget.
    REQUIRE(fj.structural_kick_moves() == 1);
    // Not inert: the kick still happened, and still moved the List. Compared
    // through a bool so a failure reports the verdict rather than dumping 20000
    // elements twice.
    const bool moved = m.var(0).elements != before;
    REQUIRE(moved);
    REQUIRE(is_permutation_of(m.var(0).elements, before));
}

TEST_CASE("a kick on one large Set is bounded by the deadline, not by the Set",
          "[fj][perturb][structural][deadline]") {
    using FJ = FeasibilityJump;
    Model m;
    // The shape #115 measured at 2.3 s for a single kick. A Set move also scans
    // the universe, so its moves are the more expensive of the two.
    add_set_vars(m, /*num_sets=*/1, /*universe=*/20000, /*min_size=*/5000, /*max_size=*/20000);
    m.close();
    RNG rng(42);
    initialize_structured_random(m, rng);  // Sets start empty until initialised
    ViolationManager vm(m);
    GFJConfig cfg = expired_deadline_config();
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);

    const std::vector<int32_t> before = m.var(0).elements;
    const int64_t unbounded_moves = static_cast<int64_t>(std::llround(kBigP * before.size()));
    REQUIRE(unbounded_moves > FJ::kMaxDeadlineStride + 1);  // the bound is not vacuous here
    fj.perturb(kBigP);

    // Exactly one, for the reason spelled out on the List case above: the stride
    // is re-armed per kick, so this value is deterministic and pinning it is what
    // keeps the re-arm alive.
    REQUIRE(fj.structural_kick_moves() == 1);
    REQUIRE(is_valid_set(m.var(0)));  // min_size/max_size and the universe still hold
}

TEST_CASE("every kick re-arms the stride, so none inherits a grown one",
          "[fj][perturb][structural][deadline]") {
    // The property the two expired-deadline tests above rest on, isolated so it
    // fails on its own if arm_structural_kick() stops resetting the stride.
    //
    // One List of 20 gives k = 2 moves per kick, and the arithmetic is then
    // forced. Move 1 is ungated; the check before move 2 finds countdown 1, reads
    // the clock ONCE, and retunes the stride to 8 (the growth cap: a ~1.5 us move
    // against a target of remaining/64 = 156 ms). So a kick that starts re-armed
    // always reads the clock exactly once here.
    //
    // Let a kick inherit the previous kick's stride instead and the second kick
    // starts at countdown 8, which k = 2 moves never exhausts -- so it reads the
    // clock ZERO times. That is the discriminator, and it is deterministic: it
    // turns on the countdown arithmetic, not on how long anything took.
    Model m;
    add_list_vars(m, /*num_lists=*/1, /*n=*/20);
    m.close();
    ViolationManager vm(m);
    RNG rng(42);
    GFJConfig cfg;
    cfg.two_phase = false;
    cfg.time_limit = 10.0;  // never reached; this is a stride-sizing input only
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);

    // Three kicks: the first would pass even without the re-arm (nothing has been
    // grown yet), so it is the second and third that carry the test.
    for (int kick = 0; kick < 3; ++kick) {
        fj.perturb(kDefaultP);
        REQUIRE(fj.structural_kick_moves() == 2);   // the whole run, never truncated
        REQUIRE(fj.structural_kick_checks() == 1);  // ...and re-armed, so it checked
        // The retune really ran. A move would have to cost 156 ms on a 20-element
        // List to leave the stride at 1, so this cannot flake.
        REQUIRE(fj.structural_kick_stride() > 1);
    }
}

TEST_CASE("a deadline that expires mid-kick stops it within a stride",
          "[fj][perturb][structural][deadline]") {
    using FJ = FeasibilityJump;
    // The guarantee's actual scenario, which the expired-deadline tests above do
    // NOT reach: there the very first check finds the deadline gone and returns
    // before the stride is ever retuned, so the countdown reload is dead code in
    // them. Here the budget outlives many moves, so the pass ramps, reloads the
    // countdown repeatedly, and is then cut off partway through one variable.
    constexpr int64_t kUnboundedMoves = 10000;  // k = round(0.5 * 20000)
    Model m;
    add_list_vars(m, /*num_lists=*/1, /*n=*/20000);
    m.close();
    ViolationManager vm(m);
    RNG rng(42);
    GFJConfig cfg;
    cfg.two_phase = false;
    // Two margins, both wide. A move on this List measures ~230 us (#115 reports
    // 1021 ms for 3000 moves on a 30000-element List), so the whole kick is
    // ~2.3 s against this 0.2 s budget -- 11x headroom before the kick could
    // finish and defeat the "stopped early" half. In the other direction the
    // budget affords ~870 moves, so the machine would have to stall for the
    // entire 0.2 s before the second move to defeat the "got going" half.
    cfg.time_limit = 0.2;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    fj.perturb(kBigP);

    // It got going: past the first check, so the stride was retuned and the
    // countdown reloaded at least once -- the wiring the other tests never touch.
    REQUIRE(fj.structural_kick_moves() > 1);
    REQUIRE(fj.structural_kick_checks() >= 2);
    // ...and the deadline, not the variable, is what ended it.
    REQUIRE(fj.structural_kick_moves() < kUnboundedMoves);
    // The guarantee itself, in the only form observable without a clock: the
    // moves between two consecutive clock reads never exceed the cap, so neither
    // can the work done past the deadline.
    //
    // Note what this does NOT catch, so nobody mistakes it for the whole story.
    // Reloading the countdown with a MULTIPLE of the stride is invisible in this
    // regime, because the tuner measures the interval it actually got: over
    // 4 * stride moves it sees 4x the elapsed time and shrinks the stride 4x to
    // match, landing on the same moves-per-check. The mutation only shows where
    // the 64-move cap binds instead of the time target -- see the small-structure
    // test below, which is the one that fails on it.
    REQUIRE(fj.structural_kick_moves() <= FJ::kMaxDeadlineStride * fj.structural_kick_checks() + 1);
}

TEST_CASE("a kick with no wall clock reads no clock at all",
          "[fj][perturb][structural][deadline]") {
    // Determinism, observed directly rather than inferred: with time_limit <= 0
    // the check short-circuits before the clock read, so nothing timing-derived
    // can reach control flow and an iteration-budgeted run stays bit-identical
    // from machine to machine. The kick also runs every move it would have run.
    auto kick = [](double time_limit, uint64_t seed) {
        Model m;
        add_list_vars(m, /*num_lists=*/3, /*n=*/20);  // k = round(0.1 * 20) = 2 each
        m.close();
        ViolationManager vm(m);
        RNG rng(seed);
        GFJConfig cfg;
        cfg.two_phase = false;
        cfg.time_limit = time_limit;
        FeasibilityJump fj(m, vm, rng, cfg);
        fj.begin(/*set_initial_x=*/true);
        fj.perturb(kDefaultP);
        return std::make_pair(fj.structural_kick_checks(), all_elements(m));
    };

    const auto [checks, elements] = kick(/*time_limit=*/0.0, /*seed=*/7);
    REQUIRE(checks == 0);                    // the clock was never read
    REQUIRE(kick(0.0, 7).second == elements);  // ...so the kick is reproducible
}

TEST_CASE("a kick on many small structures is not cut short",
          "[fj][perturb][structural][deadline]") {
    using FJ = FeasibilityJump;
    // The shape the between-variables check already handled well, and the one
    // the new per-move check must not slow down. 200 Lists x k = 2 moves.
    constexpr int kLists = 200;
    constexpr int64_t kMovesPerList = 2;
    Model m;
    add_list_vars(m, kLists, /*n=*/20);
    m.close();
    ViolationManager vm(m);
    RNG rng(42);
    GFJConfig cfg;
    cfg.two_phase = false;
    // Never reached: the kick is microseconds of work. Generous on purpose — the
    // margin is how much slower the machine may be before the tuner can no
    // longer afford the cap, and at remaining/64 = 156 ms per stride against
    // ~1.5 us moves that is a factor of ~1e5.
    cfg.time_limit = 10.0;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    fj.perturb(kDefaultP);

    // Every move the pass would have made without a deadline: the bound fires on
    // the budget, never on the structure count.
    REQUIRE(fj.structural_kick_moves() == kLists * kMovesPerList);
    // Throughput, observed as clock reads rather than timed: the stride ramps
    // 1, 8, 64 and then holds at the cap, so the pass reads the clock 8 times
    // over these 400 moves instead of 400 times. Asserted with room -- one read
    // per 8 moves rather than the 1-per-50 actually measured -- because a single
    // scheduling stall on a loaded machine throttles one growth step, and the
    // claim being made is "not once per move", not an exact ramp.
    REQUIRE(fj.structural_kick_stride() > 1);
    REQUIRE(fj.structural_kick_stride() <= FJ::kMaxDeadlineStride);
    REQUIRE(fj.structural_kick_checks() * 8 <= fj.structural_kick_moves());
    // And the countdown really is reloaded with the stride itself. This is the
    // sharpest place to say it: the stride pins at the cap here, so the bound
    // below sits just above the observed 8 reads for 400 moves. Reload with
    // stride * 4 and the ramp reads at moves 1, 5, 37, 293 -- 4 reads, whose
    // budget of 4 * 64 + 1 = 257 no longer covers 400 moves, so this fails while
    // every other assertion in the file still passes.
    REQUIRE(fj.structural_kick_moves() <= FJ::kMaxDeadlineStride * fj.structural_kick_checks() + 1);
}

TEST_CASE("a kick past its deadline stops instead of walking the remaining structures",
          "[fj][perturb][structural][deadline]") {
    // The expired-deadline cases above all have exactly ONE structural variable,
    // so none of them reaches perturb_structural's outer `break`: a pass that
    // only broke out of the inner loop would pass every one of them unchanged.
    // It must not carry on -- each remaining variable copies its whole `elements`
    // vector for the before/after comparison and consults the clock again, which
    // is O(total elements) of work past a deadline that is already gone.
    //
    // Three Lists make that observable, deterministically: the deadline is spent
    // before the pass starts, so the first check ends it after one move.
    Model m;
    add_list_vars(m, /*num_lists=*/3, /*n=*/20);
    m.close();
    ViolationManager vm(m);
    RNG rng(42);
    GFJConfig cfg = expired_deadline_config();
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);

    fj.perturb(kBigP);

    // One move on the first List, then the first check ends the pass...
    REQUIRE(fj.structural_kick_moves() == 1);
    // ...and ends it *there*. Delete the outer break and this reports 3: the pass
    // reaches each remaining variable, finds the countdown already spent, and
    // reads the clock again on its way out of it.
    REQUIRE(fj.structural_kick_checks() == 1);
}
