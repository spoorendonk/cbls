// Regression tests for #109: a diversification kick must always move
// something. perturb() used to randomise each variable independently, so at the
// default perturbation_probability = 0.1 a kick changed nothing at all with
// probability (1-p)^n — 90% on a one-variable model, 81% on two. The tests
// below run many seeds rather than one lucky draw, so a reintroduced
// no-op would fail deterministically rather than flake.
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

#include <set>
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
    SECTION("no jumpable variables at all") {
        // A List variable is not jumpable, so there is nothing for the kick to
        // force. It must return quietly rather than spin or crash.
        Model m;
        m.list_var(4);
        m.close();
        ViolationManager vm(m);
        RNG rng(3);
        FeasibilityJump fj(m, vm, rng);
        std::vector<int32_t> elements_before = m.var(0).elements;
        std::vector<double> before = assignment(m);
        for (int k = 0; k < 10; ++k) {
            fj.perturb(kDefaultP);
        }
        REQUIRE(assignment(m) == before);
        REQUIRE(m.var(0).elements == elements_before);
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
