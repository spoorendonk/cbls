#include "cbls/feasibility_jump.h"

#include "cbls/dag_ops.h"
#include "cbls/moves.h"
#include "cbls/randomize.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cbls {

namespace {

constexpr double kTol = 1e-9;
constexpr double kInf = std::numeric_limits<double>::infinity();

// The jump-candidate budgets. `int_jump_candidates` enumerates an Int domain
// whole up to `kExhaustiveJumpWidth` and otherwise offers a `kJumpGridPoints`
// grid plus a few local values; the breakpoint candidates (#186) take the same
// grid, and enumerate whole only up to `kExhaustiveBreakpoints`.
//
// Why the breakpoint cap is the grid's size and not the Int width. Measured
// with a scratch harness timing compute_var_jump of one variable that sits
// under the op(s) and in five linear rows `i*x + y <= i*W`, the op's row
// violated so every candidate is probed; Release, AMD Ryzen 5 5600H (12
// threads, simon-Legion-5-Pro-16ACH6H), load < 1.5 under an exclusive bench
// lock, three runs agreeing within ~5%, engine at af0213a. Microseconds per
// call:
//
//   Float under one ceil, edges in domain   32: 38   33 (capped): 38.7
//                                            64: 41  256: 42  10^6: 43
//   Int (domain 10^5) under ceil(x / s)      32: 42   33 (capped): 43
//                                           256: 44  4096: 44
//   Int index, Element table of n            32: 26   68: 41   69 (capped): 35
//                                         1000: 40  10^4: 28
//   K ceil nodes on one Float (shared grid)  K=1: 47  2: 57  4: 78  8: 141
//   baselines: Int grid, no breakpoint node 12.4; Float reaching no breakpoint
//   op in a model with one 1.10, same Float in a model without 1.09
//
// Every capped path costs ~35-45 us whatever the range, and enumerating 32
// edges (up to three probes each) costs about the same, so 32 is where
// enumerating whole stops being free. The earlier two-probes-per-edge form
// measured 208 us at 256 edges (engine 9c587f1), ~7x the capped path. An
// Element's own cap (enumerate while the indices are no more than its capped
// path would offer) sits at the same crossover: 41 us at 68 against 35 capped.
// K nodes cost ~13 us each beyond the first -- their near edges, which stay
// per node -- with the grid shared, so no per-variable limit is needed on top.
// The Int width stays 256 so that no existing model's trajectory moves.
constexpr double kExhaustiveJumpWidth = 256.0;
constexpr int kJumpGridPoints = 32;
constexpr double kExhaustiveBreakpoints = kJumpGridPoints;

// A constraint is violated if its residual exceeds the tolerance. Written so
// that non-finite residuals (NaN from inf-inf, or +inf) count as violated:
// !(x <= tol) is true for x > tol and for NaN. This prevents a NaN constraint
// from being silently treated as satisfied (a false "feasible").
bool is_violated(double residual) {
    return !(residual <= kTol);
}

// A row's contribution to the unproductive-batch progress measure (see
// FeasibilityJump::unweighted_violation_). Finite residuals of rows this engine
// calls VIOLATED: NaN fails the comparison, and +inf is excluded explicitly so
// that subtracting it out again in update_var cannot turn the running total into
// a NaN it can never leave. Both callers -- the fresh recomputation and the
// incremental update -- go through this, so the two definitions cannot drift
// apart.
//
// The threshold is is_violated's kTol, not a bare `> 0.0`. A residual in
// (0, kTol] is a SATISFIED row everywhere else in this file, and counting it
// here made the measure disagree with the engine's own predicate. That
// disagreement is invisible on inequality rows over integers, where a satisfied
// row's residual is a clean 0.0, and structural on EQUALITY rows over
// continuous variables, where NodeOp::Eq's |lhs - rhs| lands a few ulp off zero
// instead: the measure could then never reach its floor, and every test of the
// form "the real rows are all satisfied, so this measure has nothing left to
// say" was dead code on such a model (#102, MINLPLib ex8_6_1: 45 nonlinear
// equalities over 75 continuous variables).
//
// Rows violated between kTol and the search's feasibility_tolerance are a
// different band and are still counted. They are violated as far as FJ is
// concerned, and it is FJ's progress this measures.
double progress_residual(double residual) {
    return (residual > kTol && residual < std::numeric_limits<double>::infinity()) ? residual : 0.0;
}

double clamp_to_domain(const Variable& var, double value) {
    return std::min(std::max(value, var.lb), var.ub);
}

// --- perturbation helpers --------------------------------------------------
// `random_in_domain` itself lives in randomize.h, shared with search.cpp's
// initialisers and LNS's destroy step (#112), and so does `movable_domain`,
// which the search also reads (#201). The one below is specific to the kick and
// stays here; it reads the domain through the same `int_sample_window`, so the
// values it considers in-domain are exactly the ones `random_in_domain` can draw.
//
// Draw a random value from the domain that DIFFERS from the current one, for
// the single variable a perturbation is guaranteed to move (#109). Plain
// resampling is not enough: it redraws the current value with probability
// 1/|domain|, which on a Bool is one kick in two. Returns the current value
// unchanged only for a pinned domain, where no move exists.
double random_different_in_domain(const Variable& var, RNG& rng) {
    if (!movable_domain(var)) {
        return var.value;
    }
    const DomainWindow w = domain_window(var);
    switch (var.type) {
        case VarType::Bool:
            return var.value != 0.0 ? 0.0 : 1.0;
        case VarType::Int: {
            // Non-empty: movable_domain above rejected the empty case.
            const DomainWindow s = int_sample_window(var);
            const auto lb = static_cast<int64_t>(s.lo);
            const auto ub = static_cast<int64_t>(s.hi);
            if (!(var.value >= s.lo) || !(var.value <= s.hi)) {
                // Outside the window (or NaN), so any draw differs. Compared in
                // double: `var.value` need not be castable to int64_t at all.
                return static_cast<double>(rng.integers(lb, ub + 1));
            }
            const auto cur = static_cast<int64_t>(var.value);
            // Uniform over the domain minus the current value: draw from a
            // domain one value short, then step over the hole at `cur`.
            int64_t draw = rng.integers(lb, ub);  // [lb, ub-1]
            if (draw >= cur) {
                ++draw;
            }
            return static_cast<double>(draw);
        }
        default: {  // Float
            const double v = rng.uniform(w.lo, w.hi);
            if (v != var.value) {
                return v;
            }
            // Measure-zero in exact arithmetic, but a narrow enough domain makes
            // it reachable; fall back to the endpoint further from the current
            // value, which differs because the domain holds more than one point.
            return (var.value - w.lo >= w.hi - var.value) ? w.lo : w.hi;
        }
    }
}

// --- structural perturbation helpers ---------------------------------------
// List and Set variables are not jumpable, so none of the above can reach them:
// a kick on a model whose decisions live in structural variables randomised
// nothing, burned the stagnation counter and left the search exactly where it
// was (#111). They get their own pass, built out of the same typed move
// generators the STRUCTURAL batch uses (moves.cpp) rather than fresh mutation
// code — so the kick explores exactly the neighbourhood the search knows how to
// evaluate, and every move it applies is legal by construction: a List keeps its
// elements distinct and its length inside [min_size, max_size], a member of a
// ListPartition keeps its cover, and a Set stays inside min_size/max_size.

// How many random structural moves a kick applies to one variable.
//
// `perturbation_probability` has to keep governing how much of the model moves,
// so scale with the variable's own size: k = round(p * |elements|) applies a p
// fraction of the structure's size in MOVES. That is not the same as displacing
// a p fraction of its slots, and the gap is large for a List: list_2opt
// reverses a random sub-range (mean ~n/3), so k = 0.1n moves rewrite ~98% of
// positions on n = 1000 while breaking ~26% of adjacent pairs. The adjacency
// figure is the one that tracks p, so the scaling is right for a List read
// pairwise (pair_lambda_sum) and much coarser than p suggests for one read
// positionally (`at`).
//
// A structure has no "randomise the whole variable" analogue that is not a
// restart, so the probability sets *how much* of each structure moves rather
// than *which* structures move — which is also what keeps a structure smaller
// than 1/p slots from never moving at all.
//
// The size is the CURRENT membership, which for a List is its whole decision
// content but for a Set is not: a 3-of-1000 Set gets a kick sized on the 3, not
// on the universe. That is deliberate — the kick then rewrites a p fraction of
// the set's *state*, and p = 1 rewrites all of it — but it does mean a sparse
// Set cannot grow far in a single kick. LNS, which resamples the cardinality
// outright, is the mechanism for that.
//
// Clamped to at least one move, so a kick on a model whose decisions are all
// structural is never a no-op (#111), and to at most one move per slot, which is
// already a full scramble, so a misconfigured p > 1 cannot turn a kick into
// unbounded work. The comparison is written to reject NaN.
int32_t structural_kick_size(const Variable& var, double probability) {
    const auto n = static_cast<int32_t>(var.elements.size());
    const double scaled = std::round(probability * static_cast<double>(n));
    if (!(scaled > 1.0)) {
        return 1;
    }
    // scaled > 1 implies probability * n > 1, hence n >= 1: an empty structure
    // always took the early return, so the clamp needs no guard for it.
    return static_cast<int32_t>(std::min(scaled, static_cast<double>(n)));
}

// Did a run of structural moves leave `var` somewhere the search can tell apart
// from `before`? "Somewhere else" is type-dependent, and comparing the raw
// vectors gets a Set wrong.
//
// For a List, order IS the decision content: the DAG reads it positionally
// (`at`) and pairwise (`pair_lambda_sum`), so vector inequality is exactly the
// question. For a Set, `elements` is unordered membership — Count and Lambda
// both read it order-insensitively — so a run that removes an element and adds
// it back lands on a permuted vector holding the identical set. Calling that
// "changed" hands back a kick that moved nothing and skips the never-a-no-op
// fallback, which is the defect #111 exists to prevent, just arrived at
// sideways. Measured at 0.5% of kicks on a universe-30 Set at the default p.
bool structure_moved(const Variable& var, const std::vector<int32_t>& before) {
    if (var.type != VarType::Set) {
        return var.elements != before;
    }
    if (var.elements.size() != before.size()) {
        return true;
    }
    std::vector<int32_t> a = var.elements;
    std::vector<int32_t> b = before;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a != b;
}

// Apply one uniformly chosen structural move to `var_id` and report whether the
// variable's elements actually changed. False means this draw offered nothing
// that moves the variable — in practice a structural dead end (a List shorter
// than two elements, a Set with no legal add, remove or swap), so the caller
// stops rather than redraw. That is a heuristic rather than a proof: the
// generator picks its positions at random, so on a List holding duplicate
// elements a later draw could in principle differ.
//
// Candidates that leave the elements as they are — a relocate to the adjacent
// position reinserts the element where it was — are dropped before the draw, so
// a kick cannot silently lose a move to one.
//
// STILL NOT ROUTED THROUGH SearchConfig::move_generators (#165, unchanged by
// #164). The diversification kick draws from the same RNG as the rest of the
// search, so changing what it draws -- a registered generator proposing a
// different number of candidates, or none -- shifts every later draw and changes
// the trajectory of every model that has a structured variable, kick or no kick.
// The kick is also a RANDOMISER rather than an optimiser: it wants an arbitrary
// legal move, which is exactly what generate_standard_moves gives it, where a
// cost-aware generator would give it the opposite.
//
// What #164 adds instead is a second SOURCE of moves, not a second source of
// policy: a List in a partition also offers the partition's inter-list moves,
// anchored on this variable. That is what makes the kick able to move an
// ALL-EMPTY partition -- `structural_kick_size` asks for one move on a
// zero-length list, the five intra-list moves need two elements to reorder and
// `list_insert` is suppressed for a partition member, so before #164 the kick
// found nothing and silently did nothing, which is the exact defect #109/#111
// closed for the scalar case. `generate_partition_moves` appends at most one
// candidate and draws nothing on a model with no partition, so the draw sequence
// of every pre-#164 model is untouched.
//
// "All-empty partition" means an `AtMostOnce` one, which is the only kind that
// can be empty: `partition_insert` is what moves it, and it is `AtMostOnce`-only
// because under `Exact` an insert would double-serve. An all-empty `Exact`
// partition is unreachable rather than repairable — both initialisers lay one
// out complete, which they must, since no `Exact` move can repair an incomplete
// cover.
bool apply_random_structural_move(Model& model, int32_t var_id, RNG& rng) {
    std::vector<Move> moves = generate_standard_moves(model.var(var_id), rng);
    const int partition = model.partition_of_list(var_id);
    if (partition >= 0) {
        generate_partition_moves(model, partition, /*anchor=*/var_id, rng, moves,
                                 /*neighbours=*/nullptr);
    }
    // Keep only the candidates that actually move THIS variable: the caller
    // decides whether the kick did anything by comparing this variable's
    // elements before and after, so a candidate that moved only a sibling list
    // would be reported as a no-op kick and trigger the never-a-no-op fallback
    // on top of a move that had in fact been made. On a non-partition variable
    // every candidate is a single change to it, so this is the pre-#164
    // predicate verbatim.
    const std::vector<int32_t>& current = model.var(var_id).elements;
    moves.erase(std::remove_if(moves.begin(), moves.end(),
                               [var_id, &current](const Move& m) {
                                   return std::none_of(m.changes.begin(), m.changes.end(),
                                                       [var_id, &current](const Move::Change& c) {
                                                           return c.var_id == var_id &&
                                                                  !change_is_noop(c, current);
                                                       });
                               }),
                moves.end());
    if (moves.empty()) {
        return false;
    }
    const auto pick = static_cast<size_t>(rng.integers(0, static_cast<int64_t>(moves.size())));
    apply_move(model, moves[pick]);
    return true;
}

// Integer jump candidates: exhaustive over a small domain, else a coarse grid
// plus neighbours/endpoints. Each consider() costs O(|G_v|) when every weighted
// row of G_v is linear (LinearJumpScorer: an affine comparison or bare body),
// else one weighted_violation_delta (two delta_evaluate passes); the JumpTable cache
// amortises either across the GLS loop. Scoring is still per candidate: a
// closed-form ARGMIN over a linear column would pick from a different candidate
// set, which is a different algorithm rather than a cheaper evaluation of this
// one.
//
// Bounds are read as doubles through `domain_window` (#114): no `long`, so
// nothing overflows and an unbounded Int gets candidates instead of freezing in
// the jump table. A finite, integral bound within +/-2^53 passes through
// verbatim, so those candidates are bit-identical to the pre-#114 ones; a wider
// finite domain is narrowed by the window and its candidates do change, which is
// the point on `[-1e19, 1e19]`. Pinned by "finite Int jump candidates are
// unchanged" and "an unbounded Int is offered jump candidates"
// (tests/test_unbounded_domain.cpp, which carries the archaeology).
//
// Returns whether it offered EVERY value a jump could reach -- the exhaustive
// enumeration, or no jump at all -- so that the caller knows no further
// candidate (#186's breakpoints) can add anything.
template <class Consider>
bool int_jump_candidates(const Variable& var, double x0, Consider&& consider) {
    if (!(var.lb < var.ub)) {
        // Pinned, degenerately ordered or NaN-bounded: no jump exists. Asked of
        // the *declared* bounds, because `domain_window` swaps a reversed pair
        // and would otherwise turn an empty domain into a searchable one.
        return true;
    }
    const DomainWindow w = domain_window(var);
    const double lo = std::ceil(w.lo);
    const double hi = std::floor(w.hi);
    if (hi <= lo) {
        // Fewer than two integers in the window. Past 2^53 that also means no
        // jump is representable: `x +/- 1` is not a distinct double up there.
        return true;
    }
    // Enumerate only where `v += 1.0` actually advances. At 1e18 the ulp is 128,
    // so a width-<=256 loop over such endpoints would never terminate.
    const bool steppable = std::abs(lo) < kExactIntMagnitude && std::abs(hi) < kExactIntMagnitude;
    if (steppable && hi - lo <= kExhaustiveJumpWidth) {
        // The counter IS the candidate, and `steppable` is exactly the
        // precondition that makes `+= 1.0` exact and the loop finite.
        // NOLINTNEXTLINE(bugprone-float-loop-counter)
        for (double v = lo; v <= hi; v += 1.0) {
            consider(v);
        }
        return true;
    }
    consider(lo);
    consider(hi);
    // Clamped to the *declared* bounds, not the window, so a variable that has
    // drifted outside the window keeps its local moves.
    consider(clamp_to_domain(var, x0 - 1));
    consider(clamp_to_domain(var, x0 + 1));
    for (int k = 1; k < kJumpGridPoints; ++k) {
        const double frac = static_cast<double>(k) / kJumpGridPoints;
        consider(std::round(lo + (frac * (hi - lo))));
    }
    return false;
}

// Float jump candidates: cheap convex-ish descent — a Newton step toward the
// root of each violated constraint containing v (reverse-mode AD), plus
// midpoint/endpoints. The GLS loop iterates these to converge; the
// InnerSolverHook does the heavy continuous objective polish. This replaces a
// per-jump golden-section (~60 evals) with a handful — critical for throughput
// on continuous-heavy models. Newton candidates come FIRST so that, on a tie in
// violation delta (a feasible plateau), the gradient-informed point wins rather
// than an arbitrary endpoint (`consider` keeps the first-seen minimum). The
// objective enters here too: when `obj <= bound` is violated, chasing its root
// pulls the objective down.
// Returns whether the gradient carried usable direction here. False means this
// variable takes part in at least one violated constraint yet none of them
// yielded a Newton candidate — every gradient was ~0 (or non-finite, which
// fails the same test), i.e. it sits at a
// *stationary* point — so the three remaining candidates are box constants
// unrelated to local geometry and the variable may have no move at all.
//
// A variable in no violated constraint reports true: nothing to escape.
//
// `linear`, when given, supplies a violated row's partial from its cached slope
// where that is bit-identical to compute_partial (LinearJumpScorer::
// residual_partial_at), so the Newton candidates are exactly the ones the AD sweep
// would produce, without the sweep.
template <class Consider>
bool float_jump_candidates(Model& model, int32_t var_id, const Variable& var, double x0,
                           LinearJumpScorer* linear, Consider&& consider) {
    if (var.ub <= var.lb) {
        return true;  // fixed: nothing to escape
    }
    const auto& cids = model.constraint_ids();
    int budget = 4;
    bool any_newton = false;
    bool saw_violated = false;
    const ConstSpan<int32_t> gv = model.constraints_of_var(var_id);
    for (size_t k = 0; k < gv.size(); ++k) {
        if (budget <= 0) {
            break;
        }
        const int32_t c = gv[k];
        double residual = model.node_value(cids[c]);
        if (residual <= kTol) {
            continue;  // satisfied: no root to chase
        }
        saw_violated = true;
        double grad = 0.0;
        if (linear == nullptr || !linear->residual_partial_at(var_id, k, grad)) {
            grad = compute_partial(model, cids[c], var_id);
        }
        // A NaN residual (a domain error, #205) gives a NaN step, and an
        // infinite one on an unbounded column an infinite candidate: neither
        // is a Newton candidate.
        const double step_to = clamp_to_domain(var, x0 - (residual / grad));
        if (std::abs(grad) > 1e-12 && std::isfinite(step_to)) {
            consider(step_to);
            any_newton = true;
            --budget;
        }
    }
    consider(0.5 * (var.lb + var.ub));
    consider(var.lb);
    consider(var.ub);
    return !saw_violated || any_newton;
}

// ---------------------------------------------------------------------------
// Breakpoint candidates (#186)
// ---------------------------------------------------------------------------
//
// Ceil, Floor, Round and Element are piecewise constant, so their local
// derivative is 0 and neither a Newton step nor the Int grid knows where their
// value changes. The op itself does: Ceil and Floor change at the integers,
// Round at the half-integers, and an Element index takes the values 0..n-1.
// Mapped back through the argument u(v), each of those values of u is a value
// of v, and those are this variable's candidates.
//
// A candidate is only a PROPOSAL: `consider` scores every one exactly, by the
// same weighted violation delta as any other. The map decides where candidates
// land, never whether a jump is scored right.
//
// THE MAP. The walk collects the cone above v through every op that carries a
// slope (`carries_slope`: all but the structural ops, user code, Const and the
// breakpoint ops, which stop it), pruned to the nodes `Model::close()` marked as
// reaching a breakpoint op, and carries du/dv forward along it once per call
// (`ConeSlopes`), each edge's factor being the op's own `local_derivative`. So the
// slope of an argument is the exact derivative at the current point, every path
// from v included -- `ceil(v + exp(v))` gets 1 + exp(v). Where the argument is
// linear in v (a sum of `c * v` terms with c fixed for a single-variable jump)
// the map is exact. Two further cases:
//
//  - `u = N / D` with v in D only -- the epic's `ceil(cycle / headway)` with
//    headway the decision -- is inverted exactly, v = x0 + (N/t - D0) / D',
//    when that Div IS the breakpoint node's argument and D is affine in v.
//  - Any other nonlinearity is LINEARISED at the current point, so its
//    candidates can land off the edge they aim at. The candidate is still
//    scored exactly, so the cost of a miss is a wasted probe. A path through a
//    stopping op contributes nothing: a breakpoint op inside another's argument
//    (`ceil(v + floor(v))`) counts as constant, since its slope is 0.
//
// A Sum that names a child more than once (`x + x`) counts it that many times:
// `close()` marks such Sums (`kRepeatedChild`), and only those pay a scan of
// their children.
//
// WHICH CANDIDATES, per breakpoint t (a value of u where the op changes), with
// du a small step in u (absolute, at least 4 ulp of t, at most 1/4):
//
//  - Ceil/Floor/Round, Int variable: floor of the lower and ceil of the upper of
//    the two values of v that land at t -+ du, plus the v at t itself when that
//    is an integer. So both sides of the edge are offered even where float
//    error puts the edge integer on the wrong one.
//  - Ceil/Floor/Round, Float variable: those two values of v, each stepped by
//    nextafter until the map puts it strictly on its own side of t, plus the v
//    at t itself when the map puts it on the side the edge belongs to (Ceil:
//    below; Floor, and Round above zero: above) -- a plateau's extreme point,
//    where a Ceil/Floor optimum sits.
//  - Element, Int variable: the index value k itself (floor and ceil of its v).
//  - Element, Float variable: the middle of index k's plateau intersected with
//    the range u covers, so a plateau only partly inside the range still gets a
//    candidate. `trunc` makes plateau 0 the interval (-1, 1).
//
// HOW MANY, per argument. A Ceil/Floor/Round offers every breakpoint while
// there are at most `kExhaustiveBreakpoints` (the measurement that sets it is at
// the constant); an Element every index while there are no more than its capped
// path would offer anyway (`kNearBreakpoints` + the grid + its representatives).
// Above that: the breakpoints either side of the current argument, the Element
// representatives of the line the other index currently selects
// (`ElementTable::row_reps_by_col`), and a grid across the range. The grid is ONE
// budget, `kJumpGridPoints - 1` points, shared by every capped argument the walk
// reaches, so K breakpoint nodes do not cost K grids. Candidates are then
// deduplicated across every node and against the ones the Int/Float path
// already offered.
//
// That makes an Element over a table wider than its cap a DEVIATION from #186's
// "the candidates are the index values": an index holding a value no
// representative, grid point or near neighbour names is reachable only by the
// search's other moves. The reason is cost -- every index of a 10^4 table was
// 2x10^4 probes per jump-table refresh of that variable.
//
// COST when it runs: the gate (one flag test per dependent of v), then the
// pruned cone and its sort, O(c log c) for a cone of c nodes, plus their parent
// edges -- one pass, where a reverse AD sweep per breakpoint node made
// `ceil(sum a_i x_i)` cost O(n) per variable. It runs only on a model that HAS a
// breakpoint node, for a variable whose dependents reach one, and for an Int only
// when its domain was not already enumerated whole; every other variable's
// candidates, and every other model's, are unchanged.

constexpr double kNearBreakpoints = 5.0;  // ceil(x) - 2 .. floor(x) + 2

bool reaches_breakpoint(const Model& model, int32_t nid) {
    return (model.breakpoint_flags()[static_cast<size_t>(nid)] &
            ModelStructure::kReachesBreakpoint) != 0;
}

// Does anything above `var_id` reach a breakpoint op? One flag test per dependent.
bool variable_reaches_breakpoint(const Model& model, int32_t var_id) {
    if (model.breakpoint_flags().size() != model.num_nodes()) {
        return false;  // not closed: nothing classified, and no order to walk in
    }
    const ConstSpan<int32_t> deps = model.dependents(var_id);
    return std::any_of(deps.begin(), deps.end(),
                       [&model](int32_t d) { return reaches_breakpoint(model, d); });
}

// d(parent)/d(child) for one edge, summed over the operands naming `child`. A
// Sum pays a scan of its children only when `close()` saw it name one twice.
double edge_derivative(const Model& model, const ExprNode& parent, const ChildRef& child) {
    auto same = [&child](const ChildRef& r) {
        return r.is_var == child.is_var && r.id == child.id;
    };
    const ConstSpan<ChildRef> kids = model.children(parent);
    if (parent.op == NodeOp::Sum) {
        if ((model.breakpoint_flags()[static_cast<size_t>(parent.id)] &
             ModelStructure::kRepeatedChild) == 0) {
            return 1.0;
        }
        return static_cast<double>(std::count_if(kids.begin(), kids.end(), same));
    }
    double d = 0.0;
    for (size_t i = 0; i < kids.size(); ++i) {
        if (same(kids[i])) {
            d += local_derivative(parent, static_cast<int>(i), model);
        }
    }
    return d;
}

// The pruned cone above one variable and du/dv on it, forward mode. Per thread,
// grown to the largest model walked; entries are valid only under the current
// epoch, so a call costs its own cone and never a clear of the arrays.
class ConeSlopes {
public:
    void build(const Model& model, int32_t var_id) {
        if (stamp_.size() < model.num_nodes()) {
            stamp_.resize(model.num_nodes(), 0);
            slope_.resize(model.num_nodes(), 0.0);
        }
        if (++epoch_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0);
            epoch_ = 1;
        }
        var_id_ = var_id;
        cone_.clear();
        collect(model);
        // Topological order, so a node's slope is complete before it is pushed.
        std::sort(cone_.begin(), cone_.end(), [&model](int32_t a, int32_t b) {
            return model.topo_position(a) < model.topo_position(b);
        });
        propagate(model);
    }

    [[nodiscard]] const std::vector<int32_t>& cone() const { return cone_; }

    // du/dv for one operand: 1 for the variable itself, the carried slope for a
    // node in the cone, and 0 for anything that does not move with v.
    [[nodiscard]] double of(const ChildRef& r) const {
        if (r.is_var) {
            return r.id == var_id_ ? 1.0 : 0.0;
        }
        return stamp_[static_cast<size_t>(r.id)] == epoch_ ? slope_[static_cast<size_t>(r.id)]
                                                           : 0.0;
    }

private:
    void collect(const Model& model) {
        for (const int32_t dep : model.dependents(var_id_)) {
            if (reaches_breakpoint(model, dep)) {
                stamp_[static_cast<size_t>(dep)] = epoch_;
                slope_[static_cast<size_t>(dep)] = 0.0;
                cone_.push_back(dep);
            }
        }
        // `cone_` grows while it is walked, so it is indexed rather than iterated.
        for (size_t head = 0; head < cone_.size(); ++head) {
            if (!carries_slope(model.nodes()[static_cast<size_t>(cone_[head])].op)) {
                continue;  // a breakpoint node: piecewise constant above here
            }
            for (const int32_t parent : model.parents(cone_[head])) {
                if (stamp_[static_cast<size_t>(parent)] != epoch_ &&
                    reaches_breakpoint(model, parent)) {
                    stamp_[static_cast<size_t>(parent)] = epoch_;
                    slope_[static_cast<size_t>(parent)] = 0.0;
                    cone_.push_back(parent);
                }
            }
        }
    }

    void propagate(const Model& model) {
        const ChildRef var_ref{var_id_, true};
        for (const int32_t dep : model.dependents(var_id_)) {
            const ExprNode& nd = model.nodes()[static_cast<size_t>(dep)];
            if (stamp_[static_cast<size_t>(dep)] == epoch_ && carries_slope(nd.op)) {
                slope_[static_cast<size_t>(dep)] += edge_derivative(model, nd, var_ref);
            }
        }
        for (const int32_t nid : cone_) {
            const double s = slope_[static_cast<size_t>(nid)];
            if (s == 0.0 || !carries_slope(model.nodes()[static_cast<size_t>(nid)].op)) {
                continue;
            }
            const ChildRef ref{nid, false};
            for (const int32_t parent : model.parents(nid)) {
                const ExprNode& pn = model.nodes()[static_cast<size_t>(parent)];
                if (stamp_[static_cast<size_t>(parent)] == epoch_ && carries_slope(pn.op)) {
                    slope_[static_cast<size_t>(parent)] += edge_derivative(model, pn, ref) * s;
                }
            }
        }
    }

    std::vector<uint32_t> stamp_;
    std::vector<double> slope_;
    std::vector<int32_t> cone_;
    uint32_t epoch_ = 0;
    int32_t var_id_ = -1;
};

double operand_value(const Model& model, const ChildRef& r) {
    return r.is_var ? model.variables()[static_cast<size_t>(r.id)].value
                    : model.node_values()[static_cast<size_t>(r.id)];
}

// u as a function of v, and its inverse, for one argument of a breakpoint node:
// linear, u = u0 + a (v - x0), or the exact reciprocal u = N / (D0 + D' (v - x0)).
struct ArgMap {
    bool inverse = false;
    double x0 = 0.0;
    double u0 = 0.0;
    double slope = 0.0;
    double num = 0.0;
    double d0 = 0.0;
    double d_slope = 0.0;

    [[nodiscard]] double to_v(double t) const {
        if (inverse) {
            return x0 + (((num / t) - d0) / d_slope);
        }
        return x0 + ((t - u0) / slope);
    }
    [[nodiscard]] double to_u(double v) const {
        if (inverse) {
            return num / (d0 + (d_slope * (v - x0)));
        }
        return u0 + (slope * (v - x0));
    }
    // Whether u grows with v. For the reciprocal, du/dv = -N D' / D^2.
    [[nodiscard]] bool increasing() const { return inverse ? num * d_slope < 0.0 : slope > 0.0; }
};

// The map for argument `kid` of a breakpoint node, or false when the argument
// does not move with v (or moves by a non-finite amount).
bool argument_map(const Model& model, const ConeSlopes& cone, const ChildRef& kid, double x0,
                  ArgMap& map) {
    map.x0 = x0;
    map.u0 = operand_value(model, kid);
    if (!kid.is_var) {
        const ExprNode& arg = model.nodes()[static_cast<size_t>(kid.id)];
        if (arg.op == NodeOp::Div) {
            const ConstSpan<ChildRef> nd = model.children(arg);
            const double d_slope = cone.of(nd[1]);
            if (cone.of(nd[0]) == 0.0 && d_slope != 0.0 && std::isfinite(d_slope)) {
                map.inverse = true;
                map.num = operand_value(model, nd[0]);
                map.d0 = operand_value(model, nd[1]);
                map.d_slope = d_slope;
                return std::isfinite(map.num) && map.num != 0.0 && std::isfinite(map.d0);
            }
        }
    }
    map.slope = cone.of(kid);
    return std::abs(map.slope) > 1e-12 && std::isfinite(map.slope) && std::isfinite(map.u0);
}

// The breakpoints of one argument, as integers k over [k_min, k_max]: breakpoint
// t_k = k + offset for Ceil/Floor/Round, index k for an Element. `u_lo`/`u_hi` are
// the values u takes at the ends of v's domain window. Empty (k_max < k_min) when
// the reciprocal's denominator can reach zero inside the window.
struct BreakpointRange {
    double offset = 0.0;
    double k_min = 0.0;
    double k_max = -1.0;
    double u_lo = 0.0;
    double u_hi = 0.0;
};

BreakpointRange breakpoint_range(const Model& model, const ExprNode& nd, size_t child_idx,
                                 const Variable& var, const ArgMap& map) {
    const DomainWindow w = domain_window(var);
    BreakpointRange r;
    if (map.inverse) {
        const double d_lo = map.d0 + (map.d_slope * (w.lo - map.x0));
        const double d_hi = map.d0 + (map.d_slope * (w.hi - map.x0));
        if (!(d_lo * d_hi > 0.0)) {
            return r;
        }
    }
    const double ua = map.to_u(w.lo);
    const double ub = map.to_u(w.hi);
    r.u_lo = std::min(ua, ub);
    r.u_hi = std::max(ua, ub);
    if (nd.op == NodeOp::Element) {
        const ElementTable& tbl = model.element_table(nd.lambda_func_id);
        const auto n = static_cast<double>(child_idx == 0 ? tbl.rows : tbl.cols);
        // An Int lands on index values; a Float on every plateau the range
        // touches, which `trunc` numbers trunc(u_lo)..trunc(u_hi).
        const bool is_int = var.type == VarType::Int;
        r.k_min = std::max(is_int ? std::ceil(r.u_lo) : std::trunc(r.u_lo), 0.0);
        r.k_max = std::min(is_int ? std::floor(r.u_hi) : std::trunc(r.u_hi), n - 1.0);
    } else {
        r.offset = nd.op == NodeOp::Round ? 0.5 : 0.0;
        r.k_min = std::ceil(r.u_lo - r.offset);
        r.k_max = std::floor(r.u_hi - r.offset);
    }
    if (!std::isfinite(r.k_min) || !std::isfinite(r.k_max)) {
        r.k_max = r.k_min - 1.0;
    }
    return r;
}

// `v` truncated toward zero as an index into [0, n), or -1: Element's own rule.
int32_t current_index(double v, int32_t n) {
    const double t = std::trunc(v);
    if (std::isnan(t) || t < 0.0 || t >= static_cast<double>(n)) {
        return -1;
    }
    return static_cast<int32_t>(t);
}

// The representatives for argument `c` of an Element: those of the line the
// OTHER index currently selects. A one-index table has a single column.
const std::vector<int32_t>* element_reps(const Model& model, const ExprNode& nd, size_t c) {
    const ElementTable& tbl = model.element_table(nd.lambda_func_id);
    const ConstSpan<ChildRef> kids = model.children(nd);
    if (kids.size() == 1) {
        return &tbl.row_reps_by_col.front();
    }
    const size_t other = 1 - c;
    const int32_t line =
        current_index(operand_value(model, kids[other]), other == 0 ? tbl.rows : tbl.cols);
    if (line < 0) {
        return nullptr;
    }
    return c == 0 ? &tbl.row_reps_by_col[static_cast<size_t>(line)]
                  : &tbl.col_reps_by_row[static_cast<size_t>(line)];
}

// One argument of one breakpoint node that moves with v.
struct BreakpointJob {
    const ExprNode* node = nullptr;
    ArgMap map;
    BreakpointRange range;
    const std::vector<int32_t>* reps = nullptr;
    bool capped = false;
};

// Whether enumerating `job`'s range whole costs more than its capped path.
bool is_capped(const BreakpointJob& job) {
    const double count = job.range.k_max - job.range.k_min + 1.0;
    if (job.node->op != NodeOp::Element) {
        return count > kExhaustiveBreakpoints;
    }
    const double reps = job.reps != nullptr ? static_cast<double>(job.reps->size()) : 0.0;
    return count > kNearBreakpoints + (kJumpGridPoints - 1) + reps;
}

// The k values to offer from one job: every one, or the breakpoints near the
// current argument, the representatives, and `grid_points` of the shared grid.
void breakpoint_ks(const BreakpointJob& job, int grid_points, std::vector<double>& ks) {
    ks.clear();
    const BreakpointRange& range = job.range;
    const double count = range.k_max - range.k_min + 1.0;
    if (!(count >= 1.0)) {
        return;
    }
    if (!job.capped) {
        const auto n = static_cast<int>(count);
        for (int i = 0; i < n; ++i) {
            ks.push_back(range.k_min + static_cast<double>(i));
        }
        return;
    }
    // Strictly below the current argument: ceil(x) - 1, ceil(x) - 2; strictly
    // above: floor(x) + 1, floor(x) + 2; and x itself when it is a breakpoint.
    const double x = job.map.u0 - range.offset;
    if (std::isfinite(x)) {
        const double first = std::ceil(x) - 2.0;
        const double last = std::floor(x) + 2.0;
        for (int dk = 0; dk <= 4; ++dk) {
            const double k = first + static_cast<double>(dk);
            if (k <= last && k >= range.k_min && k <= range.k_max) {
                ks.push_back(k);
            }
        }
    }
    for (int g = 1; g <= grid_points; ++g) {
        const double frac = static_cast<double>(g) / (grid_points + 1);
        ks.push_back(std::round(range.k_min + (frac * (range.k_max - range.k_min))));
    }
    if (job.reps != nullptr) {
        for (const int32_t k : *job.reps) {
            if (k >= range.k_min && k <= range.k_max) {
                ks.push_back(static_cast<double>(k));
            }
        }
    }
}

// Step `v` by ulps toward `dir` until the map puts it on the wanted side of t.
// Bounded: the map is monotone, so a few steps suffice unless it is flat at
// double precision, where nothing will.
double step_off_edge(const ArgMap& map, double v, double t, bool want_below, double dir) {
    for (int i = 0; i < 8; ++i) {
        const double u = map.to_u(v);
        if (want_below ? u < t : u > t) {
            break;
        }
        v = std::nextafter(v, dir);
    }
    return v;
}

// The values of v one Ceil/Floor/Round edge t proposes (see the section comment).
void offer_edge(const ArgMap& map, const Variable& var, NodeOp op, double t,
                std::vector<double>& out) {
    const double vm = map.to_v(t);
    const double at = std::abs(t);
    const double du = std::min(std::max(1e-9, 4.0 * (std::nextafter(at, kInf) - at)), 0.25);
    double lo = map.to_v(t - du);
    double hi = map.to_v(t + du);
    if (!std::isfinite(vm) || !std::isfinite(lo) || !std::isfinite(hi)) {
        return;
    }
    if (lo > hi) {
        std::swap(lo, hi);
    }
    if (var.type == VarType::Int) {
        out.push_back(clamp_to_domain(var, std::floor(lo)));
        out.push_back(clamp_to_domain(var, std::ceil(hi)));
        if (std::abs(vm - std::round(vm)) <= 1e-9 * std::max(1.0, std::abs(vm))) {
            out.push_back(clamp_to_domain(var, std::round(vm)));
        }
        return;
    }
    // The lower v lands below t when u grows with v, above it otherwise.
    const bool up = map.increasing();
    lo = step_off_edge(map, std::min(lo, std::nextafter(vm, -kInf)), t, up, -kInf);
    hi = step_off_edge(map, std::max(hi, std::nextafter(vm, kInf)), t, !up, kInf);
    out.push_back(clamp_to_domain(var, lo));
    out.push_back(clamp_to_domain(var, hi));
    // The edge itself, when the map puts it in the plateau the edge belongs to.
    const bool belongs_above = op == NodeOp::Floor || (op == NodeOp::Round && t > 0.0);
    const double um = map.to_u(vm);
    if (belongs_above ? um >= t : um <= t) {
        out.push_back(clamp_to_domain(var, vm));
    }
}

// The values of v an Element index k proposes (see the section comment).
void offer_index(const ArgMap& map, const BreakpointRange& range, const Variable& var, double k,
                 std::vector<double>& out) {
    if (var.type == VarType::Int) {
        const double vm = map.to_v(k);
        if (std::isfinite(vm)) {
            out.push_back(clamp_to_domain(var, std::floor(vm)));
            out.push_back(clamp_to_domain(var, std::ceil(vm)));
        }
        return;
    }
    const double lo = std::max(k == 0.0 ? -1.0 : k, range.u_lo);
    const double hi = std::min(k == 0.0 ? 1.0 : k + 1.0, range.u_hi);
    if (lo > hi) {
        return;
    }
    const double vm = map.to_v(0.5 * (lo + hi));
    if (std::isfinite(vm)) {
        out.push_back(clamp_to_domain(var, vm));
    }
}

// Every breakpoint candidate for `var_id`, in the order the cone reaches them,
// appended to `out` (not yet deduplicated).
void breakpoint_jump_candidates(const Model& model, int32_t var_id, const Variable& var, double x0,
                                std::vector<double>& out) {
    if (!(var.lb < var.ub) || !variable_reaches_breakpoint(model, var_id)) {
        return;  // pinned, or nothing above it to propose from
    }
    thread_local ConeSlopes cone;
    thread_local std::vector<BreakpointJob> jobs;
    thread_local std::vector<double> ks;
    cone.build(model, var_id);
    jobs.clear();
    for (const int32_t nid : cone.cone()) {
        const ExprNode& nd = model.nodes()[static_cast<size_t>(nid)];
        if (!is_breakpoint_op(nd.op)) {
            continue;
        }
        const ConstSpan<ChildRef> kids = model.children(nd);
        for (size_t c = 0; c < kids.size(); ++c) {
            BreakpointJob job;
            job.node = &nd;
            if (!argument_map(model, cone, kids[c], x0, job.map)) {
                continue;  // this argument does not move with the variable
            }
            job.range = breakpoint_range(model, nd, c, var, job.map);
            if (nd.op == NodeOp::Element) {
                job.reps = element_reps(model, nd, c);
            }
            job.capped = is_capped(job);
            jobs.push_back(job);
        }
    }
    const auto capped = static_cast<int>(
        std::count_if(jobs.begin(), jobs.end(), [](const BreakpointJob& j) { return j.capped; }));
    const int grid_points = capped == 0 ? 0 : std::max(1, (kJumpGridPoints - 1) / capped);
    for (const BreakpointJob& job : jobs) {
        breakpoint_ks(job, grid_points, ks);
        for (const double k : ks) {
            if (job.node->op == NodeOp::Element) {
                offer_index(job.map, job.range, var, k, out);
            } else {
                offer_edge(job.map, var, job.node->op, k + job.range.offset, out);
            }
        }
    }
}

// Offer `cands` to `consider` in their order, skipping any value offered earlier
// in this list or already in `seen` (the Int/Float path's finite candidates).
// Sorting indices rather than stamping values: a double has no index to stamp,
// and the lists are a few hundred long at most. The index tie-break makes the
// sort's order total without `stable_sort`'s buffer allocation.
template <class Consider>
void consider_unique(std::vector<double>& cands, std::vector<double>& seen, Consider&& consider) {
    thread_local std::vector<size_t> order;
    thread_local std::vector<uint8_t> keep;
    std::sort(seen.begin(), seen.end());
    order.resize(cands.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&cands](size_t a, size_t b) {
        return cands[a] < cands[b] || (cands[a] == cands[b] && a < b);
    });
    keep.assign(cands.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        const double v = cands[order[i]];
        const bool first = i == 0 || cands[order[i - 1]] != v;
        keep[order[i]] = static_cast<uint8_t>(first && std::isfinite(v) &&
                                              !std::binary_search(seen.begin(), seen.end(), v));
    }
    for (size_t i = 0; i < cands.size(); ++i) {
        if (keep[i] != 0) {
            consider(cands[i]);
        }
    }
}

// Relative probe steps. 1e-6 breaks an *exact* stationary point, where the
// first-order model says nothing and any nonzero step is information; 1e-2
// then covers ground, because the gain at a quadratic stationary point is
// O(h^2) and a 1e-6 step alone makes the search crawl.
constexpr std::array<double, 2> kEscapeRelSteps = {1e-6, 1e-2};

// The local move Float otherwise lacks entirely. `int_jump_candidates` always
// offers x0 +/- 1; Float had only a Newton step (length set by the target, and
// vanishing with the gradient) plus three box constants, so a Float at an
// interior stationary point had an empty neighbourhood and froze there.
//
// Two-sided because at a saddle the descent direction is exactly what a zero
// gradient cannot supply — it has to be sampled.
template <class Consider>
void float_escape_candidates(const Variable& var, double x0, Consider&& consider) {
    // Scaled on |x0|, deliberately NOT on box width: for an unbounded NL column
    // that width is the inf_clamp artifact rather than information. The +1 keeps
    // the step nonzero at x0 == 0 and keeps x0 +/- h distinct from x0.
    const double scale = std::abs(x0) + 1.0;
    for (double rel : kEscapeRelSteps) {
        const double h = rel * scale;
        const double up = clamp_to_domain(var, x0 + h);
        const double down = clamp_to_domain(var, x0 - h);
        if (up != var.ub) {  // lb/ub are already candidates; don't pay twice
            consider(up);
        }
        if (down != var.lb) {
            consider(down);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

JumpResult compute_var_jump(Model& model, const std::vector<double>& weights, int32_t var_id,
                            bool allow_escape_probe, LinearJumpScorer* linear) {
    const Variable& var = model.var(var_id);
    const double x0 = var.value;

    // f(j) = weighted violation delta of moving var_id to j (0 at the current
    // value). The best jump minimises f; score is the reduction -min f.
    //
    // In closed form when every weighted row of G_v allows it (prepared once per
    // call, O(|G_v|)); a non-finite candidate or step -- an infinite Float bound
    // -- is one the closed form does not model, so that one candidate takes the
    // probe. The probe restores the assignment exactly, so the snapshot survives.
    const bool fast = linear != nullptr && linear->prepare(var_id, weights);
    auto f = [&](double j) {
        if (fast && std::isfinite(j - x0)) {
            return linear->delta(j);
        }
        return model.weighted_violation_delta(var_id, j, weights);
    };

    double best_j = x0;
    double best_f = 0.0;  // f(x0) == 0
    auto consider = [&](double j) {
        if (j == x0) {
            return;
        }
        double fv = f(j);
        if (fv < best_f) {
            best_f = fv;
            best_j = j;
        }
    };
    // A variable that reaches a breakpoint node (#186) records what the Int/Float
    // path offered, so its breakpoint candidates can skip those values; every
    // other variable pays one predictable branch per candidate and records
    // nothing. Only FINITE values are recorded: a free Float's box midpoint is
    // NaN, and a NaN in `seen` would make its sort undefined.
    const bool breakpoints =
        model.has_breakpoint_nodes() && variable_reaches_breakpoint(model, var_id);
    thread_local std::vector<double> seen;
    thread_local std::vector<double> extra;
    if (breakpoints) {
        seen.clear();
        extra.clear();
    }
    auto consider_seen = [&](double j) {
        if (breakpoints && std::isfinite(j)) {
            seen.push_back(j);
        }
        consider(j);
    };

    if (var.type == VarType::Bool) {
        consider(1.0 - x0);
    } else if (var.type == VarType::Int) {
        const bool complete = int_jump_candidates(var, x0, consider_seen);
        if (!complete && breakpoints) {
            breakpoint_jump_candidates(model, var_id, var, x0, extra);
            consider_unique(extra, seen, consider);
        }
    } else if (var.type == VarType::Float) {
        // The probe is a LAST RESORT and must stay one. Firing it whenever a
        // variable is stationary and nothing improved — which is the steady
        // state of local search — measured ~9x worse on shiporig across every
        // seed: the drip of tiny improvements suppresses stagnation, so
        // diversification never fires. Hence `allow_escape_probe`, which the
        // search loop arms only once it is genuinely stuck.
        const bool gradient_usable =
            float_jump_candidates(model, var_id, var, x0, linear, consider_seen);
        // After the Newton candidates, so those keep winning a tie (#186).
        if (breakpoints) {
            breakpoint_jump_candidates(model, var_id, var, x0, extra);
            consider_unique(extra, seen, consider);
        }
        if (allow_escape_probe && !gradient_usable && best_f >= 0.0) {
            float_escape_candidates(var, x0, consider);
        }
    }

    return {best_j, -best_f};
}

void gls_update_weights(ViolationManager& vm, double rho) {
    const size_t nc = vm.weights.size();
    for (size_t c = 0; c < nc; ++c) {
        vm.weights[c] *= rho;
        // Bump only active (weight > 0) constraints that are currently violated;
        // masked constraints (weight 0) stay 0.
        if (vm.weights[c] > 0.0 && vm.constraint_violation(static_cast<int>(c)) > kTol) {
            vm.weights[c] += 1.0;
        }
    }
    vm.invalidate_cache();
}

// ---------------------------------------------------------------------------
// LazyWeightDecay (#175): see the header for the representation and its bounds
// ---------------------------------------------------------------------------

double LazyWeightDecay::decay(std::vector<double>& w, double rho) {
    const double next = scale_ * rho;
    // Written so that NaN takes the fold: it fails both comparisons.
    if (next >= kMinScale && next <= kMaxScale) {
        scale_ = next;
        step_ = 1.0 / scale_;
        return 1.0;
    }
    return fold(w, next);
}

double LazyWeightDecay::materialise(std::vector<double>& w) {
    if (scale_ == 1.0) {
        return 1.0;  // already effective: a rho = 1 batch never pays the sweep
    }
    return fold(w, scale_);
}

double LazyWeightDecay::fold(std::vector<double>& w, double factor) {
    constexpr double kFloor = std::numeric_limits<double>::denorm_min();
    for (double& x : w) {
        const double y = x * factor;
        // A positive weight decayed by a positive factor is positive in exact
        // arithmetic; do not let an underflow mask the row for good. 0 stays 0,
        // and a factor of exactly 0 (rho = 0) zeroes it as the eager form does.
        x = (y == 0.0 && x > 0.0 && factor > 0.0) ? kFloor : y;
    }
    scale_ = 1.0;
    step_ = 1.0;
    return factor;
}

// ---------------------------------------------------------------------------
// FeasibilityJump
// ---------------------------------------------------------------------------

FeasibilityJump::FeasibilityJump(Model& model, ViolationManager& vm, RNG& rng, GFJConfig config)
    : model_(model), vm_(vm), rng_(rng), config_(config), jumps_(model.num_vars()), linear_(model) {
    const size_t nc = model_.constraint_ids().size();
    violated_.assign(nc, 0);
    violated_pos_.assign(nc, -1);
    active_violated_of_var_.assign(model_.num_vars(), 0);
    in_queue_.assign(model_.num_vars(), 0);
    is_linear_.assign(nc, 0);
    linear_.resize_rows(nc);
    vars_of_constraint_.assign(nc, {});

    objective_ci_ = model_.objective_constraint_idx();

    compute_linear_constraints();
    build_row_slots();

    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v)) {
            continue;
        }
        for (int32_t c : model_.constraints_of_var(v)) {
            vars_of_constraint_[c].push_back(v);
        }
    }
}

// The complement of `is_structured` (dag.h): between them the two partition
// VarType, and `solve()` relies on that to initialise every variable exactly once
// (#108) — FJ sets the scalars here, `initialize_structured_random` sets the rest.
// Deliberately a whitelist, not `!is_structured(t)`: a VarType added later must
// opt in to being jumped rather than default into it.
bool FeasibilityJump::jumpable(int32_t var_id) const {
    auto t = model_.var(var_id).type;
    return t == VarType::Bool || t == VarType::Int || t == VarType::Float;
}

bool FeasibilityJump::active(int32_t constraint_idx) const {
    return vm_.weights[constraint_idx] > 0.0;
}

// ---- Incremental violated-row state (#174); see violated_ in the header ----

void FeasibilityJump::reconcile_counted(int32_t c) {
    const auto ci = static_cast<size_t>(c);
    const bool want = (violated_[ci] & kInV) != 0 && active(c);
    const bool have = (violated_[ci] & kCounted) != 0;
    if (want == have) {
        return;
    }
    const int32_t step = want ? 1 : -1;
    for (const int32_t v : vars_of_constraint_[ci]) {
        active_violated_of_var_[static_cast<size_t>(v)] += step;
    }
    violated_[ci] = static_cast<uint8_t>(want ? (kInV | kCounted) : (violated_[ci] & kInV));
}

void FeasibilityJump::uncount(int32_t c) {
    const auto ci = static_cast<size_t>(c);
    if ((violated_[ci] & kCounted) == 0) {
        return;
    }
    for (const int32_t v : vars_of_constraint_[ci]) {
        --active_violated_of_var_[static_cast<size_t>(v)];
    }
    violated_[ci] = kInV;
}

void FeasibilityJump::set_violated(int32_t c, bool now) {
    const auto ci = static_cast<size_t>(c);
    const bool was = (violated_[ci] & kInV) != 0;
    if (now) {
        if (!was) {
            violated_pos_[ci] = static_cast<int32_t>(violated_rows_.size());
            violated_rows_.push_back(c);
            violated_[ci] = kInV;
        }
        reconcile_counted(c);
        return;
    }
    if (!was) {
        return;
    }
    uncount(c);
    violated_[ci] = 0;
    const auto pos = static_cast<size_t>(violated_pos_[ci]);
    const int32_t last = violated_rows_.back();
    if (pos + 1 != violated_rows_.size()) {
        violated_rows_[pos] = last;
        violated_pos_[static_cast<size_t>(last)] = static_cast<int32_t>(pos);
    }
    violated_rows_.pop_back();
    violated_pos_[ci] = -1;
}

void FeasibilityJump::reconcile_all_counted() {
    for (const int32_t c : violated_rows_) {
        reconcile_counted(c);
    }
}

void FeasibilityJump::rebuild_violated_index() {
    std::fill(active_violated_of_var_.begin(), active_violated_of_var_.end(), 0);
    violated_rows_.clear();
    const size_t nc = violated_.size();
    for (size_t c = 0; c < nc; ++c) {
        violated_[c] = static_cast<uint8_t>(violated_[c] & kInV);
        if (violated_[c] == 0) {
            violated_pos_[c] = -1;
            continue;
        }
        violated_pos_[c] = static_cast<int32_t>(violated_rows_.size());
        violated_rows_.push_back(static_cast<int32_t>(c));
        reconcile_counted(static_cast<int32_t>(c));
    }
}

bool FeasibilityJump::row_violated(int32_t ci) const {
    if (ci < 0 || static_cast<size_t>(ci) >= violated_.size()) {
        throw std::out_of_range("FeasibilityJump::row_violated: no such row");
    }
    return (violated_[static_cast<size_t>(ci)] & kInV) != 0;
}

std::vector<int32_t> FeasibilityJump::violated_rows() const {
    std::vector<int32_t> rows = violated_rows_;
    std::sort(rows.begin(), rows.end());
    return rows;
}

int32_t FeasibilityJump::active_violated_rows_of(int32_t var_id) const {
    if (var_id < 0 || static_cast<size_t>(var_id) >= active_violated_of_var_.size()) {
        throw std::out_of_range("FeasibilityJump::active_violated_rows_of: no such variable");
    }
    return active_violated_of_var_[static_cast<size_t>(var_id)];
}

void FeasibilityJump::enqueue(int32_t var_id) {
    if (in_queue_[var_id] == 0) {
        in_queue_[var_id] = 1;
        queue_.push_back(var_id);
    }
}

namespace {

// True when no child of `nd` reaches a variable, i.e. the whole subtree is a
// compile-time constant. Vacuously true for a leaf (a Const node has no
// children); a variable child is never constant, since its value is search
// state.
bool children_all_const(ConstSpan<ChildRef> children, const std::vector<uint8_t>& is_const) {
    return std::all_of(children.begin(), children.end(),
                       [&](const ChildRef& ch) { return !ch.is_var && is_const[ch.id] != 0; });
}

// Is this node affine in the variables, given the same classification already
// settled for every node below it? Only called for nodes that are NOT wholly
// constant, so `children` is populated for every op that indexes it.
bool node_is_affine(NodeOp op, ConstSpan<ChildRef> children, const std::vector<uint8_t>& is_const,
                    const std::vector<uint8_t>& is_affine) {
    auto child_const = [&](const ChildRef& c) -> bool {
        return c.is_var ? false : static_cast<bool>(is_const[c.id]);
    };
    auto child_affine = [&](const ChildRef& c) -> bool {
        return c.is_var ? true : static_cast<bool>(is_affine[c.id]);
    };
    switch (op) {
        case NodeOp::Const:
        case NodeOp::Neg:
        case NodeOp::Sum:
            return std::all_of(children.begin(), children.end(), child_affine);
        case NodeOp::Prod:  // affine if at most one child non-constant
            return (child_const(children[0]) && child_affine(children[1])) ||
                   (child_const(children[1]) && child_affine(children[0]));
        case NodeOp::Div:  // affine / const
            return child_affine(children[0]) && child_const(children[1]);
        case NodeOp::Leq:
        case NodeOp::Geq:
        case NodeOp::Lt:
        case NodeOp::Gt:  // residual lhs-rhs is affine if both sides affine
            return child_affine(children[0]) && child_affine(children[1]);
        // #186's Element/Ceil/Floor/Round (piecewise constant) and LambdaExtra/
        // PairLambdaExtra (user code) land here DELIBERATELY: never affine, so a
        // row over one is never scored in closed form by LinearJumpScorer and
        // takes the probe. Pinned by "a row over a piecewise-constant op is not
        // scored in closed form" (tests/test_element_rounding.cpp).
        default:  // Eq (abs), Neq (step), Pow, Min, Max, trig, etc.
            return false;
    }
}

// Can LinearJumpScorer score this row in closed form? Two shapes qualify.
//
// A comparison whose two children are both affine in the variables. Asked of the
// CHILDREN, not of the row's own affineness, because Eq is |lhs - rhs| -- not
// affine, yet exactly computable from two affine sides.
//
// A bare body (`add_constraint(expr)`, read as expr <= 0) that is itself affine
// (#190): the scorer models it as the residual `expr - 0`, which is the node
// value for every input (linear_jump.h). Asked of the row's own affineness:
// Sum/Neg/Prod-by-constant/Div-by-constant over affine children. Neq (a step),
// Custom, and every nonlinear op never qualify either way.
//
// Both shapes read `cf_affine`, not node_is_affine's verdict: a comparison
// NESTED below the row is not affine for scoring, because its value is a
// residual that reads a literal +/-inf bound as a sentinel (0, whatever moves
// the other side) -- `p + r D` would move the row while the DAG keeps it. A
// constant subtree stays affine: it cannot move.
bool closed_form_row(const ExprNode& nd, int32_t nid, ConstSpan<ChildRef> children,
                     const std::vector<uint8_t>& cf_affine) {
    if (!is_comparison_op(nd.op)) {
        return cf_affine[nid] != 0;
    }
    return std::all_of(children.begin(), children.end(),
                       [&](const ChildRef& c) { return c.is_var || cf_affine[c.id] != 0; });
}

}  // namespace

void FeasibilityJump::compute_linear_constraints() {
    const auto& nodes = model_.nodes();
    const size_t nn = nodes.size();
    std::vector<uint8_t> is_const(nn, 0);
    std::vector<uint8_t> is_affine(nn, 0);
    // As is_affine, except that a non-constant comparison node is not affine:
    // the closed form's notion (closed_form_row). is_affine keeps comparisons,
    // since is_linear_ (two-phase GLS's mask) classifies a ROW by its own node.
    std::vector<uint8_t> cf_affine(nn, 0);

    // topo_order has children before parents.
    for (int32_t nid : model_.topo_order()) {
        const ExprNode& nd = nodes[nid];
        const ConstSpan<ChildRef> children = model_.children(nd);
        const bool all_const = children_all_const(children, is_const);
        is_const[nid] = static_cast<uint8_t>(all_const);
        // A constant subtree is affine, and short-circuiting there is what keeps
        // node_is_affine from indexing the children of a childless leaf.
        is_affine[nid] =
            static_cast<uint8_t>(all_const || node_is_affine(nd.op, children, is_const, is_affine));
        cf_affine[nid] = static_cast<uint8_t>(
            all_const ||
            (!is_comparison_op(nd.op) && node_is_affine(nd.op, children, is_const, cf_affine)));
    }

    const auto& cids = model_.constraint_ids();
    for (size_t c = 0; c < cids.size(); ++c) {
        is_linear_[c] = is_affine[cids[c]];
        const ExprNode& nd = nodes[cids[c]];
        linear_.set_row_eligible(static_cast<int32_t>(c),
                                 closed_form_row(nd, cids[c], model_.children(nd), cf_affine));
    }
}

// Every per-row and per-variable table here is sized once, at construction, and
// never grown. The model can still gain a row afterwards: the objective row, which
// `add_objective_soft_constraint` appends -- and `freeze()` and the first `solve()`
// of a model with an objective both run it. So can the ViolationManager's weights
// fall out of step, if the manager was built before that row. After either,
// `violated_[ci]`, `is_linear_[ci]`, `vars_of_constraint_[ci]` and `active(ci)` are
// indexed by the NEW row count, unchecked -- so the mismatch is refused here rather
// than read past; build the ViolationManager and this object after the row exists.
//
// Checked at the entry points a driver calls once per batch or kick, never per row:
// eight size compares against a body that then runs thousands of iterations.
void FeasibilityJump::require_tables_in_step() const {
    const size_t nc = model_.constraint_ids().size();
    const size_t nv = model_.num_vars();
    if (violated_.size() != nc || is_linear_.size() != nc || linear_.num_rows() != nc ||
        vars_of_constraint_.size() != nc || violated_pos_.size() != nc || in_queue_.size() != nv ||
        active_violated_of_var_.size() != nv || vm_.weights.size() != nc) {
        throw std::logic_error(
            "FeasibilityJump: the model or its ViolationManager no longer has the row and "
            "variable counts this object was built for -- a row (the objective row that freeze() "
            "and solve() add) was added after it or the manager was built. Build both after the "
            "model's last row");
    }
}

void FeasibilityJump::set_initial_assignment() {
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v)) {
            continue;
        }
        Variable& var = model_.var_mut(v);
        double target = clamp_to_domain(var, 0.0);
        if (var.type == VarType::Bool || var.type == VarType::Int) {
            target = std::round(target);
        }
        var.value = target;
    }
}

// Must stay definitionally identical to the incremental maintenance in
// update_var: same rows skipped, same predicate on the residual. The whole point
// of re-grounding is that the exact sum and the accumulated one mean the same
// thing, and tests/test_feasibility_jump.cpp pins them against each other.
void FeasibilityJump::refresh_unweighted_violation() {
    const auto& cids = model_.constraint_ids();
    const size_t nc = cids.size();
    double total = 0.0;
    for (size_t c = 0; c < nc; ++c) {
        const auto ci = static_cast<int32_t>(c);
        if (ci == objective_ci_ || !active(ci)) {
            continue;  // objective row: see unweighted_violation_. Masked: not being solved.
        }
        total += progress_residual(model_.node_value(cids[c]));
    }
    unweighted_violation_ = total;
}

void FeasibilityJump::rebuild_violated_and_scan_set() {
    const auto& cids = model_.constraint_ids();
    const size_t nc = cids.size();
    for (const int32_t c : uncertain_rows_) {
        in_uncertain_[static_cast<size_t>(c)] = 0;
    }
    uncertain_rows_.clear();
    for (size_t c = 0; c < nc; ++c) {
        const double residual = model_.node_value(cids[c]);
        violated_[c] = is_violated(residual) ? kInV : 0;
        note_row_certainty(static_cast<int32_t>(c), residual);
    }
    rebuild_violated_index();
    refresh_unweighted_violation();
    std::fill(in_queue_.begin(), in_queue_.end(), 0);
    queue_.clear();
    // A counted row is exactly one in V and active. The rebuild has just laid V
    // out ascending, so this visits rows in index order -- a by-product of the
    // O(#rows) rebuild, not a requirement: any deterministic order would do.
    for (const int32_t c : violated_rows_) {
        if ((violated_[static_cast<size_t>(c)] & kCounted) != 0) {
            for (int32_t v : vars_of_constraint_[static_cast<size_t>(c)]) {
                enqueue(v);
            }
        }
    }
    jumps_.invalidate_all();
}

void FeasibilityJump::update_var(int32_t var_id) {
    const auto& cids = model_.constraint_ids();
    const ConstSpan<int32_t> gv = model_.constraints_of_var(var_id);
    // Only the rows this variable takes part in can move, so the running
    // unweighted total is maintained over `gv` rather than recomputed over every
    // row. The "before" side has to be read here, ahead of delta_evaluate.
    double violation_delta = 0.0;
    for (int32_t c : gv) {
        if (c != objective_ci_ && active(c)) {
            violation_delta -= progress_residual(model_.node_value(cids[c]));
        }
    }

    const double j = jumps_.jump_value(var_id);
    Variable& var = model_.var_mut(var_id);
    const double old_value = var.value;
    var.value = j;
    // Each incremental Sum in the cone moves by its one changed term instead of
    // re-summing (#177, #188): the re-sum's bits wherever the update is exact,
    // and otherwise a drift its bound accounts for.
    commit_scalar_move(model_, var_id, old_value);
    jumps_.invalidate(var_id);

    // Every row of gv is settled (in V or not, counted or not) before any
    // neighbour is tested below, so a neighbour sharing several rows with var_id
    // sees all of them -- as the O(|G_vp|) rescan this count replaced did. A
    // row whose verdict its Sum's drift now leaves undecided is noted for the
    // local-minimum gate (#188): one row-to-slot load per row, and for a row
    // over an incremental Sum one load of its state, which ends the test while
    // the Sum is not drifting.
    for (int32_t c : gv) {
        const double after = model_.node_value(cids[c]);
        if (c != objective_ci_ && active(c)) {
            violation_delta += progress_residual(after);
        }
        set_violated(c, is_violated(after));
        note_row_certainty(c, after);
    }
    unweighted_violation_ += violation_delta;
    // A vp sharing several rows with var_id is visited once per shared row. Not
    // deduplicated (#174 proposed a stamp): with the participation test now O(1),
    // a repeat visit costs an idempotent invalidate on a line the first visit
    // already touched, a byte load and at most one 4-byte count load -- no more
    // than a stamp check-and-set, which would also charge every FIRST visit and
    // 4 B per variable.
    for (int32_t c : gv) {
        for (int32_t vp : vars_of_constraint_[c]) {
            if (vp == var_id) {
                continue;
            }
            jumps_.invalidate(vp);
            if (in_queue_[vp] == 0 && participates_in_active_violated(vp)) {
                enqueue(vp);
            }
        }
    }
}

// ---- The incremental Sums' drift (#188) ----
//
// A committed move updates each incremental Sum in its cone by its term's
// change (commit_scalar_move), and an inexact update leaves the Sum off the
// real sum of its terms by at most its drift bound. A row reads its Sum
// directly (classify_incremental_sums), so the row's residual is off by at most
// that bound too. FJ acts on residuals in three places, and each must not act
// on drift:
//
//   - the GLS bump at a local minimum raises the weight of every row in V. A
//     row in V only by drift would get a weight the reference algorithm never
//     gives it -- and weights persist across batches, so the bias would too.
//     The mirror case is a row really violated but out of V by drift, which the
//     bump would skip.
//   - "Feasible", read off an empty V.
//   - the batch end, and anything that reads the model after it.
//
// The gate (settle_undecided_rows) handles the first two at O(1) in the common
// case: update_var notes every row whose verdict -- residual against kTol --
// the bound leaves undecided, in either direction, and at a local minimum only
// those rows' Sums are re-summed. A row whose residual clears kTol by more
// than its bound has the same verdict on the real sum; on integral data no
// update is inexact, no row is ever noted, and the gate costs one empty-list
// test. The re-groundings (reground_drifted_rows) re-sum every drifted Sum
// before a Feasible verdict and at the end of every batch, so the verdict is
// the re-sums' own and nothing outside a batch sees drift.

bool FeasibilityJump::verdict_undecided(int32_t c, double residual) const {
    const int32_t slot = row_slot_[static_cast<size_t>(c)];
    if (slot < 0) {
        return false;
    }
    const IncSumState& st = model_.inc_sums().slots[static_cast<size_t>(slot)];
    if (st.drifting == 0 || !std::isfinite(residual)) {
        return false;
    }
    // The row computes fl(S - q) (or |S - q|, or S - q plus the strict margin,
    // rounded once more) where the real sum T would give T - q: they differ by
    // at most M = drift_bound + 2^-51 |residual| (two half-ulps of the
    // residual). The verdict is decided when |residual - kTol| > M in real
    // arithmetic. Both sides are computed in floating point: the difference
    // rounds by at most a factor (1 + u), and M's one add by as much again, so
    // the margin is nudged up the way the drift bound is (see
    // `round_bound_up` in src/dag_ops.cpp), to at least M (1 + u) -- then a
    // computed "decided" is decided exactly.
    const double m = st.drift_bound + (std::fabs(residual) * 0x1p-51);
    const double margin = m + (m * 0x1p-51);
    return std::fabs(residual - kTol) <= margin;
}

void FeasibilityJump::note_row_certainty(int32_t c, double residual) {
    const auto ci = static_cast<size_t>(c);
    if (in_uncertain_[ci] == 0 && verdict_undecided(c, residual)) {
        in_uncertain_[ci] = 1;
        uncertain_rows_.push_back(c);
    }
}

void FeasibilityJump::capture_rows_of_slot(int32_t slot) {
    const auto& cids = model_.constraint_ids();
    const auto s = static_cast<size_t>(slot);
    for (uint32_t k = slot_rows_begin_[s]; k < slot_rows_begin_[s + 1]; ++k) {
        const int32_t c = slot_rows_[k];
        row_before_.emplace_back(c, model_.node_values()[cids[static_cast<size_t>(c)]]);
    }
}

// O(|uncertain_rows_|) plus, per Sum it re-sums, that Sum's length and its
// rows' variables. Where it loses: a model whose rows sit within their drift
// bound of kTol at most local minima -- big-M rows at equality, whose bound is
// ulps of M -- pays a re-sum per such row per minimum, which is what the
// exact-only commit paid per commit touching it.
bool FeasibilityJump::settle_undecided_rows() {
    if (uncertain_rows_.empty()) {
        return false;
    }
    const auto& cids = model_.constraint_ids();
    slot_scratch_.clear();
    row_before_.clear();
    for (const int32_t c : uncertain_rows_) {
        in_uncertain_[static_cast<size_t>(c)] = 0;
        // Re-checked: a later commit may have decided it -- or re-summed its Sum,
        // which a second row over the same Sum finds here no longer drifting.
        if (!verdict_undecided(c, model_.node_values()[cids[static_cast<size_t>(c)]])) {
            continue;
        }
        const int32_t slot = row_slot_[static_cast<size_t>(c)];
        capture_rows_of_slot(slot);
        reground_inc_sum(model_, slot);
        slot_scratch_.push_back(slot);
    }
    uncertain_rows_.clear();
    if (slot_scratch_.empty()) {
        return false;
    }
    IncrementalSumCounters& counts = incremental_sum_counters();
    ++counts.gated_minima;
    const bool flipped = settle_regrounded_rows();
    if (flipped) {
        ++counts.gate_flips;
    }
    return flipped;
}

bool FeasibilityJump::reground_drifted_rows() {
    for (const int32_t c : uncertain_rows_) {
        in_uncertain_[static_cast<size_t>(c)] = 0;
    }
    uncertain_rows_.clear();
    IncSums& sums = model_.inc_sums();
    if (sums.drifted.empty()) {
        return false;
    }
    slot_scratch_.clear();
    row_before_.clear();
    // Exactly the slots reground_drifted_sums re-sums: listed and still drifting.
    for (const int32_t slot : sums.drifted) {
        if (sums.slots[static_cast<size_t>(slot)].drifting != 0) {
            capture_rows_of_slot(slot);
        }
    }
    reground_drifted_sums(model_, slot_scratch_);
    return settle_regrounded_rows();
}

bool FeasibilityJump::settle_regrounded_rows() {
    const auto& cids = model_.constraint_ids();
    bool flipped = false;
    size_t n_moved = 0;
    // Compacted in place to the rows that moved: the write index never passes
    // the read one.
    for (const auto& [c, before] : row_before_) {
        const auto ci = static_cast<size_t>(c);
        const double after = model_.node_values()[cids[ci]];
        uint64_t before_bits = 0;
        uint64_t after_bits = 0;
        std::memcpy(&before_bits, &before, sizeof before_bits);
        std::memcpy(&after_bits, &after, sizeof after_bits);
        if (before_bits == after_bits) {
            continue;
        }
        if (c != objective_ci_ && active(c)) {
            unweighted_violation_ += progress_residual(after) - progress_residual(before);
        }
        const bool was = (violated_[ci] & kInV) != 0;
        const bool now = is_violated(after);
        set_violated(c, now);
        flipped = flipped || was != now;
        row_before_[n_moved++].first = c;
    }
    // As in update_var: every moved row is settled in V before any neighbour
    // is tested, so a neighbour sharing several of them sees all of them.
    for (size_t k = 0; k < n_moved; ++k) {
        const int32_t c = row_before_[k].first;
        for (const int32_t vp : vars_of_constraint_[static_cast<size_t>(c)]) {
            jumps_.invalidate(vp);
            if (in_queue_[vp] == 0 && participates_in_active_violated(vp)) {
                enqueue(vp);
            }
        }
    }
    row_before_.clear();
    return flipped;
}

void FeasibilityJump::build_row_slots() {
    const auto& cids = model_.constraint_ids();
    const auto& nodes = model_.nodes();
    const size_t nc = cids.size();
    const size_t ns = model_.inc_sum_nodes().size();
    // A closed model's state is sized to its slots by the full_evaluate that
    // ends every close; verdict_undecided indexes it by slot unchecked.
    assert(model_.inc_sums().slots.size() == ns);
    row_slot_.assign(nc, -1);
    in_uncertain_.assign(nc, 0);
    uncertain_rows_.clear();
    slot_rows_begin_.assign(ns + 1, 0);
    for (size_t c = 0; c < nc; ++c) {
        for (const ChildRef& ref : model_.children(nodes[cids[c]])) {
            if (!ref.is_var && (nodes[ref.id].inc_sum_flags & ExprNode::kIncSum) != 0) {
                const int32_t slot = nodes[ref.id].lambda_func_id;
                row_slot_[c] = slot;
                ++slot_rows_begin_[static_cast<size_t>(slot) + 1];
                break;  // a row reads at most one: its other side is a var or a Const
            }
        }
    }
    for (size_t s = 0; s < ns; ++s) {
        slot_rows_begin_[s + 1] += slot_rows_begin_[s];
    }
    slot_rows_.assign(slot_rows_begin_[ns], 0);
    std::vector<uint32_t> cursor(slot_rows_begin_.begin(), slot_rows_begin_.end() - 1);
    for (size_t c = 0; c < nc; ++c) {
        if (row_slot_[c] >= 0) {
            slot_rows_[cursor[static_cast<size_t>(row_slot_[c])]++] = static_cast<int32_t>(c);
        }
    }
}

bool FeasibilityJump::apply_jump(int sample_size) {
    // Sample up to `sample_size` DISTINCT positive-score variables from the scan
    // set Q and apply the best one's jump (ViolationLS, Davies et al. CPAIOR
    // 2024, Algorithm 2; OR-Tools' ScanRelevantVariables). A variable with a
    // non-positive score is removed from Q permanently (swap-remove) and does
    // not count toward the sample; positive ones stay in Q.
    //
    // There is no draw cap, and returning false therefore MEANS that Q holds no
    // positive-score variable -- the local minimum the GLS loop answers with a
    // weight bump. A cap on draws (#206: `sample_size * 8 + 16`, counting the
    // removals too) gave up while improving variables were still in Q whenever
    // they were rare in it, so a long violated row of non-improving variables
    // read as a false minimum, the bump re-queued the whole row, and the false
    // minimum repeated -- inflating every long row's weight far faster than the
    // reference allows.
    //
    // Distinctness without redraws: Q's prefix [0, n) holds the positives
    // sampled so far this call, and each draw is uniform over the unsampled
    // suffix [n, |Q|). A positive is swapped into the prefix; a non-positive is
    // swap-removed from the suffix (Q's back is in the suffix, so the prefix is
    // undisturbed). Every draw either grows the prefix or shrinks Q, so the loop
    // ends after at most |Q| draws and needs no backstop. Each removal is paid
    // for by the enqueue that put the variable there, so the uncapped scan
    // costs no more amortised than the capped one did.
    //
    // Reordering Q is harmless: in_queue_ is a flag, nothing reads Q's order
    // except this draw, and the draw is uniform over a set, so the sample's
    // distribution does not depend on where in Q a variable sits.
    int32_t best_v = -1;
    double best_score = 0.0;
    size_t n = 0;
    const auto want = static_cast<size_t>(std::max(sample_size, 0));
    while (n < queue_.size() && n < want) {
        const auto idx =
            n + static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(queue_.size() - n)));
        const int32_t v = queue_[idx];
        if (!jumps_.valid(v)) {
            JumpResult r = compute_var_jump(model_, vm_.weights, v, escape_probe_, &linear_);
            jumps_.set(v, r.jump_value, r.score);
        }
        const double s = jumps_.score(v);
        if (s <= 0.0) {
            in_queue_[v] = 0;
            queue_[idx] = queue_.back();
            queue_.pop_back();
            continue;
        }
        std::swap(queue_[n], queue_[idx]);
        ++n;
        if (s > best_score) {
            best_score = s;
            best_v = v;
        }
    }
    if (best_v < 0) {
        return false;
    }
    update_var(best_v);
    return true;
}

// O(|V|) over the violated list, reading each row's weight live -- so it answers
// correctly even for a weight changed since the row was last evaluated, which the
// counted bits would only catch at the next reconcile.
bool FeasibilityJump::any_active_violated() const {
    return std::any_of(violated_rows_.begin(), violated_rows_.end(),
                       [this](int32_t c) { return active(c); });
}

// Next deadline-check stride, given how long the last one actually took.
//
// Growth is capped at kStrideGrowth so the stride ramps up through
// progressively longer — and therefore more accurate — measurements instead of
// extrapolating the whole way from one short one. The cap never weakens the
// bound: an x8 growth is only taken when 8 x elapsed is still inside the
// target, so the predicted duration of the next stride is <= target either way.
//
// Shrinking is deliberately NOT capped: iterations that got more expensive must
// be caught on the very next check.
//
// An uncapped shrink is NOT, on its own, what stops the stride ratcheting upward
// and going silent — the failure mode that got an earlier self-tuning stride
// removed from this engine. It cannot be, because a shrink is only APPLIED at a
// check and the next check is a whole stride away: a stride grown while
// iterations were cheap is spent in full on the first expensive one, and the
// tuner learns nothing until after the damage. What bounds that is
// kMaxDeadlineStride, the hard iteration cap; see its comment.
int64_t FeasibilityJump::next_deadline_stride(int64_t stride, double elapsed_seconds,
                                              double target_seconds) {
    // A non-finite measurement carries no information; take the floor rather
    // than the growth cap, which is what an `elapsed > 0.0` test alone would
    // silently do with a NaN. Unreachable from a monotonic steady_clock, so this
    // is defensive, not load-bearing.
    if (std::isnan(elapsed_seconds) || std::isnan(target_seconds)) {
        return 1;
    }
    // elapsed <= 0 means the interval was below the clock's resolution, i.e.
    // far inside the target: grow by the cap.
    auto scale = static_cast<double>(kStrideGrowth);
    if (elapsed_seconds > 0.0) {
        scale = std::min(scale, target_seconds / elapsed_seconds);
    }
    const double next = static_cast<double>(stride) * scale;
    if (!(next > 1.0)) {
        return 1;  // never fewer than one iteration per check
    }
    if (next >= static_cast<double>(kMaxDeadlineStride)) {
        return kMaxDeadlineStride;
    }
    return static_cast<int64_t>(next);
}

// Arm (or disarm) the wall-clock deadline and reset the stride tuner. Called
// from both entry points, begin() and run(), so the tuner state cannot be left
// stale from a previous run.
void FeasibilityJump::arm_deadline() {
    has_deadline_ = config_.time_limit > 0.0;
    deadline_checks_ = 0;
    if (!has_deadline_) {
        return;  // no clock is read, and no clock-derived state exists, at all
    }
    const auto now = std::chrono::steady_clock::now();
    deadline_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(config_.time_limit));
    last_deadline_check_ = now;
    // Start at one iteration and let the tuner grow it. The first stride is the
    // one no measurement has bounded yet, and starting it at 64 on a model whose
    // iterations cost 100ms is the whole of #113.
    deadline_stride_ = 1;
    deadline_countdown_ = 1;
}

// The GLS inner loop (ApplyJump + stagnation weight bump), reusing the current
// state (Q, weights, jump table). Returns Feasible as soon as no active
// constraint is violated; Unsolved when the batch / global iteration budget or
// the deadline is hit. batch_iter_limit <= 0 means "no per-call limit".
//
// ---- How the deadline is observed (#113) ----
//
// Reading the clock is not free: steady_clock::now() measures 1408 ns/call on
// this project's reference machine, whose clocksource is HPET (it is ~20-25 ns
// through the vDSO on a TSC clocksource), against GLS iterations that can be a
// few microseconds. Checking every iteration measured 2996 -> 5228 ns per
// iteration, a 1.75x throughput loss, on a small Bool model — so the loop
// cannot simply check every time.
//
// It used to check on a fixed stride of 64 iterations, which is not a time
// bound at all: one GLS iteration is O(sampled vars x candidate values x
// constraints touched), so 64 of them are microseconds on a small model and
// seconds on a large one. Measured on 400 Int vars with 20 000 rows of 8: every
// budget from 0.05s to 3s took ~7s, always after exactly 64 iterations.
//
// So the stride is sized in *time* instead. Each check measures how long the
// previous stride took, which is the current cost of an iteration, and sizes
// the next stride to kStrideBudgetFraction of the total budget. The guarantee:
//
//   a batch returns at most one stride past the deadline, and a stride costs at
//   most 1/64 of the budget -- or one GLS iteration, whichever is larger,
//   because an iteration is atomic and cannot be pre-empted from the inside.
//
// Sizing the stride in time also bounds the clock overhead without having to
// measure the clock at all: one read per stride, against a stride costing
// budget/64, is 1.4us / (budget/64) even on the expensive clocksource — 0.45%
// of a 20ms budget, 0.009% of a 1s one. It degrades only for budgets so small
// (well under a millisecond) that the run is over before throughput matters.
//
// Two honest caveats. The prediction is a measurement, so an iteration whose
// cost jumps mid-stride overruns the target and is corrected only at the next
// check. And the interval between checks can span a batch boundary, so work the
// *outer* loop does between batches (hook, LNS, structural sweep) is charged to
// the stride, which shrinks it; that is conservative, never the reverse.
// No improving jump anywhere in the scan set. Bump the GLS weights, then
// invalidate and re-queue every variable of every active violated constraint, so
// the next iteration re-scores them against the new penalty landscape.
//
// The weight update is gls_update_weights' -- decay every row by rho, add 1 to
// every active violated row -- in LazyWeightDecay's representation (#175): the
// decay is O(1) on the global scale, and only the rows in V are touched, so the
// bump is O(|V|) plus the requeue below (the nonzeros of V's counted rows)
// rather than O(#rows). The algorithm's weights are unchanged (ViolationLS, Davies et al. CPAIOR
// 2024, Algorithm 3); only their storage is. The violated rows are V's, which is the eager form's
// `constraint_violation(c) > kTol` row for row: is_violated is the same kTol test and a NaN or +inf
// clamps to kInfPenalty there. That holds only while V is current -- update_var keeps it so inside
// the loop, and a caller that mutates the assignment outside FJ must resync() before the next batch
// (solve() does). A stale V now mis-weights rows, not just mis-queues them.
void FeasibilityJump::bump_weights_and_requeue() {
    const double folded = weight_decay_.decay(vm_.weights, config_.rho);
    if (folded != 1.0) {
        // Cached scores live in the scaled space too, and this rescale is
        // REQUIRED, not cosmetic. A valid entry that survives a bump has no
        // counted violated row, so its score is <= 0 -- except in the (0, kTol]
        // residual band, where it is positive. Such an entry left in Q
        // unrescaled would be up to 1 / folded (~1e30) times too large against
        // every score computed after the fold, and apply_jump's argmax would take
        // it over any real improving jump. Reachable wherever one gls_loop makes
        // more than 1347 decays at rho = 0.95 -- run()/gls(), or LNS repair's
        // fj_nl_initialize on continuous equality rows (#102) -- but only for an
        // entry sampling happened to miss, which is why no test pins this site;
        // the batch-API fence pins the rescale in materialise_weights.
        jumps_.scale_scores(folded);
    }
    vm_.invalidate_cache();
    // V in its list order. The order variables enter Q is the order apply_jump's
    // draw indexes, so it matters for the trajectory, but only that it is
    // deterministic: the list order is a function of the flip history alone
    // (appends and swap-removes), so a seeded run reproduces. It is NOT
    // ascending, and keeping it ascending would cost a sort per bump (#174 did,
    // for bit-identity with the whole-row sweep; the lazy decay gave that up).
    // Each row's counted bit is re-read against its bumped weight on the way
    // past (a fold by rho = 0 deactivates every row), which is also the only
    // weight read the bump's own bookkeeping needs -- rows outside V are never
    // looked at.
    for (const int32_t c : violated_rows_) {
        weight_decay_.bump(vm_.weights, static_cast<size_t>(c));
        reconcile_counted(c);
        if ((violated_[static_cast<size_t>(c)] & kCounted) != 0) {
            for (int32_t v : vars_of_constraint_[static_cast<size_t>(c)]) {
                jumps_.invalidate(v);
                enqueue(v);
            }
        }
    }
}

// Progress accounting for one GLS iteration. Returns true when the batch must
// end because it has stopped reducing the real rows' violation. See the
// unproductive-streak discussion on unweighted_violation_ in the header.
bool FeasibilityJump::track_batch_progress(double& batch_best_violation, bool watch_progress) {
    // A new minimum has to beat the incumbent by more than the accumulator can
    // drift within one batch, or ulp noise resets the streak and the exit never
    // fires. Relative, because an absolute floor does not survive scale (#118).
    auto improves = [](double v, double best) {
        return v < best - (kProgressRelEps * std::max(1.0, best));
    };

    // Holding ground counts as unproductive -- holding ground is exactly what
    // the cycling case does -- so only a strict new minimum, by more than the
    // drift floor, resets the streak.
    if (improves(unweighted_violation_, batch_best_violation)) {
        batch_best_violation = unweighted_violation_;
        unproductive_streak_ = 0;
        return false;
    }
    if (!watch_progress || ++unproductive_streak_ < config_.unproductive_iterations) {
        return false;
    }

    // The streak was accumulated on the incremental measure; ending the batch is
    // the one decision it drives, so make that decision on an exact sum instead.
    // This also re-grounds the accumulator, which is how a run that drifted (or
    // that lost a row to +inf and back) gets its measure repaired rather than
    // staying wrong for the whole run.
    refresh_unweighted_violation();
    if (improves(unweighted_violation_, batch_best_violation)) {
        batch_best_violation = unweighted_violation_;
        unproductive_streak_ = 0;
        return false;
    }
    if (unweighted_violation_ <= 0.0) {
        // The measure has no usable signal, which covers TWO states and
        // deliberately treats them alike. Either every real row is
        // satisfied (progress_residual counts only residuals > kTol, the
        // same threshold is_violated uses, so a zero sum means exactly
        // that for finite rows) -- or every violated real row is +inf or
        // NaN, which progress_residual also contributes 0 for. Both are
        // "nothing to measure", and neither is evidence of a stall.
        //
        // Testing "is any real row violated" instead was tried and is
        // WRONG: on a non-convex body that has gone non-finite it is
        // permanently true while the measure is permanently pinned, so
        // improves() can never fire and the batch reports stuck at every
        // streak limit for the rest of the run -- pre-feasibility, where
        // the kick still draws LNS. That trades a silently inert exit for
        // a budget-burning one on exactly the elec-class instances.
        //
        // The measure then sums to zero and can never improve on itself,
        // so every batch in the objective-descent phase would
        // report stuck at exactly unproductive_iterations, turning a
        // stall detector into an unconditional one. The search is not
        // stalled there; it is descending against the artificial
        // objective row, which this measure deliberately cannot see.
        // Having no signal is not evidence of being stuck, so hand the
        // batch back to its own limit and let perturbation_period keep
        // owning that regime.
        unproductive_streak_ = 0;
        return false;
    }
    return true;
}

// Deadline observation, reached only when the stride countdown has expired.
// Returns true if the deadline has passed; otherwise re-sizes the stride from
// what the last one actually cost. See the long comment above gls_loop.
bool FeasibilityJump::deadline_passed_and_retune() {
    const auto now = std::chrono::steady_clock::now();
    ++deadline_checks_;  // every clock read, including the one that stops the run
    if (now >= deadline_) {
        return true;
    }
    // Size the next stride against the budget that is LEFT, not the
    // budget that was given. A fraction of the total permits an overrun
    // of budget/64 right up to the deadline — 9.4 s on a 600 s
    // benchmark run, by design — whereas remaining/64 tightens as the
    // deadline approaches and costs nothing to compute.
    const double remaining = std::chrono::duration<double>(deadline_ - now).count();
    deadline_stride_ = next_deadline_stride(
        deadline_stride_, std::chrono::duration<double>(now - last_deadline_check_).count(),
        remaining * kStrideBudgetFraction);
    deadline_countdown_ = deadline_stride_;
    last_deadline_check_ = now;
    return false;
}

void FeasibilityJump::materialise_weights() {
    const double folded = weight_decay_.materialise(vm_.weights);
    if (folded != 1.0) {
        // A positive weight stays positive and 0 stays 0 (see fold), so no row's
        // active bit moves and the counted bits need no reconcile.
        jumps_.scale_scores(folded);
        vm_.invalidate_cache();
    }
}

GFJStatus FeasibilityJump::gls_loop(int sample_size, int64_t batch_iter_limit) {
    // The lazy decay's scale is local to the loop: whatever way the loop is left,
    // vm_.weights are effective weights again afterwards (#175).
    try {
        const GFJStatus status = gls_loop_scaled(sample_size, batch_iter_limit);
        // No drift leaves a batch (#188): the search, the pool, LNS and the
        // inner solver read re-summed rows. A no-op where the exit already
        // re-grounded (Feasible, and batch_end_status). The deadline and the
        // iteration budget return Unsolved without looking at V, and that
        // status stands: the re-grounding settles V for the caller, and the
        // caller reads feasibility off the re-summed rows, not off this status.
        reground_drifted_rows();
        materialise_weights();
        return status;
    } catch (...) {
        materialise_weights();
        throw;
    }
}

GFJStatus FeasibilityJump::gls_loop_scaled(int sample_size, int64_t batch_iter_limit) {
    int64_t batch_iters = 0;
    // Re-ground before taking the batch's reference minimum. Consecutive
    // non-improving FJ batches reach here without a rebuild in between, so
    // without this the accumulator would carry a whole stagnant run's rounding
    // into the comparison that decides whether the run IS stagnant.
    refresh_unweighted_violation();
    // The weights may have been changed between batches by something other than
    // the GLS bump; re-read them for every row in V before update_var trusts the
    // counts. O(|V|).
    reconcile_all_counted();
    double batch_best_violation = unweighted_violation_;
    unproductive_streak_ = 0;
    // Only the batch API has an outer loop to hand control back to; gls()/run()
    // passes no limit and must run its budget out.
    // watch_progress_ is the second witness: solve() arms it only once its own
    // stagnation count says the search has stopped improving, because this
    // measure on its own is unconditionally true on any plateau at positive
    // violation (see THE SECOND WITNESS on unweighted_violation_).
    const bool watch_progress =
        watch_progress_ && batch_iter_limit > 0 && config_.unproductive_iterations > 0;

    while (true) {
        // At a local minimum, every row whose verdict drift could have flipped
        // is re-summed before the verdicts are acted on (#188): a row violated
        // only by drift leaves V rather than being bumped, and one satisfied
        // only by drift enters it. If that moved a row across, the minimum was
        // found on stale values, so the loop samples again instead.
        if (!apply_jump(sample_size) && !settle_undecided_rows()) {
            if (!any_active_violated()) {
                // The gate settled every undecided row; the re-grounding makes
                // the verdict the re-sums' own, as it would be with no drift.
                reground_drifted_rows();
                if (!any_active_violated()) {
                    return GFJStatus::Feasible;
                }
            } else {
                bump_weights_and_requeue();
            }
        }

        ++iterations_;
        ++batch_iters;
        if (track_batch_progress(batch_best_violation, watch_progress)) {
            batch_stuck_ = true;
            return batch_end_status();
        }
        if (batch_iter_limit > 0 && batch_iters >= batch_iter_limit) {
            return batch_end_status();
        }
        if (config_.max_iterations > 0 && iterations_ >= config_.max_iterations) {
            return GFJStatus::Unsolved;
        }
        // Short-circuited on has_deadline_, so a run with no wall clock neither
        // reads the clock nor touches any of the tuner state: iteration-budgeted
        // runs stay bit-identical. The countdown decrement is likewise inside the
        // short circuit, and the clock read stays behind it, in the helper.
        if (has_deadline_ && --deadline_countdown_ <= 0 && deadline_passed_and_retune()) {
            return GFJStatus::Unsolved;
        }
    }
}

GFJStatus FeasibilityJump::gls(int sample_size) {
    rebuild_violated_and_scan_set();
    return gls_loop(sample_size, 0);
}

// ---- Batch API (drives ViolationLS Algorithm 6 from an outer loop) ----

void FeasibilityJump::begin(bool set_initial_x) {
    require_tables_in_step();
    iterations_ = 0;
    // A fresh run starts with the escape probe disarmed, alongside the iteration
    // count and the deadline. solve() constructs a FeasibilityJump per call so
    // this cannot matter today; it is here so a caller that reuses one instance
    // does not inherit the previous run's stagnation state (#117).
    escape_probe_ = false;
    unproductive_streak_ = 0;
    batch_stuck_ = false;
    watch_progress_ = true;
    arm_deadline();
    if (set_initial_x) {
        set_initial_assignment();
    }
    full_evaluate(model_);
    std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
    vm_.invalidate_cache();
    rebuild_violated_and_scan_set();
}

bool FeasibilityJump::batch(int64_t batch_iterations) {
    require_tables_in_step();
    batch_stuck_ = false;
    return gls_loop(config_.sample_size_general, batch_iterations) == GFJStatus::Feasible;
}

void FeasibilityJump::reset_weights() {
    require_tables_in_step();
    std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
    vm_.invalidate_cache();
    rebuild_violated_and_scan_set();
}

void FeasibilityJump::resync() {
    require_tables_in_step();
    rebuild_violated_and_scan_set();
}

int32_t FeasibilityJump::pick_forced_perturb_var() {
    const auto num_vars = static_cast<int32_t>(model_.num_vars());
    auto eligible = [this](int32_t v) { return jumpable(v) && movable_domain(model_.var(v)); };

    int32_t count = 0;
    for (int32_t v = 0; v < num_vars; ++v) {
        count += eligible(v) ? 1 : 0;
    }
    if (count == 0) {
        return -1;  // nothing can move; a no-op kick is the correct outcome
    }
    // Two passes rather than materialising the candidate list: the O(n) scan is
    // dwarfed by the full_evaluate the perturbation ends with.
    int64_t k = rng_.integers(0, count);
    for (int32_t v = 0; v < num_vars; ++v) {
        if (!eligible(v)) {
            continue;
        }
        if (k == 0) {
            return v;
        }
        --k;
    }
    return -1;  // unreachable: k < count eligible variables were skipped
}

// Reset the structural kick's move counter and stride tuner. Starting the stride
// at one move is the point of the whole bound: the first move of a kick is the
// one no measurement has sized yet, and on a model whose structure is a single
// large Set the entire quadratic run happens inside that first variable. A
// stride inherited from a previous kick would be spent there.
void FeasibilityJump::arm_structural_kick() {
    kick_moves_ = 0;
    kick_checks_ = 0;
    kick_stride_ = 1;
    kick_countdown_ = 1;
    if (has_deadline_) {
        last_kick_check_ = std::chrono::steady_clock::now();
    }
}

// Stop the structural pass? Checked between MOVES, which is what makes the
// bound mean anything on a model with one large structure (#115).
//
// The unit here is one structural move — one move-set generation plus one apply,
// O(|elements| + universe) element copies — which is the same unit the STRUCTURAL
// batch's sweep caps its overrun at (#105). Checking between *variables*, as this
// pass used to, capped nothing: a variable costs k = round(p * |elements|) of
// those moves, quadratic in its size, so a model whose structure lives in one
// List or Set ran the whole quadratic pass and then consulted the clock on its
// way to a variable that did not exist. Measured at 2.3 s for one kick on a
// 41k-element Set, with solve(time_limit=1.0) returning in 1.29 s.
//
// A move is not cheap enough to check the clock before every one of them: the
// shape this pass already handles well is many small structures, where a move on
// a 100-element List is ~1.5us against 1408 ns for steady_clock::now() on this
// project's HPET reference machine. So the check strides, and the stride is the
// one the GLS loop already uses — FeasibilityJump::next_deadline_stride, sized in
// time from the last measurement, growth capped at 8x, shrink uncapped, and hard
// capped at kMaxDeadlineStride moves.
//
// Sharing that tuner shares the lesson it encodes (#113): a stride sized in time
// ALONE goes silent exactly when it is needed, because the shrink can only be
// applied at a check and the next check is a whole stride away, so a stride grown
// over many cheap small structures would be spent in full on the first move of a
// large one. The hard cap is what bounds that, and it is why bounding k against
// the remaining budget instead — the other direction #115 named — was not taken:
// k is chosen once, before the run, so a per-move cost that rises inside the run
// is never re-observed at all.
//
//   Guarantee: perturb()'s structural pass applies at most kMaxDeadlineStride
//   (64) further moves after the deadline passes, and a stride costs at most
//   1/64 of the remaining budget in predicted time — or one move, whichever is
//   larger, since a move is atomic and cannot be pre-empted from the inside.
//
// The prediction is a measurement, so a move whose cost jumps mid-stride is
// absorbed by the 64-move half of the bound, not the time half.
//
// One honest gap, PRE-EXISTING and deliberately not closed here. The guarantee is
// stated in moves, and the cost model behind it assumes cost is proportional to
// moves — but a move that is never applied is not free. generate_standard_moves
// on a Set allocates a vector<bool> over the universe, copies the membership and
// builds the complement, O(|elements| + universe), before discovering there is no
// legal move; the pass then breaks out of that variable having applied nothing.
// Since the check short-circuits on kick_moves_ == 0 without decrementing, a
// model of M saturated Sets (min_size == max_size == universe_size) does O(M * U)
// work with zero clock reads. The old between-variables check was gated on
// `changed` and was equally blind to it, so this is not a regression, and the
// cost is linear in the model rather than quadratic in one variable — the shape
// #115 is about. Closing it means bounding failed ATTEMPTS as well as moves.
bool FeasibilityJump::kick_past_deadline() {
    // Short-circuited on has_deadline_, so a run with no wall clock neither reads
    // the clock nor touches any tuner state: iteration-budgeted runs stay
    // bit-identical. Gated on having applied a move as well, so a deadline
    // already crossed on entry still leaves the kick something to have done —
    // the never-a-no-op contract of #109/#111, which perturb()'s fallback then
    // completes if that one move happened to cancel out.
    if (!has_deadline_ || kick_moves_ == 0) {
        return false;
    }
    if (--kick_countdown_ > 0) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    // Counted before the deadline test — as the GLS loop also does — so the read
    // that stops the pass is included rather than dropped. The count is every
    // read made by THIS check, not by the kick as a whole: arm_structural_kick()
    // takes one more, uncounted. So `structural_kick_checks() == 0` states
    // exactly that the strided check never consulted the clock, and it is the
    // pairing with `has_deadline_ == false` — which also silences the arm — that
    // makes a no-wall-clock run read no clock at all.
    ++kick_checks_;
    if (now >= deadline_) {
        return true;
    }
    // Sized against the budget that is LEFT, as the GLS loop is: a fraction of
    // the total would permit budget/64 of overrun right up to the deadline.
    const double remaining = std::chrono::duration<double>(deadline_ - now).count();
    kick_stride_ = next_deadline_stride(
        kick_stride_, std::chrono::duration<double>(now - last_kick_check_).count(),
        remaining * kStrideBudgetFraction);
    kick_countdown_ = kick_stride_;
    last_kick_check_ = now;
    return false;
}

bool FeasibilityJump::perturb_structural(double probability) {
    // Cost per structural variable is O(k * (|elements| + universe)),
    // k = round(p * |elements|). Since #164 a candidate is a POSITIONAL EDIT
    // rather than a whole element vector, so the per-candidate term is the
    // MEMBERSHIP SCAN the length-changing moves need — a Set's complement, a
    // variable-length List's absent-element sweep, a partition's unassigned pool
    // — rather than an O(|elements|) copy. Still superlinear in a single
    // structure's size, which is why the deadline is checked between moves rather
    // than between variables — see kick_past_deadline() for the bound that buys
    // and what it costs. The at-least-ninefold drop it brought is what forced
    // tests/test_perturb.cpp's mid-kick deadline case onto a ten-times-longer
    // List to keep outrunning its budget.
    arm_structural_kick();
    bool changed = false;
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!is_structured(model_.var(v).type)) {
            continue;  // no RNG draw: a scalar-only model keeps its draw sequence
        }
        // Whether the run moved the variable is a question about its NET effect,
        // not about how many moves were applied: a run of two can add an element
        // and remove it again, and calling that "changed" would hand back a kick
        // that changed nothing — the very thing #111 is about.
        const std::vector<int32_t> before = model_.var(v).elements;
        const int32_t k = structural_kick_size(model_.var(v), probability);
        bool out_of_time = false;
        for (int32_t i = 0; i < k; ++i) {
            if (kick_past_deadline()) {
                out_of_time = true;
                break;
            }
            if (!apply_random_structural_move(model_, v, rng_)) {
                break;  // nothing can move this variable; further tries cannot either
            }
            ++kick_moves_;
        }
        // Recorded even when the budget cut the run short: a truncated run still
        // moved the variable, and the caller's never-a-no-op fallback keys off it.
        changed = changed || structure_moved(model_.var(v), before);
        if (out_of_time) {
            break;
        }
    }
    return changed;
}

bool FeasibilityJump::force_structural_move() {
    std::vector<int32_t> structured;
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (is_structured(model_.var(v).type)) {
            structured.push_back(v);
        }
    }
    if (structured.empty()) {
        return false;
    }
    // Walk from a random structure and take the first that moves. Not uniform
    // over the movable ones — one sitting behind a run of dead ends is favoured
    // — but this runs only on a kick that would otherwise have changed nothing,
    // where any movable structure will do. A single applied move always changes
    // the variable: the no-op candidates were filtered out before the draw.
    const size_t n = structured.size();
    const auto start = static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(n)));
    for (size_t i = 0; i < n; ++i) {
        if (apply_random_structural_move(model_, structured[(start + i) % n], rng_)) {
            return true;
        }
    }
    return false;  // every structure is a dead end
}

void FeasibilityJump::perturb(double probability) {
    require_tables_in_step();
    // Randomise each jumpable variable independently, then make sure the kick
    // actually moved something. Independent draws alone leave the assignment
    // untouched with probability (1-p)^n, which at the default p = 0.1 is 81% on
    // a two-variable model — exactly the small models most likely to be stuck in
    // one basin, where the kick then burned the stagnation counter and the
    // search resumed where it was (#109).
    //
    // The guarantee is a FALLBACK, not a variable forced on every kick. That
    // matters for fidelity: on a model big enough for the per-variable
    // probability to do its job a no-op kick is vanishingly rare, so the scan
    // below never runs, no extra RNG draw is taken, and the kick keeps exactly
    // the distribution — and the exact draw sequence — it had before. Forcing a
    // variable unconditionally instead would shift the draw sequence on every
    // model.
    bool changed = false;
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v) || rng_.random() >= probability) {
            continue;
        }
        Variable& var = model_.var_mut(v);
        const double previous = var.value;
        var.value = random_in_domain(var, rng_);
        changed = changed || var.value != previous;
    }
    // The loop above only reaches jumpable (scalar) variables. List and Set
    // variables get their own pass, so a kick on a model whose decision
    // structure is structural is a real kick rather than a no-op (#111). The
    // pass draws no random numbers on a model without List/Set variables, so
    // scalar-only models keep the exact draw sequence — and hence the exact
    // runs — they had before.
    const bool structural_changed = perturb_structural(probability);
    changed = changed || structural_changed;
    if (!changed) {
        // Nothing moved: pick one variable that CAN move and move it. Scalars
        // first, because that is the cheap answer and the common one; if every
        // scalar is pinned (-1), force a structure instead — the structural pass
        // reaching here means either that every structure is a dead end, in
        // which case this fails too and a kick that changes nothing is the
        // correct outcome, or that a run of moves cancelled itself out, which a
        // single further move undoes.
        const int32_t forced = pick_forced_perturb_var();
        if (forced >= 0) {
            Variable& var = model_.var_mut(forced);
            var.value = random_different_in_domain(var, rng_);
        } else {
            force_structural_move();
        }
    }
    full_evaluate(model_);
    reset_weights();
}

bool FeasibilityJump::all_satisfied() const {
    return !any_active_violated();
}

// ---------------------------------------------------------------------------
// Novelty Jump (paper Algorithms 4-5)
// ---------------------------------------------------------------------------

void FeasibilityJump::init_novelty_weights() {
    // W'[c] = W[c] for constraints violated at entry, else kCompoundDiscount*W[c]
    // (the "novelty" weights make breaking a not-violated-since-best constraint
    // cheap, prioritising chains that fix the initially-broken constraints).
    const size_t nc = vm_.weights.size();
    novelty_weights_.resize(nc);
    for (size_t c = 0; c < nc; ++c) {
        novelty_weights_[c] =
            violated_[c] != 0 ? vm_.weights[c] : kCompoundDiscount * vm_.weights[c];
    }
}

void FeasibilityJump::nj_enqueue(int32_t var_id) {
    if (nj_in_queue_[var_id] == 0) {
        nj_in_queue_[var_id] = 1;
        nj_queue_.push_back(var_id);
    }
}

// O(|Q'| + |V|) rather than O(#vars + #rows) (#174). nj_in_queue_[v] is set
// exactly for the v in nj_queue_ (nj_enqueue sets it with the push,
// select_novelty_var clears it with the swap-remove), so clearing it through the
// queue is the same as clearing the whole vector. The rows are visited in V's
// list order (deterministic; see bump_weights_and_requeue) with a live weight
// read. That order feeds select_novelty_var's draw, so it is part of the
// trajectory, but no longer ascending.
void FeasibilityJump::seed_novelty_scan_set() {
    for (const int32_t v : nj_queue_) {
        nj_in_queue_[static_cast<size_t>(v)] = 0;
    }
    nj_queue_.clear();
    for (const int32_t c : violated_rows_) {
        if (active(c)) {
            for (int32_t v : vars_of_constraint_[static_cast<size_t>(c)]) {
                nj_enqueue(v);
            }
        }
    }
}

// Best of up to 3 sampled vars in Q\T satisfying the filter F (paper §4):
// F = (s_m + novelty_score > 0)  OR  (score > s_c). "Best" = highest original
// score. The chosen var is removed from Q (paper Algorithm 5 line 6).
//
// Only variables that PASS F count toward the 3, and the draw goes on until 3
// have passed or Q\T is exhausted (#206). Counting the filtered draws too --
// with a 32-draw cap besides -- made three non-passing draws in a row look like
// a dead end and forced a backtrack while passing variables were still in Q.
//
// A variable that fails F, or is on the stack (T), leaves only this call's
// sample pool, not Q: F depends on s_m and s_c, which differ at the next call.
// The pool is Q's suffix: [0, k) holds everything drawn this call, each draw is
// uniform over [k, |Q|) and swaps its pick to position k. So every draw is a
// distinct variable, the loop ends after at most |Q| draws, and the chosen
// var's position stays put (the prefix is never touched again) for the
// swap-remove at the end. Q's order is read by nothing but this draw.
FeasibilityJump::NoveltyPick FeasibilityJump::select_novelty_var(double s_m, double s_c) {
    NoveltyPick best;
    size_t best_idx = 0;
    int passed = 0;
    size_t k = 0;
    while (k < nj_queue_.size() && passed < 3) {
        const auto idx =
            k + static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(nj_queue_.size() - k)));
        const int32_t v = nj_queue_[idx];
        std::swap(nj_queue_[k], nj_queue_[idx]);
        const size_t pos = k++;
        if (on_stack_[v] != 0) {
            continue;  // on the stack (T)
        }
        // W'-argmin, then its score under W. Both in closed form where the rows
        // allow it, exactly as apply_jump scores.
        JumpResult nr = compute_var_jump(model_, novelty_weights_, v, false, &linear_);
        double score = 0.0;
        if (linear_.prepare(v, vm_.weights) && std::isfinite(nr.jump_value - model_.var(v).value)) {
            score = -linear_.delta(nr.jump_value);
        } else {
            score = -model_.weighted_violation_delta(v, nr.jump_value, vm_.weights);
        }
        const bool passes = (s_m + nr.score > 0.0) || (score > s_c);
        if (!passes) {
            continue;
        }
        ++passed;
        if (best.var < 0 || score > best.score) {
            best = {v, nr.jump_value, score, nr.score};
            best_idx = pos;
        }
    }
    if (best.var >= 0) {
        // Remove the chosen var from Q (swap-remove).
        nj_in_queue_[best.var] = 0;
        nj_queue_[best_idx] = nj_queue_.back();
        nj_queue_.pop_back();
    }
    return best;
}

// NoveltyJumpSearch (Algorithm 5), recursive with the explicit move_stack_ for
// T-membership and commit/revert. s_m is the cumulative original-weight score of
// the moves currently on the stack. Returns true once a compound move with
// positive cumulative score is found (left applied); false leaves the assignment
// as it was on entry (every move it applied is reverted).
bool FeasibilityJump::novelty_jump_search(double s_m, int budget) {
    if (budget < 0 || nj_work_remaining_ <= 0) {
        return false;
    }
    double s_c = 0.0;  // best explored child score at this level
    const auto& cids = model_.constraint_ids();
    while (true) {
        NoveltyPick pick = select_novelty_var(s_m, s_c);
        if (pick.var < 0) {
            return false;
        }
        s_c = std::max(s_c, pick.score);

        const int32_t v = pick.var;
        const double old_value = model_.var(v).value;
        model_.var_mut(v).value = pick.jump;
        delta_evaluate(model_, &v, 1);
        move_stack_.push_back({v, old_value});
        on_stack_[v] = 1;
        --nj_work_remaining_;  // bound total moves applied per apply_novelty_jump

        // Refresh violated_ for v's constraints; promote any now-broken
        // constraint to full novelty weight and add its vars to the scan set.
        for (int32_t c : model_.constraints_of_var(v)) {
            set_violated(c, is_violated(model_.node_value(cids[c])));
            if (violated_[c] != 0 && novelty_weights_[c] != vm_.weights[c]) {
                novelty_weights_[c] = vm_.weights[c];
                for (int32_t vp : vars_of_constraint_[c]) {
                    nj_enqueue(vp);
                }
            }
        }

        if (s_m + pick.score > 0.0) {
            return true;  // commit (leave applied)
        }
        if (novelty_jump_search(s_m + pick.score, budget)) {
            return true;
        }

        // Backtrack: revert this move and try a sibling (consumes a discrepancy).
        on_stack_[v] = 0;
        move_stack_.pop_back();
        model_.var_mut(v).value = old_value;
        delta_evaluate(model_, &v, 1);
        for (int32_t c : model_.constraints_of_var(v)) {
            set_violated(c, is_violated(model_.node_value(cids[c])));
        }
        budget -= 1;
    }
}

bool FeasibilityJump::apply_novelty_jump() {
    require_tables_in_step();
    // Its legs are plain commits, which add no drift, so once whatever an
    // earlier batch left is re-grounded -- nothing, after a batch that ended
    // normally -- every "reached feasibility" below is read off re-summed rows
    // (#188).
    reground_drifted_rows();
    const size_t nv = model_.num_vars();
    // Flags stay set only for entries of nj_queue_ / move_stack_, which
    // seed_novelty_scan_set and clear_stack clear through below -- also across
    // calls, since an early return leaves both populated. resize sizes them on
    // the first call and is a no-op after.
    nj_in_queue_.resize(nv, 0);
    on_stack_.resize(nv, 0);
    nj_work_remaining_ = kNoveltyWorkBudget;  // bound the compound-move search

    // on_stack_[v] is set exactly for the v on move_stack_ (pushed together,
    // popped together, and select_novelty_var never picks a var already on it),
    // so clearing through the stack is the whole-vector clear at O(|T|) (#174).
    auto clear_stack = [this]() {
        for (const StackMove& m : move_stack_) {
            on_stack_[static_cast<size_t>(m.var)] = 0;
        }
        move_stack_.clear();
    };
    int b = 0;
    while (b <= 2) {
        init_novelty_weights();
        seed_novelty_scan_set();
        clear_stack();
        while (novelty_jump_search(0.0, b)) {
            if (!any_active_violated()) {
                return true;  // reached feasibility
            }
            // Committed a compound move; start a fresh one from the new state
            // (reset budget per Algorithm 4 line 8, keep evolving W').
            b = 0;
            seed_novelty_scan_set();
            clear_stack();
        }
        b += 1;
    }
    return false;
}

GFJStatus FeasibilityJump::run() {
    require_tables_in_step();
    iterations_ = 0;
    arm_deadline();

    if (config_.set_initial_x) {
        set_initial_assignment();
    }
    full_evaluate(model_);
    vm_.invalidate_cache();

    const size_t nc = model_.constraint_ids().size();
    bool has_nonlinear =
        std::any_of(is_linear_.begin(), is_linear_.end(), [](uint8_t lin) { return lin == 0; });

    if (config_.two_phase && has_nonlinear) {
        // Phase 1: GLS on the linear submodel. Non-linear constraint weights are
        // masked to 0 (a mask, not a learned weight); active() == weight>0 then
        // excludes them, and the GLS decay leaves 0-weights at 0. Phase 2
        // restores all weights to 1 (the paper uses fresh weights per phase).
        for (size_t c = 0; c < nc; ++c) {
            vm_.weights[c] = is_linear_[c] != 0 ? 1.0 : 0.0;
        }
        vm_.invalidate_cache();
        gls(config_.sample_size_linear);
        // Restore full weights for the general phase.
        std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
        vm_.invalidate_cache();
    } else {
        std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
        vm_.invalidate_cache();
    }

    GFJStatus status =
        gls(config_.two_phase ? config_.sample_size_general : config_.sample_size_linear);
    vm_.invalidate_cache();
    return status;
}

}  // namespace cbls
