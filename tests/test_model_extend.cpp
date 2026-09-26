// Model::extend (#167): growing a closed model without rebuilding it.
//
// The acceptance criterion at the heart of the issue is EQUIVALENCE: a model
// extended after close() must be indistinguishable from the same model built
// whole and closed. "Indistinguishable" cannot mean "the same node ids", because
// a term appended to an existing Sum is created AFTER that Sum in the staged
// build and BEFORE it in the whole one -- which is exactly the case that had no
// representation at all before this change. So both builders record a name for
// every node they make, and the comparison runs over those names.

#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"
#include "cbls/model.h"
#include "cbls/model_extension.h"
#include "cbls/violation.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cbls;

namespace {

// ---------------------------------------------------------------------------
// A logical model: variables, rows of the form `sum_i coef_i * x_i <= rhs`, some
// of whose terms arrive with the extension, plus whole rows that do.
// ---------------------------------------------------------------------------

struct VarSpec {
    int kind = 0;  // 0 Bool, 1 Int, 2 Float
    double lb = 0.0;
    double ub = 1.0;
    double initial = 0.0;
};

struct Term {
    double coef = 1.0;
    int32_t var = 0;  // logical variable index; == var id in both builds
};

struct RowSpec {
    std::vector<Term> base_terms;
    std::vector<Term> new_terms;  // appended to this row's Sum by the extension
    double rhs = 0.0;
};

struct Spec {
    std::vector<VarSpec> base_vars;
    std::vector<VarSpec> new_vars;
    std::vector<RowSpec> rows;
    std::vector<RowSpec> extra_rows;  // whole new rows; only `base_terms` is used
};

// What a builder produced: the model plus name -> node id.
struct Built {
    Model model;
    std::map<std::string, int32_t> nodes;
};

std::string term_name(size_t row, const char* which, size_t idx, const char* part) {
    return "r" + std::to_string(row) + "." + which + std::to_string(idx) + "." + part;
}

int32_t make_var(Model& m, const VarSpec& v) {
    if (v.kind == 0) {
        return m.bool_var();
    }
    if (v.kind == 1) {
        return m.int_var(static_cast<int>(v.lb), static_cast<int>(v.ub));
    }
    return m.float_var(v.lb, v.ub);
}

int32_t var_handle(int32_t var_id) {
    return -(var_id + 1);
}

// Emit `terms` as coef*var products, recording each node under a name, and return
// the product handles in order.
std::vector<int32_t> emit_terms(Model& m, Built& out, const std::vector<Term>& terms, size_t row,
                                const char* which) {
    std::vector<int32_t> prods;
    prods.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        const int32_t c = m.constant(terms[i].coef);
        const int32_t p = m.prod(c, var_handle(terms[i].var));
        out.nodes[term_name(row, which, i, "c")] = c;
        out.nodes[term_name(row, which, i, "p")] = p;
        prods.push_back(p);
    }
    return prods;
}

void emit_row_tail(Model& m, Built& out, const std::vector<int32_t>& prods, double rhs,
                   const std::string& prefix) {
    const int32_t s = m.sum(prods);
    const int32_t r = m.constant(rhs);
    const int32_t leq = m.leq(s, r);
    out.nodes[prefix + ".sum"] = s;
    out.nodes[prefix + ".rhs"] = r;
    out.nodes[prefix + ".leq"] = leq;
    m.add_constraint(leq);
}

Built build_whole(const Spec& spec) {
    Built out;
    Model& m = out.model;
    for (const VarSpec& v : spec.base_vars) {
        make_var(m, v);
    }
    for (const VarSpec& v : spec.new_vars) {
        const int32_t h = make_var(m, v);
        m.var_mut(-(h + 1)).value = v.initial;
    }
    for (size_t r = 0; r < spec.rows.size(); ++r) {
        std::vector<int32_t> prods = emit_terms(m, out, spec.rows[r].base_terms, r, "b");
        const std::vector<int32_t> extra = emit_terms(m, out, spec.rows[r].new_terms, r, "n");
        prods.insert(prods.end(), extra.begin(), extra.end());
        emit_row_tail(m, out, prods, spec.rows[r].rhs, "r" + std::to_string(r));
    }
    for (size_t e = 0; e < spec.extra_rows.size(); ++e) {
        const size_t row = spec.rows.size() + e;
        const std::vector<int32_t> prods =
            emit_terms(m, out, spec.extra_rows[e].base_terms, row, "b");
        emit_row_tail(m, out, prods, spec.extra_rows[e].rhs, "r" + std::to_string(row));
    }
    m.close();
    return out;
}

Built build_staged(const Spec& spec, ExtensionResult& res) {
    Built out;
    Model& m = out.model;
    for (const VarSpec& v : spec.base_vars) {
        make_var(m, v);
    }
    for (size_t r = 0; r < spec.rows.size(); ++r) {
        const std::vector<int32_t> prods = emit_terms(m, out, spec.rows[r].base_terms, r, "b");
        emit_row_tail(m, out, prods, spec.rows[r].rhs, "r" + std::to_string(r));
    }
    m.close();

    ModelExtension ext(m);
    for (const VarSpec& v : spec.new_vars) {
        int32_t h = 0;
        if (v.kind == 0) {
            h = ext.bool_var();
        } else if (v.kind == 1) {
            h = ext.int_var(static_cast<int>(v.lb), static_cast<int>(v.ub));
        } else {
            h = ext.float_var(v.lb, v.ub);
        }
        ext.set_initial(h, v.initial);
    }
    for (size_t r = 0; r < spec.rows.size(); ++r) {
        const int32_t target = out.nodes.at("r" + std::to_string(r) + ".sum");
        for (size_t i = 0; i < spec.rows[r].new_terms.size(); ++i) {
            const Term& t = spec.rows[r].new_terms[i];
            const int32_t c = ext.constant(t.coef);
            const int32_t p = ext.prod(c, var_handle(t.var));
            out.nodes[term_name(r, "n", i, "c")] = c;
            out.nodes[term_name(r, "n", i, "p")] = p;
            ext.append_to_sum(target, p);
        }
    }
    for (size_t e = 0; e < spec.extra_rows.size(); ++e) {
        const size_t row = spec.rows.size() + e;
        std::vector<int32_t> prods;
        for (size_t i = 0; i < spec.extra_rows[e].base_terms.size(); ++i) {
            const Term& t = spec.extra_rows[e].base_terms[i];
            const int32_t c = ext.constant(t.coef);
            const int32_t p = ext.prod(c, var_handle(t.var));
            out.nodes[term_name(row, "b", i, "c")] = c;
            out.nodes[term_name(row, "b", i, "p")] = p;
            prods.push_back(p);
        }
        const int32_t s = ext.sum(prods);
        const int32_t r = ext.constant(spec.extra_rows[e].rhs);
        const int32_t leq = ext.leq(s, r);
        const std::string prefix = "r" + std::to_string(row);
        out.nodes[prefix + ".sum"] = s;
        out.nodes[prefix + ".rhs"] = r;
        out.nodes[prefix + ".leq"] = leq;
        ext.add_constraint(leq);
    }
    res = m.extend(ext);
    return out;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

std::unordered_map<int32_t, std::string> invert(const std::map<std::string, int32_t>& names) {
    std::unordered_map<int32_t, std::string> inv;
    for (const auto& kv : names) {
        inv[kv.second] = kv.first;
    }
    return inv;
}

// Every list `parents`, `dependents` and `constraints_of_var` hand out is
// contractually STRICTLY ASCENDING and distinct; weighted_delta_from requires it
// of rows and FJ's scan order over G_v feeds the trajectory.
void require_strictly_ascending(ConstSpan<int32_t> ids) {
    for (size_t i = 1; i < ids.size(); ++i) {
        REQUIRE(ids[i - 1] < ids[i]);
    }
}

std::set<std::string> names_of(ConstSpan<int32_t> ids,
                               const std::unordered_map<int32_t, std::string>& inv) {
    std::set<std::string> out;
    for (const int32_t id : ids) {
        const auto it = inv.find(id);
        if (it == inv.end()) {
            FAIL("node id " << id << " has no recorded name");
            return out;
        }
        out.insert(it->second);
    }
    return out;
}

void require_valid_topo_order(const Model& m) {
    const std::vector<int32_t>& order = m.topo_order();
    REQUIRE(order.size() == m.num_nodes());
    std::vector<int32_t> seen(m.num_nodes(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        const int32_t nid = order[i];
        REQUIRE(nid >= 0);
        REQUIRE(static_cast<size_t>(nid) < m.num_nodes());
        REQUIRE(seen[nid] == 0);
        seen[nid] = 1;
        REQUIRE(m.topo_position(nid) == static_cast<int32_t>(i));
        for (const ChildRef& child : m.children(m.nodes()[nid])) {
            if (!child.is_var) {
                // A child must already have been emitted.
                REQUIRE(seen[child.id] == 1);
            }
        }
    }
}

void require_equivalent(const Built& staged, const Built& whole) {
    const Model& a = staged.model;
    const Model& b = whole.model;
    REQUIRE(a.num_vars() == b.num_vars());
    REQUIRE(a.num_nodes() == b.num_nodes());
    REQUIRE(a.constraint_ids().size() == b.constraint_ids().size());
    REQUIRE(staged.nodes.size() == whole.nodes.size());
    REQUIRE(staged.nodes.size() == a.num_nodes());

    const std::unordered_map<int32_t, std::string> inv_a = invert(staged.nodes);
    const std::unordered_map<int32_t, std::string> inv_b = invert(whole.nodes);

    require_valid_topo_order(a);
    require_valid_topo_order(b);

    for (const auto& kv : staged.nodes) {
        const auto it = whole.nodes.find(kv.first);
        if (it == whole.nodes.end()) {
            FAIL("staged model has an unmatched node name: " << kv.first);
            return;
        }
        const int32_t ia = kv.second;
        const int32_t ib = it->second;
        REQUIRE(a.nodes()[ia].op == b.nodes()[ib].op);
        // Bitwise: the two builds sum the same terms in the same order, so an
        // inequality here is a real divergence and not rounding.
        REQUIRE(a.node_value(ia) == b.node_value(ib));

        // Children, as a SEQUENCE -- this is what pins the appended terms landing
        // in the right place and in the right order.
        const ConstSpan<ChildRef> ca = a.children(a.nodes()[ia]);
        const ConstSpan<ChildRef> cb = b.children(b.nodes()[ib]);
        REQUIRE(ca.size() == cb.size());
        for (size_t k = 0; k < ca.size(); ++k) {
            REQUIRE(ca[k].is_var == cb[k].is_var);
            if (ca[k].is_var) {
                REQUIRE(ca[k].id == cb[k].id);
            } else {
                REQUIRE(inv_a.at(ca[k].id) == inv_b.at(cb[k].id));
            }
        }

        require_strictly_ascending(a.parents(ia));
        require_strictly_ascending(b.parents(ib));
        REQUIRE(names_of(a.parents(ia), inv_a) == names_of(b.parents(ib), inv_b));
    }

    for (int32_t v = 0; v < static_cast<int32_t>(a.num_vars()); ++v) {
        REQUIRE(a.var(v).type == b.var(v).type);
        REQUIRE(a.var(v).value == b.var(v).value);
        require_strictly_ascending(a.dependents(v));
        require_strictly_ascending(b.dependents(v));
        REQUIRE(names_of(a.dependents(v), inv_a) == names_of(b.dependents(v), inv_b));

        const ConstSpan<int32_t> ga = a.constraints_of_var(v);
        const ConstSpan<int32_t> gb = b.constraints_of_var(v);
        require_strictly_ascending(ga);
        require_strictly_ascending(gb);
        REQUIRE(std::vector<int32_t>(ga.begin(), ga.end()) ==
                std::vector<int32_t>(gb.begin(), gb.end()));
    }
}

// ---------------------------------------------------------------------------
// Random spec generation
// ---------------------------------------------------------------------------

VarSpec random_var(std::mt19937& rng) {
    VarSpec v;
    v.kind = static_cast<int>(rng() % 3);
    if (v.kind == 0) {
        v.lb = 0.0;
        v.ub = 1.0;
    } else if (v.kind == 1) {
        v.lb = -2.0;
        v.ub = 5.0;
    } else {
        v.lb = -1.5;
        v.ub = 3.25;
    }
    // A value strictly inside the domain, so the comparison is not trivially
    // comparing two lower bounds.
    v.initial = v.lb + ((v.ub - v.lb) * static_cast<double>(rng() % 5) / 4.0);
    if (v.kind != 2) {
        v.initial = std::round(v.initial);
    }
    return v;
}

Spec random_spec(std::mt19937& rng) {
    Spec spec;
    const size_t n_base = 2 + (rng() % 5);
    const size_t n_new = 1 + (rng() % 4);
    for (size_t i = 0; i < n_base; ++i) {
        spec.base_vars.push_back(random_var(rng));
    }
    for (size_t i = 0; i < n_new; ++i) {
        spec.new_vars.push_back(random_var(rng));
    }
    const auto total_vars = static_cast<int32_t>(n_base + n_new);
    auto coef = [&rng]() { return 0.5 + static_cast<double>(rng() % 7); };

    const size_t n_rows = 1 + (rng() % 4);
    for (size_t r = 0; r < n_rows; ++r) {
        RowSpec row;
        const size_t n_terms = 1 + (rng() % 3);
        for (size_t i = 0; i < n_terms; ++i) {
            row.base_terms.push_back({coef(), static_cast<int32_t>(rng() % n_base)});
        }
        const size_t n_appended = rng() % 3;
        for (size_t i = 0; i < n_appended; ++i) {
            // Deliberately drawn over ALL variables, existing ones included: a
            // term over an existing variable makes that variable gain a row index
            // that can sit BELOW rows it is already in, which is the case the CSR
            // splice has to merge rather than append.
            row.new_terms.push_back(
                {coef(), static_cast<int32_t>(rng() % static_cast<size_t>(total_vars))});
        }
        row.rhs = static_cast<double>(rng() % 11) - 5.0;
        spec.rows.push_back(row);
    }
    const size_t n_extra = rng() % 3;
    for (size_t e = 0; e < n_extra; ++e) {
        RowSpec row;
        const size_t n_terms = 1 + (rng() % 3);
        for (size_t i = 0; i < n_terms; ++i) {
            row.base_terms.push_back(
                {coef(), static_cast<int32_t>(rng() % static_cast<size_t>(total_vars))});
        }
        row.rhs = static_cast<double>(rng() % 11) - 5.0;
        spec.extra_rows.push_back(row);
    }
    return spec;
}

}  // namespace

TEST_CASE("an extended model matches the same model built whole", "[extend]") {
    for (uint32_t seed = 1; seed <= 64; ++seed) {
        std::mt19937 rng(seed);
        const Spec spec = random_spec(rng);
        ExtensionResult res;
        const Built staged = build_staged(spec, res);
        const Built whole = build_whole(spec);
        REQUIRE(res.num_new_vars == static_cast<int32_t>(spec.new_vars.size()));
        REQUIRE(res.num_new_constraints == static_cast<int32_t>(spec.extra_rows.size()));
        // Nothing this generator produces needs a re-sort: every appended term is
        // a new node over a new constant and a variable.
        REQUIRE_FALSE(res.topo_order_rebuilt);
        require_equivalent(staged, whole);
    }
}

TEST_CASE("an extension whose terms only appear in new rows is equivalent too", "[extend]") {
    // The pure lazy-cut shape: no append at all, so extend skips the constraint
    // scan entirely. Worth its own case because the random generator almost always
    // produces at least one append.
    Spec spec;
    spec.base_vars = {{0, 0.0, 1.0, 1.0}, {1, 0.0, 4.0, 3.0}};
    spec.new_vars = {{2, -1.0, 1.0, 0.5}};
    spec.rows.push_back({{{2.0, 0}, {3.0, 1}}, {}, 5.0});
    spec.extra_rows.push_back({{{1.0, 0}, {1.0, 2}}, {}, 1.0});
    ExtensionResult res;
    const Built staged = build_staged(spec, res);
    const Built whole = build_whole(spec);
    REQUIRE(res.touched_constraints.empty());
    require_equivalent(staged, whole);
}

TEST_CASE("appending a term the Sum already names adds no second back-reference", "[extend]") {
    // rebuild_back_references dedups a parent that names the same child twice
    // (`prod(x, x)`), and the splice has to agree or a spliced array and a rebuilt
    // one are different arrays.
    Model m;
    const int32_t x = m.bool_var();
    const int32_t t = m.prod(m.constant(2.0), x);
    const int32_t s = m.sum({t});
    m.add_constraint(m.leq(s, m.constant(1.0)));
    m.close();
    m.var_mut(0).value = 1.0;
    full_evaluate(m);
    REQUIRE(m.parents(t).size() == 1);

    ModelExtension ext(m);
    ext.append_to_sum(s, t);  // the same node again
    const ExtensionResult res = m.extend(ext);
    REQUIRE(res.num_new_nodes == 0);
    // Two children now, one parent still.
    REQUIRE(m.children(m.node(s)).size() == 2);
    REQUIRE(m.parents(t).size() == 1);
    REQUIRE(m.parents(t)[0] == s);
    // 2*1 twice.
    REQUIRE(m.node_value(s) == 4.0);
}

TEST_CASE("extend re-sorts when the existing order cannot absorb the new nodes", "[extend]") {
    // An EXISTING node appended as a term to a Sum that already precedes it: the
    // order has to change, not merely grow. Nothing in the tree builds this, but
    // a caller can, and the answer must be a correct re-sort rather than a wrong
    // evaluation order.
    Model m;
    const int32_t x = m.float_var(0.0, 10.0);
    const int32_t y = m.float_var(0.0, 10.0);
    const int32_t s = m.sum({x});
    m.add_constraint(m.leq(s, m.constant(100.0)));
    const int32_t t = m.prod(m.constant(2.0), y);  // made after s, ordered after it
    m.close();
    m.var_mut(0).value = 3.0;
    m.var_mut(1).value = 4.0;
    full_evaluate(m);
    REQUIRE(m.topo_position(t) > m.topo_position(s));

    ModelExtension ext(m);
    ext.append_to_sum(s, t);
    const ExtensionResult res = m.extend(ext);
    REQUIRE(res.topo_order_rebuilt);
    REQUIRE(m.topo_position(t) < m.topo_position(s));
    require_valid_topo_order(m);
    REQUIRE(m.node_value(s) == 11.0);  // 3 + 2*4

    // And it stays right through an ordinary delta_evaluate.
    m.var_mut(1).value = 1.0;
    delta_evaluate(m, {1});
    REQUIRE(m.node_value(s) == 5.0);
}

TEST_CASE("repeated appends compact the child array instead of leaking holes", "[extend]") {
    Model m;
    std::vector<int32_t> prods;
    for (int i = 0; i < 4; ++i) {
        const int32_t v = m.float_var(0.0, 1.0);
        prods.push_back(m.prod(m.constant(1.0), v));
    }
    const int32_t s = m.sum(prods);
    m.add_constraint(m.leq(s, m.constant(100.0)));
    m.close();
    for (int32_t v = 0; v < 4; ++v) {
        m.var_mut(v).value = 1.0;
    }
    full_evaluate(m);
    REQUIRE(m.node_value(s) == 4.0);

    // Relocation leaves the grown row's old slice behind as a hole, and compaction
    // is what stops those accumulating. `child_ref_holes()` is the counter, so the
    // firing is observable rather than merely inferred from the arithmetic: holes
    // only ever GROW between extends, so a drop is a compaction and nothing else.
    REQUIRE(m.child_ref_holes() == 0);
    bool compacted = false;
    size_t holes_before = 0;

    for (int round = 0; round < 12; ++round) {
        ModelExtension ext(m);
        const int32_t h = ext.float_var(0.0, 1.0);
        ext.set_initial(h, 1.0);
        ext.append_to_sum(s, ext.prod(ext.constant(1.0), h));
        const ExtensionResult res = m.extend(ext);
        REQUIRE(res.num_new_vars == 1);
        if (m.child_ref_holes() < holes_before) {
            compacted = true;
        }
        holes_before = m.child_ref_holes();
        require_valid_topo_order(m);
        REQUIRE(m.node_value(s) == 4.0 + static_cast<double>(round + 1));
        REQUIRE(m.children(m.node(s)).size() == static_cast<size_t>(5 + round));
        // Every variable is in the row, exactly once, in ascending row order.
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            const ConstSpan<int32_t> g = m.constraints_of_var(v);
            REQUIRE(g.size() == 1);
            REQUIRE(g[0] == 0);
        }
    }
    REQUIRE(compacted);
}

TEST_CASE("extend refuses what it cannot represent", "[extend]") {
    Model open_model;
    const int32_t x = open_model.bool_var();
    const int32_t s = open_model.sum({x});
    open_model.add_constraint(open_model.leq(s, open_model.constant(1.0)));

    SECTION("an open model") {
        REQUIRE_THROWS_AS(ModelExtension(open_model), std::logic_error);
    }

    open_model.close();
    Model& m = open_model;

    SECTION("a non-Sum append target") {
        ModelExtension ext(m);
        REQUIRE_THROWS_AS(ext.append_to_sum(m.constraint_ids()[0], ext.constant(1.0)),
                          std::invalid_argument);
    }
    SECTION("a target created by the extension itself") {
        ModelExtension ext(m);
        const int32_t fresh = ext.sum({ext.constant(1.0)});
        REQUIRE_THROWS_AS(ext.append_to_sum(fresh, ext.constant(1.0)), std::invalid_argument);
    }
    SECTION("a var handle where a node is required") {
        ModelExtension ext(m);
        REQUIRE_THROWS_AS(ext.add_constraint(ext.bool_var()), std::invalid_argument);
    }
    SECTION("a handle naming nothing") {
        ModelExtension ext(m);
        REQUIRE_THROWS_AS(ext.neg(9999), std::out_of_range);
        REQUIRE_THROWS_AS(ext.neg(-9999), std::out_of_range);
    }
    SECTION("set_initial on an existing variable, or out of bounds") {
        ModelExtension ext(m);
        REQUIRE_THROWS_AS(ext.set_initial(x, 1.0), std::invalid_argument);
        const int32_t h = ext.int_var(0, 3);
        REQUIRE_THROWS_AS(ext.set_initial(h, 4.0), std::invalid_argument);
    }
    SECTION("an extension replayed against a model that has since grown") {
        ModelExtension ext(m);
        ext.add_constraint(ext.leq(ext.constant(1.0), ext.constant(2.0)));
        ModelExtension other(m);
        other.bool_var();
        (void)m.extend(other);
        REQUIRE_THROWS_AS(m.extend(ext), std::invalid_argument);
    }
    SECTION("an empty extension changes nothing") {
        const size_t nodes_before = m.num_nodes();
        const ModelExtension ext(m);
        const ExtensionResult res = m.extend(ext);
        REQUIRE(res.num_new_vars == 0);
        REQUIRE(res.num_new_nodes == 0);
        REQUIRE(m.num_nodes() == nodes_before);
    }
}

TEST_CASE("a padded pre-extension state is still a valid restart point", "[extend]") {
    Model m;
    const int32_t x = m.int_var(0, 5);
    const int32_t y = m.int_var(0, 5);
    const int32_t s = m.sum({m.prod(m.constant(1.0), x), m.prod(m.constant(1.0), y)});
    m.add_constraint(m.leq(s, m.constant(6.0)));
    m.close();
    m.var_mut(0).value = 2.0;
    m.var_mut(1).value = 3.0;
    full_evaluate(m);
    const Model::State incumbent = m.copy_state();

    ModelExtension ext(m);
    const int32_t z = ext.int_var(0, 5);
    ext.set_initial(z, 4.0);
    ext.append_to_sum(s, ext.prod(ext.constant(2.0), z));
    const ExtensionResult res = m.extend(ext);
    REQUIRE(m.node_value(s) == 2.0 + 3.0 + 8.0);

    // The size check still bites, which is the point of pad_state being strict.
    Model::State stale = incumbent;
    REQUIRE_THROWS_AS(m.restore_state(stale), std::invalid_argument);
    pad_state(stale, res);
    REQUIRE_THROWS_AS(pad_state(stale, res), std::invalid_argument);  // not twice

    // Move the assignment away, then restore the padded incumbent.
    m.var_mut(0).value = 5.0;
    m.var_mut(1).value = 5.0;
    m.var_mut(2).value = 0.0;
    full_evaluate(m);
    m.restore_state(stale);
    full_evaluate(m);
    REQUIRE(m.var(0).value == 2.0);
    REQUIRE(m.var(1).value == 3.0);
    REQUIRE(m.var(2).value == 4.0);  // the initial the extension declared
    REQUIRE(m.node_value(s) == 2.0 + 3.0 + 8.0);
}

TEST_CASE("extending a running FJ keeps the existing rows' GLS weights", "[extend]") {
    // The issue's third criterion, pinned where it can be pinned in this slice:
    // at component level, on a ViolationManager and a FeasibilityJump driven
    // directly. The end-to-end "between batches of a running search" version needs
    // a hook surface inside ViolationLSLoop, which is #168's API.
    //
    // The model is deliberately infeasible so the GLS dynamics bump the weights
    // well away from 1: an extension that quietly reset them would look identical
    // on a model FJ solves immediately.
    Model m;
    const int32_t a = m.bool_var();
    const int32_t b = m.bool_var();
    const int32_t lhs = m.sum({m.prod(m.constant(1.0), a), m.prod(m.constant(1.0), b)});
    m.add_constraint(m.leq(lhs, m.constant(0.0)));  // a + b <= 0
    m.add_constraint(m.geq(lhs, m.constant(2.0)));  // a + b >= 2
    m.close();

    ViolationManager vm(m);
    RNG rng(42);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    for (int i = 0; i < 6; ++i) {
        (void)fj.batch(40);
    }
    const std::vector<double> weights_before = vm.weights;
    REQUIRE(weights_before.size() == 2);
    const bool diverged = std::any_of(weights_before.begin(), weights_before.end(),
                                      [](double w) { return w != 1.0; });
    REQUIRE(diverged);

    const Model::State incumbent = m.copy_state();

    ModelExtension ext(m);
    const int32_t c = ext.bool_var();
    ext.set_initial(c, 1.0);
    // A new column entering an existing row, and a whole new row over it.
    ext.append_to_sum(lhs, ext.prod(ext.constant(1.0), c));
    ext.add_constraint(ext.leq(ext.prod(ext.constant(1.0), c), ext.constant(0.0)));
    const ExtensionResult res = m.extend(ext);

    REQUIRE(res.num_new_vars == 1);
    REQUIRE(res.num_new_constraints == 1);
    // Both existing rows read the grown Sum, so both are reported as touched.
    REQUIRE(res.touched_constraints == std::vector<int32_t>{0, 1});

    vm.on_extended(res);
    fj.on_extended(res);

    REQUIRE(vm.weights.size() == 3);
    REQUIRE(vm.weights[0] == weights_before[0]);
    REQUIRE(vm.weights[1] == weights_before[1]);
    REQUIRE(vm.weights[2] == 1.0);

    // The grown model is consistent and the search keeps running on it.
    REQUIRE(m.num_vars() == 3);
    REQUIRE(m.constraints_of_var(2).size() == 3);
    const int64_t iterations_before = fj.iterations();
    for (int i = 0; i < 4; ++i) {
        (void)fj.batch(40);
    }
    // The search carried on from where it was rather than being restarted: the
    // iteration count is cumulative since begin(), and begin() is what resets it.
    REQUIRE(fj.iterations() > iterations_before);
    // Nothing is asserted about where the weights went NEXT. rho = 0.95 decays a
    // row that has become satisfied -- which the new column can make happen -- so
    // "they kept growing" is not a property of the mechanism. What is pinned is
    // that on_extended itself did not touch them, above.

    // And the pre-extension incumbent is still restorable once padded.
    Model::State padded = incumbent;
    pad_state(padded, res);
    m.restore_state(padded);
    full_evaluate(m);
    REQUIRE(m.var(2).value == 1.0);
}

TEST_CASE("on_extended rejects a result that does not describe this model", "[extend]") {
    Model m;
    const int32_t x = m.bool_var();
    m.add_constraint(m.leq(m.prod(m.constant(1.0), x), m.constant(0.0)));
    m.close();
    ViolationManager vm(m);
    RNG rng(7);
    FeasibilityJump fj(m, vm, rng);

    ExtensionResult bogus;
    bogus.first_new_constraint = 5;
    bogus.num_new_constraints = 2;
    REQUIRE_THROWS_AS(vm.on_extended(bogus), std::invalid_argument);
    REQUIRE_THROWS_AS(fj.on_extended(bogus), std::invalid_argument);
}

TEST_CASE("a new row over existing variables enters their G_v in order", "[extend]") {
    // The lazily separated cut: no new variable, no append, one new row. The
    // existing variables' G_v lists gain a row index above everything they hold,
    // which is the append case; the merge case is covered by the property test.
    Model m;
    const int32_t x = m.int_var(0, 3);
    const int32_t y = m.int_var(0, 3);
    m.add_constraint(m.leq(m.prod(m.constant(1.0), x), m.constant(3.0)));
    m.add_constraint(m.leq(m.prod(m.constant(1.0), y), m.constant(3.0)));
    m.close();
    m.var_mut(0).value = 3.0;
    m.var_mut(1).value = 3.0;
    full_evaluate(m);

    ModelExtension ext(m);
    const int32_t cut =
        ext.leq(ext.sum({ext.prod(ext.constant(1.0), x), ext.prod(ext.constant(1.0), y)}),
                ext.constant(4.0));
    ext.add_constraint(cut);
    const ExtensionResult res = m.extend(ext);

    REQUIRE(res.num_new_vars == 0);
    REQUIRE(res.first_new_constraint == 2);
    REQUIRE(res.touched_constraints.empty());
    REQUIRE(m.node_value(cut) == 2.0);  // 3 + 3 - 4
    for (int32_t v = 0; v < 2; ++v) {
        const ConstSpan<int32_t> g = m.constraints_of_var(v);
        REQUIRE(g.size() == 2);
        REQUIRE(g[0] == v);
        REQUIRE(g[1] == 2);
    }
    // The new row is dirtied by an ordinary move on an existing variable, which is
    // only true if `dependents` picked up the new product nodes.
    m.var_mut(0).value = 0.0;
    delta_evaluate(m, {0});
    REQUIRE(m.node_value(cut) == -1.0);
}

TEST_CASE("a column added mid-search is actually reachable by FJ", "[extend]") {
    // The behavioural half of on_extended. The base model is INFEASIBLE and only
    // the new column can fix it, so reaching feasibility after the extension needs
    // all of it: the jump table grown, `violated_` set for the changed row,
    // `vars_of_constraint_` carrying the new variable (which is what re-queues it
    // after a weight bump), and G_v spliced so the probe sees the row at all.
    Model m;
    const int32_t a = m.bool_var();
    const int32_t lhs = m.sum({m.prod(m.constant(1.0), a)});
    m.add_constraint(m.geq(lhs, m.constant(2.0)));  // a >= 2, unsatisfiable
    m.close();

    ViolationManager vm(m);
    RNG rng(11);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    for (int i = 0; i < 5; ++i) {
        REQUIRE_FALSE(fj.batch(50));
    }

    ModelExtension ext(m);
    const int32_t c = ext.bool_var();
    ext.append_to_sum(lhs, ext.prod(ext.constant(1.0), c));
    const ExtensionResult res = m.extend(ext);
    vm.on_extended(res);
    fj.on_extended(res);

    bool feasible = false;
    for (int i = 0; i < 20 && !feasible; ++i) {
        feasible = fj.batch(50);
    }
    REQUIRE(feasible);
    REQUIRE(m.var(0).value == 1.0);
    REQUIRE(m.var(1).value == 1.0);
}

TEST_CASE("extend reclassifies the linearity of the rows it changed", "[extend]") {
    // Appending a term can make a linear row NON-linear, which is the case a
    // "classify the new rows only" update would miss. `run()`'s first phase
    // descends the linear submodel, so a stale classification would put a
    // nonlinear row in it.
    Model m;
    const int32_t x = m.float_var(0.0, 2.0);
    const int32_t y = m.float_var(0.0, 2.0);
    const int32_t linear_row = m.sum({m.prod(m.constant(1.0), x)});
    m.add_constraint(m.leq(linear_row, m.constant(1.0)));    // row 0: linear
    m.add_constraint(m.leq(m.prod(x, y), m.constant(1.0)));  // row 1: bilinear
    m.close();

    ViolationManager vm(m);
    RNG rng(3);
    FeasibilityJump fj(m, vm, rng);
    REQUIRE(fj.row_is_linear(0));
    REQUIRE_FALSE(fj.row_is_linear(1));

    ModelExtension ext(m);
    const int32_t z = ext.float_var(0.0, 2.0);
    // A bilinear term into the linear row, plus one new row of each kind.
    ext.append_to_sum(linear_row, ext.prod(z, y));
    ext.add_constraint(ext.leq(ext.prod(ext.constant(3.0), z), ext.constant(1.0)));
    ext.add_constraint(ext.leq(ext.exp_expr(z), ext.constant(1.0)));
    const ExtensionResult res = m.extend(ext);
    vm.on_extended(res);
    fj.on_extended(res);

    REQUIRE_FALSE(fj.row_is_linear(0));  // reclassified by the appended term
    REQUIRE_FALSE(fj.row_is_linear(1));
    REQUIRE(fj.row_is_linear(2));
    REQUIRE_FALSE(fj.row_is_linear(3));
}

TEST_CASE("a resynced FJ still reaches a column the extension added", "[extend]") {
    // The same reachability check as above, but routed through resync() -- which
    // rebuilds the scan set from `vars_of_constraint_` alone. It is therefore the
    // pin on that table having gained the new variable: without it the column is
    // never queued, never re-queued by a weight bump, and the row stays violated
    // for ever.
    Model m;
    const int32_t a = m.bool_var();
    const int32_t lhs = m.sum({m.prod(m.constant(1.0), a)});
    m.add_constraint(m.geq(lhs, m.constant(2.0)));
    m.close();

    ViolationManager vm(m);
    RNG rng(5);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    REQUIRE_FALSE(fj.batch(50));

    ModelExtension ext(m);
    const int32_t c = ext.bool_var();
    ext.append_to_sum(lhs, ext.prod(ext.constant(1.0), c));
    const ExtensionResult res = m.extend(ext);
    vm.on_extended(res);
    fj.on_extended(res);
    fj.resync();

    bool feasible = false;
    for (int i = 0; i < 20 && !feasible; ++i) {
        feasible = fj.batch(50);
    }
    REQUIRE(feasible);
}

TEST_CASE("append_to_sum refuses a term that would close a cycle", "[extend]") {
    // The FIRST operation in this engine that can make the DAG cyclic. `close()`
    // is safe by construction -- a node can only name children that already exist
    // -- but appending a term to an existing Sum adds an edge in the other
    // direction, so a term that already reads the target closes a loop.
    //
    // Nothing downstream survives that. `plan_topo_insert` refuses the splice, and
    // `detail::compute_topo_order` is Kahn's over `parents`, which on a cyclic
    // graph returns a SHORT order: measured before this check, a 5-node model came
    // back with a 2-node order, every missing node left at `topo_pos == 0`, so
    // `full_evaluate` never recomputed it again and `evaluate_dirty_in_topo_order`
    // put it before its own inputs -- at exit code 0 with no diagnostic.
    //
    // Refused at RECORD time, which is what keeps the model untouched: `extend`
    // has no rollback (see Model::extend).
    Model m;
    const int32_t x = m.float_var(0.0, 10.0);
    const int32_t s = m.sum({x});
    const int32_t t = m.prod(s, m.constant(2.0));  // t reads s
    m.add_constraint(m.leq(s, m.constant(100.0)));
    m.close();
    const size_t nodes_before = m.num_nodes();

    SECTION("a term that reads the target") {
        ModelExtension ext(m);
        REQUIRE_THROWS_AS(ext.append_to_sum(s, t), std::invalid_argument);
        // The refusal is at record time, so neither the extension nor the model
        // kept anything: replaying it is a no-op.
        const ExtensionResult res = m.extend(ext);
        REQUIRE(res.num_new_nodes == 0);
        REQUIRE(m.num_nodes() == nodes_before);
        REQUIRE(m.topo_order().size() == m.num_nodes());
    }
    SECTION("the target itself") {
        ModelExtension ext(m);
        REQUIRE_THROWS_AS(ext.append_to_sum(s, s), std::invalid_argument);
    }
    SECTION("a term that reaches the target through a node this extension adds") {
        ModelExtension ext(m);
        const int32_t n = ext.neg(t);  // n -> t -> s
        REQUIRE_THROWS_AS(ext.append_to_sum(s, n), std::invalid_argument);
    }
    SECTION("a legal append to the same Sum still goes through") {
        ModelExtension ext(m);
        const int32_t y = ext.float_var(0.0, 10.0);
        ext.set_initial(y, 2.0);
        ext.append_to_sum(s, ext.prod(ext.constant(3.0), y));
        const ExtensionResult res = m.extend(ext);
        REQUIRE_FALSE(res.topo_order_rebuilt);
        require_valid_topo_order(m);
    }
}

TEST_CASE("append_to_sum sees the cycle two appends close together", "[extend]") {
    // Neither append reads the other's target in the BASE model; the loop exists
    // only once both edges are recorded, which is why the check runs against the
    // extension's own pending appends and not just against the closed model.
    Model m;
    const int32_t x = m.float_var(0.0, 10.0);
    const int32_t s1 = m.sum({x});
    const int32_t s2 = m.sum({x});
    m.add_constraint(m.leq(s1, m.constant(100.0)));
    m.add_constraint(m.leq(s2, m.constant(100.0)));
    m.close();

    ModelExtension ext(m);
    ext.append_to_sum(s1, ext.prod(s2, ext.constant(1.0)));  // s1 -> .. -> s2, fine
    const int32_t back = ext.prod(s1, ext.constant(1.0));
    REQUIRE_THROWS_AS(ext.append_to_sum(s2, back), std::invalid_argument);
}

TEST_CASE("FJ::on_extended refuses to run before the ViolationManager grew", "[extend]") {
    // `refresh_unweighted_violation` loops over the GROWN constraint count and
    // asks `active(ci)`, which is an unchecked `vm_.weights[ci] > 0.0`. Called in
    // the wrong order that is an out-of-bounds read per new row, and
    // `unweighted_violation_` comes out of whatever was past the end -- with
    // `!active(ci)` able to mask real rows on the way. The order was documented on
    // the declaration and enforced nowhere.
    Model m;
    const int32_t a = m.bool_var();
    const int32_t lhs = m.sum({m.prod(m.constant(1.0), a)});
    m.add_constraint(m.geq(lhs, m.constant(2.0)));
    m.close();

    ViolationManager vm(m);
    RNG rng(13);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    REQUIRE_FALSE(fj.batch(20));

    ModelExtension ext(m);
    const int32_t c = ext.bool_var();
    ext.append_to_sum(lhs, ext.prod(ext.constant(1.0), c));
    ext.add_constraint(ext.leq(ext.prod(ext.constant(1.0), c), ext.constant(1.0)));
    const ExtensionResult res = m.extend(ext);

    REQUIRE_THROWS_AS(fj.on_extended(res), std::invalid_argument);
    // In the documented order both go through, and the second call is the one that
    // was refused a moment ago.
    vm.on_extended(res);
    fj.on_extended(res);
    REQUIRE(vm.weights.size() == 2);
}

TEST_CASE("on_extended refuses a result whose row indices name nothing", "[extend]") {
    // `ExtensionResult` is a plain struct with public members and a default
    // constructor, and the case above already treats a hand-built one as a
    // reachable input. The COUNT fields were validated; the index vectors were
    // used raw -- `is_linear_[ci]`, `violated_[ci]`, `cids[ci]` and
    // `vars_of_constraint_[ci]` are all unchecked, so a stray index is a heap
    // write, not an exception.
    Model m;
    const int32_t x = m.bool_var();
    m.add_constraint(m.leq(m.prod(m.constant(1.0), x), m.constant(0.0)));
    m.close();
    ViolationManager vm(m);
    RNG rng(7);
    FeasibilityJump fj(m, vm, rng);

    // Describes this model's counts exactly -- one variable, one constraint, no
    // additions -- so every count check passes.
    ExtensionResult base;
    base.first_new_var = 1;
    base.first_new_constraint = 1;

    SECTION("a touched constraint out of range") {
        ExtensionResult bogus = base;
        bogus.touched_constraints = {999};
        REQUIRE_THROWS_AS(fj.on_extended(bogus), std::out_of_range);
    }
    SECTION("a touched constraint naming a NEW row") {
        // touched_constraints is documented as existing rows only; a new row is
        // already covered by the id range and would be classified twice.
        ExtensionResult bogus = base;
        bogus.touched_constraints = {1};
        REQUIRE_THROWS_AS(fj.on_extended(bogus), std::out_of_range);
    }
    SECTION("touched constraints out of order") {
        ExtensionResult bogus = base;
        bogus.touched_constraints = {0, 0};
        REQUIRE_THROWS_AS(fj.on_extended(bogus), std::invalid_argument);
    }
    SECTION("an incidence naming a row that does not exist") {
        ExtensionResult bogus = base;
        bogus.new_incidences = {{999, 0}};
        REQUIRE_THROWS_AS(fj.on_extended(bogus), std::out_of_range);
    }
    SECTION("an incidence naming a variable that does not exist") {
        ExtensionResult bogus = base;
        bogus.new_incidences = {{0, 999}};
        REQUIRE_THROWS_AS(fj.on_extended(bogus), std::out_of_range);
    }
    SECTION("a result that does describe the model is accepted") {
        REQUIRE_NOTHROW(vm.on_extended(base));
        REQUIRE_NOTHROW(fj.on_extended(base));
    }
}

TEST_CASE("appending a bare variable to a Sum merges into its dependents", "[extend]") {
    // The one way a variable gains a dependent with a LOWER node id than one it
    // already has, so the one shape where `dependent_ids` must MERGE rather than
    // append. The property test never reaches it: every term it appends is a fresh
    // `prod` whose id is above everything, and for a variable the merge is only
    // ever into G_v. Appending a variable handle straight into the row is also the
    // cheapest form of the column-generation append -- a unit coefficient needs no
    // product node at all -- and it adds no nodes and no variables, so it exercises
    // an extension whose only effect is on the CSR arrays.
    Model m;
    const int32_t x = m.float_var(0.0, 5.0);
    const int32_t y = m.float_var(0.0, 5.0);
    const int32_t row0 = m.sum({m.prod(m.constant(1.0), y)});
    m.add_constraint(m.leq(row0, m.constant(100.0)));
    // A HIGHER-id node reading x, made after the Sum, so x's dependents already
    // hold an id above the one the append will add.
    const int32_t high = m.prod(m.constant(2.0), x);
    m.add_constraint(m.leq(high, m.constant(100.0)));
    m.close();
    m.var_mut(0).value = 3.0;  // x
    m.var_mut(1).value = 4.0;  // y
    full_evaluate(m);
    REQUIRE(m.node_value(row0) == 4.0);
    REQUIRE(m.dependents(0).size() == 1);
    REQUIRE(m.dependents(0)[0] == high);
    REQUIRE(high > row0);

    ModelExtension ext(m);
    ext.append_to_sum(row0, x);  // a bare variable handle, no term node
    const ExtensionResult res = m.extend(ext);

    REQUIRE(res.num_new_nodes == 0);
    REQUIRE(res.num_new_vars == 0);
    REQUIRE(res.num_new_constraints == 0);
    REQUIRE(res.touched_constraints == std::vector<int32_t>{0});
    REQUIRE_FALSE(res.topo_order_rebuilt);
    require_valid_topo_order(m);

    // Merged, not appended: the new dependent sorts in front of the one x had.
    const ConstSpan<int32_t> deps = m.dependents(0);
    REQUIRE(deps.size() == 2);
    REQUIRE(deps[0] == row0);
    REQUIRE(deps[1] == high);
    // And the same shape in G_v: x was in row 1 only, and gains row 0 below it.
    const ConstSpan<int32_t> g = m.constraints_of_var(0);
    REQUIRE(g.size() == 2);
    REQUIRE(g[0] == 0);
    REQUIRE(g[1] == 1);

    REQUIRE(m.node_value(row0) == 7.0);  // y + x
    // The row is now dirtied by an ordinary move on x, which is what the merged
    // dependents entry buys.
    m.var_mut(0).value = 1.0;
    delta_evaluate(m, {0});
    REQUIRE(m.node_value(row0) == 5.0);
    REQUIRE(m.node_value(high) == 2.0);
}

TEST_CASE("an extension can read an existing List or Set variable", "[extend]") {
    // `at` and `count` over a structured variable that already exists are the two
    // builders `ModelExtension` deliberately offers (a NEW List or Set is refused,
    // because its starting assignment is laid out above this layer). Untested until
    // now: the class comment promises them.
    Model m;
    const int32_t lv = m.list_var(5);
    const int32_t sv = m.set_var(10, 0, 10);
    const int32_t z = m.float_var(0.0, 10.0);
    m.add_constraint(m.leq(m.prod(m.constant(1.0), z), m.constant(100.0)));
    m.close();
    const int32_t lv_id = -(lv + 1);
    const int32_t sv_id = -(sv + 1);
    m.var_mut(lv_id).elements = {4, 3, 2, 1, 0};
    m.var_mut(sv_id).elements = {1, 3, 5, 7};
    m.var_mut(-(z + 1)).value = 2.0;
    full_evaluate(m);

    ModelExtension ext(m);
    const int32_t second = ext.at(lv, ext.constant(1.0));
    const int32_t size = ext.count(sv);
    ext.add_constraint(ext.leq(second, ext.constant(3.0)));
    ext.add_constraint(ext.leq(size, ext.constant(2.0)));
    const ExtensionResult res = m.extend(ext);

    REQUIRE(res.num_new_constraints == 2);
    require_valid_topo_order(m);
    REQUIRE(m.node_value(second) == 3.0);  // elements[1]
    REQUIRE(m.node_value(size) == 4.0);
    // The structured variables entered G_v and the new nodes entered their
    // dependents, which is what makes a structural move dirty the new rows.
    REQUIRE(m.constraints_of_var(lv_id).size() == 1);
    REQUIRE(m.constraints_of_var(lv_id)[0] == 1);
    REQUIRE(m.constraints_of_var(sv_id).size() == 1);
    REQUIRE(m.constraints_of_var(sv_id)[0] == 2);
    REQUIRE(m.dependents(lv_id).size() == 1);
    REQUIRE(m.dependents(lv_id)[0] == second);

    m.var_mut(lv_id).elements = {0, 1, 2, 3, 4};
    m.var_mut(sv_id).elements = {2, 4};
    delta_evaluate(m, {lv_id, sv_id});
    REQUIRE(m.node_value(second) == 1.0);
    REQUIRE(m.node_value(size) == 2.0);
}

TEST_CASE("a ViolationManager out of step with a grown model refuses to read", "[extend]") {
    // The window between `Model::extend` returning and `on_extended`: the model has
    // more rows than the manager has weights, and every read indexes both by
    // constraint index -- `bump_weights` WRITES. It is the window #168's in-loop
    // hook will sit in, so it throws rather than overreading the heap.
    Model m;
    const int32_t x = m.bool_var();
    m.add_constraint(m.leq(m.prod(m.constant(1.0), x), m.constant(0.0)));
    m.close();
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.total_violation() == 0.0);

    ModelExtension ext(m);
    const int32_t c = ext.bool_var();
    ext.add_constraint(ext.geq(ext.prod(ext.constant(1.0), c), ext.constant(1.0)));
    const ExtensionResult res = m.extend(ext);

    std::vector<double> snapshot;
    REQUIRE_THROWS_AS(vm.total_violation(), std::logic_error);
    REQUIRE_THROWS_AS(vm.augmented_objective(), std::logic_error);
    REQUIRE_THROWS_AS(vm.snapshot_violations(snapshot), std::logic_error);
    REQUIRE_THROWS_AS(vm.bump_weights(), std::logic_error);
    REQUIRE_THROWS_AS(vm.weighted_violation_delta(0, 1.0), std::logic_error);

    SECTION("a weight that is not a weight is refused too") {
        REQUIRE_THROWS_AS(vm.on_extended(res, std::nan("")), std::invalid_argument);
        REQUIRE_THROWS_AS(vm.on_extended(res, -1.0), std::invalid_argument);
        // Zero is legitimate: active() is weight > 0, so the row starts masked --
        // which is how run()'s linear-submodel phase masks the nonlinear rows.
        vm.on_extended(res, 0.0);
        REQUIRE(vm.weights.size() == 2);
        REQUIRE(vm.weights[1] == 0.0);
    }
    SECTION("and reads again once it has grown") {
        vm.on_extended(res);
        REQUIRE(vm.weights.size() == 2);
        REQUIRE(vm.total_violation() == 1.0);  // the new row is violated at c = 0
        REQUIRE_NOTHROW(vm.bump_weights());
    }
}
