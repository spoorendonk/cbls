#pragma once

// Set covering, modelled two ways on purpose.
//
//   min  sum_j c_j x_j   s.t. every row is covered by at least one chosen column
//
// `Encoding::Set` is the model this benchmark exists for (issue #93): the choice
// of columns is ONE collection-typed `Set` variable over the column universe,
// each row's coverage is a `lambda_sum` over the chosen columns, and the search
// moves through the Set add/remove/swap generators in the STRUCTURAL batch.
//
// `Encoding::Bool` is the control: the same instance as one Bool per column,
// which is the ordinary linear encoding CP-SAT's violation-based LS worker also
// accepts. It goes through Generalised Feasibility Jump like any scalar model.
// Running both on the same instance is what turns "the engine supports Set
// variables" into a measurement — see benchmarks/instances/setcover/README.md.
//
// LIFETIME: the model's lambdas hold pointers into the instance. The instance
// must outlive the model.

#include "data.h"

#include <algorithm>
#include <cbls/expr.h>
#include <cbls/model.h>
#include <cstdint>
#include <string>
#include <vector>

namespace cbls {
namespace setcover {

enum class Encoding { Set, Bool };

inline const char* encoding_name(Encoding e) {
    return e == Encoding::Set ? "set" : "bool";
}

struct SetCoverModel {
    Model model;
    Encoding encoding = Encoding::Set;
    int32_t chosen = 0;      // Set encoding: handle of the Set variable
    std::vector<int32_t> x;  // Bool encoding: one handle per column

    // The columns the current assignment selects, ascending. Same meaning under
    // both encodings, so verification and reporting are encoding-agnostic.
    std::vector<int> selected_columns() const {
        std::vector<int> selected;
        if (encoding == Encoding::Set) {
            const Variable& v = model.var(handle_to_var_id(chosen));
            selected.assign(v.elements.begin(), v.elements.end());
        } else {
            for (size_t j = 0; j < x.size(); ++j) {
                if (model.var(handle_to_var_id(x[j])).value > 0.5) {
                    selected.push_back(static_cast<int>(j));
                }
            }
        }
        std::sort(selected.begin(), selected.end());
        return selected;
    }
};

// One `Set` variable over the columns; one Lambda coverage row per row.
//
// The cardinality upper bound is `min(cols, rows)`, which is valid rather than
// tuned: every row needs one column, so no *minimal* cover holds more columns
// than there are rows, and any cover with more is dominated by one of its
// subsets. That domination step assumes costs are non-negative — with a negative
// cost, dropping a redundant column can raise the objective and an optimum need
// not be minimal. OR-Library set-covering costs are positive integers, so the
// bound preserves an optimum here; a reader porting this to a signed-cost
// instance needs to drop it. It matters because `initialize_structured_random` draws the initial
// size uniformly from [min_size, max_size] — without the bound the search starts
// from ~cols/2 columns and spends its budget shedding them.
inline SetCoverModel build_set_model(const SetCoverInstance& inst) {
    SetCoverModel scm;
    scm.encoding = Encoding::Set;
    Model& m = scm.model;

    const int max_size = std::min(inst.cols, inst.rows);
    scm.chosen = m.set_var(inst.cols, /*min_size=*/0, max_size, "columns");

    for (int i = 0; i < inst.rows; ++i) {
        const uint8_t* row = inst.covers.data() + static_cast<size_t>(i) * inst.cols;
        int32_t covered = m.lambda_sum(scm.chosen, [row](int col) { return row[col] ? 1.0 : 0.0; });
        m.add_constraint(m.geq(covered, m.constant(1.0)));
    }

    const double* cost = inst.cost.data();
    m.minimize(m.lambda_sum(scm.chosen, [cost](int col) { return cost[col]; }));
    m.close();
    return scm;
}

// One Bool per column; one linear coverage row per row. The control encoding.
inline SetCoverModel build_bool_model(const SetCoverInstance& inst) {
    SetCoverModel scm;
    scm.encoding = Encoding::Bool;
    Model& m = scm.model;

    scm.x.reserve(static_cast<size_t>(inst.cols));
    for (int j = 0; j < inst.cols; ++j) {
        scm.x.push_back(m.bool_var("x" + std::to_string(j)));
    }

    for (int i = 0; i < inst.rows; ++i) {
        std::vector<int32_t> terms;
        terms.reserve(inst.row_cols[static_cast<size_t>(i)].size());
        for (int col : inst.row_cols[static_cast<size_t>(i)]) {
            terms.push_back(scm.x[static_cast<size_t>(col)]);
        }
        m.add_constraint(m.geq(m.sum(terms), m.constant(1.0)));
    }

    std::vector<int32_t> cost_terms;
    cost_terms.reserve(static_cast<size_t>(inst.cols));
    for (int j = 0; j < inst.cols; ++j) {
        cost_terms.push_back(
            m.prod(m.constant(inst.cost[static_cast<size_t>(j)]), scm.x[static_cast<size_t>(j)]));
    }
    m.minimize(m.sum(cost_terms));
    m.close();
    return scm;
}

inline SetCoverModel build_model(const SetCoverInstance& inst, Encoding encoding) {
    return encoding == Encoding::Set ? build_set_model(inst) : build_bool_model(inst);
}

}  // namespace setcover
}  // namespace cbls
