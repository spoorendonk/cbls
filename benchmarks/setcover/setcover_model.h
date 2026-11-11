#pragma once

// Set covering, modelled two ways on purpose.
//
//   min  sum_j c_j x_j   s.t. every row is covered by at least one chosen column
//
// `Encoding::Set` (#93): ONE `Set` variable over the columns, each row's
// coverage a `lambda_sum` over the chosen ones, searched by the Set
// add/remove/swap generators in the structural batch.
// `Encoding::Bool` is the control: one Bool per column, the linear encoding
// CP-SAT's LS also takes, searched by GFJ. Both on one instance make Set support
// a measurement (benchmarks/instances/setcover/README.md).
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

namespace cbls::setcover {

enum class Encoding : std::uint8_t { Set, Bool };

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
    [[nodiscard]] std::vector<int> selected_columns() const {
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
// Cardinality bound `min(cols, rows)`, valid not tuned: a minimal cover has at
// most one column per row, and with non-negative costs (OR-Library's are
// positive) some optimum is minimal. Drop it for signed costs. It matters:
// `initialize_structured_random` draws the size uniformly from [min, max], so
// without it the search starts near cols/2 and spends its budget shedding.
inline SetCoverModel build_set_model(const SetCoverInstance& inst) {
    SetCoverModel scm;
    scm.encoding = Encoding::Set;
    Model& m = scm.model;

    const int max_size = std::min(inst.cols, inst.rows);
    scm.chosen = m.set_var(inst.cols, /*min_size=*/0, max_size, "columns");

    for (int i = 0; i < inst.rows; ++i) {
        const uint8_t* row = inst.covers.data() + (static_cast<size_t>(i) * inst.cols);
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

}  // namespace cbls::setcover
