#pragma once

// Feasibility check for a set-covering solution, recomputed from the parsed
// instance rather than read off the DAG's node values, so a model builder that
// wires up the wrong rows is caught.
//
// How independent it is depends on the encoding. The Bool model reads
// `inst.row_cols` while this reads `inst.covers`, so that pairing is a genuine
// two-representation cross-check. The Set model's coverage lambda reads
// `inst.covers` too, so for that encoding a *parser* bug would be invisible to
// both — the parser unit tests and the SHA-256 manifest gate are what cover
// that, not this file. `verify_model` (the generic bound/constraint check) is
// folded in on top.

#include "data.h"
#include "setcover_model.h"

#include <cbls/verify.h>
#include <cmath>
#include <string>
#include <vector>

namespace cbls {
namespace setcover {

struct CoverCheck {
    bool covered = false;       // every row covered by at least one chosen column
    int uncovered_rows = 0;
    int duplicate_columns = 0;  // a column selected twice (a malformed Set value)
    int invalid_columns = 0;    // a column index outside 0..cols-1
    double cost = 0.0;          // recomputed from the instance costs
    int num_columns = 0;
};

// Pure instance-level check: does `columns` cover every row, and what does it
// cost? Independent of the Model, so it also scores solutions read from a file.
inline CoverCheck check_cover(const SetCoverInstance& inst, const std::vector<int>& columns) {
    CoverCheck check;
    check.num_columns = static_cast<int>(columns.size());
    std::vector<uint8_t> seen(static_cast<size_t>(inst.cols), 0);
    std::vector<uint8_t> covered(static_cast<size_t>(inst.rows), 0);
    for (int col : columns) {
        if (col < 0 || col >= inst.cols) {
            ++check.invalid_columns;
            continue;
        }
        if (seen[static_cast<size_t>(col)]) {
            ++check.duplicate_columns;
            continue;  // a repeat neither adds coverage nor should be paid for twice
        }
        seen[static_cast<size_t>(col)] = 1;
        check.cost += inst.cost[static_cast<size_t>(col)];
        for (int i = 0; i < inst.rows; ++i) {
            if (inst.covers_row(i, col)) {
                covered[static_cast<size_t>(i)] = 1;
            }
        }
    }
    for (int i = 0; i < inst.rows; ++i) {
        if (!covered[static_cast<size_t>(i)]) {
            ++check.uncovered_rows;
        }
    }
    check.covered = (check.uncovered_rows == 0);
    return check;
}

// Full verification of a solved model: generic model check, plus the recomputed
// cover, plus agreement between the recomputed cost and the DAG's objective.
inline VerifyResult verify_setcover(const SetCoverModel& scm, const SetCoverInstance& inst,
                                    double tol = 1e-6) {
    VerifyResult result = verify_model(scm.model);
    const CoverCheck check = check_cover(inst, scm.selected_columns());

    if (!check.covered) {
        result.add_error({VerifyError::Kind::ConstraintViolation, "coverage",
                          0.0, static_cast<double>(check.uncovered_rows),
                          std::to_string(check.uncovered_rows) + " rows are covered by no "
                          "selected column"});
    }
    if (check.duplicate_columns + check.invalid_columns > 0) {
        result.add_error({VerifyError::Kind::Custom, "columns", 0.0,
                          static_cast<double>(check.duplicate_columns + check.invalid_columns),
                          "selected column list has " +
                              std::to_string(check.duplicate_columns) + " repeated and " +
                              std::to_string(check.invalid_columns) + " out-of-range entries"});
    }
    const double dag_objective = scm.model.node(scm.model.objective_id()).value;
    if (std::abs(dag_objective - check.cost) > tol * (1.0 + std::abs(check.cost))) {
        result.add_error({VerifyError::Kind::ObjectiveMismatch, "cost", check.cost, dag_objective,
                          "objective node disagrees with the cost recomputed from the instance"});
    }
    return result;
}

}  // namespace setcover
}  // namespace cbls
