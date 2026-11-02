#pragma once

// MPS / .solu reader interface for CBLS.
//
// The reader implementation is vendored from spoorendonk/mipx
// (https://github.com/spoorendonk/mipx) and adapted to populate the
// minimal POD `MpsProblem` defined here instead of the heavier
// `mipx::LpProblem`.
//
// The MPS-to-Model adapter (`mps_to_model`) builds a closed CBLS
// `Model` from an `MpsProblem`.

#include "bound_propagation.h"
#include "model.h"
#include "presolve.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cbls {

inline constexpr double kMpsInf = std::numeric_limits<double>::infinity();

/// Variable kind from MPS COLUMNS / BOUNDS sections.
enum class MpsVarKind : uint8_t {
    Continuous,
    Integer,
    Binary,
};

/// Constraint sense from MPS ROWS section.
/// 'N' rows are not stored as constraints — the first 'N' row becomes the
/// objective; any further 'N' rows are ignored.
enum class MpsRowSense : uint8_t {
    L,  ///< <= rhs
    G,  ///< >= rhs
    E,  ///< == rhs
};

struct MpsVar {
    std::string name;
    double lb = 0.0;      ///< MPS default 0; -inf if FR/MI applied.
    double ub = kMpsInf;  ///< MPS default +inf; finite if UP/UI/BV/FX.
    MpsVarKind kind = MpsVarKind::Continuous;
};

struct MpsRow {
    std::string name;
    MpsRowSense sense = MpsRowSense::L;
    double rhs = 0.0;
    /// Range value (0 if absent). Interpretation depends on `sense`:
    ///   L: lower = rhs - |range|, upper = rhs
    ///   G: lower = rhs,           upper = rhs + |range|
    ///   E: range > 0 -> [rhs, rhs+range]
    ///       range < 0 -> [rhs+range, rhs]
    double range = 0.0;
};

/// One nonzero coefficient of the constraint matrix or objective.
/// `row_idx == -1` denotes the objective row (the first MPS 'N' row).
struct MpsNonzero {
    int32_t row_idx = -1;
    int32_t col_idx = -1;
    double value = 0.0;
};

struct MpsProblem {
    std::string name;
    std::string objective_row_name;  ///< First 'N' row name; empty if none.
    double objective_offset = 0.0;   ///< From RHS of the objective row.
    /// True if the MPS had `OBJSENSE MAX|MAXIMIZE`. The reader records this
    /// faithfully; `mps_to_model` currently rejects it (CBLS expects min).
    bool maximize = false;

    std::vector<MpsVar> vars;
    std::vector<MpsRow> rows;
    std::vector<MpsNonzero> nonzeros;  ///< (row_idx, col_idx, value); row_idx = -1 = objective.
};

/// Read an MPS file (fixed or free format). Auto-detects `.gz` for gzip.
/// If CBLS was built with `CBLS_USE_BZIP2`, also handles `.bz2`.
MpsProblem read_mps(const std::string& filename);

/// Entry from a MIPLIB `.solu` file.
struct SoluEntry {
    std::string name;
    double value = 0.0;
    bool is_infeasible = false;  ///< true for `=inf=` lines.
    bool is_optimal = false;     ///< true for `=opt=` lines (false otherwise).
};

/// Read a MIPLIB-style `.solu` file with known optimal / infeasible tags.
std::vector<SoluEntry> read_solu(const std::string& filename);

/// Build a closed CBLS `Model` from an `MpsProblem`.
///
/// One CBLS variable per MPS column (Int when kind != Continuous, otherwise
/// Float). One linear constraint per row, sense translated from `MpsRowSense`.
/// Objective is the linear cost expression from the 'N' row (plus `offset`).
///
/// CBLS variables require finite bounds, so a column with an infinite one has
/// to acquire a finite substitute. `propagate_bounds` derives that bound from
/// the constraints (see `bound_propagation.h`), which cannot remove a feasible
/// point; `inf_clamp` is the fallback where propagation finds no finite bound,
/// and *can* lose solutions, so it is only ever consulted last.
///
/// With `presolve.enabled` the rows and column boxes go through `presolve`
/// (presolve.h) first and the model is built from the reduced problem. Its
/// `propagate_bounds` and `propagation.max_passes` are then taken from the two
/// fields below, so propagation runs once, inside presolve, not twice.
struct MpsToModelOptions {
    double inf_clamp = 1.0e9;
    /// Derive implied bounds from the rows before falling back on `inf_clamp`.
    bool propagate_bounds = true;
    /// Fixed-point iteration cap for that pass; see `BoundPropagationOptions`.
    int max_propagation_passes = 10;
    /// Off by default; off, the model is the one built without presolve.
    PresolveOptions presolve;
};

struct MpsToModelResult {
    Model model;
    /// `var_handles[i]` is the CBLS variable handle (negative encoding
    /// `-(var_id + 1)`) for MPS column `i` without presolve, and for *reduced*
    /// column `i` with it: map a point back with `original_values`. Code that
    /// assumes one handle per MPS column must check the size first.
    std::vector<int32_t> var_handles;
    /// `constraint_node_ids[i]` is the constraint expression node id for MPS
    /// row `i`; -1 for a row presolve removed. A range row, and an `E` row with
    /// a RANGES entry, adds a second constraint not listed here. The listed one
    /// is the same side with or without presolve: `>=` for a `G` row, `<=` for
    /// `L` and ranged `E` (unless presolve dropped that side as redundant).
    std::vector<int32_t> constraint_node_ids;
    int32_t objective_node_id = -1;  ///< -1 if the MPS had no 'N' row.
    /// Outcome of the propagation pass without presolve; all-zero when it was
    /// disabled or ran inside presolve (counted in `presolve_stats` there).
    BoundPropagationStats bound_stats;
    /// Columns of the built model (reduced columns with presolve) whose bounds
    /// `inf_clamp` or the int32 clip still had to supply after propagation,
    /// i.e. where the unsound fallback was actually used.
    int n_clamped_columns = 0;
    /// All-zero (status `Unchanged`) without presolve. `Infeasible` means
    /// presolve proved the problem empty and the model was built without it.
    PresolveStats presolve_stats;
    /// Maps reduced values back to the MPS columns; empty (n_original 0)
    /// unless the model was built from a presolved problem.
    Postsolve postsolve;
    /// MPS column of each reduced column; empty unless built from presolve.
    std::vector<int32_t> reduced_to_original;

    /// The model's current point as one value per MPS column: the variable
    /// values, mapped back through `postsolve` when the model was presolved.
    /// `model` must be this result's model (or a copy of it).
    [[nodiscard]] std::vector<double> original_values(const Model& model) const;
};

MpsToModelResult mps_to_model(const MpsProblem& prob, const MpsToModelOptions& opts = {});

}  // namespace cbls
