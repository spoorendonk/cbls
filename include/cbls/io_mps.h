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

#include "model.h"

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
/// Variables with infinite bounds are clamped to `[-inf_clamp, +inf_clamp]`
/// (default 1e9) since CBLS variables require finite bounds.
struct MpsToModelOptions {
    double inf_clamp = 1.0e9;
};

struct MpsToModelResult {
    Model model;
    /// `var_handles[i]` is the CBLS variable handle for MPS column `i`
    /// (negative encoding `-(var_id + 1)`).
    std::vector<int32_t> var_handles;
    /// `constraint_node_ids[i]` is the constraint expression node id for MPS row `i`.
    std::vector<int32_t> constraint_node_ids;
    int32_t objective_node_id = -1;  ///< -1 if the MPS had no 'N' row.
};

MpsToModelResult mps_to_model(const MpsProblem& prob, const MpsToModelOptions& opts = {});

}  // namespace cbls
