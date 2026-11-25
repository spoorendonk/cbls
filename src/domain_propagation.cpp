// Incremental, undoable domain propagation over linear rows. See
// include/cbls/domain_propagation.h.

#include "cbls/domain_propagation.h"

#include "activity_bounds.h"
#include "cbls/dag_propagation.h"
#include "cbls/linear_jump.h"
#include "cbls/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace cbls {

using activity_detail::Activity;
using activity_detail::eff_hi;
using activity_detail::eff_lo;
using activity_detail::improves;
using activity_detail::max_contribution;
using activity_detail::min_contribution;
using activity_detail::relaxation;
using activity_detail::roundoff_per_unit;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr double kPosInf = std::numeric_limits<double>::infinity();

}  // namespace

// ---------------------------------------------------------------------------
// LinearRowStore

LinearRowStore::LinearRowStore(const Model& model, double row_tolerance)
    : LinearRowStore(model, row_tolerance, nullptr, nullptr) {}

std::optional<LinearRowStore> LinearRowStore::build_until(
    const Model& model, double row_tolerance, std::chrono::steady_clock::time_point deadline) {
    bool complete = true;
    LinearRowStore store(model, row_tolerance, &deadline, &complete);
    if (!complete) {
        return std::nullopt;
    }
    return store;
}

LinearRowStore::LinearRowStore(const Model& model, double row_tolerance,
                               const std::chrono::steady_clock::time_point* deadline,
                               bool* complete)
    : row_tolerance_(row_tolerance) {
    const auto past = [deadline, complete] {
        if (deadline == nullptr || complete == nullptr ||
            std::chrono::steady_clock::now() < *deadline) {
            return false;
        }
        *complete = false;
        return true;
    };
    const auto n_vars = static_cast<int32_t>(model.num_vars());
    if (!model.is_closed()) {
        throw std::logic_error("LinearRowStore: the model must be closed and evaluated");
    }
    if (!std::isfinite(row_tolerance) || row_tolerance < 0.0) {
        throw std::invalid_argument("LinearRowStore: row_tolerance must be finite and >= 0");
    }
    const auto n_cons = static_cast<int32_t>(model.constraint_ids().size());
    std::vector<std::pair<int32_t, double>> terms;
    std::vector<size_t> col_count(static_cast<size_t>(n_vars) + 1, 0);
    row_start_.push_back(0);
    for (int32_t c = 0; c < n_cons; ++c) {
        if ((c & 15) == 15 && past()) {
            return;
        }
        if (c == model.objective_constraint_idx()) {
            continue;  // soft: propagating it would make the cutoff hard
        }
        double constant = 0.0;
        if (!affine_row_terms(model, c, terms, constant)) {
            continue;
        }
        if (terms.empty()) {
            const int32_t cid = model.constraint_ids()[static_cast<size_t>(c)];
            const bool is_eq = model.nodes()[static_cast<size_t>(cid)].op == NodeOp::Eq;
            constant_rows_.push_back(ConstantRow{c, constant, is_eq});
            continue;
        }
        const bool structured = std::any_of(terms.begin(), terms.end(), [&](const auto& t) {
            return is_structured(model.var(t.first).type);
        });
        if (structured) {
            continue;
        }
        const int32_t nid = model.constraint_ids()[static_cast<size_t>(c)];
        const bool eq = model.nodes()[static_cast<size_t>(nid)].op == NodeOp::Eq;
        // residual = sum + constant, and residual <= 0 (= 0 for Eq), each side
        // widened by the tolerance.
        constraint_.push_back(c);
        row_hi_.push_back(-constant + row_tolerance);
        row_lo_.push_back(eq ? -constant - row_tolerance : kNegInf);
        for (const auto& [v, a] : terms) {
            cols_.push_back(v);
            coefs_.push_back(a);
            ++col_count[static_cast<size_t>(v) + 1];
        }
        row_start_.push_back(cols_.size());
    }
    if (past()) {
        return;
    }
    build_column_index(n_vars, col_count);
}

void LinearRowStore::build_column_index(int32_t n_vars, const std::vector<size_t>& col_count) {
    // Counting sort of the row entries by variable, so each variable's
    // entries are in row order.
    col_start_.assign(static_cast<size_t>(n_vars) + 1, 0);
    for (size_t v = 1; v <= static_cast<size_t>(n_vars); ++v) {
        col_start_[v] = col_start_[v - 1] + col_count[v];
    }
    col_entries_.assign(cols_.size(), 0);
    col_rows_.assign(cols_.size(), 0);
    std::vector<size_t> next(col_start_.begin(), col_start_.end() - 1);
    for (int32_t r = 0; r < num_rows(); ++r) {
        for (size_t k = row_start(r); k < row_start(r + 1); ++k) {
            const size_t slot = next[static_cast<size_t>(cols_[k])]++;
            col_entries_[slot] = k;
            col_rows_[slot] = r;
        }
    }
}

// ---------------------------------------------------------------------------
// DomainPropagator

struct DomainPropagator::RowEntry {
    int32_t row;
    Activity min;
    Activity max;
};

DomainPropagator::~DomainPropagator() = default;

DomainPropagator::DomainPropagator(const LinearRowStore& store, const Model& model,
                                   DomainPropagationOptions opts, const DagRowStore* dag)
    : store_(store), dag_(dag), opts_(opts) {
    const auto n_vars = static_cast<size_t>(store.num_vars());
    if (model.num_vars() != n_vars) {
        throw std::invalid_argument("DomainPropagator: the store has " + std::to_string(n_vars) +
                                    " variables, the model " + std::to_string(model.num_vars()));
    }
    if (dag != nullptr && static_cast<size_t>(dag->num_vars()) != n_vars) {
        throw std::invalid_argument("DomainPropagator: the DAG store has " +
                                    std::to_string(dag->num_vars()) + " variables, the model " +
                                    std::to_string(n_vars));
    }
    integral_.resize(n_vars);
    lo_.resize(n_vars);
    hi_.resize(n_vars);
    for (size_t v = 0; v < n_vars; ++v) {
        const Variable& var = model.var(static_cast<int32_t>(v));
        integral_[v] = static_cast<uint8_t>(var.type == VarType::Bool || var.type == VarType::Int);
        lo_[v] = var.lb;
        hi_[v] = var.ub;
    }
    const auto n_rows = static_cast<size_t>(store.num_rows());
    min_act_.resize(n_rows);
    max_act_.resize(n_rows);
    max_variation_.assign(n_rows, 0.0);
    drift_scale_.assign(n_rows, 0.0);
    updates_.assign(n_rows, 0);
    const double inf = opts_.bounds.infinity;
    for (int32_t r = 0; r < store.num_rows(); ++r) {
        const auto ri = static_cast<size_t>(r);
        for (size_t k = store.row_start(r); k < store.row_start(r + 1); ++k) {
            const auto j = static_cast<size_t>(store.cols()[k]);
            const double a = store.coefs()[k];
            const double lo = eff_lo(lo_[j], inf);
            const double hi = eff_hi(hi_[j], inf);
            min_act_[ri].add(min_contribution(a, lo, hi));
            max_act_[ri].add(max_contribution(a, lo, hi));
            const double reach = std::abs(a) * (hi - lo);
            // NaN (an inf - inf box) bounds nothing: never filter the row.
            max_variation_[ri] = std::isnan(reach) ? kPosInf : std::max(max_variation_[ri], reach);
            // Bounds only shrink, so no later finite contribution exceeds this.
            drift_scale_[ri] += std::abs(a) * std::max(std::abs(lo), std::abs(hi));
        }
    }
    int64_t work = store.nnz();
    size_t n_dag = 0;
    if (dag != nullptr) {
        n_dag = static_cast<size_t>(dag->num_rows());
        work += dag->size();
        dag_node_fwd_.resize(dag->max_row_nodes());
        dag_node_bwd_.resize(dag->max_row_nodes());
        dag_var_fwd_.resize(dag->max_row_vars());
        dag_var_bwd_.resize(dag->max_row_vars());
    }
    queued_stamp_.assign(n_rows + n_dag, 0);
    constexpr int64_t kValvePerNonzero = 100;
    cascade_limit_ =
        opts_.max_cascade_terms >= 0 ? opts_.max_cascade_terms : kValvePerNonzero * work;
}

int32_t DomainPropagator::checkpoint() {
    levels_.push_back(Level{bound_trail_.size(), row_trail_.size()});
    return static_cast<int32_t>(levels_.size()) - 1;
}

void DomainPropagator::backtrack(int32_t level) {
    if (level < 0 || static_cast<size_t>(level) >= levels_.size()) {
        throw std::out_of_range("DomainPropagator::backtrack: no checkpoint at level " +
                                std::to_string(level));
    }
    const Level to = levels_[static_cast<size_t>(level)];
    // Reverse order, so the oldest saved value of each item is the one left.
    while (row_trail_.size() > to.rows) {
        const RowEntry& e = row_trail_.back();
        min_act_[static_cast<size_t>(e.row)] = e.min;
        max_act_[static_cast<size_t>(e.row)] = e.max;
        --updates_[static_cast<size_t>(e.row)];
        row_trail_.pop_back();
    }
    while (bound_trail_.size() > to.bounds) {
        const BoundEntry& e = bound_trail_.back();
        lo_[static_cast<size_t>(e.var)] = e.lo;
        hi_[static_cast<size_t>(e.var)] = e.hi;
        bound_trail_.pop_back();
    }
    levels_.resize(static_cast<size_t>(level));
}

void DomainPropagator::changed_since(int32_t level, std::vector<int32_t>& out) const {
    if (level < 0 || static_cast<size_t>(level) >= levels_.size()) {
        throw std::out_of_range("DomainPropagator::changed_since: no checkpoint at level " +
                                std::to_string(level));
    }
    out.clear();
    for (size_t i = levels_[static_cast<size_t>(level)].bounds; i < bound_trail_.size(); ++i) {
        out.push_back(bound_trail_[i].var);
    }
    // Distinct, in first-change order. O(t log t) for t trail entries.
    std::vector<int32_t> seen(out);
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    std::vector<uint8_t> taken(seen.size(), 0);
    size_t n = 0;
    for (const int32_t v : out) {
        const auto pos =
            static_cast<size_t>(std::lower_bound(seen.begin(), seen.end(), v) - seen.begin());
        if (taken[pos] == 0) {
            taken[pos] = 1;
            out[n++] = v;
        }
    }
    out.resize(n);
}

double DomainPropagator::min_activity_sum(int32_t r) const {
    return min_act_[static_cast<size_t>(r)].sum;
}
int DomainPropagator::min_activity_infinite(int32_t r) const {
    return min_act_[static_cast<size_t>(r)].n_infinite;
}
double DomainPropagator::max_activity_sum(int32_t r) const {
    return max_act_[static_cast<size_t>(r)].sum;
}
int DomainPropagator::max_activity_infinite(int32_t r) const {
    return max_act_[static_cast<size_t>(r)].n_infinite;
}

// Sets var's bounds, trailing the old ones and every row activity it moves.
// O(rows of var).
void DomainPropagator::set_bounds(int32_t var, double lo, double hi) {
    const auto j = static_cast<size_t>(var);
    const double inf = opts_.bounds.infinity;
    const double old_lo = eff_lo(lo_[j], inf);
    const double old_hi = eff_hi(hi_[j], inf);
    const double new_lo = eff_lo(lo, inf);
    const double new_hi = eff_hi(hi, inf);
    // At depth 0 nothing can backtrack, so nothing is trailed.
    const bool trail = !levels_.empty();
    if (trail) {
        bound_trail_.push_back(BoundEntry{var, lo_[j], hi_[j]});
    }
    lo_[j] = lo;
    hi_[j] = hi;
    for (size_t e = store_.col_start(var); e < store_.col_start(var + 1); ++e) {
        const int32_t r = store_.col_rows()[e];
        const auto ri = static_cast<size_t>(r);
        const double a = store_.coefs()[store_.col_entries()[e]];
        if (trail) {
            row_trail_.push_back(RowEntry{r, min_act_[ri], max_act_[ri]});
        }
        ++updates_[ri];  // the drift is real either way
        min_act_[ri].remove(min_contribution(a, old_lo, old_hi));
        min_act_[ri].add(min_contribution(a, new_lo, new_hi));
        max_act_[ri].remove(max_contribution(a, old_lo, old_hi));
        max_act_[ri].add(max_contribution(a, new_lo, new_hi));
    }
}

void DomainPropagator::enqueue_rows_of(int32_t var) {
    for (size_t e = store_.col_start(var); e < store_.col_start(var + 1); ++e) {
        const int32_t r = store_.col_rows()[e];
        if (queued_stamp_[static_cast<size_t>(r)] != stamp_) {
            queued_stamp_[static_cast<size_t>(r)] = stamp_;
            queue_.push_back(r);
        }
    }
    if (dag_ == nullptr) {
        return;
    }
    for (size_t e = dag_->var_row_start(var); e < dag_->var_row_start(var + 1); ++e) {
        const int32_t id = store_.num_rows() + dag_->var_rows()[e];
        if (queued_stamp_[static_cast<size_t>(id)] != stamp_) {
            queued_stamp_[static_cast<size_t>(id)] = stamp_;
            queue_.push_back(id);
        }
    }
}

bool DomainPropagator::conflict() {
    ++stats_.conflicts;
    return false;
}

// Applies `x[var] <= value` (`>=` when `upper` is false) as `propagate_bounds`
// does: relaxed outward by the safety or roundoff margin, rounded inward on an
// integral variable, kept only if it improves the bound materially. A crossing
// within the safety margin collapses to a point; a wider one is a conflict.
bool DomainPropagator::apply(int32_t var, double value, bool upper, double roundoff) {
    const BoundPropagationOptions& b = opts_.bounds;
    if (!std::isfinite(value) || std::abs(value) >= b.infinity) {
        return true;
    }
    const double eps = std::max(relaxation(value, b), roundoff);
    value = upper ? value + eps : value - eps;
    const auto j = static_cast<size_t>(var);
    if (integral_[j] != 0) {
        value = upper ? std::floor(value + eps) : std::ceil(value - eps);
    }
    double lo = lo_[j];
    double hi = hi_[j];
    if (upper) {
        if (!improves(value, eff_hi(hi, b.infinity), b)) {
            return true;
        }
        hi = value;
    } else {
        if (!improves(-value, -eff_lo(lo, b.infinity), b)) {
            return true;
        }
        lo = value;
    }
    if (lo > hi) {
        if (lo > hi + relaxation(lo, b)) {
            return false;
        }
        if (upper) {
            hi = lo;
        } else {
            lo = hi;
        }
    }
    ++stats_.bounds_derived;
    set_bounds(var, lo, hi);
    if (enqueue_derived_) {
        enqueue_rows_of(var);
    }
    return true;
}

// Row r's activities summed afresh from the current bounds, in term order:
// the arithmetic `propagate_bounds` derives with. O(nnz).
void DomainPropagator::rebuild_activities(int32_t r, Activity& min_act, Activity& max_act) const {
    const double inf = opts_.bounds.infinity;
    min_act = Activity{};
    max_act = Activity{};
    for (size_t k = store_.row_start(r); k < store_.row_start(r + 1); ++k) {
        const auto j = static_cast<size_t>(store_.cols()[k]);
        const double a = store_.coefs()[k];
        const double lo = eff_lo(lo_[j], inf);
        const double hi = eff_hi(hi_[j], inf);
        min_act.add(min_contribution(a, lo, hi));
        max_act.add(max_contribution(a, lo, hi));
    }
}

// True if the slack filter may skip row r: on each side that could derive
// something, the slack is at least the max variation plus a bound on the
// rounding drift of the incremental sum. Each add or remove on a running sum
// whose finite terms are at most drift_scale_ in total is off by at most
// DBL_EPSILON * drift_scale_; the row took nnz adds at the start and two per
// live update since, and the slack subtraction adds one more operation, over
// |row bound| as well. O(1).
bool DomainPropagator::filtered(int32_t r) const {
    const auto ri = static_cast<size_t>(r);
    const Activity& min_act = min_act_[ri];
    const Activity& max_act = max_act_[ri];
    const double row_lo = eff_lo(store_.row_lo(r), opts_.bounds.infinity);
    const double row_hi = eff_hi(store_.row_hi(r), opts_.bounds.infinity);
    const auto ops = static_cast<double>(store_.row_start(r + 1) - store_.row_start(r) +
                                         (2 * static_cast<size_t>(updates_[ri])) + 2);
    const double eps = std::numeric_limits<double>::epsilon();
    const double mv = max_variation_[ri];
    // One infinite term forces the scan: it is the term that side can bound.
    const bool upper_tight =
        std::isfinite(row_hi) && min_act.n_infinite <= 1 &&
        (min_act.n_infinite == 1 ||
         row_hi - min_act.sum < mv + (ops * eps * (drift_scale_[ri] + std::abs(row_hi))));
    const bool lower_tight =
        std::isfinite(row_lo) && max_act.n_infinite <= 1 &&
        (max_act.n_infinite == 1 ||
         max_act.sum - row_lo < mv + (ops * eps * (drift_scale_[ri] + std::abs(row_lo))));
    return !upper_tight && !lower_tight;
}

// One row against the current bounds. The incremental activities only decide
// the filter; a scanned row's activities are summed afresh, since a running
// sum keeps the rounding of what it removed (#334). They are read once, at the start, as
// `propagate_bounds` reads them: a bound derived mid-row only tightens later terms' rest-of-row, so
// the stale copy is sound, merely weaker. O(1) when the filter skips the row, else O(nnz) plus the
// derived bounds' set_bounds.
bool DomainPropagator::scan_row(int32_t r) {
    if (r >= store_.num_rows()) {
        return scan_dag_row(r - store_.num_rows());
    }
    if (opts_.filter && filtered(r)) {
        ++stats_.rows_filtered;
        return true;
    }
    const double row_lo = eff_lo(store_.row_lo(r), opts_.bounds.infinity);
    const double row_hi = eff_hi(store_.row_hi(r), opts_.bounds.infinity);
    Activity min_act;
    Activity max_act;
    rebuild_activities(r, min_act, max_act);
    // A side derives something only with a finite row bound and at most one
    // infinite term in the activity it is read against.
    const bool upper_side = std::isfinite(row_hi) && min_act.n_infinite <= 1;
    const bool lower_side = std::isfinite(row_lo) && max_act.n_infinite <= 1;
    if (!upper_side && !lower_side) {
        return true;
    }
    ++stats_.rows_scanned;
    const double inf = opts_.bounds.infinity;
    const size_t begin = store_.row_start(r);
    const size_t end = store_.row_start(r + 1);
    const size_t nnz = end - begin;
    call_terms_ += static_cast<int64_t>(nnz);
    stats_.terms_scanned += static_cast<int64_t>(nnz);
    for (size_t k = begin; k < end; ++k) {
        const int32_t col = store_.cols()[k];
        const double a = store_.coefs()[k];
        const auto j = static_cast<size_t>(col);
        const double lo = eff_lo(lo_[j], inf);
        const double hi = eff_hi(hi_[j], inf);
        // a x <= row_hi - min_rest  and  a x >= row_lo - max_rest.
        const double min_rest = min_act.without(min_contribution(a, lo, hi), kNegInf);
        const double max_rest = max_act.without(max_contribution(a, lo, hi), kPosInf);
        const double upper_slack =
            (std::isfinite(row_hi) && std::isfinite(min_rest)) ? row_hi - min_rest : kPosInf;
        const double lower_slack =
            (std::isfinite(row_lo) && std::isfinite(max_rest)) ? row_lo - max_rest : kNegInf;
        const double per_unit = roundoff_per_unit(nnz, a);
        const double hi_side_roundoff = per_unit * (std::abs(row_hi) + min_act.magnitude);
        const double lo_side_roundoff = per_unit * (std::abs(row_lo) + max_act.magnitude);
        const bool pos = a > 0.0;
        if (!apply(col, upper_slack / a, /*upper=*/pos, hi_side_roundoff) ||
            !apply(col, lower_slack / a, /*upper=*/!pos, lo_side_roundoff)) {
            return false;
        }
    }
    return true;
}

bool DomainPropagator::fix(int32_t var, double value) {
    return tighten(var, value, value);
}

bool DomainPropagator::tighten(int32_t var, double lo, double hi) {
    switch (narrow(var, lo, hi, "tighten")) {
        case Narrowed::kConflict:
            return conflict();
        case Narrowed::kUnchanged:
            return true;
        case Narrowed::kChanged:
            break;
    }
    begin_call();
    enqueue_rows_of(var);
    return run_queue();
}

bool DomainPropagator::fix_all(const std::vector<std::pair<int32_t, double>>& fixings) {
    // Validate first, so a throw leaves the state untouched.
    for (const auto& [var, value] : fixings) {
        check_request(var, value, value, "fix_all");
    }
    bool ok = true;
    begin_call();
    for (const auto& [var, value] : fixings) {
        switch (narrow(var, value, value, "fix_all")) {
            case Narrowed::kConflict:
                ok = conflict();
                break;
            case Narrowed::kUnchanged:
                break;
            case Narrowed::kChanged:
                enqueue_rows_of(var);
                break;
        }
    }
    // Propagated even after a crossing fixing, as the reference does.
    return run_queue() && ok;
}

void DomainPropagator::check_request(int32_t var, double lo, double hi, const char* caller) const {
    if (var < 0 || var >= store_.num_vars()) {
        throw std::out_of_range(std::string("DomainPropagator::") + caller + ": variable " +
                                std::to_string(var));
    }
    if (std::isnan(lo) || std::isnan(hi)) {
        throw std::invalid_argument(std::string("DomainPropagator::") + caller +
                                    ": NaN bound on variable " + std::to_string(var));
    }
}

// Intersects `var`'s domain with `[lo, hi]` (inward on an integral variable),
// without propagating. A crossing beyond the margin changes nothing.
DomainPropagator::Narrowed DomainPropagator::narrow(int32_t var, double lo, double hi,
                                                    const char* caller) {
    check_request(var, lo, hi, caller);
    ++stats_.calls;
    if (lo == kPosInf || hi == kNegInf) {
        // No point satisfies it; the margins below would read inf - inf.
        return Narrowed::kConflict;
    }
    const auto j = static_cast<size_t>(var);
    const BoundPropagationOptions& b = opts_.bounds;
    if (integral_[j] != 0) {
        // Inward, after the same margin `apply` allows a derived bound.
        lo = std::ceil(lo - relaxation(lo, b));
        hi = std::floor(hi + relaxation(hi, b));
    }
    double new_lo = std::max(lo, lo_[j]);
    double new_hi = std::min(hi, hi_[j]);
    if (new_lo > new_hi) {
        if (new_lo > new_hi + relaxation(new_lo, b)) {
            return Narrowed::kConflict;
        }
        // A crossing within the margin: a point, inside the current box.
        const double point = (lo_[j] == new_lo) ? new_lo : new_hi;
        new_lo = new_hi = std::min(std::max(point, lo_[j]), hi_[j]);
    }
    if (new_lo == lo_[j] && new_hi == hi_[j]) {
        return Narrowed::kUnchanged;
    }
    set_bounds(var, new_lo, new_hi);
    return Narrowed::kChanged;
}

bool DomainPropagator::propagate_all() {
    ++stats_.propagate_all_calls;
    const BoundPropagationOptions& b = opts_.bounds;
    for (const LinearRowStore::ConstantRow& row : store_.constant_rows()) {
        // residual = -(row bound); a positive one is a bound below 0, finite
        // on its own side however large.
        const double excess = row.eq ? std::abs(row.residual) : row.residual;
        if (excess > relaxation(row.residual, b) + store_.row_tolerance()) {
            return conflict();
        }
    }
    const int64_t derived_before = stats_.bounds_derived;
    begin_call();
    const int32_t n_queued =
        store_.num_rows() + (dag_ != nullptr ? dag_->num_rows() : 0);  // DAG rows after
    for (int32_t r = 0; r < n_queued; ++r) {
        queued_stamp_[static_cast<size_t>(r)] = stamp_;
        queue_.push_back(r);
    }
    const bool ok = run_queue();
    stats_.propagate_all_bounds_derived += stats_.bounds_derived - derived_before;
    return ok;
}

void DomainPropagator::begin_call() {
    // A fresh stamp per call; on wrap-around, clear the stamps once.
    if (++stamp_ == 0) {
        std::fill(queued_stamp_.begin(), queued_stamp_.end(), 0U);
        stamp_ = 1;
    }
    queue_.clear();
    call_terms_ = 0;
    enqueue_derived_ = false;
}

// Scans the queued rows; with cascade on, rows of derived bounds join the
// queue, until it empties or, past the call's own rows, the valve trips.
bool DomainPropagator::run_queue() {
    enqueue_derived_ = opts_.cascade;
    const size_t own = queue_.size();  // the call's own rows: never cut
    // By index: with cascade on, scan_row appends to queue_ as it goes.
    size_t q = 0;
    while (q < queue_.size()) {
        if (q >= own && call_terms_ >= cascade_limit_) {
            ++stats_.valve_stops;
            break;
        }
        const int32_t r = queue_[q];
        // Dequeued: a bound derived later in this call may queue it again.
        queued_stamp_[static_cast<size_t>(r)] = 0;
        if (!scan_row(r)) {
            return conflict();
        }
        ++q;
    }
    return true;
}

}  // namespace cbls
