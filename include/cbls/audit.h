#pragma once

#include <cstdint>

namespace cbls {

class FeasibilityJump;

/// Where in `solve()`'s outer loop a `SearchAuditor` is being called (#225).
enum class AuditPoint : std::uint8_t {
    /// At the top of every outer iteration, before the batch runs: after the
    /// previous batch's outcome (bound tightening, weight reset, resync), any
    /// diversification kick (perturb, LNS repair, adoption from the pool) and
    /// the shared-bound sync have all been applied. Every incremental structure
    /// the next batch reads must agree with the assignment here.
    BeforeBatch,
    /// Straight after a Feasibility-Jump batch returns.
    AfterFeasibilityJump,
    /// Straight after a Novelty-Jump batch returns.
    AfterNoveltyJump,
};

/// A test-facing consistency hook (#225): `solve()` hands it the search's
/// `FeasibilityJump` at each `AuditPoint`, so a test can compare the engine's
/// incremental state against a from-scratch recompute
/// (`FeasibilityJump::audit`) at points no public API otherwise reaches.
///
/// Not a host integration point -- that is `Tracer`. The default is a null
/// `SearchConfig::auditor`, which costs one predictable null compare per
/// batch. A non-null auditor must not change the assignment: it is handed a
/// const view, and an auditor that reaches around it voids the trajectory.
class SearchAuditor {
public:
    SearchAuditor() = default;
    SearchAuditor(const SearchAuditor&) = default;
    SearchAuditor& operator=(const SearchAuditor&) = default;
    SearchAuditor(SearchAuditor&&) = default;
    SearchAuditor& operator=(SearchAuditor&&) = default;
    virtual ~SearchAuditor();
    virtual void audit(AuditPoint point, const FeasibilityJump& fj) = 0;
};

}  // namespace cbls
