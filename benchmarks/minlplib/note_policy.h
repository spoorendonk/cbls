#pragma once

#include <cstdint>

// What to do with a curated root-cause note when a row is scored.
//
// analysis_notes.csv explains why an instance the runner could NOT solve came
// back infeasible. Three outcomes are possible once a row has been measured,
// and conflating them is how the "now solved" warning became dead code:
//
//   * the row is infeasible          -- the note still describes it, so merge it
//                                       into the published note column;
//   * the row solved and verified    -- the note's verdict is contradicted by the
//                                       data, so warn loudly and mark it stale
//                                       rather than publishing a claim we know
//                                       is wrong;
//   * the row solved but FAILED      -- neither. The failure is a solver
//     verification                     bookkeeping mismatch, not the
//                                       infeasibility mechanism the note
//                                       describes, so the note is neither
//                                       retired nor pasted on.
//
// Kept as a pure function so the three-way split is testable without running a
// solve: the previous form was a nested conditional inside main() whose "now
// solved" branch could not be reached at all.

namespace cbls::minlplib {

enum class NoteAction : std::uint8_t {
    kNone,   ///< no curated note, or a feasible-but-unverified row
    kMerge,  ///< row is infeasible: the note still applies
    kStale,  ///< row solved and verified: warn, the note is out of date
};

/// `has_note` -- analysis_notes.csv carries a row for this instance.
/// `feasible` -- the search reported a feasible incumbent.
/// `verified` -- the independent re-check of that incumbent passed.
inline NoteAction note_action(bool has_note, bool feasible, bool verified) {
    if (!has_note) {
        return NoteAction::kNone;
    }
    if (!feasible) {
        return NoteAction::kMerge;
    }
    return verified ? NoteAction::kStale : NoteAction::kNone;
}

}  // namespace cbls::minlplib
