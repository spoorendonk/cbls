#pragma once

#include <cstdint>

// What to do with a curated root-cause note (analysis_notes.csv: why an
// instance the runner could NOT solve came back infeasible) once a row is scored:
//
//   * infeasible              -- the note still applies: merge it into the
//                                published note column;
//   * solved and verified     -- the data contradicts the note: warn and mark
//                                it stale rather than publish a known-wrong claim;
//   * solved, failed verify   -- neither: a verification mismatch is not the
//                                infeasibility mechanism the note describes.
//
// A pure function so all three branches are testable without a solve.

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
