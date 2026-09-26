#include "cbls/custom_invariant.h"

#include "cbls/model.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace cbls {

// Out of line rather than inline in the header, so that `custom_invariant.h`
// needs only a forward declaration of `Model` and `model.h` can include it --
// which is what lets `Model` hold `std::vector<CustomInvariantSlot>` with a
// defaulted destructor in the header. The cost is a call per input read, next
// to the virtual `delta()` the reads sit inside.

double InvariantInputs::value(int32_t i) const {
    const ChildRef& ref = children_[static_cast<size_t>(i)];
    if (ref.is_var) {
        return model_->variables()[ref.id].value;
    }
    return model_->node_values()[ref.id];
}

ConstSpan<int32_t> InvariantInputs::elements(int32_t i) const {
    const ChildRef& ref = children_[static_cast<size_t>(i)];
    if (!ref.is_var) {
        return {};
    }
    const Variable& var = model_->variables()[ref.id];
    if (!is_structured(var.type)) {
        return {};
    }
    return {var.elements.data(), var.elements.size()};
}

bool InvariantInputs::is_structured_input(int32_t i) const {
    const ChildRef& ref = children_[static_cast<size_t>(i)];
    return ref.is_var && is_structured(model_->variables()[ref.id].type);
}

ConstSpan<PositionalEdit> InputEdits::list() const {
    if (!available_) {
        throw std::logic_error(
            "InputEdits::list(): no positional information for this input -- check available() "
            "and re-read InvariantInputs::elements(i) when it is false");
    }
    return edits_;
}

InputEdits InvariantInputs::edits(int32_t i) const {
    if (!in_delta_ || !is_structured_input(i)) {
        return {};
    }
    // `changed` is ascending (see `CustomInvariant::delta`).
    if (!std::binary_search(changed_.begin(), changed_.end(), i)) {
        // Not recomputed this pass, so -- `changed` being exact for variable
        // inputs -- it did not move: a complete, empty description.
        return InputEdits(ConstSpan<PositionalEdit>());
    }
    if (journal_ == nullptr) {
        return {};
    }
    ConstSpan<PositionalEdit> list;
    const int32_t var_id = children_[static_cast<size_t>(i)].id;
    if (journal_->lookup(var_id, list) != EditJournal::Status::Known) {
        return {};
    }
    return InputEdits(list);
}

// ---------------------------------------------------------------------------
// EditJournal (#172)
// ---------------------------------------------------------------------------

void EditJournal::begin(int32_t var_id) {
    for (size_t r = 0; r < records_.size(); ++r) {
        if (records_[r].var_id != var_id) {
            continue;
        }
        open_ = r;
        if (r + 1 != records_.size()) {
            // Its edits would no longer be contiguous with the ones about to be
            // pushed; say "re-read it" rather than hand back half the story.
            records_[r].known = false;
        }
        return;
    }
    Record record;
    record.var_id = var_id;
    record.begin = static_cast<uint32_t>(edits_.size());
    record.end = record.begin;
    records_.push_back(record);
    open_ = records_.size() - 1;
}

void EditJournal::push(const PositionalEdit& edit) {
    assert(open_ != kNone && "EditJournal::push without begin()");
    Record& record = records_[open_];
    if (!record.known) {
        return;  // nothing an Unknown record says is read
    }
    // Only the LAST record can grow in place; `begin` has already marked any
    // other re-opened record Unknown, so this is the only live case.
    assert(open_ + 1 == records_.size() && record.end == edits_.size());
    edits_.push_back(edit);
    record.end = static_cast<uint32_t>(edits_.size());
}

void EditJournal::mark_unknown() noexcept {
    assert(open_ != kNone && "EditJournal::mark_unknown without begin()");
    records_[open_].known = false;
}

void EditJournal::append_inverse(const EditJournal& applied, int32_t var_id) {
    assert(&applied != this);
    ConstSpan<PositionalEdit> edits;
    switch (applied.lookup(var_id, edits)) {
        case Status::Absent:
            return;
        case Status::Unknown:
            mark_unknown();
            return;
        case Status::Known:
            break;
    }
    for (size_t k = edits.size(); k > 0; --k) {
        push(inverse_edit(edits[k - 1]));
    }
}

void EditJournal::append_forward(const EditJournal& applied, int32_t var_id) {
    assert(&applied != this);
    ConstSpan<PositionalEdit> edits;
    switch (applied.lookup(var_id, edits)) {
        case Status::Absent:
            return;
        case Status::Unknown:
            mark_unknown();
            return;
        case Status::Known:
            break;
    }
    for (const PositionalEdit& edit : edits) {
        push(edit);
    }
}

EditJournal::Status EditJournal::lookup(int32_t var_id,
                                        ConstSpan<PositionalEdit>& edits) const noexcept {
    for (const Record& record : records_) {
        if (record.var_id != var_id) {
            continue;
        }
        if (!record.known) {
            return Status::Unknown;
        }
        edits = ConstSpan<PositionalEdit>(edits_.data() + record.begin, record.end - record.begin);
        return Status::Known;
    }
    return Status::Absent;
}

}  // namespace cbls
