#include "cbls/custom_invariant.h"

#include "cbls/model.h"

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

}  // namespace cbls
