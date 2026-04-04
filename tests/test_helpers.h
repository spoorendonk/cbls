#pragma once

#include <cbls/model.h>

// Short alias for tests — delegates to the canonical core function.
inline int32_t vid(int32_t handle) { return cbls::handle_to_var_id(handle); }
