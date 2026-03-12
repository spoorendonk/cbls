#pragma once

#include <cstdint>

// Convert a variable handle (negative) to internal var ID
static inline int32_t vid(int32_t handle) { return -(handle + 1); }
