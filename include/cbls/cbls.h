#pragma once

namespace cbls {
inline constexpr const char* kVersion = "0.1.0";
}

#include "dag.h"
#include "dag_ops.h"
#include "expr.h"
#include "feasibility_jump.h"
#include "formatter.h"
#include "inner_solver.h"
#include "io.h"
#include "io_mps.h"
#include "lns.h"
#include "model.h"
#include "move_generator.h"
#include "moves.h"
#include "pool.h"
#include "randomize.h"
#include "rng.h"
#include "search.h"
// structural_batch.h is deliberately NOT here: StructuralBatch is the engine's
// own sweep, constructed by solve(). A caller registers generators through
// SearchConfig and never builds one. The tests that drive it directly include it
// directly.
#include "verify.h"
#include "violation.h"
