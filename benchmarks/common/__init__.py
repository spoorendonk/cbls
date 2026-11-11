"""What every benchmark driver shares, whatever it measures (#160).

* `provenance` -- what produced a result beyond the code: the engine commit, the
  build directory it was compiled in, the machine it ran on;
* `jobs` -- running a solver process: a timeout, an address-space cap, its own
  process group, a log, and bounded concurrency with a serial tail;
* `records` -- writing and reading a run's files so that a kill at any moment
  leaves either the previous file or the new one, never half of one; and where
  each benchmark's record schema is declared;
* `pins` -- hashing downloads against a committed pin table, so a library
  revision cannot silently redefine a benchmark's reference values (#196).

Each benchmark's resume rule and record shape stays in its driver: they differ
for measurement reasons (`mipfeas` runs four-up and verifies every point,
`minlplib` publishes a table and must solve serially, the ablation interleaves
arms per instance), and #160 forbids changing them. `records` names them in one
place.

The C++ headers beside these modules (`runner_args.h`, `search_config_flags.h`)
are the runner-side counterpart and are unrelated to this package.
"""
