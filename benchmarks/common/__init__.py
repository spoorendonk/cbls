"""What every benchmark driver shares, whatever it measures (issue #160).

Three modules, one per stage of a run's lifecycle that is not specific to a
benchmark:

* `provenance` -- what produced a result beyond the code: the engine commit, the
  build directory it was compiled in, the machine it ran on;
* `jobs` -- running a solver process: a timeout, an address-space cap, its own
  process group, a log, and bounded concurrency with a serial tail;
* `records` -- writing and reading the files a run leaves behind so that a kill
  at any moment leaves either the previous file or the new one, never half of
  one; and where each benchmark's record schema is declared.

What is deliberately NOT here is each benchmark's resume rule and record shape.
They differ for reasons of measurement, not of implementation -- `mipfeas` runs
four-up and verifies every point, `minlplib` publishes a table and must solve
serially, the ablation interleaves arms per instance -- and issue #160's own
acceptance criteria forbid changing any of them. `records` names them in one
place instead.

The C++ headers beside these modules (`runner_args.h`, `search_config_flags.h`)
are the runner-side counterpart and are unrelated to this package.
"""
