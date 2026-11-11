# benchmarks/common

`benchmarks/common/` is **not** a benchmark. It holds what runners and drivers
share: C++ headers for the runners (`runner_args.h`, `search_config_flags.h`,
`published_guard.h`, `runner_io.h`) and a Python package for the drivers (#160).

`runner_args.h` holds three things. First the flag-value *reporting policy* (a
bad double reports and returns NaN for a later positivity guard to turn into
exit 2; a bad integer reports and exits 2), layered over the pure parsing *rule*
in `include/cbls/arg_parse.h`. Second `ArgCursor`, the value-flag matching rule:
a flag that takes a value matches only when a value actually follows it, so a
trailing `--budget` falls through to the runner's unknown-argument path instead
of reading past `argv` or silently keeping a default that a published table would
then be built on. Third `same_file`, the published-table guard's path
comparison — canonicalised where the file exists, so that naming the published
table explicitly (which the documented commands do) is held to the same rule as
defaulting into it. `mipfeas`, `minlplib`, `uc-chped` and `cvrp` share the first two;
`setcover` deliberately does not, and says why at its own `parse_args`. The
policy split is deliberate: the rule is a library concern, the policy writes to
stderr and calls `std::exit` and is pinned by
`tests/python/test_run_benchmark.py`, so it lives next to the programs those
tests run.

`published_guard.h` is `check_published_write`, the refusals `minlplib` and
`uc-chped` share before writing `<inst_dir>/comparison.csv` or
`anytime_trace.csv`: `--out`/`--trace` naming one file or the table's temp file,
a crossed artifact, a non-protocol run, a missing `--commit`. Its messages are
pinned by `tests/python/test_cli.py`. `runner_io.h` is `file_exists`,
`seconds_since`, `peak_rss_kib` and the temp-then-rename writes
(`rename_into_place`, `write_atomically`) the four runners use.

`search_config_flags.h` is the per-arm search configuration (#136): the flag
table, its parse, its validation, its application onto `cbls::SearchConfig` and
`solve()`'s arguments, and the canonical `search_config` cell every row carries.
`minlplib`, `uc-chped` and `cvrp` share it; `mipfeas` takes five of its flags (#267:
`--no-lns`, `--lns-interval`, `--novelty-prob`, `--unproductive-iters`,
`--perturbation-period`), parsed and validated here but recorded under its own
JSON keys. Two rules it exists to enforce: **a flag
whose field the engine ignores must not ship** (the header names the two
rejected — the GLS ρ and `use_fj`, whose argument `solve()` opens by discarding),
and **a non-default arm may never write a published comparison table**, which is
why `first_non_default_search_flag` feeds `minlplib`'s and `uc-chped`'s
`resolve_out_csv`. Every
flag has a behavioural probe in `tests/test_bench_flags.cpp`; add one with the
next flag, and check it fails when the apply step is neutered.

Whether the engine reads a flag depends on the model too, so every flag belongs
to a **family** (common, scalar, structural, neighbour; #198) and every runner
declares the families its models can use: `minlplib`, `uc-chped` and `mipfeas`
declare `kScalarRunnerSearchFlags`, a List runner the common, structural and — when it
supplies the cost — neighbour families. A flag outside the declared set is
refused by `validate_search_flags`, and the `search_config` cell and `--help`
carry the declared families only, which keeps a scalar runner's cell
byte-identical to its committed tables. `--neighbours K` carries only K; the
runner hands its cost to `neighbours_argument`. `setcover` keeps its own parser
(one invocation spans a Set and a Bool model, which no per-runner mask can
describe) but uses the table's spellings and value rules for its three
structural flags.

Neither header has a CMake target — runners reach them through the
`target_include_directories(<target> PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})` line
each already carries, so a new runner target that omits that line will not
compile `<benchmarks/common/...>`. `cbls_tests` carries the same line for
`${CMAKE_SOURCE_DIR}`, which is what lets a Catch2 test include them without a
runner `main()`.

The Python package is the driver mechanics every benchmark shares: `jobs.py`
runs one solver process (timeout, address-space cap, own session, log) and a
plan (bounded workers, serial tail); `records.py` writes and reads the files a
run leaves behind so that a kill leaves the old file or the new one, and lists
where each benchmark's record schema is declared; `provenance.py` is the commit,
the build-dir refusals and the machine record; `pins.py` hashes downloaded files
against a committed pin table (the `mipfeas` and `cvrp` downloaders); `fetch.py`
is the downloaders' HTTP fetch and HTML-page sniff; `stats.py` is the Student t
quantile and the exact sign test the score and ablation reports use. What it deliberately does **not**
hold is any benchmark's resume rule or record shape — `mipfeas`, the MINLPLib
re-run and the ablation campaign resume on three different keys for reasons of
measurement, and a single loop with per-benchmark hooks would only relocate
them. The MINLPLib runner's contract (columns, exit status, note vocabulary) is
declared once in `benchmarks/minlplib/runner.py`. The mechanics are pinned in
`tests/python/test_benchmark_common.py`; don't re-test the mechanics per driver,
but do pin each driver's use of them — which jobs go in the tail, what a killed
job records, which timeout a call carries — since that wiring is what a
refactor silently drops. Drivers run as scripts reach the package through a
`sys.path` shim on the repository root, as `run_ablation.py` always did.

All four runners take `--threads N` (`parse_thread_count` in `runner_args.h`),
the `ParallelSearch` portfolio above 1. `uc-chped` refuses that onto its
published table, which measures the single-threaded `solve()`; `minlplib`
publishes at 1 or at the campaign count 16 (#255, #256), one count per table.
(The `xk tune` trial driver `tune_trial.py` and its study files, #261-#266, are
in git history: `git show 047f2d4c:benchmarks/common/tune_trial.py`.)
