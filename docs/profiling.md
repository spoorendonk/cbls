# Profiling, heap attribution and sanitizers

CLAUDE.md says to profile before micro-optimizing. This document is how.

It exists because performance and memory claims in this repository have been
made by arithmetic and subtraction rather than by measurement, and were wrong
often enough to matter — a published pass cost that was pure run-to-run noise
(off by ~8x once timed directly), a several-hundred-MB "memory risk" that
measurement showed did not exist, and a MIPfeas reader that reportedly reaches
the whole process's peak RSS with nothing attributed to it (issue #122).
Subtraction of two whole-program timings is not a measurement; neither is a
review estimate of an allocation.

Everything below was verified on the machine described under
[Tool availability](#tool-availability-verified-not-assumed). Nothing here is
written from memory of what a tool usually does.

## Which tool for which question

| Question | Tool | Section |
|---|---|---|
| How much memory does this peak at? | `/usr/bin/time -v`, or the benchmark's own `peak_rss_kib` | [Heap: magnitude](#heap-magnitude) |
| *Which call site* owns that peak? | jemalloc heap profiler, `LD_PRELOAD`, no rebuild | [Heap: attribution](#heap-attribution) |
| Where does the wall clock go? | `perf record` (needs a sysctl), else `gprof` | [CPU](#cpu) |
| Exact instruction counts, clean call graph | `valgrind --tool=callgrind` (~50x slowdown, not installed) | [CPU](#callgrind) |
| Is this undefined behaviour / a memory error? | `-DCBLS_SANITIZE=...` | [Sanitizers](#sanitizers) |

Two rules apply to every recipe here:

- **Never configure a sanitizer or profiling build in `build/`.** That directory
  is what the git hooks rebuild and gate on, and both flags are *cache* entries:
  set one there once and every later flag-less `cmake -B build` keeps it, with
  pre-commit gating on the result. Measurement builds get their own directory
  (`build-asan/`, `build-profile/`); `.gitignore` covers `build*/`.
- **Never run a profile next to a benchmark comparison, a parallel build, or
  another profile.** CLAUDE.md already forbids concurrent time-limited
  comparisons; a profiler competing for cores and memory invalidates both. Check
  `uptime` first.

## Build flags

| Flag | Default | Adds | For |
|---|---|---|---|
| `-DCBLS_PROFILE=ON` | OFF | `-g -fno-omit-frame-pointer` on top of the build type | `perf`, heap-dump symbolisation |
| `-DCBLS_SANITIZE=<comma,list>` | empty (off) | `-fsanitize=<list> -g -fno-omit-frame-pointer`, and the same `-fsanitize` at link | ASan/UBSan runs |

Both are declared before the first target in the root `CMakeLists.txt`, so the
flags reach `cbls_lib`, the CLI, the benchmark runners *and* Catch2. Neither
reaches a fresh gated build, which passes neither flag — but see the cache
warning above before reusing a `build/` that once had one.

The profiling build:

```bash
cmake -B build-profile -DCBLS_PROFILE=ON
cmake --build build-profile -j"$(nproc)"
```

**State the build type in any profile you publish.** `CMakeLists.txt` defaults
to Release, and the difference is not cosmetic: the same suite runs in ~40s
optimized and ~304s unoptimized, so a profile of a `-DCMAKE_BUILD_TYPE=Debug`
binary measures a program nobody runs. `CBLS_PROFILE` deliberately does *not*
change the build type — it only adds back the symbols and frame pointers that
Release drops.

## Tool availability: verified, not assumed

Measured 2026-09-02 on Ubuntu 26.04, kernel 7.0.0-30-generic, GCC 15.2.0. Every
"present" row below was exercised, not just `command -v`'d; every "absent" row
carries the exact install command. Re-check on a different machine rather than
trusting this table.

| Tool | State | Notes |
|---|---|---|
| `perf` | **present but blocked** | `/proc/sys/kernel/perf_event_paranoid` is `4`; `perf record` fails to open *any* event, including software ones. See [CPU](#cpu). |
| `gprof` | **present, works** | GCC's own; needs a `-pg` build. The no-privilege CPU fallback. |
| jemalloc heap profiler | **present, works** | Stock `libjemalloc.so.2` (5.3.0-4) is built **with** `--enable-prof`, so profiling needs no rebuild and no root. |
| `gdb`, `strace`, `addr2line`, `ccache` | present | `ptrace_scope` is `1`: `gdb` may only attach to its own descendants, so launch under `gdb`, don't `gdb -p`. |
| GCC sanitizers (`libasan`) | present, works | Stack traces symbolise correctly without `llvm-symbolizer`. |
| `valgrind` (massif, callgrind) | **absent** | `sudo apt install valgrind` |
| `heaptrack` | **absent** | `sudo apt install heaptrack` (`heaptrack-gui` for the flame view) |
| `jeprof` | **absent** | `sudo apt install libjemalloc-dev` — the [symbolisation fallback](#symbolising-a-heap-dump) needs no install |
| `llvm-symbolizer`, `clang` | absent | Not needed; GCC symbolises its own sanitizer output. |

## Heap: magnitude

*How much, and when* — on the unmodified Release binary, because the answer must
describe the program that actually runs.

```bash
cmake -B build && cmake --build build -j"$(nproc)"
# Instances are downloaded, not tracked. `--subset smoke` fetches 11 named
# instances and 50v-10 is NOT one of them, so either take the full roster
# (~546 MiB) or point --inst-dir at a checkout that already has it:
#     .venv/bin/python benchmarks/instances/mipfeas/download.py --subset full
/usr/bin/time -v ./build/cbls_mipfeas --instance 50v-10 \
    --inst-dir benchmarks/instances/mipfeas --out-dir /tmp/prof --budget 5 \
    2>&1 | grep -E 'Maximum resident|Elapsed'
```

`Maximum resident set size (kbytes)` is the process high-water mark. To see
*when* it is reached rather than only how high, sample the live process:

```bash
./build/cbls_mipfeas ... & pid=$!
while kill -0 $pid 2>/dev/null; do
    awk '/VmHWM/{hwm=$2} /VmRSS/{rss=$2}
         END{printf "%s hwm=%s rss=%s kB\n", strftime("%T"), hwm, rss}' \
        /proc/$pid/status
    sleep 0.2
done
```

The MIPfeas runner already records `peak_rss_kib` in every result JSON, which is
the right input for sizing `--jobs` on a benchmark run. Neither figure attributes
anything to a call site — that is the next section's job.

## Heap: attribution

jemalloc's heap profiler, preloaded over the Release binary. No rebuild, no
root, no `valgrind`.

```bash
mkdir -p /tmp/heapprof && cd /tmp/heapprof
MALLOC_CONF=prof:true,prof_gdump:true,prof_final:true,prof_prefix:jeheap \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    /path/to/build-profile/cbls_mipfeas --instance 50v-10 \
    --inst-dir /path/to/benchmarks/instances/mipfeas --out-dir . --budget 5

# The PEAK dump is the highest-numbered gdump -- sort -V, not plain sort.
ls jeheap.*.u*.heap | sort -V | tail -1
```

`prof_gdump:true` writes a dump every time the process reaches a new
**mapped-virtual-memory** high-water mark, so the highest-numbered `.u*.heap`
file is the peak of what jemalloc had mapped. That is the peak wanted here, with
one limit worth knowing: jemalloc does not unmap on free, so a later live-bytes
peak that fits inside already-mapped space triggers no dump. Take the magnitude
from `/usr/bin/time -v` and the attribution from the dumps. This is the setting that matters and the easy one to get wrong:
`prof_final` alone dumps what is *live at exit*, which by construction misses
every transient that was freed before then — i.e. exactly the shape the MIPfeas
reader question has. Verified on a synthetic two-phase program (a 300 MB
transient that is freed, then a 40 MB live footprint): the final dump attributed
36 MB to the surviving phase and the peak gdump attributed 268 MB to the freed
one.

Each record is an `@ 0x... 0x...` stack — innermost frame first — *followed* by
its own `t*: <count>: <bytes>` line; the `t*` line before the first `@` is the
whole-process total, not a record. Byte totals are *sampled* estimates
(one sample per 512 KB by default, `lg_prof_sample:19`); the 268 MB above is the
sampler's estimate of 300 MB. Lower `lg_prof_sample` for small allocations, at
the cost of overhead.

Caveats, all load-bearing:

- **`LD_PRELOAD` replaces the allocator.** Never quote RSS from a jemalloc run
  as the program's memory use — take magnitude from the plain Release binary
  ([above](#heap-magnitude)) and attribution from here.
- **Never combine this with an ASan build.** Two allocators, one process.
- `prof_gdump` can write hundreds of files on a long run (318 for the toy
  above). Always run in a scratch directory, keep the peak dump, delete the rest.
- Dumps land relative to the *current* directory, hence the `cd`.

This pipeline is verified end to end on a real binary, not only on a toy. At
`3ef721f`, `cbls_mipfeas --instance 50v-10 --budget 5` under the preload wrote
83 gdumps, and the largest stack in the last one symbolised to

```
cbls::ExprNode::ExprNode(ExprNode&&)      include/cbls/dag.h:78
cbls::Model::constant(double)             src/model.cpp:115
cbls::mps_to_model(...)                   src/io/mps_to_model.cpp:232
```

— i.e. call-site attribution of the kind issue #122 point 3 asks for, on an
instance small enough to run anywhere — `/usr/bin/time -v` on the plain
Release binary puts its whole-process peak at 6.3 MB.

### Symbolising a heap dump

With `jeprof` (after `sudo apt install libjemalloc-dev`) — the readable path:

```bash
jeprof --text /path/to/build-profile/cbls_mipfeas jeheap.<N>.u<N>.heap | head -20
```

Without it, `addr2line` and the dump's own `MAPPED_LIBRARIES` section do the
job. The binary is PIE, so subtract its load base before resolving:

```bash
BIN=/path/to/build-profile/cbls_mipfeas
DUMP=$(ls jeheap.*.u*.heap | sort -V | tail -1)
BASE=$(grep -m1 " r--p 00000000 .* $BIN\$" "$DUMP" | cut -d- -f1)
[ -n "$BASE" ] || { echo "no mapping for $BIN in $DUMP"; exit 1; }
# Records are NOT ordered by size, so pick the stack with the most bytes rather
# than the first one. Each `@ 0x...` line is followed by its own `t*:` totals.
STACK=$(awk '/^@/{s=$0}
             /^  t\*:/{split($0,f,": "); b=f[3]+0; if (s && b>max){max=b; best=s}}
             END{print best}' "$DUMP")
for a in $(echo "$STACK" | tr ' ' '\n' | grep '^0x'); do
    addr2line -f -C -i -e "$BIN" "$(printf '0x%x' $((a - 0x$BASE)))"
done
```

`-i` matters at `-O3`: without it a frame resolves to whichever function the
optimizer inlined into that address, which is routinely a `std::vector` internal
rather than the caller you want.

Frames inside libc or jemalloc itself resolve to `??` — expected, and harmless:
the frames worth reading are ours. This needs `-DCBLS_PROFILE=ON` for the
symbols; a plain Release binary still yields function names from the symbol
table but no file or line, and the frame pointers keep the stacks walkable.

## CPU

### perf (blocked here by a sysctl)

`perf` is installed, and on this machine it cannot record anything:

```
Failure to open event 'cpu/cycles/Pu' on PMU 'cpu'
Access to performance monitoring and observability operations is limited.
perf_event_paranoid setting is 4
```

`perf record` needs `kernel.perf_event_paranoid <= 2` for user-space profiling
(`<= 1` to also profile kernel code, with `kernel.kptr_restrict=0` to symbolise
it; `<= 0` for CPU-wide events) — the ladder `perf` itself printed above. At `>= 3` — this machine ships `4` — unprivileged
`perf_event_open` is refused outright, software events included, so there is no
event selection that works around it. The fix needs root, once per boot:

```bash
sudo sysctl -w kernel.perf_event_paranoid=2      # persist in /etc/sysctl.conf
```

If root is not available, use [gprof](#gprof-the-no-privilege-fallback) instead;
`sudo perf record` also works but writes a root-owned `perf.data`.

With the sysctl set, on the profiling build:

```bash
cmake -B build-profile -DCBLS_PROFILE=ON && cmake --build build-profile -j"$(nproc)"
perf record -g --call-graph=fp -F 999 -o /tmp/perf.data -- \
    ./build-profile/cbls_mipfeas --instance 50v-10 \
    --inst-dir benchmarks/instances/mipfeas --out-dir /tmp/prof --budget 60
perf report -i /tmp/perf.data --stdio --sort=symbol | head -40
```

`--call-graph=fp` is why `CBLS_PROFILE` re-enables frame pointers; without them
use `--call-graph=dwarf` (bigger files, slower). Profile at a *realistic*
budget: the hot loops here are the ones under a wall-clock deadline, and a
one-second run profiles model building instead.

### gprof: the no-privilege fallback

Instrumentation rather than sampling, entirely in-process, no kernel
involvement. `-pg` is passed as raw flags rather than as a third CMake option —
it changes codegen and needs to reach the linker too:

```bash
cmake -B build-gprof -DCMAKE_CXX_FLAGS="-pg -g -fno-omit-frame-pointer" \
      -DCMAKE_EXE_LINKER_FLAGS="-pg"
cmake --build build-gprof -j"$(nproc)"
cd /tmp/prof && /path/to/build-gprof/cbls_mipfeas ... # writes ./gmon.out
gprof /path/to/build-gprof/cbls_mipfeas gmon.out | head -40
```

Smoke-checked at `8f5f1cb` on `50v-10` with a 5s budget: `gmon.out` appears in
the working directory and the flat profile is led by
`cbls::delta_evaluate` (46%), `cbls::evaluate` (13%) and
`cbls::compute_partial` (9%) — the expected hot loops. The second entry,
`std::vector<unsigned char>::_M_fill_assign` at 25% over 521M calls, is the
distortion in person: `-pg` blocks the inlining that Release relies on and then
charges the call overhead to the callee. Directional, not a wall-clock model.

Read the numbers knowing what they are:

- **Single-threaded only.** glibc's `gmon` records the main thread. The MIPfeas
  runner is single-threaded already and has no `--threads` flag (it exits 2 on an
  unknown argument); under `cbls_cli` keep its default `--threads 1` or the
  profile is a fiction.
- `gmon.out` lands in the *current* directory and is overwritten per run.
- Instrumentation inhibits inlining decisions and adds per-call overhead, so
  attribution is directional, not a wall-clock model.

### callgrind

Exact instruction counts and a clean call graph, at roughly **50x slowdown** —
usable only on a small instance or a heavily reduced budget. Pointing it at a
600s solve and concluding the engine is slow is a mistake this section exists to
prevent. Not installed: `sudo apt install valgrind`, then

```bash
valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out ./build-profile/cbls_cli ...
callgrind_annotate /tmp/cg.out | head -40
```

`heaptrack` (`sudo apt install heaptrack`) is the same story for the heap: a far
nicer allocation-site tree than raw jemalloc dumps, at a cost the preload path
does not have.

## Sanitizers

```bash
cmake -B build-asan -DCBLS_SANITIZE=address,undefined,float-cast-overflow
cmake --build build-asan -j4
UBSAN_OPTIONS=print_stacktrace=1 ctest --test-dir build-asan -LE slow --output-on-failure -j4
```

Spell out **`float-cast-overflow`**. GCC 15's `-fsanitize=undefined` does *not*
include it — verified: `(long long)1e30` returns `LLONG_MIN` silently under
`undefined` alone, and reports `1e+30 is outside the range of representable
values of type 'long long int'` once the check is named. That conversion is the
exact class of bug that motivated wiring sanitizers up in the first place, so
omitting the flag would give a sanitizer build that misses its own founding
example.

- UBSan **prints and continues** by default, so a green `ctest` is not proof of
  a clean run — grep the output for `runtime error`. Add
  `UBSAN_OPTIONS=halt_on_error=1` to turn findings into failures.
- Use `-j4` and `-LE slow`, not `-j$(nproc)`: ASan multiplies both memory and
  runtime, and the `[slow]` CHPED solves under ASan are how you exhaust a shared
  box.
- Leave `CBLS_BUILD_PYTHON` off. Importing an ASan-instrumented extension module
  into a clean interpreter needs `LD_PRELOAD` of `libasan`, which is its own
  exercise.
- ASan's redzones, quarantine and shadow memory make this build's RSS and
  timings meaningless. It answers "is this correct", never "is this fast".

### Verified state

At `f511b8d`, `-DCBLS_SANITIZE=address,undefined,float-cast-overflow` puts the
flag on every translation unit — 155 of 155, `grep -c fsanitize
build-asan/compile_commands.json` against `grep -c '"file"'` on the same file — and links both `libasan.so.8` and
`libubsan.so.1`. `ctest --test-dir build-asan -LE slow -j3` (`-j3` rather than the recipe's
`-j4`, because the box was shared) was **302/302 green in 84.5s** (the fast set was 302 tests at that commit; it is 306 now — this is a record of that run, not a current count), with zero `runtime error` lines, zero AddressSanitizer reports and
no leaks — LeakSanitizer is on by default and would have said otherwise. Check
the flags reached the compiler before trusting a green run: a mis-spelled
`CBLS_SANITIZE` value fails at compile time, but an option that silently did not
apply looks exactly like a clean suite.

The 6 `[slow]` CHPED/UC-CHPED solves were **not** run under ASan — they are the
suite's heaviest, and ASan multiplies their footprint. So the claim is "the fast
set is clean", not "the suite is clean". The value of this build is on the next
change anyway, not on this one.

## Recording what a profile found

Per CLAUDE.md's rule that measurements go stale silently:

- **Name the engine commit** next to any number you publish. `git rev-parse
  --short HEAD` at the time of the run, in the prose or in a `commit_sha`
  column.
- Findings go in prose — this file, a benchmark README, or the issue — not as a
  committed table nobody reads. Raw `perf.data`, `gmon.out` and `.heap` files
  stay in scratch; they are large, machine-specific, and unreadable without the
  exact binary that produced them.
- Say which build type and which flags produced the profile.
- **A `build-profile` wall-clock is not a benchmark number.**
  `-fno-omit-frame-pointer` costs throughput and the binary is otherwise
  indistinguishable from Release in a results table. Published timings come from
  a plain `build/`.

## Open question: the MIPfeas reader peak

Issue #122 point 3: on the largest instances in the MIPLIB-based feasibility
roster, reading the file alone is reported to reach the whole process's peak RSS
before any model is built, and nothing in the tree says why. The issue puts that
peak at ~3.2 GB; **that figure has no source in this tree and is not reproduced
here** — the only 3.2 GB the repository records is CP-SAT's peak on
`neos-5114902-kasavu` in `benchmarks/instances/mipfeas/README.md`, where CBLS
peaked at 1.2 GB. Treat the magnitude as unestablished until the capture below
runs; the recipe settles both the number and its owner. **The profile has not
been taken yet** — it is tracked as #127, deliberately split out of #122 because
a capture is a quiet-box measurement run and has to be scheduled against the
other pending benchmark runs rather than run beside them. Replace this section
with the finding, and its commit, when #127 executes. All three outcomes are
publishable: the reader owns the peak, something else does, or there is no such
peak for this engine and the ~3.2 GB was the reference solver's number.

The capture, on an otherwise idle machine — check `uptime` first, and do not run
it beside a build:

```bash
# 1. Profiling build (Release + symbols + frame pointers).
cmake -B build-profile -DCBLS_PROFILE=ON && cmake --build build-profile -j"$(nproc)"

REPO=$(git rev-parse --show-toplevel)

# 2. Magnitude, on the plain gated Release binary and the unmodified allocator.
#    square47 carries the roster's most nonzeros (27.4M) and spends ~100-150s in
#    read + build before its first iteration, so a short budget still covers the
#    whole reader phase.
mkdir -p /tmp/prof
/usr/bin/time -v "$REPO"/build/cbls_mipfeas --instance square47 \
    --inst-dir "$REPO"/benchmarks/instances/mipfeas --out-dir /tmp/prof --budget 10 \
    2>&1 | grep -E 'Maximum resident|Elapsed'

# 3. Attribution, jemalloc preload over the profiling build, peak dump.
mkdir -p /tmp/heapprof && cd /tmp/heapprof
MALLOC_CONF=prof:true,prof_gdump:true,prof_prefix:jeheap \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    "$REPO"/build-profile/cbls_mipfeas --instance square47 \
    --inst-dir "$REPO"/benchmarks/instances/mipfeas --out-dir . --budget 10
ls jeheap.*.u*.heap | sort -V | tail -1     # the peak
```

Then symbolise it ([above](#symbolising-a-heap-dump)) and write the top few
sites down here with the commit they were measured at. `neos-5114902-kasavu`
(710k columns, 4.9M nonzeros, ~1.2 GB under CBLS) is the cheaper second data
point if square47 is too heavy for the machine at hand.
