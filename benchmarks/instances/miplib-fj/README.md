# MIPLIB-FJ benchmark subset

> **Retired — superseded by `benchmarks/instances/mipfeas/` (issue #90).**
>
> This benchmark's roster (11 self-selected instances), metric (`gap_to_opt%`)
> and absence of any baseline solver were all replaced: MIPfeas runs the full
> 233-instance MIPLIB 2017 benchmark-set roster, scores the Primal Integral, and
> compares against CP-SAT at its default parameters (#284). The runner and its CMake
> target are gone; `tests/test_io_mps.cpp` still covers the MPS reader and
> adapter, which MIPfeas depends on.
>
> The vendored instances are also a live test dependency, because the MIPfeas
> roster's own instances are not vendored (~546 MiB): `tests/test_mipfeas.cpp`
> reads `pk1.mps.gz` and `miplib2017-v22.solu` from this directory for its
> end-to-end solve, and `tests/test_throughput.cpp` and
> `tests/python/test_verify_solution.py` read from it too. Don't prune them.
>
> The data here is kept, not deleted: `comparison.csv` is the pre-port-SA vs
> post-port-ViolationLS record, which answers a different question (did the
> ViolationLS port change anything?) and is not reproducible from the current
> tree. The vendored instances back it up.

The 11 instances are a self-selected MIPLIB 2017 subset in the spirit of
Luteberget & Sartor 2023 (*Feasibility Jump*), not its roster: `enlight_hard`,
`gen-ip054`, `gen-ip002`, `markshare1`, `markshare2`, `flugpl`, `pk1`, `mas76`,
`neos5`, `mad` and `binkar10_1`, with reference values from
`miplib2017-v22.solu`. All are vendored, so a fresh clone needs no download.
The original description is in git history (`benchmarks/instances/miplib-fj/README.md`
before #360); so is the downloader that fetched them (`download.py`, removed by
#373).

## Layout

```
benchmarks/instances/miplib-fj/
  README.md                # this file
  comparison.csv           # pre-port SA and post-port ViolationLS record (not reproducible)
  miplib2017-v22.solu      # vendored
  *.mps.gz                 # the 11 vendored instances (tracked)
```
