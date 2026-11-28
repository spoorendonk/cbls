# MIP: a feasible point of a MIPLIB `.mps.gz` file

This example reads a MIPLIB instance, searches for a feasible point, prints it
and checks it. The instance is
[`neos5`](https://github.com/spoorendonk/cbls/blob/main/benchmarks/instances/miplib-fj/neos5.mps.gz)
from MIPLIB 2017: 63 columns, 63 rows and a minimised objective whose optimum
is 15. The repository vendors the file in the retired `miplib-fj` set.

## Run it

From a checkout, with the module installed (`pip install .`; see
[Use from Python](https://github.com/spoorendonk/cbls#use-from-python)):

```sh
python examples/mip.py benchmarks/instances/miplib-fj/neos5.mps.gz
```

In C++, build the example and run it:

```sh
cmake -B build
cmake --build build --target cbls_example_mip
build/cbls_example_mip benchmarks/instances/miplib-fj/neos5.mps.gz
```

Both solve for 2 s on one thread with seed 42. The Python script prints:

```text
neos5: 63 columns, 63 rows
feasible  = True
objective = 15.000000
nonzero   = 17 of 63: d1=1 d8=1 d10=1 d14=1 d15=1 d18=1 d20=1 d27=1 ...
verify: PASS
```

The C++ program prints `true` in place of `True` and only the count of nonzero
columns. The solve stops on wall-clock time, so the point can differ from run
to run, or on a slower machine; the tests check that it is feasible and
verifies. The test suite runs both commands against the build tree:
`example_mip` in ctest, and `tests/python/test_examples.py`.

## From the command line

The `cbls` command solves the file with no script, here with the same 2 s, one
thread and seed 42:

```sh
cbls benchmarks/instances/miplib-fj/neos5.mps.gz --time-limit 2 --threads 1 --seed 42 --quiet
```

```text
Status:     feasible
Objective:  15.000000
Time:       2.00s (117250 iterations, stopped on time_limit)
Solution:
  d1 = 1.00
  d2 = 0.00
  ...
  d63 = 0.00
```

It prints a blank line first, and the solution lists all 63 columns by name,
zeros too, at two decimals. `--quiet` prints the result only. Without it, a
header and progress lines come first: a line for each new best point, marked
`*`, and one about every second between them. `--format jsonl` prints the
result as one JSON line, with the values at full precision and the columns
sorted as text (`d1`, `d10`, `d11`, ...):

```sh
cbls benchmarks/instances/miplib-fj/neos5.mps.gz --time-limit 2 --threads 1 --seed 42 --format jsonl --quiet
```

```text
{"event":"result","feasible":true,"iterations":117641,"objective":15.0,"solution":{"d1":1.0,"d10":1.0,"d11":0.0, ... ,"d9":0.0},"status":"feasible","termination":"time_limit","time":2.0}
```

The iteration count and the point depend on the machine. The command exits
with 0 when it found a feasible point, else 1. It does not run the checks of
[Verifying it](#verifying-it); the scripts do.

This is the `cbls` a pip install provides, also run as `python -m cbls`. It
reads `.cbls`, MPS (`.mps`, `.mps.gz`) and `.nl` files. The `cbls` binary of a
CMake build or install (`build/cbls`, `<prefix>/bin/cbls`) reads `.cbls` files
only and stops on this file with `Error loading model`. With both installed,
the first on `PATH` runs, and `cbls --help` lists the formats it reads (see
[Command line](https://github.com/spoorendonk/cbls#command-line)).
`tests/python/test_examples.py` runs the two `cbls` commands above as
`python -m cbls`.

## The code

```{literalinclude} ../../examples/mip.py
:language: python
```

```{literalinclude} ../../examples/mip.cpp
:language: cpp
```

## Reading the result

- **`mps_to_model(read_mps(path))`** builds a model with one variable per MPS
  column, and one linear constraint per row (two for a range row). `read_mps`
  reads `.mps` and `.mps.gz`. cbls variables need finite bounds: an infinite
  one is derived from the rows where it can be, else clamped to 1e9, which can
  cut off solutions. `n_clamped_columns` counts the columns clamped.
- **`objective`** is the MPS objective, minimised. `mps_to_model` rejects
  `OBJSENSE MAX` with an error; negate the costs to maximise.
- **`original_values()`** gives the point as one value per MPS column, in the
  file's order. The script prints the nonzero ones by column name.
- **`feasible`** means every row and bound holds within the engine's tolerance
  of 1e-6. It says nothing about optimality: here the search happens to reach
  the optimum, 15, but cbls cannot prove it.

## Verifying it

`cbls::verify_model` re-checks the point the model is left at:
- each variable is within its bounds, and each integer variable is integral;
- each constraint's violation is at most 1e-6, and a domain row allows none;
- the stored node values match a fresh evaluation.

It then prints `verify: PASS`, or each failure. The Python `verify` function
makes the first two checks through the public `Model` accessors.

## Where cbls stands

For a model whose faithful form is a MILP, a MIP solver is the right tool. On
the MIPfeas roster, 211 held-out MIPLIB instances for 60 s, CP-SAT at its
defaults is ahead of cbls:
- it finds a feasible point on **175**, against cbls's **140**;
- its primal integral (shifted geometric mean, lower is better) is **0.168**,
  against cbls's **0.390**.

CP-SAT's `fj` and `ls` workers run the same algorithm as cbls, but at its
defaults CP-SAT also brings presolve and its full portfolio, so the table
compares the tools as shipped. It is no case for cbls on MILPs (#255). The
[MIPfeas table](../results.md#mipfeas) has the full numbers and run record.
cbls's case is the terms a MILP cannot state: see the
[MINLPLib example](minlplib.md).
