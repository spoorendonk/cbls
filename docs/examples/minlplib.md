# MINLPLib: a nonlinear model from an `.nl` file

This example solves a MINLPLib instance, prints the point found and checks it.
The instance is [`st_e38`](https://github.com/spoorendonk/cbls/blob/main/benchmarks/instances/minlplib/st_e38.nl),
the pressure-vessel design problem. It has four variables, two of them
integer, and minimises a cost that is non-convex in the vessel's radius and
length. MINLPLib's best known objective is 7197.727149, and it is proved
optimal. The repository vendors the file.

## Run it

From a checkout, with the module installed (`pip install .`; see
[Use from Python](https://github.com/spoorendonk/cbls#use-from-python)):

```sh
python examples/minlplib.py benchmarks/instances/minlplib/st_e38.nl
```

In C++, build the example and run it:

```sh
cmake -B build
cmake --build build --target cbls_example_minlplib
build/cbls_example_minlplib benchmarks/instances/minlplib/st_e38.nl
```

Both solve for 2 s on one thread with seed 42, and print:

```text
feasible  = True
objective = 7197.727149
  x0 = 58.290155
  x1 = 43.692654
  x2 = 18.000000
  x3 = 10.000000
verify: PASS
```

The C++ program prints `true` in place of `True`. The solve stops on
wall-clock time, so the digits can differ on a slower machine. The test suite
runs both commands against the build tree: `example_minlplib` in ctest, and
`tests/python/test_examples.py`, which also checks the objective against the
best known.

## From the command line

The `cbls` command solves the file with no script, here with the same 2 s, one
thread and seed 42:

```sh
cbls benchmarks/instances/minlplib/st_e38.nl --time-limit 2 --threads 1 --seed 42 --quiet
```

```text
Status:     feasible
Objective:  7197.727149
Time:       2.00s (194466 iterations, stopped on time_limit)
Solution:
  x0 = 58.29
  x1 = 43.69
  x2 = 18.00
  x3 = 10.00
```

It prints a blank line first. `--quiet` prints the result only. Without it, a
header and progress lines come first: a line for each new best point, marked
`*`, and one about every second between them. `--format jsonl` prints the
result as one JSON line, with the values at full precision:

```sh
cbls benchmarks/instances/minlplib/st_e38.nl --time-limit 2 --threads 1 --seed 42 --format jsonl --quiet
```

```json
{"event":"result","feasible":true,"iterations":194360,"objective":7197.727148524341,"solution":{"x0":58.2901554404145,"x1":43.692654445213535,"x2":18.0,"x3":10.0},"status":"feasible","termination":"time_limit","time":2.0}
```

The iteration count depends on the machine, and as above the digits can too.
The command exits with 0 when it found a feasible point, else 1. It does not
run the checks of [Verifying it](#verifying-it); the scripts do.

This is the `cbls` a pip install provides, also run as `python -m cbls`. It
reads `.cbls`, MPS and `.nl` files. The `cbls` binary of a CMake build or
install (`build/cbls`, `<prefix>/bin/cbls`) reads `.cbls` files only and stops
on this file with `Error loading model`. With both installed, the first on
`PATH` runs, and `cbls --help` lists the formats it reads (see
[Command line](https://github.com/spoorendonk/cbls#command-line)).
`tests/python/test_examples.py` runs the two `cbls` commands above as
`python -m cbls`.

## The code

```{literalinclude} ../../examples/minlplib.py
:language: python
```

```{literalinclude} ../../examples/minlplib.cpp
:language: cpp
```

## Reading the result

- **`nl_to_model(read_nl(path))`** builds a closed model, with one variable per
  `.nl` column. If the file uses an operator cbls does not model, `supported`
  is false and `skipped_reasons` says why. Such a model must not be solved.
- **`objective`** is in the model's own sense, as the CLI prints it. The engine
  minimises, so for a maximisation model C++'s `SearchResult::objective` (in
  Python `result.search.objective`) is the negation; the C++ program flips it
  back with `is_maximizing()`.
- **`original_values()`** gives the point as one value per `.nl` column, in the
  file's order. `x2` and `x3` are the integer variables.
- **`feasible`** means every constraint holds within the engine's tolerance of
  1e-6.

## Verifying it

The engine's own `feasible` flag is not the whole check. `cbls::verify_model`
re-checks the point the model is left at:
- each variable is within its bounds, and each integer variable is integral;
- each constraint's violation is at most 1e-6, and a domain row allows none;
- the stored node values match a fresh evaluation.

It then prints `verify: PASS`, or each failure. The Python `verify` function
makes the first two checks through the public `Model` accessors.

A second check is against the instance's reference value. 7197.727149 is
MINLPLib's best known and proved optimal (`bounds.csv` in the instance
directory). A point feasible within 1e-6 can land a hair below it, as SCIP's
7197.727140 does; an objective clearly below would mean a verification bug,
not a record.

## Where cbls stands

cbls cannot prove optimality. SCIP 10.0.2 reaches the same objective on
`st_e38` and proves it optimal in 0.07 s (the instance's row in
[`scip_baseline.csv`](https://github.com/spoorendonk/cbls/blob/main/benchmarks/instances/minlplib/scip_baseline.csv)). Over Mittelmann's nontrivial MINLPLib roster, cbls
is behind SCIP:
- its GM(gap+1) is **1.676 times SCIP's**;
- it finds a feasible point on 169 instances, against SCIP's 187 of 195.

That run gave cbls 16 threads and SCIP one, both for 60 s (#256). The
[MINLPLib table](../results.md#minlplib) has the full numbers and run record,
and it also lists the open instances where cbls improves on MINLPLib's best
known.
