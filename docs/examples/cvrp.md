# CVRP: routes as `List` variables

This example models a capacitated vehicle routing problem (CVRP) with `List`
variables, solves it and reads the routes. A fleet of vehicles leaves one
depot; each customer has a demand and is visited by exactly one vehicle, no
vehicle carries more than its capacity, and the total route length is
minimised. The instance is
[`X-n115-k10`](https://github.com/spoorendonk/cbls/blob/main/benchmarks/instances/cvrp/X/X-n115-k10.vrp)
from CVRPLIB's X set (Uchoa et al. 2017): 114 customers, capacity 169, at least
10 vehicles, and a best-known cost of 12747
([its `.sol` file](https://github.com/spoorendonk/cbls/blob/main/benchmarks/instances/cvrp/X/X-n115-k10.sol)).
It is the instance of the figure on the [home page](../index.md).

## Run it

From a checkout, with the module installed (`pip install .`; see
[Use from Python](https://github.com/spoorendonk/cbls#use-from-python)):

```sh
python examples/cvrp.py benchmarks/instances/cvrp/X/X-n115-k10.vrp
```

In C++, build the example and run it:

```sh
cmake -B build
cmake --build build --target cbls_example_cvrp
build/cbls_example_cvrp benchmarks/instances/cvrp/X/X-n115-k10.vrp
```

Both solve for 5 s on one thread with seed 1. The Python script prints:

```text
X-n115-k10: 114 customers, capacity 169, fleet 13
feasible  = True
objective = 13619
Route #1: 50 52 21 82 88 9 7 63 64
Route #2: 15 32 17 106 3 93 18 42 59
Route #3: 72 102 24 71 37 1 98 34 26 46
Route #4: 95 107 35 99 49 109 47 79 87 73 86 60 48 6 108 19 51 25
Route #5: 29 53 78 54 62 94 20 101 4 92 57 100 28
Route #6: 10
Route #7: 104 41 91 2
Route #8: 70 103 61 33 81 65 23 22 96 45 12 113 58 77 38 97 80 55 76 69
Route #9: 36 11 31 56 13
Route #10: 111 110 112 105 30 114 44 27 84 75 83 8 14 68 43 40 67 39 90 74 85
Route #11: 16 5 66 89
verify: PASS
```

The C++ program prints `true` in place of `True`. The routes above are 6.8 %
above the best-known cost, with 11 vehicles where the best-known solution uses
10. The solve stops on wall-clock time, so the routes can differ from run to
run, or on a slower machine; the tests check that they are feasible and
verify. The test suite runs both commands against the build tree:
`example_cvrp` in ctest, and `tests/python/test_examples.py`.

## The code

```{literalinclude} ../../examples/cvrp.py
:language: python
```

```{literalinclude} ../../examples/cvrp.cpp
:language: cpp
```

## The model

- **One `List` per vehicle**, `list_var(n, 0, n, ...)`: an ordered sequence of
  distinct customers `0..n-1`, of length 0 to `n`. The depot is not an
  element. The fleet is the instance's minimum `k` plus 3, a size the search
  may leave partly unused: it is not a vehicle count.
- **`add_list_partition(routes, Cover.Exact)`** puts every customer on exactly
  one route. It is not a penalised constraint but an invariant: the search
  starts from a random complete partition (the Lists' `ListInit` is ignored)
  and its moves, which relocate, swap and reverse customers within and between
  routes, keep it complete.
- **Capacity:** per route, `lambda_sum` (in Python `lambda_table_sum`, from a
  NumPy array) adds the demands of its elements, and a constraint holds that
  sum at most the capacity.
- **Length:** per route, `pair_lambda_sum` (in Python `pair_table_sum`, from a
  distance matrix) adds the distance between each pair of consecutive
  customers, and its head and tail terms add the legs from and back to the
  depot. An empty route costs 0. The objective minimises the sum over routes.
- **Search:** `nearest_neighbours` lists each customer's 20 nearest, and the
  routing moves (relocate, swap, 2-opt, 2-opt*) pair a customer with one of
  them, plus a move into an empty route. LNS with a destroy fraction of 0.2
  diversifies the search.

Distances are TSPLIB's `EUC_2D`, the Euclidean distance rounded to the nearest
integer, the metric CVRPLIB's best-known costs are stated in.

## Reading the result

- **`feasible`** means every route is within capacity; the partition holds
  throughout. It says nothing about optimality: cbls has no lower bound.
- **`objective`** is the total route length.
- **The routes** are the Lists' elements in visiting order. In Python they are
  in `result.elements`, by variable id; a List handle `h` is variable
  `-(h + 1)`. In C++, `restore_state(result.best_state)` loads the best point
  into the model, and each List's `elements` are read from it. The scripts
  print each non-empty route as node numbers, customer element `e` as `e + 1`,
  in the numbering of CVRPLIB's `.sol` files.

## Verifying it

Each script's `verify` re-checks the routes it printed, independently of the
model: every customer is on exactly one route, each route's load is within the
capacity, and the length of the routes, recomputed from the distance matrix,
equals the reported objective. It then prints `verify: PASS`, or each failure.

## Where cbls stands

This is a worked example of `List` variables, not a benchmark result: **no
`List` result is published**. CVRP is the model `List` will be measured on,
and until that campaign reports, there is no evidence on how cbls compares
with a routing solver here (README,
[Known limitations](https://github.com/spoorendonk/cbls#known-limitations)).
For a CVRP itself, a dedicated routing solver is the tool to reach for. The
[benchmark runner](https://github.com/spoorendonk/cbls/tree/main/benchmarks/cvrp)
builds the same model in C++.
