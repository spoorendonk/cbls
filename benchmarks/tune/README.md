# Tuning studies

One `xk tune` study file per benchmark class (#261): the parameter space, the
reference configuration, the instance sets and the cost. Each trial is one
`benchmarks/common/tune_trial.py` run (its `cost.json`, see
`benchmarks/common/README.md`).

| File | Issue | Cost |
|---|---|---|
| `cvrp.study.toml` | #263 | `gap_percent` to the pinned BKS |

The plan they follow (shape, params, splits, core-hours) is on #261. Results
go on each class's issue; a setting that is confirmed lands as an engine or
runner default change, never by editing a study file.

## Running one

They need experiment-kit at or after #141 (`[study] threads`). Run them on the
workstation's local queue only, one study at a time:

```sh
export XK_SERVER=local XK_PROJECT=cbls-tune
xk submit tune benchmarks/tune/cvrp.study.toml --trials 24
xk submit tune benchmarks/tune/cvrp.study.toml --stage elites
xk submit tune benchmarks/tune/cvrp.study.toml --stage confirm
xk submit tune benchmarks/tune/cvrp.study.toml --stage fidelity
xk tune importance benchmarks/tune/cvrp.study.toml
```

`rev = "main"` is resolved on the first run and kept. A study is frozen like a
suite: changing its commands, space, cost or screen and confirm instances needs
a `version` bump, which starts a new study.

## Conventions

- **Sets.** `screen` tunes at a short budget, `intermediate` is the same
  instances at a longer one (what `--stage fidelity` compares with the screen),
  `confirm` is held out and never seen while tuning. The wall budget is an
  instance param (`secs`), so one instance can sit in two sets.
- **Reference.** Every param has a `default`; together they render the
  runner's current protocol, which runs as trial 0. Known arms from earlier
  issues are `[[study.enqueue]]`d after it.
- **Flags.** A param's `format` value is shell-quoted, so a flag with no value
  is a one-element `choice` gated on the bool that switches it on (`no_lns`).
  A flag read under several settings of another param gets one param per
  setting, each with its own `when` (`sample_*`).
- **Width.** `threads = 16` on cores 0-15, `exclusive = true`: each run is the
  16-thread `ParallelSearch` portfolio, alone on the machine. The build is the
  same command in every study, so it is built once per commit.
