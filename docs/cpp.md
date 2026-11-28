# C++ reference

The supported part of the C++ library, in namespace `cbls`: what
`<cbls/cbls.h>` brings in, plus `<cbls/io_nl.h>` for AMPL `.nl` models (README,
[Use from C++](index.md#use-from-c)). Link `cbls::cbls`. Build a `Model` with
its variable and constraint methods, or load one with `load_model`,
`mps_to_model(read_mps(path))` or `nl_to_model(read_nl(path))`, then call
`solve` for one search or `ParallelSearch::solve` for the portfolio. Other
installed headers serve this repository's runners and tests and are not
documented here.

## Model

```{doxygenclass} cbls::Model
```

```{doxygenclass} cbls::Expr
```

```{doxygenstruct} cbls::ListPartition
```

```{doxygenenum} cbls::Cover
```

### Expression functions

```{doxygenfunction} cbls::sin
```

```{doxygenfunction} cbls::cos
```

```{doxygenfunction} cbls::tan
```

```{doxygenfunction} cbls::exp
```

```{doxygenfunction} cbls::log
```

```{doxygenfunction} cbls::sqrt
```

```{doxygenfunction} cbls::abs
```

```{doxygenfunction} cbls::pow
```

```{doxygenfunction} cbls::min
```

```{doxygenfunction} cbls::max
```

```{doxygenfunction} cbls::if_then_else
```

```{doxygenfunction} cbls::ceil
```

```{doxygenfunction} cbls::floor
```

```{doxygenfunction} cbls::round
```

```{doxygenfunction} cbls::element(const std::vector<double> &table, const Expr &index)
```

```{doxygenfunction} cbls::element(const std::vector<std::vector<double>> &table, const Expr &row, const Expr &col)
```

## Solving

```{doxygenfunction} cbls::solve
```

```{doxygenstruct} cbls::SearchConfig
```

```{doxygenstruct} cbls::SearchResult
```

```{doxygenenum} cbls::TerminationReason
```

```{doxygenstruct} cbls::WorkerFailure
```

```{doxygenclass} cbls::SolveCallback
```

```{doxygenstruct} cbls::SolveProgress
```

```{doxygenclass} cbls::StopToken
```

```{doxygenclass} cbls::StopRef
```

## Portfolio

```{doxygenclass} cbls::ParallelSearch
```

```{doxygenstruct} cbls::ParallelConfig
```

## Model files

```{doxygenfunction} cbls::load_model(const std::string &path)
```

```{doxygenfunction} cbls::load_model(std::istream &input)
```

```{doxygenfunction} cbls::save_model(const Model &model, const std::string &path)
```

```{doxygenfunction} cbls::save_model(const Model &model, std::ostream &out)
```

### MPS

```{doxygenfunction} cbls::read_mps
```

```{doxygenfunction} cbls::mps_to_model
```

```{doxygenstruct} cbls::MpsProblem
```

```{doxygenstruct} cbls::MpsToModelOptions
```

```{doxygenstruct} cbls::MpsToModelResult
```

### AMPL `.nl`

```{doxygenfunction} cbls::read_nl
```

```{doxygenfunction} cbls::parse_nl
```

```{doxygenfunction} cbls::nl_to_model
```

```{doxygenstruct} cbls::NlProblem
```

```{doxygenstruct} cbls::NlToModelOptions
```

```{doxygenstruct} cbls::NlToModelResult
```

## Checking a solution

```{doxygenfunction} cbls::verify_model
```

```{doxygenstruct} cbls::VerifyResult
```

```{doxygenstruct} cbls::VerifyError
```
