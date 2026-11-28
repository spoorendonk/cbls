"""The `cbls` console script and `python -m cbls`: src/cli.cpp over the module (#465).

Same options, output and exit codes as the C++ CLI, through its own formatters
and flag parsing (bound in `_cbls_core`). One difference: the model file may
also be MPS (`.mps`, `.mps.gz`) or AMPL `.nl`, chosen by extension.
"""

from __future__ import annotations

import math
import signal
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import _cbls_core as _core
from ._solve import _effective_threads, solve

if TYPE_CHECKING:
    from collections.abc import Sequence

HELP = """\
Usage: cbls [OPTIONS] MODEL

  Constraint-Based Local Search solver. Reads a model file (.cbls JSONL,
  .mps, .mps.gz or AMPL .nl, by extension) and finds optimal or feasible
  variable assignments.

Arguments:
  MODEL                 Path to the model file (.cbls, .mps, .mps.gz, .nl)

Options:
  --time-limit SECS     Maximum solve time in seconds (default: 10.0)
  --seed INT            Random seed for reproducibility (default: 42)
  --lns FRACTION        Enable LNS with destroy fraction, e.g. 0.3
  --lns-interval INT    LNS fires every N diversification kicks (default: 3)
  --intensify           Enable float intensification hook
  --threads N           Number of search threads (default: 0 = one per core)
  --format human|jsonl  Output format (default: human)
  --quiet               Suppress progress, print only final result
  --help                Show this help message
  --version             Show version number
"""

_INT_MIN = -(2**31)
_INT_MAX = 2**31 - 1


class _UsageError(Exception):
    """A malformed command line: `Error: <message>`, exit 1."""


@dataclass
class _Options:
    """src/cli.cpp's CliOptions, with its defaults."""

    model_path: str = ""
    time_limit: float = 10.0
    seed: int = 42
    intensify: bool = False
    lns_fraction: float = 0.0
    config: _core.SearchConfig = field(default_factory=_core.SearchConfig)  # lns_interval
    format: str = "human"
    quiet: bool = False
    threads: int = 0


def _out_of_range(flag: str, text: str) -> _UsageError:
    return _UsageError(f"{flag}: '{text}' is out of range")


def _number(flag: str, text: str) -> float:
    status, value = _core._parse_double(text)
    if status == "ok" and not math.isnan(value):  # nan reads as malformed, as the CLI
        return value
    if status == "out_of_range":
        raise _out_of_range(flag, text)
    raise _UsageError(f"{flag}: '{text}' is not a number")


def _integer(flag: str, text: str, unsigned: bool = False) -> int:
    status, value = _core._parse_uint64(text) if unsigned else _core._parse_int64(text)
    if status == "ok":
        return value
    if status == "out_of_range":
        raise _out_of_range(flag, text)
    raise _UsageError(f"{flag}: '{text}' is not an integer")


def _int(flag: str, text: str) -> int:
    value = _integer(flag, text)
    if not _INT_MIN <= value <= _INT_MAX:
        raise _out_of_range(flag, text)
    return value


def _take_value(opt: _Options, flag: str, text: str) -> None:
    """One `--flag VALUE` option, `flag` one of `_VALUE_FLAGS`."""
    if flag == "--format":
        if text not in ("human", "jsonl"):
            raise _UsageError("--format must be 'human' or 'jsonl'")
        opt.format = text
    elif flag == "--time-limit":
        opt.time_limit = _number(flag, text)
    elif flag == "--seed":
        opt.seed = _integer(flag, text, unsigned=True)
    elif flag == "--lns":
        opt.lns_fraction = _number(flag, text)
    elif flag == "--lns-interval":
        opt.config.lns_interval = _int(flag, text)
    else:
        opt.threads = _int(flag, text)


_VALUE_FLAGS = ("--format", "--time-limit", "--seed", "--lns", "--lns-interval", "--threads")


def _parse_args(argv: Sequence[str]) -> _Options | None:
    """The options, or None when --help or --version has been answered.

    As src/cli.cpp's loop: the last positional is the model, and a value flag
    with nothing after it is an unknown option.
    """
    opt = _Options()
    i = 0
    while i < len(argv):
        arg = argv[i]
        i += 1
        if arg in ("--help", "-h"):
            sys.stdout.write(HELP)
            return None
        if arg == "--version":
            sys.stdout.write(f"cbls {_core.version}\n")
            return None
        if arg == "--intensify":
            opt.intensify = True
        elif arg == "--quiet":
            opt.quiet = True
        elif arg in _VALUE_FLAGS and i < len(argv):
            _take_value(opt, arg, argv[i])
            i += 1
        elif arg.startswith("-"):
            raise _UsageError(f"unknown option '{arg}'")
        else:
            opt.model_path = arg
    return opt


class _LoadError(Exception):
    """A model file that could not be read: `Error loading model: <message>`."""


def _load(path: str) -> _core.Model:
    """The model in `path`, read by its extension; anything else is .cbls, as the CLI.

    Case-sensitive, as read_mps's own `.gz` test: `.MPS.GZ` read as MPS would be
    read as plain text, an empty model reported feasible.
    """
    if path.endswith((".mps", ".mps.gz")):
        return _core.mps_to_model(_core.read_mps(path)).model
    if path.endswith(".nl"):
        converted = _core.nl_to_model(_core.read_nl(path))
        if not converted.supported:
            raise _LoadError("unsupported .nl model: " + "; ".join(converted.skipped_reasons))
        return converted.model
    return _core.load_model(path)


def _run(opt: _Options) -> int:
    if not opt.model_path:
        sys.stderr.write("Error: no model file specified. Use --help for usage.\n")
        return 1
    try:
        model = _load(opt.model_path)
    except Exception as e:
        sys.stderr.write(f"Error loading model: {e}\n")
        return 1

    # The C++ bytes as they are, to the binary stream under sys.stdout.
    sys.stdout.flush()
    formatter: _core.HumanFormatter | _core.JsonlFormatter
    if opt.format == "human":
        formatter = _core.HumanFormatter(sys.stdout.buffer)
    else:
        formatter = _core.JsonlFormatter(sys.stdout.buffer)
    if not opt.quiet:
        formatter.print_header(opt.model_path, model, opt.seed, opt.time_limit)

    # As the CLI: a count below 1 runs the single-threaded search, 0 one per core.
    threads = max(opt.threads, 1) if opt.threads != 0 else 0
    n_threads = _effective_threads(threads)
    try:
        result = solve(
            model,
            opt.time_limit,
            opt.seed,
            threads=threads,
            lns=opt.lns_fraction if opt.lns_fraction > 0.0 else None,
            intensify=opt.intensify,
            callback=None if opt.quiet else formatter,
            config=opt.config,
        )
    except Exception as e:
        if n_threads == 1:
            raise
        sys.stderr.write(f"Error: parallel search failed: {e}\n")
        return 1
    search = result.search
    if n_threads > 1 and search.workers_completed < n_threads:
        sys.stderr.write(
            f"Warning: {search.workers_completed} of {n_threads} portfolio workers completed\n"
        )
        for failure in search.worker_failures:
            sys.stderr.write(f"  worker {failure.worker}: {failure.reason}\n")

    formatter.print_result(search, model)
    return 0 if result.feasible else 1


def _default_signals() -> None:
    """SIGINT and SIGPIPE kill the process, as they do the C++ CLI.

    Python's SIGINT handler only sets a flag, which nothing reads while the
    search runs with the GIL released, so Ctrl-C would wait out the time
    limit; an ignored SIGPIPE turns `| head` into a BrokenPipeError.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if hasattr(signal, "SIGPIPE"):  # POSIX only
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI on `argv`; returns the exit code.

    With no `argv` it is the process (the console script, `python -m cbls`):
    it reads sys.argv and takes the default SIGINT and SIGPIPE handling.
    """
    if argv is None:
        _default_signals()
    args = sys.argv[1:] if argv is None else argv
    try:
        opt = _parse_args(args)
        return 0 if opt is None else _run(opt)
    except Exception as e:  # a _UsageError, or as the CLI's main: no traceback, exit 1
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
