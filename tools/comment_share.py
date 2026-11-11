"""Print the comment share (comment lines / non-blank lines) per source area.

Counts tracked files only (`git ls-files`). C++: `//` and `/* */`; Python:
`#` and docstrings. A line with code and a trailing comment counts as code.
Usage: `.venv/bin/python tools/comment_share.py [--by-file include/cbls]`.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import io
import subprocess
import tokenize
from collections.abc import Callable
from pathlib import Path

CPP_SUFFIXES = (".h", ".hpp", ".cpp", ".cc")
VENDORED = frozenset({"src/io/mps_reader.cpp", "src/io/solu_reader.cpp"})


def cpp_counts(text: str) -> tuple[int, int]:
    """Return (non-blank lines, comment-only lines) of C++ source."""
    starts = [0]
    starts.extend(i + 1 for i, c in enumerate(text) if c == "\n")
    code: set[int] = set()
    comments: set[int] = set()

    def mark(target: set[int], begin: int, end: int) -> None:
        first = bisect.bisect_right(starts, begin) - 1
        last = bisect.bisect_right(starts, max(begin, end - 1)) - 1
        target.update(range(first, last + 1))

    i = 0
    size = len(text)
    while i < size:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = size if end < 0 else end
            mark(comments, i, end)
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = size if end < 0 else end + 2
            mark(comments, i, end)
            i = end
        elif text.startswith('R"', i) and (i == 0 or not _ident_char(text[i - 1])):
            open_paren = text.find("(", i + 2)
            delim = text[i + 2 : open_paren]
            end = text.find(")" + delim + '"', open_paren)
            end = size if end < 0 else end + len(delim) + 2
            mark(code, i, end)
            i = end
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            # The isalnum test skips digit separators (1'000).
            j = i + 1
            while j < size and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, size)
            mark(code, i, end)
            i = end
        else:
            if not c.isspace():
                mark(code, i, i + 1)
            i += 1
    lines = text.split("\n")
    nonblank = sum(1 for line in lines if line.strip())
    comment = sum(
        1 for n, line in enumerate(lines) if line.strip() and n in comments and n not in code
    )
    return nonblank, comment


def _ident_char(c: str) -> bool:
    # u8R"( LR"( uR"( UR"( are raw strings too; fooR"( is not.
    return (c.isalnum() or c == "_") and c not in "8LuU"


def docstring_lines(tree: ast.AST) -> set[int]:
    """Return the 1-based line numbers covered by docstrings."""
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            end = body[0].end_lineno or body[0].lineno
            covered.update(range(body[0].lineno, end + 1))
    return covered


def py_counts(text: str) -> tuple[int, int]:
    """Return (non-blank lines, comment-only lines) of Python source."""
    lines = text.splitlines()
    docs = docstring_lines(ast.parse(text))
    code: set[int] = set()
    comments: set[int] = set()
    skip = {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            comments.add(tok.start[0])
        elif tok.type not in skip:
            code.update(range(tok.start[0], tok.end[0] + 1))
    code -= docs
    comments |= docs
    nonblank = 0
    comment = 0
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        nonblank += 1
        if n in comments and n not in code:
            comment += 1
    return nonblank, comment


Area = tuple[str, Callable[[str], bool], Callable[[str], tuple[int, int]]]

AREAS: list[Area] = [
    ("include/cbls", lambda p: p.startswith("include/cbls/"), cpp_counts),
    (
        "src",
        lambda p: p.startswith("src/") and p.endswith(CPP_SUFFIXES) and p not in VENDORED,
        cpp_counts,
    ),
    ("python/bindings.cpp", lambda p: p == "python/bindings.cpp", cpp_counts),
    (
        "benchmarks C++",
        lambda p: p.startswith("benchmarks/") and p.endswith(CPP_SUFFIXES),
        cpp_counts,
    ),
    (
        "benchmarks Python",
        lambda p: p.startswith("benchmarks/") and p.endswith(".py"),
        py_counts,
    ),
    (
        "tests C++",
        lambda p: p.startswith("tests/") and p.endswith(CPP_SUFFIXES),
        cpp_counts,
    ),
    (
        "tests/python",
        lambda p: p.startswith("tests/python/") and p.endswith(".py"),
        py_counts,
    ),
]

ROOT = Path(__file__).resolve().parent.parent


def row(name: str, files: int, nonblank: int, comment: int) -> str:
    share = comment / nonblank if nonblank else 0.0
    return f"{name:<40} {files:>5} {nonblank:>9} {comment:>8} {share:>6.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Comment share per source area.")
    parser.add_argument(
        "--by-file", metavar="AREA", help="also list each file of AREA (e.g. include/cbls)"
    )
    args = parser.parse_args()
    files = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    print(f"{'area':<40} {'files':>5} {'non-blank':>9} {'comment':>8} {'share':>6}")
    detail: list[str] = []
    for name, selects, count in AREAS:
        chosen = [p for p in files if selects(p)]
        nonblank = 0
        comment = 0
        for p in chosen:
            nb, cm = count((ROOT / p).read_text(encoding="utf-8"))
            nonblank += nb
            comment += cm
            if name == args.by_file:
                detail.append(row(p, 1, nb, cm))
        print(row(name, len(chosen), nonblank, comment))
    if detail:
        print()
        print("\n".join(detail))


if __name__ == "__main__":
    main()
