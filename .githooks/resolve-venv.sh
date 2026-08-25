#!/bin/bash
# Resolve Python venv. Sourced by other hooks.
#
# This directory is the source for the git hooks — edit them here.
#
# On success (venv present): VENV_BIN is set to the venv's bin directory and
# callers can run `"$VENV_BIN/ruff"`, `"$VENV_BIN/pytest"`, etc.
#
# If the project has Python infrastructure (pyproject.toml / setup.py /
# setup.cfg / requirements.txt) but no venv: exit 2, which propagates through
# `source` and blocks the enclosing hook.
#
# If the project has no Python metadata (e.g. a C++ project with a stray
# utility script): set VENV_BIN="" and return. Callers already guard every
# Python tool invocation with `[ -x "$VENV_BIN/foo" ]`, so those steps skip
# naturally without the hook failing.
#
# Usage:
#   source "$(dirname "$0")/resolve-venv.sh"
#
# SC2034: VENV_BIN looks unused here because its only consumers are the hooks
# that source this file.  SC2317: the `return 0 2>/dev/null || exit 0` line is
# deliberately dual-mode (sourced vs. executed) and only one half ever runs.
# shellcheck disable=SC2034,SC2317

VENV_DIR=""
[ -d ".venv" ] && VENV_DIR=".venv"
[ -d "venv" ] && VENV_DIR="venv"

if [ -z "$VENV_DIR" ]; then
	if [ -f "pyproject.toml" ] || [ -f "setup.py" ] || [ -f "setup.cfg" ] || [ -f "requirements.txt" ]; then
		echo "FAILED: No virtualenv found. Create one first:"
		echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
		exit 2
	fi
	# No Python metadata — treat staged .py files as standalone utilities and
	# let Python tool steps skip via the `-x "$VENV_BIN/..."` guards in callers.
	VENV_BIN=""
	return 0 2>/dev/null || exit 0
fi

VENV_BIN="$VENV_DIR/bin"
