#!/bin/bash
# Proves the clang-tidy gate reports compiler diagnostics (issue #171).
#
# Usage: .githooks/tidy-probe.sh CLANG_TIDY BUILD_DIR
#
# Run by the pre-push hook whenever the push could change what the gate sees
# (a .clang-tidy, a CMakeLists.txt, or this probe), and by ctest as
# `clang_tidy_gate_probe` on every full or fast suite run.
#
# The gate needs two things that each fail OPEN when missing, which is why a
# probe rather than a review guards them:
#
#   1. `clang-diagnostic-*` named after the `-*` in .clang-tidy's Checks.
#      clang-tidy prepends it, and a leading `-*` then switches it straight off
#      -- the state this repository was in until #171. No warning, no error.
#   2. -Wall -Wextra in every first-party compile command. clang-tidy reads its
#      flags from compile_commands.json, so a diagnostic the build does not turn
#      on can never be reported, whatever the check list says.
#
# Check 1 lints .githooks/tidy-probe/incomplete_switch.cpp under the EFFECTIVE
# config of each config directory (dumped for a canary file in it, then passed
# back with --config), so a nested .clang-tidy that re-breaks the list is caught
# as surely as the root one. Check 2 reads the compile database: the probe's own
# entry proves the flags reach clang-tidy, and the scan proves every first-party
# target carries them -- the flags are per target (cbls_enable_warnings in
# CMakeLists.txt), so a new target that forgets the call would otherwise compile
# without them in silence.
#
# Exit 0: the gate bites. Exit 1: it does not, or the probe could not run.
# Exit 77: no clang-tidy to run (ctest reports that as Skipped, not Passed).

set -u

CLANG_TIDY="${1:-}"
BUILD_DIR="${2:-build}"

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)
case "$BUILD_DIR" in
/*) ;;
*) BUILD_DIR="$ROOT/$BUILD_DIR" ;;
esac

PROBE=.githooks/tidy-probe/incomplete_switch.cpp
# One canary per clang-tidy config directory -- the same pair the pre-push
# hook seeds its lint with when a .clang-tidy changes.
CANARIES="src/search.cpp src/io/mps_to_model.cpp"
EXPECTED="clang-diagnostic-switch clang-diagnostic-unused-parameter"
DB="$BUILD_DIR/compile_commands.json"

if [ -z "$CLANG_TIDY" ] || ! command -v "$CLANG_TIDY" >/dev/null 2>&1; then
	echo "tidy-probe: SKIPPED -- no clang-tidy ('$CLANG_TIDY'). Install the"
	echo "pinned one with .venv/bin/pip install -e '.[dev]' and reconfigure."
	exit 77
fi
if [ ! -f "$DB" ]; then
	echo "tidy-probe: FAIL -- no $DB. Configure the build first."
	exit 1
fi
if ! grep -qF "$ROOT/$PROBE" "$DB"; then
	echo "tidy-probe: FAIL -- $PROBE has no entry in $DB, so it would be"
	echo "linted with guessed flags. The cbls_tidy_probe target in"
	echo "CMakeLists.txt puts it there; reconfigure the build."
	exit 1
fi

FAIL=0

# --- Check 2: every first-party compile command carries the flags ---------
# CMake writes one "command" line per translation unit, ending in
# `-c <absolute source path>`. First-party is named by directory rather than
# "anything under this checkout", because FetchContent sources live in
# build*/_deps and, in a checkout whose .venv is a real directory, nanobind's
# own sources under .venv/ -- neither is ours to warn on.
# A fixed-string prefix match, not a regex: the checkout path is arbitrary text.
FIRST_PARTY=$(grep '"command"' "$DB" | awk -v root="$ROOT/" '{
	i = index($0, " -c " root)
	if (i && substr($0, i + 4 + length(root)) ~ /^(src|tests|benchmarks|examples|python|\.githooks)\//) print
}')
if [ -z "$FIRST_PARTY" ]; then
	echo "tidy-probe: FAIL -- no first-party compile command found in $DB."
	exit 1
fi
MISSING=""
for flag in -Wall -Wextra; do
	MISSING="${MISSING}$(echo "$FIRST_PARTY" |
		grep -vE -- " $flag( |\")" |
		sed -E 's/.* -c ([^"]*)".*/\1/' || true)"$'\n'
done
MISSING=$(echo "$MISSING" | sed '/^$/d')
if [ -n "$MISSING" ]; then
	echo "tidy-probe: FAIL -- compiled without -Wall/-Wextra, so clang-tidy"
	echo "cannot report their compiler warnings. Give the target"
	echo "cbls_enable_warnings() in CMakeLists.txt:"
	echo "$MISSING" | sort -u | sed "s|^$ROOT/|  |"
	FAIL=1
fi

# --- Check 1: the probe is reported under each directory's config ---------
for canary in $CANARIES; do
	if [ ! -f "$canary" ]; then
		echo "tidy-probe: FAIL -- canary $canary is gone; name a surviving"
		echo "file in its config directory in this script and in pre-push."
		FAIL=1
		continue
	fi
	# stdout only: anything clang-tidy says on stderr would become config text.
	CONFIG=$("$CLANG_TIDY" -p "$BUILD_DIR" --dump-config "$canary" 2>/dev/null) && RC=0 || RC=$?
	if [ "$RC" -ne 0 ]; then
		echo "tidy-probe: FAIL -- --dump-config for $canary exited $RC:"
		"$CLANG_TIDY" -p "$BUILD_DIR" --dump-config "$canary" 2>&1 | head -5
		FAIL=1
		continue
	fi
	# The dump is a YAML document with `---`/`...` markers; --config takes the
	# bare mapping and rejects `...` as an unknown key.
	CONFIG=$(echo "$CONFIG" | sed -e '/^---$/d' -e '/^\.\.\.$/d')
	OUT=$("$CLANG_TIDY" -p "$BUILD_DIR" --config="$CONFIG" --quiet "$PROBE" 2>&1) && RC=0 || RC=$?
	if [ "$RC" -ne 0 ]; then
		# Nonzero is a config fault ("Error: no checks enabled.") or a compile
		# command with -Werror, which turns every finding into a compiler
		# error the gate reports as a config fault instead of a finding.
		echo "tidy-probe: FAIL -- clang-tidy exited $RC on the probe under"
		echo "$canary's config:"
		echo "$OUT" | head -5
		FAIL=1
		continue
	fi
	for diag in $EXPECTED; do
		if ! echo "$OUT" | grep -qF "[$diag]"; then
			echo "tidy-probe: FAIL -- [$diag] not reported under the config"
			echo "that applies to $canary. The gate is silent on that"
			echo "diagnostic: check that .clang-tidy names clang-diagnostic-*"
			echo "AFTER its -* (verify with --dump-config), and that the"
			echo "compile command carries -Wall -Wextra."
			FAIL=1
		fi
	done
done

if [ "$FAIL" -eq 0 ]; then
	echo "tidy-probe: OK -- $EXPECTED reported under the config of each of:"
	echo "  $CANARIES"
fi
exit "$FAIL"
