#!/bin/bash
# Proves the clang-tidy gate reports compiler diagnostics (issue #171).
#
# Usage: .githooks/tidy-probe.sh CLANG_TIDY BUILD_DIR
#
# Run by ctest as `clang_tidy_gate_probe` on every full or fast suite run, and
# by the pre-push hook whenever the push could change what the gate sees (a
# .clang-tidy, a CMakeLists.txt, or this probe).
#
# The gate needs two things that each fail OPEN when missing, which is why a
# probe rather than a review guards them:
#
#   1. `clang-diagnostic-*` named after the `-*` in .clang-tidy's Checks.
#      clang-tidy prepends it, and a leading `-*` then switches it straight off
#      -- the state this repository was in until #171. No warning, no error.
#   2. -Wall -Wextra in every first-party compile command, and nothing that
#      cancels them. clang-tidy reads its flags from compile_commands.json, so
#      a diagnostic the build does not turn on can never be reported, whatever
#      the check list says. (Not -Wswitch itself: clang enables that by
#      default, so for the NodeOp switches item 1 was the whole hole on the
#      clang-tidy side. GCC, which compiles the build, needs -Wall for it --
#      pre-push gates GCC's warnings separately.)
#
# Check A (flags) reads only the compile database, so it runs even where there
# is no clang-tidy: CI and a checkout without .venv still get it.
#
# Check B (diagnostics) puts a copy of .githooks/tidy-probe/incomplete_switch.cpp
# into EVERY directory that holds a .clang-tidy, inside a scratch mirror of the
# tree's configs, and lints each copy through clang-tidy's ordinary directory
# lookup. Its three diagnostics pin the check list, -Wall and -Wextra
# respectively. A mirror rather than the real tree, because a probe written
# into src/ would be swept and committed; a real lookup rather than
# `--dump-config | --config`, because that round trip does not reproduce the
# config (it warns "invalid identifier naming option 'FunctionHungarianPrefix'"
# on read-back). The config files are byte copies at the same relative paths,
# and the mirror's root carries the root config, which does not inherit, so the
# lookup stops there exactly as it does in the tree. A new config directory is
# probed automatically.
#
# Exit 0: the gate bites. Exit 1: it does not, or the probe could not run.
# Exit 77: Check A passed but there is no clang-tidy for Check B -- ctest
# reports that as Skipped, not Passed.

set -u

CLANG_TIDY="${1:-}"
BUILD_DIR="${2:-build}"

HOOK_DIR=$(cd "$(dirname "$0")" && pwd -P) || exit 1
# shellcheck source=.githooks/gate-lib.sh
. "$HOOK_DIR/gate-lib.sh"

cd "$HOOK_DIR/.." || exit 1
case "$BUILD_DIR" in
/*) ;;
*) BUILD_DIR="$(pwd -P)/$BUILD_DIR" ;;
esac
CACHE="$BUILD_DIR/CMakeCache.txt"
# The source and binary dirs exactly as CMake recorded them -- the spelling
# every path in compile_commands.json uses. `pwd -P` is only the fallback:
# through a symlinked path the two can differ, and a prefix match on the wrong
# one would see no first-party TU at all.
ROOT=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE" 2>/dev/null)
[ -n "$ROOT" ] || ROOT=$(pwd -P)
BIN=$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$CACHE" 2>/dev/null)
[ -n "$BIN" ] || BIN="$BUILD_DIR"

PROBE=.githooks/tidy-probe/incomplete_switch.cpp
EXPECTED="clang-diagnostic-switch clang-diagnostic-unused-variable clang-diagnostic-unused-parameter"
DB="$BUILD_DIR/compile_commands.json"

if [ ! -f "$DB" ]; then
	echo "tidy-probe: FAIL -- no $DB. Configure the build first."
	exit 1
fi
PROBE_CMD=$(grep '"command"' "$DB" | grep -F -- " -c $ROOT/$PROBE\"" | head -1)
if [ -z "$PROBE_CMD" ]; then
	echo "tidy-probe: FAIL -- $PROBE has no entry in $DB. The cbls_tidy_probe"
	echo "target in CMakeLists.txt puts it there: if this tree has no such"
	echo "target, merge main; otherwise reconfigure with CBLS_BUILD_TESTS=ON."
	exit 1
fi

FAIL=0

# --- Check A: every first-party compile command carries the flags ----------
FIRST_PARTY=$(first_party_commands "$DB" "$ROOT" "$BIN")
if [ -z "$FIRST_PARTY" ]; then
	echo "tidy-probe: FAIL -- no first-party compile command found in $DB."
	exit 1
fi
MISSING=$(echo "$FIRST_PARTY" | missing_warning_flags)
if [ -n "$MISSING" ]; then
	echo "tidy-probe: FAIL -- compiled without -Wall/-Wextra, so clang-tidy"
	echo "cannot report their compiler warnings. Give the target"
	echo "cbls_enable_warnings() in CMakeLists.txt:"
	echo "${MISSING//"$ROOT"\//  }"
	FAIL=1
fi
# Flags CMake seeded from the user's environment (CXXFLAGS -> CMAKE_CXX_FLAGS,
# and the per-config set) are the machine's, not the project's: exempt them.
BUILD_TYPE=$(sed -n 's/^CMAKE_BUILD_TYPE:[A-Z]*=//p' "$CACHE" 2>/dev/null | tr '[:lower:]' '[:upper:]')
ENV_FLAGS=$(sed -n -e 's/^CMAKE_CXX_FLAGS:[A-Z]*=//p' \
	-e "s/^CMAKE_CXX_FLAGS_${BUILD_TYPE:-NONE}:[A-Z]*=//p" "$CACHE" 2>/dev/null | tr '\n' ' ')
CANCELLED=$(echo "$FIRST_PARTY" | cancelled_warning_flags "$ENV_FLAGS")
if [ -n "$CANCELLED" ]; then
	echo "tidy-probe: FAIL -- a target's compile options cancel the warnings"
	echo "(any -Wno-*, -w or -Werror; fix the code rather than the flags)."
	echo "Flags in CMAKE_CXX_FLAGS -- where CXXFLAGS from your environment"
	echo "land -- are exempt, so these come from the project or a -D:"
	echo "${CANCELLED//"$ROOT"\//  }"
	FAIL=1
fi

if [ -z "$CLANG_TIDY" ] || ! command -v "$CLANG_TIDY" >/dev/null 2>&1; then
	if [ "$FAIL" -ne 0 ]; then
		exit 1
	fi
	echo "tidy-probe: flags OK; diagnostics SKIPPED -- no clang-tidy"
	echo "('$CLANG_TIDY'). Install the pinned one with"
	echo ".venv/bin/pip install -e '.[dev]' and reconfigure."
	exit 77
fi

# --- Check B: the probe is reported under every config directory ----------
# The mirror lives inside the build dir, i.e. inside the checkout, so a copy
# that is missing would not fail: clang-tidy's upward lookup would climb out
# of the mirror into the real tree and find the ROOT config, and the probe
# would pass while that directory's own config was never tested. Two guards:
# every copy is checked, and each probe copy's effective config
# (--dump-config) must equal the one the real directory gets.
# Tracked configs where git can say; otherwise every .clang-tidy outside the
# trees that hold other people's code (an exported tarball).
CONFIGS=$(git -C "$ROOT" ls-files -- '*.clang-tidy' 2>/dev/null)
if [ -z "$CONFIGS" ]; then
	CONFIGS=$(cd "$ROOT" && find . \( -path "./build*" -o -path ./.venv -o -path ./.claude -o -path ./.git \) -prune \
		-o -name .clang-tidy -print | sed 's|^\./||')
fi
if [ -z "$CONFIGS" ]; then
	echo "tidy-probe: FAIL -- found no .clang-tidy under $ROOT."
	exit 1
fi

MIRROR="$BIN/tidy-probe-mirror"
rm -rf "$MIRROR"
mkdir -p "$MIRROR"
# The probe's own compile command, retargeted at each copy, so every copy is
# linted with the flags the build really gives it.
PROBE_ARGS=$(echo "$PROBE_CMD" | sed -E 's/^ *"command": "(.*)",?$/\1/')
ENTRIES=""
PROBED=""
for config in $CONFIGS; do
	dir=$(dirname "$config")
	copy="$MIRROR/$dir/cbls_tidy_probe.cpp"
	if ! mkdir -p "$MIRROR/$dir" || ! cp "$ROOT/$config" "$MIRROR/$config" ||
		! cp "$ROOT/$PROBE" "$copy"; then
		echo "tidy-probe: FAIL -- could not mirror $config and the probe."
		exit 1
	fi
	args=$(awk -v cmd="$PROBE_ARGS" -v from="$ROOT/$PROBE" -v to="$copy" 'BEGIN {
		i = index(cmd, from)
		print substr(cmd, 1, i - 1) to substr(cmd, i + length(from))
	}')
	[ -z "$ENTRIES" ] || ENTRIES="${ENTRIES},"
	ENTRIES="${ENTRIES}{\"directory\": \"$BIN\", \"command\": \"$args\", \"file\": \"$copy\"}"
	PROBED="${PROBED} $dir"
done
echo "[${ENTRIES}]" >"$MIRROR/compile_commands.json"

for dir in $PROBED; do
	copy="$MIRROR/$dir/cbls_tidy_probe.cpp"
	# The file need not exist for --dump-config: only its directory is read.
	WANT=$("$CLANG_TIDY" --dump-config "$ROOT/$dir/cbls_tidy_probe.cpp" 2>/dev/null)
	GOT=$("$CLANG_TIDY" --dump-config "$copy" 2>/dev/null)
	if [ -z "$WANT" ] || [ "$WANT" != "$GOT" ]; then
		echo "tidy-probe: FAIL -- the probe copy for $dir/ does not see the"
		echo "config the real directory gets (its lookup escaped the mirror,"
		echo "or a copy is wrong), so a result there would test the wrong file."
		FAIL=1
		continue
	fi
	OUT=$("$CLANG_TIDY" -p "$MIRROR" --quiet "$copy" 2>&1) && RC=0 || RC=$?
	if [ "$RC" -ne 0 ]; then
		# Nonzero is a config fault ("Error: no checks enabled.") or a compile
		# command with -Werror, which turns every finding into a compiler
		# error the gate reports as a config fault instead of a finding.
		echo "tidy-probe: FAIL -- clang-tidy exited $RC on the probe under"
		echo "the config that applies in $dir/:"
		echo "$OUT" | head -5
		FAIL=1
		continue
	fi
	for diag in $EXPECTED; do
		if ! echo "$OUT" | grep -qF "[$diag]"; then
			echo "tidy-probe: FAIL -- [$diag] not reported under the config"
			echo "that applies in $dir/. The gate is silent on that"
			echo "diagnostic: check that .clang-tidy names clang-diagnostic-*"
			echo "AFTER its -* (verify with --dump-config), and that the"
			echo "compile command carries -Wall -Wextra."
			FAIL=1
		fi
	done
done
rm -rf "$MIRROR"

if [ "$FAIL" -eq 0 ]; then
	echo "tidy-probe: OK -- flags on every first-party TU; $EXPECTED"
	echo "reported under the config of each of:$PROBED"
fi
exit "$FAIL"
