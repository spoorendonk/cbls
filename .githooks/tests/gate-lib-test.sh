#!/bin/bash
# Tests for the compiler-warning gates' filters (.githooks/gate-lib.sh) and for
# tidy-probe.sh's skip/fail contract, on synthetic input. Registered in ctest as
# `gate_lib_shell_test`. Usage: gate-lib-test.sh BUILD_DIR
#
# Each filter decides what a blocking gate sees, and a filter that matches
# nothing makes that gate pass -- so each case below is one way the filter
# could go blind, or over-reach, and a mutation of the filter turns one red.

set -u
BUILD_DIR="${1:?usage: gate-lib-test.sh BUILD_DIR}"
HERE=$(cd "$(dirname "$0")" && pwd -P) || exit 1
REPO=$(cd "$HERE/../.." && pwd -P) || exit 1
# shellcheck source=.githooks/gate-lib.sh
. "$HERE/../gate-lib.sh"

SCRATCH="$BUILD_DIR/gate-lib-test"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
FAILS=0

expect() { # expect NAME EXPECTED ACTUAL
	if [ "$2" != "$3" ]; then
		echo "FAIL: $1"
		echo "  expected: $(printf '%q' "$2")"
		echo "  actual:   $(printf '%q' "$3")"
		FAILS=$((FAILS + 1))
	else
		echo "ok: $1"
	fi
}

# --- first_party_warnings -----------------------------------------------------
ESC=$'\x1b'
cat >"$SCRATCH/build.log" <<EOF
/R/src/a.cpp:1:2: warning: plain [-Wunused]
${ESC}[01m${ESC}[K/R/tests/b.cpp:3:4:${ESC}[m${ESC}[K ${ESC}[01;35m${ESC}[Kwarning: ${ESC}[m${ESC}[Kcoloured [-Wswitch]
../tools/c.cpp:5:6: warning: relative, ccache base_dir [-Wunused]
/R/buildtools/g.cpp:1:1: warning: a source dir that merely starts with build
/R/newdir/d.cpp:7:8: warning: a new top-level dir
/R/include/cbls/h.h:9: warning: no column
../build/_deps/json-src/x.hpp:1:1: warning: dependency, relative
/R/build/_deps/catch2-src/y.cpp:1:1: warning: dependency, absolute
/R/.venv/lib/nb.cpp:1:1: warning: nanobind
/R/.claude/worktrees/w/src/e.cpp:1:1: warning: another worktree
/usr/bin/ld: warning: a linker note
/other/z.cpp:1:1: warning: outside the checkout
CMake Warning: not a compiler diagnostic
/R/src/f.cpp:1:1: error: an error, not a warning
EOF
expect "first_party_warnings keeps exactly our warnings" \
	"/R/src/a.cpp:1:2: warning: plain [-Wunused]
/R/tests/b.cpp:3:4: warning: coloured [-Wswitch]
../tools/c.cpp:5:6: warning: relative, ccache base_dir [-Wunused]
/R/buildtools/g.cpp:1:1: warning: a source dir that merely starts with build
/R/newdir/d.cpp:7:8: warning: a new top-level dir
/R/include/cbls/h.h:9: warning: no column" \
	"$(first_party_warnings "$SCRATCH/build.log" /R /R/build)"

# --- first_party_commands / missing / cancelled ------------------------------
cmd() { # cmd PATH FLAGS...
	local path=$1
	shift
	printf '  "command": "/usr/bin/c++ %s -o x.o -c %s",\n' "$*" "$path"
}
{
	echo '['
	cmd /R/src/ok.cpp -O3 -Wall -Wextra
	cmd /R/src/no_wall.cpp -O3 -Wextra
	cmd /R/src/no_wextra.cpp -Wall -O3
	cmd /R/buildtools/t.cpp -Wall -Wextra -w
	cmd /R/src/werror.cpp -Wall -Wextra -Werror
	cmd /R/src/werror_eq.cpp -Wall -Wextra -Werror=switch
	cmd /R/src/wno.cpp -Wall -Wextra -Wno-sign-compare
	cmd /R/src/wall_last.cpp -O3 -Wextra -Wall
	cmd /R/build/_deps/dep.cpp -Wno-everything
	cmd /R/.venv/nb.cpp -O3
	cmd /other/x.cpp -O3
	echo ']'
} >"$SCRATCH/compile_commands.json"
FP=$(first_party_commands "$SCRATCH/compile_commands.json" /R /R/build)
expect "first_party_commands selects our TUs, including buildtools/" \
	"/R/src/ok.cpp
/R/src/no_wall.cpp
/R/src/no_wextra.cpp
/R/buildtools/t.cpp
/R/src/werror.cpp
/R/src/werror_eq.cpp
/R/src/wno.cpp
/R/src/wall_last.cpp" \
	"$(echo "$FP" | command_source)"
expect "missing_warning_flags names TUs without -Wall or -Wextra" \
	"/R/src/no_wall.cpp
/R/src/no_wextra.cpp" \
	"$(echo "$FP" | missing_warning_flags)"
expect "cancelled_warning_flags names any -w, -Werror[=x] or -Wno-*" \
	"/R/buildtools/t.cpp
/R/src/werror.cpp
/R/src/werror_eq.cpp
/R/src/wno.cpp" \
	"$(echo "$FP" | cancelled_warning_flags)"

# --- tidy-probe.sh: Check A runs without clang-tidy; skip only when clean ----
# A fake build dir whose cache names the real checkout, so the probe's own
# entry resolves; its compile database is synthetic.
fake_build() { # fake_build DIR FLAGS...
	local dir=$1
	shift
	mkdir -p "$dir"
	printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\nCMAKE_CACHEFILE_DIR:INTERNAL=%s\n' "$REPO" "$dir" >"$dir/CMakeCache.txt"
	{
		echo '['
		cmd "$REPO/.githooks/tidy-probe/incomplete_switch.cpp" "$@"
		cmd "$REPO/src/dag.cpp" "$@"
		echo ']'
	} >"$dir/compile_commands.json"
}
run_probe() { # run_probe BUILD -> exit code
	bash "$HERE/../tidy-probe.sh" "" "$1" >"$SCRATCH/probe.out" 2>&1
	echo $?
}
fake_build "$SCRATCH/clean" -Wall -Wextra
expect "probe without clang-tidy, flags clean: skip (77)" 77 "$(run_probe "$SCRATCH/clean")"
fake_build "$SCRATCH/noflags" -O3
expect "probe without clang-tidy, flags missing: fail (1)" 1 "$(run_probe "$SCRATCH/noflags")"
fake_build "$SCRATCH/cancel" -Wall -Wextra -Wno-switch
expect "probe without clang-tidy, flags cancelled: fail (1)" 1 "$(run_probe "$SCRATCH/cancel")"
mkdir -p "$SCRATCH/noprobe"
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "$REPO" >"$SCRATCH/noprobe/CMakeCache.txt"
{
	echo '['
	cmd "$REPO/src/dag.cpp" -Wall -Wextra
	echo ']'
} >"$SCRATCH/noprobe/compile_commands.json"
expect "probe with no probe entry: fail (1)" 1 "$(run_probe "$SCRATCH/noprobe")"
expect "  ... and says to merge main" 1 "$(grep -c 'merge main' "$SCRATCH/probe.out")"

rm -rf "$SCRATCH"
if [ "$FAILS" -ne 0 ]; then
	echo "$FAILS case(s) failed"
	exit 1
fi
echo "all cases passed"
