#!/bin/bash
# Tests for the compiler-warning gates' filters (.githooks/gate-lib.sh) and for
# tidy-probe.sh's skip/fail contract, on synthetic input. Registered in ctest as
# `gate_lib_shell_test`. Usage: gate-lib-test.sh BUILD_DIR
#
# Each filter decides what a blocking gate sees, and a filter that matches
# nothing makes that gate pass -- so each case below is one way the filter
# could go blind, or over-reach, and a mutation of the filter turns one red.

# SC2016: the stub clang-tidy bodies below are script text for the stub to
# expand, so single quotes are intended.
# shellcheck disable=SC2016

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
	"$(echo "$FP" | cancelled_warning_flags | LC_ALL=C sort)" # a set; macOS's locale sorts werror_eq first (#355)
expect "cancelled_warning_flags exempts the environment's CMAKE_CXX_FLAGS tokens" \
	"/R/buildtools/t.cpp
/R/src/werror.cpp" \
	"$(echo "$FP" | cancelled_warning_flags "-O2 -Werror=switch -Wno-sign-compare")"

# --- scan_build_log: the pre-push decision -----------------------------------
CONTROL='/R/.githooks/tidy-probe/incomplete_switch.cpp:31:12: warning: enumeration value not handled in switch [-Wswitch]'
scan() { # scan LOG_TEXT -> "rc:output"
	printf '%s\n' "$1" >"$SCRATCH/scan.log"
	local out rc=0
	out=$(scan_build_log "$SCRATCH/scan.log" /R /R/build) || rc=$?
	echo "$rc:$out"
}
expect "scan: control only is clean (0)" "0:" "$(scan "$CONTROL")"
expect "scan: control plus our warning blocks (1) and names only ours" \
	"1:/R/src/a.cpp:1:2: warning: x [-Wunused]" \
	"$(scan "$CONTROL
/R/src/a.cpp:1:2: warning: x [-Wunused]
/R/build/_deps/d.cpp:1:1: warning: dep")"
expect "scan: no control is blind (2), even with warnings present" "2:" \
	"$(scan "/R/src/a.cpp:1:2: warning: x [-Wunused]")"
expect "scan: an empty log is blind (2)" "2:" "$(scan "")"

# --- pre-push's path classification -----------------------------------------
CHANGED='.githooks/pre-push
.githooks/pre-commit
.githooks/commit-msg
.githooks/gate-lib.sh
.githooks/tidy-probe.sh
.githooks/tidy-probe/incomplete_switch.cpp
.githooks/tests/gate-lib-test.sh
.clang-tidy
src/io/.clang-tidy
CMakeLists.txt
tests/CMakeLists.txt
src/dag.cpp
README.md'
expect "hook_files_that_are_code: pre-push, the lib, the probe and its tests" \
	".githooks/pre-push
.githooks/gate-lib.sh
.githooks/tidy-probe.sh
.githooks/tidy-probe/incomplete_switch.cpp
.githooks/tests/gate-lib-test.sh" \
	"$(echo "$CHANGED" | hook_files_that_are_code)"
expect "probe_trigger_files: configs, CMakeLists, the probe and the lib" \
	".githooks/gate-lib.sh
.githooks/tidy-probe.sh
.githooks/tidy-probe/incomplete_switch.cpp
.clang-tidy
src/io/.clang-tidy
CMakeLists.txt
tests/CMakeLists.txt" \
	"$(echo "$CHANGED" | probe_trigger_files)"

# --- restore_probe_exclusion_in: reconfigures WITHOUT the probe variable -----
mkdir -p "$SCRATCH/bin" "$SCRATCH/restore"
printf '#!/bin/bash\necho "${CBLS_GATE_PROBE_IN_ALL:-unset}|$*" >>"%s"\n' "$SCRATCH/cmake.calls" >"$SCRATCH/bin/cmake"
chmod +x "$SCRATCH/bin/cmake"
touch "$SCRATCH/restore/CMakeCache.txt"
(
	export PATH="$SCRATCH/bin:$PATH" CBLS_GATE_PROBE_IN_ALL=1
	restore_probe_exclusion_in "$SCRATCH/restore"
	restore_probe_exclusion_in "$SCRATCH/no-such-build"
)
expect "restore_probe_exclusion_in reconfigures once, with the variable unset" \
	"unset|$SCRATCH/restore" "$(cat "$SCRATCH/cmake.calls")"

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

# --- tidy-probe.sh Check B, against stub clang-tidy binaries -----------------
# Hermetic: no real clang-tidy. Every stub answers --dump-config with the same
# text for any path, except `escape`, which answers differently inside the
# mirror -- what a lookup that climbed out of it looks like.
stub() { # stub NAME BODY
	printf '#!/bin/bash\n%s\n' "$2" >"$SCRATCH/$1"
	chmod +x "$SCRATCH/$1"
}
DUMP='for a in "$@"; do [ "$a" = --dump-config ] && { echo "Checks: stub"; exit 0; }; done'
stub silent "$DUMP"$'\nexit 0'
stub reports "$DUMP"$'\n'"echo 'w [clang-diagnostic-switch]'; echo 'w [clang-diagnostic-unused-variable]'; echo 'w [clang-diagnostic-unused-parameter]'"
stub partial "$DUMP"$'\n'"echo 'w [clang-diagnostic-switch]'; echo 'w [clang-diagnostic-unused-parameter]'"
stub crashes "$DUMP"$'\n'"echo 'e [clang-diagnostic-switch]'; echo 'e [clang-diagnostic-unused-variable]'; echo 'e [clang-diagnostic-unused-parameter]'; exit 1"
stub escape "$(printf '%s\n' \
	'for a in "$@"; do [ "$a" = --dump-config ] && d=1; f=$a; done' \
	'if [ -n "${d:-}" ]; then case $f in *tidy-probe-mirror*) echo "Checks: root";; *) echo "Checks: own";; esac; exit 0; fi' \
	"echo 'w [clang-diagnostic-switch]'; echo 'w [clang-diagnostic-unused-variable]'; echo 'w [clang-diagnostic-unused-parameter]'")"
run_stub() { # run_stub STUB -> exit code
	bash "$HERE/../tidy-probe.sh" "$SCRATCH/$1" "$SCRATCH/clean" >"$SCRATCH/probe.out" 2>&1
	echo $?
}
expect "probe: clang-tidy reporting all three diagnostics passes (0)" 0 "$(run_stub reports)"
expect "probe: clang-tidy reporting nothing fails (1)" 1 "$(run_stub silent)"
expect "probe: one diagnostic missing fails (1)" 1 "$(run_stub partial)"
expect "probe: clang-tidy exiting nonzero fails (1), even with all tags" 1 "$(run_stub crashes)"
expect "probe: a lookup that escapes the mirror fails (1)" 1 "$(run_stub escape)"

# --- pre-commit's Python half (#371) ------------------------------------------
# It must run the pinned ruff and mypy from .venv/bin (this checkout's, then the
# main checkout's from a linked worktree), never through uv (whose sync rebuilt
# the C++ extension in the hook), and refuse the commit when they are missing or
# off their pin, even with another ruff or mypy on PATH. A subshell without git's
# hook variables: ctest may be running under pre-push.
PC="$SCRATCH/pc"
mkdir -p "$PC/repo/.venv/bin" "$PC/global"
tool_stub() { # tool_stub DIR NAME VERSION
	printf '#!/bin/sh\necho "%s $*" >>"%s/calls.log"\n[ "$1" = --version ] && echo "%s %s (stub)"\nexit 0\n' \
		"$1/$2" "$PC" "$2" "$3" >"$1/$2"
	chmod +x "$1/$2"
}
tool_stub "$PC/repo/.venv/bin" ruff 0.16.1
tool_stub "$PC/repo/.venv/bin" mypy 2.3.0
tool_stub "$PC/global" ruff 0.16.1
tool_stub "$PC/global" mypy 2.3.0
printf '[project]\nname = "x"\ndev = [\n    "ruff==0.16.1",\n    "mypy==2.3.0",\n]\n' >"$PC/repo/pyproject.toml"
run_pc() { # run_pc DIR -> exit code; output in $PC/out, calls appended to calls.log
	(
		for v in $(env | sed -n 's/^\(GIT_[A-Za-z_]*\)=.*/\1/p'); do unset "$v"; done
		cd "$1" && PATH="$PC/global:$PATH" sh "$REPO/.githooks/pre-commit" >"$PC/out" 2>&1
	)
	echo $?
}
(
	for v in $(env | sed -n 's/^\(GIT_[A-Za-z_]*\)=.*/\1/p'); do unset "$v"; done
	cd "$PC/repo" && git init -q && git add pyproject.toml &&
		git -c user.name=t -c user.email=t@t commit -q --no-verify -m init &&
		git worktree add -q "$PC/wt" 2>/dev/null
) || echo "FAIL: pre-commit fixture: git setup"
: >"$PC/calls.log"
expect "pre-commit: pinned .venv tools pass (0)" 0 "$(run_pc "$PC/repo")"
expect "pre-commit: runs .venv's ruff format, ruff check and mypy, not PATH's" \
	"$PC/repo/.venv/bin/ruff format --check .|$PC/repo/.venv/bin/ruff check .|$PC/repo/.venv/bin/mypy ." \
	"$(grep -v -- '--version' "$PC/calls.log" | paste -sd '|' -)"
expect "pre-commit: a linked worktree without .venv uses the main checkout's (0)" 0 "$(run_pc "$PC/wt")"
tool_stub "$PC/repo/.venv/bin" ruff 0.15.0
expect "pre-commit: ruff off its pyproject pin refuses (1)" 1 "$(run_pc "$PC/repo")"
rm -rf "$PC/repo/.venv"
expect "pre-commit: no .venv refuses (1), with ruff and mypy on PATH" 1 "$(run_pc "$PC/repo")"
expect "pre-commit: the refusal names the uv sync to run" 1 "$(grep -c 'no mypy in .venv/bin.*uv sync' "$PC/out")"

rm -rf "$SCRATCH"
if [ "$FAILS" -ne 0 ]; then
	echo "$FAILS case(s) failed"
	exit 1
fi
echo "all cases passed"
