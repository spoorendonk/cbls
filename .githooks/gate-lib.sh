#!/bin/bash
# Shared, pure text filters for the compiler-warning gates (issue #171).
#
# Sourced by pre-push and tidy-probe.sh, and exercised on synthetic input by
# .githooks/tests/gate-lib-test.sh (ctest `gate_lib_shell_test`). They live
# apart from the hooks precisely so they can be tested: each decides what a
# gate sees, and a filter that silently matches nothing makes the gate pass.
#
# "First-party" everywhere below means: under the source root CMake recorded,
# and NOT under the binary dir (FetchContent's _deps live there), .venv/
# (nanobind's sources, where .venv is a real directory) or .claude/ (agent
# worktrees). Everything else counts as ours, so a TU in a new top-level
# directory is gated rather than skipped. The binary dir is excluded by its
# real path, not a `build*` pattern, so a source dir like `buildtools/` is not.

# first_party_warnings LOG SRC_ROOT BIN_DIR
# Prints the compiler warnings in LOG that are in first-party code. ANSI
# colour is stripped first (CMAKE_COLOR_DIAGNOSTICS=ON splits ": warning:"
# with ESC[m). A relative path -- ccache's base_dir rewrites them -- counts as
# ours unless it names one of the excluded trees.
first_party_warnings() {
	sed 's/\x1b\[[0-9;]*[mK]//g' "$1" | awk -v root="$2/" -v bin="$3/" '
		/^[^ :]+:[0-9]+:([0-9]+:)? warning:/ {
			path = substr($0, 1, index($0, ":") - 1)
			if (substr(path, 1, 1) == "/") {
				if (index(path, root) != 1) next
				if (index(path, bin) == 1) next
				if (substr(path, length(root) + 1) ~ /^(\.venv|\.claude)\//) next
			} else if (path ~ /(^|\/)(_deps|\.venv|site-packages)\//) next
			print
		}'
}

# first_party_commands DB SRC_ROOT BIN_DIR
# Prints the "command" lines of a compile_commands.json whose translation unit
# (`-c <absolute path>`, which CMake always writes) is first-party. The prefix
# is matched as a fixed string: the checkout path is arbitrary text.
first_party_commands() {
	grep '"command"' "$1" | awk -v root="$2/" -v bin="$3/" '{
		i = index($0, " -c " root)
		if (!i) next
		path = substr($0, i + 4)
		if (index(path, bin) == 1) next
		if (substr(path, length(root) + 1) ~ /^(\.venv|\.claude)\//) next
		print
	}'
}

# command_source: the TU path of each "command" line on stdin.
command_source() {
	sed -E 's/.* -c ([^"]*)".*/\1/'
}

# missing_warning_flags: TUs on stdin ("command" lines) lacking -Wall or -Wextra.
missing_warning_flags() {
	local cmds
	cmds=$(cat)
	for flag in -Wall -Wextra; do
		echo "$cmds" | grep -vE -- " $flag( |\")" | grep '"command"' | command_source
	done | sort -u
}

# cancelled_warning_flags: TUs on stdin ("command" lines) that cancel the
# warnings. The rule is total, not a list of the dangerous ones: ANY -Wno-*,
# -w, or -Werror / -Werror=*. A -Wno- on a first-party target is a suppression
# with no reason written at the site -- the same thing a NOLINT or a disabled
# check is, and closed for the same reason; fix the code instead. -Werror is
# refused because it turns every clang-tidy finding into a compiler error the
# gate reports as a configuration fault (CMakeLists.txt, cbls_enable_warnings).
#
# EXEMPT is a space-separated token list not to police: the caller passes the
# cache's CMAKE_CXX_FLAGS and CMAKE_CXX_FLAGS_<CONFIG>, which CMake seeds from
# the user's CXXFLAGS (dpkg-buildflags' -Werror=format-security, -Wno-psabi on
# ARM). The rule is about the project's own target flags, not the machine it
# is built on. -Wall/-Wextra are still required by missing_warning_flags, and
# an environment flag that really blinds clang-tidy (-w, -Werror) still turns
# the probe's diagnostics check red.
cancelled_warning_flags() { # [EXEMPT]
	awk -v exempt="${1:-}" '
		BEGIN { n = split(exempt, e, " "); for (i = 1; i <= n; i++) skip[e[i]] = 1 }
		{
			line = $0
			sub(/^ *"command": "/, "", line)
			sub(/",?$/, "", line)
			m = split(line, tok, " ")
			src = ""
			for (i = 1; i < m; i++) if (tok[i] == "-c") src = tok[i + 1]
			for (i = 1; i <= m; i++) {
				t = tok[i]
				if (t in skip) continue
				if (t == "-w" || t ~ /^-Werror(=.*)?$/ || t ~ /^-Wno-./) { print src; break }
			}
		}' | sort -u
}

# scan_build_log LOG SRC_ROOT BIN_DIR
# The pre-push compiler-warning decision on a build log that the gate probe
# TU was compiled into. Returns 2 when the probe's -Wswitch is absent (the
# scan cannot see this build's compiler output, so a clean result would mean
# nothing); 1, printing them, when first-party warnings other than the
# probe's remain; 0 when the log is clean apart from the control.
scan_build_log() {
	local all
	all=$(first_party_warnings "$1" "$2" "$3")
	if ! echo "$all" | grep -F 'tidy-probe/incomplete_switch.cpp' | grep -qF -- '-Wswitch'; then
		return 2
	fi
	local rest
	rest=$(echo "$all" | grep -vF 'tidy-probe/incomplete_switch.cpp' | sed '/^$/d')
	if [ -n "$rest" ]; then
		echo "$rest"
		return 1
	fi
	return 0
}

# hook_files_that_are_code: of the changed paths on stdin, the .githooks/ files
# a push must still build and test for, though the hooks directory is otherwise
# treated as not-code by pre-push. The probe and its lib feed a CMake target and
# two ctests; pre-push itself holds the compiler-warning scan, which only a
# build exercises.
hook_files_that_are_code() {
	grep -E '^\.githooks/(tidy-probe|gate-lib|tests/|pre-push$)' || true
}

# probe_trigger_files: of the changed paths on stdin, those that can change
# what the clang-tidy gate sees, so pre-push runs tidy-probe.sh for them.
probe_trigger_files() {
	grep -E '(^|/)\.clang-tidy$|(^|/)CMakeLists\.txt$|^\.githooks/(tidy-probe|gate-lib\.sh$)' || true
}

# restore_probe_exclusion_in BUILD_DIR
# Reconfigure BUILD_DIR without CBLS_GATE_PROBE_IN_ALL, so the dirty probe TU
# is out of its default build again. Idempotent; a missing build dir is fine.
restore_probe_exclusion_in() {
	unset CBLS_GATE_PROBE_IN_ALL
	[ -f "$1/CMakeCache.txt" ] || return 0
	(unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE && cmake "$1") >/dev/null 2>&1 ||
		echo "note: could not reconfigure $1 without the gate probe"
}
