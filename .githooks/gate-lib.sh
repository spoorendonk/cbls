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
cancelled_warning_flags() {
	grep -E -- ' (-w|-Werror(=[^ "]*)?|-Wno-[^ "]+)( |")' | command_source | sort -u
}
