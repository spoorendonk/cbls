#!/usr/bin/env bash
#
# Set-covering A/B: the `Set` encoding under two structural-selection policies
# (#165). Same instances, same seeds, same budget; the only difference is the
# policy.
#
#   benchmarks/setcover/ab_selection.sh                     # 5 seeds, 10s, whole roster
#   benchmarks/setcover/ab_selection.sh --seeds 5 --time 10 --out results/ab
#   benchmarks/setcover/ab_selection.sh --arms first_improving,best_of_sample
#
# RUN IT SERIALLY, ON AN IDLE MACHINE. Objective quality at a fixed wall-clock
# budget is a function of how many iterations the process gets, so two arms
# sharing cores produce numbers that are not comparable to each other or to the
# committed table. The script runs one solve at a time for exactly that reason
# and refuses to help you parallelise it; check `uptime` first, and re-run
# anything anomalous.
#
# Output: one CSV per arm plus a merged `ab_selection.csv`, all per-seed --- the
# runner's own `--csv` rows, which carry the arm in their `selection` column,
# the verifier's verdict, the wall time and the iteration count. A summary is
# printed, not committed: per CLAUDE.md the reference values (the published
# optima, which `download.py` carries) are the artifact, and an engine-vs-engine
# comparison is re-derived from the result records whenever the question is
# asked.
#
# READ THE RESULT WITH THIS CAVEAT. At a fixed wall-clock budget the comparison
# is policy quality MINUS representation cost, and the two arms do not pay the
# same representation cost: `best_of_sample` / `violation_guided` score up to
# `--sample-size` candidates per variable per pass where `first_improving`
# scores the generator's 3-5.
#
# The RECORDED null result predates #164, which made a candidate much cheaper --
# a structured change is now a positional edit rather than a whole element
# vector, so scoring one costs no allocation and no O(|elements|) copy where it
# used to cost two of each. That does not make the two arms equal, but it shrinks
# the term that confounded them, so the A/B is worth re-running before the null
# is read as "the policy does not help" rather than "the policy helps less than
# its per-candidate cost". Say which you measured.
#
# Quote the result with the engine commit the script prints.
set -euo pipefail

bin="build/cbls_setcover"
dir="benchmarks/instances/setcover"
time_limit="10"
seeds="5"
first_seed="42"
arms="first_improving,violation_guided"
out=""

usage() {
	cat <<'USAGE'
usage: ab_selection.sh [--bin PATH] [--dir DIR] [--time SECONDS] [--seeds N]
                       [--seed FIRST] [--arms a,b[,c]] [--out DIR]
USAGE
}

while [ $# -gt 0 ]; do
	case "$1" in
	--bin)
		bin="${2:?--bin needs a path}"
		shift 2
		;;
	--dir)
		dir="${2:?--dir needs a path}"
		shift 2
		;;
	--time)
		time_limit="${2:?--time needs seconds}"
		shift 2
		;;
	--seeds)
		seeds="${2:?--seeds needs a count}"
		shift 2
		;;
	--seed)
		first_seed="${2:?--seed needs a value}"
		shift 2
		;;
	--arms)
		arms="${2:?--arms needs a comma-separated list}"
		shift 2
		;;
	--out)
		out="${2:?--out needs a directory}"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "unknown argument '$1'" >&2
		usage >&2
		exit 2
		;;
	esac
done

if [ ! -x "$bin" ]; then
	echo "no runner at '$bin' -- build it first:" >&2
	echo "  cmake -B build && cmake --build build -j\$(nproc) --target cbls_setcover" >&2
	exit 2
fi
if [ ! -d "$dir" ]; then
	echo "no instance directory at '$dir'" >&2
	exit 2
fi
if [ -z "$out" ]; then
	out="results/setcover-ab-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$out"

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
# `git diff --quiet` exits 1 for a dirty tree and 128+ outside a repository, so
# only 1 means dirty. Untracked files are not considered -- the question this
# answers is whether the binary matches the named commit.
dirty=""
git diff --quiet HEAD -- 2>/dev/null || [ $? -ne 1 ] || dirty=" (working tree dirty)"

{
	echo "engine commit: ${commit}${dirty}"
	echo "date (UTC):    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "host:          $(uname -srm) / $(hostname)"
	echo "load at start: $(uptime)"
	echo "budget:        ${time_limit}s per run, ${seeds} seeds from ${first_seed}"
	echo "arms:          ${arms}"
} | tee "$out/run.txt"

merged="$out/ab_selection.csv"
: >"$merged"

# An unverified or infeasible run makes the runner exit 1. That is a finding
# about the arm, not a reason to abandon the other one, so the status is
# recorded and the sweep continues -- with the script exiting non-zero at the
# end so a caller cannot read a partial comparison as a clean one.
failed_arms=""
IFS=',' read -r -a arm_list <<<"$arms"
# `--arms ""` never reaches here -- `${2:?...}` at parse time fires on null as
# well as unset. What DOES get through is an empty FIELD, as in `--arms ","` or
# a trailing comma, which would hand the runner an arm named "".
for arm in ${arm_list[@]+"${arm_list[@]}"}; do
	if [ -z "$arm" ]; then
		echo "empty arm name in --arms '$arms'" >&2
		exit 2
	fi
done
for arm in "${arm_list[@]}"; do
	echo
	echo "=== arm: $arm ==="
	csv="$out/$arm.csv"
	# One process at a time, `set` encoding only: the Bool encoding has no
	# structural batch at all (build_bool_model creates no List or Set), so it
	# would be the same run in both arms.
	status=0
	"$bin" --dir "$dir" --encoding set --selection "$arm" \
		--time "$time_limit" --seeds "$seeds" --seed "$first_seed" --csv "$csv" || status=$?
	# The runner exits 2 on a bad flag value and 1 on infeasible/unverified runs.
	# Reporting a mistyped --arms as "infeasible" would send the reader looking
	# at the search instead of at the command line, so the two are kept apart.
	if [ "$status" -eq 2 ]; then
		echo "arm '$arm' was rejected by the runner (bad --selection value?)" >&2
		failed_arms="$failed_arms $arm"
		continue
	fi
	if [ "$status" -ne 0 ]; then
		echo "arm '$arm' reported infeasible or unverified runs" >&2
		failed_arms="$failed_arms $arm"
	fi
	if [ ! -s "$csv" ]; then
		echo "arm '$arm' wrote no rows" >&2
		failed_arms="$failed_arms $arm"
		continue
	fi
	if [ ! -s "$merged" ]; then
		head -n 1 "$csv" >"$merged"
	fi
	tail -n +2 "$csv" >>"$merged"
done

echo
echo "=== per arm: mean gap% / best objective / feasible-and-verified runs ==="
# Only verified-feasible rows are aggregated. An infeasible run has an
# objective that is not a cover's cost, so averaging it in would report a
# number for a solution that does not exist -- which is how a short budget
# makes a bad arm look good.
awk -F, 'NR > 1 {
    arm = $3; inst = $1
    if (!(arm in arms_seen)) { arms_order[++arm_n] = arm; arms_seen[arm] = 1 }
    if (!(inst in inst_seen)) { inst_order[++inst_n] = inst; inst_seen[inst] = 1 }
    key = inst SUBSEP arm
    total[key]++
    if ($8 == 1 && $9 == 1) {
        sum[key] += $7
        ok[key]++
        if (!(key in best) || $5 + 0 < best[key]) best[key] = $5 + 0
    }
}
END {
    printf "%-10s", "instance"
    for (a = 1; a <= arm_n; a++) printf "  %24s", arms_order[a]
    printf "\n"
    for (i = 1; i <= inst_n; i++) {
        printf "%-10s", inst_order[i]
        for (a = 1; a <= arm_n; a++) {
            key = inst_order[i] SUBSEP arms_order[a]
            if (ok[key] > 0)
                printf "  %9.1f%% %7.0f %2d/%-2d", sum[key] / ok[key], best[key], ok[key], total[key]
            else
                printf "  %24s", "none feasible"
        }
        printf "\n"
    }
}' "$merged" | tee -a "$out/run.txt"

echo
echo "per-seed rows: $merged"
echo "run record:    $out/run.txt"
if [ -n "$failed_arms" ]; then
	echo "arms with infeasible/unverified runs:$failed_arms" >&2
	exit 1
fi
