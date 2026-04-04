#!/bin/bash
# Validate a ROADEF 2010 solution using the official checker
# Usage: check_solution.sh <data_file> <solution_file>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER_DIR="${SCRIPT_DIR}/../instances/nuclear-outage/CHECKER"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_file> <solution_file>"
    echo "Example: $0 data0.txt solution0.txt"
    exit 1
fi

DATA_FILE="$1"
SOLUTION_FILE="$2"

# Find checker.jar
CHECKER_JAR=""
for f in "$CHECKER_DIR"/checker.jar "$CHECKER_DIR"/*.jar; do
    if [ -f "$f" ]; then
        CHECKER_JAR="$f"
        break
    fi
done

if [ -z "$CHECKER_JAR" ]; then
    echo "ERROR: checker.jar not found in $CHECKER_DIR"
    echo "Place the official checker.jar in: $CHECKER_DIR/"
    exit 1
fi

echo "Data file:     $DATA_FILE"
echo "Solution file: $SOLUTION_FILE"
echo "Checker:       $CHECKER_JAR"
echo "---"

java -Xmx512m -jar "$CHECKER_JAR" -d "$DATA_FILE" -r "$SOLUTION_FILE"

# Show results
if [ -f constraint_check.txt ]; then
    echo "--- Constraint check results ---"
    cat constraint_check.txt
fi

if [ -f objective_check.txt ]; then
    echo "--- Objective check results ---"
    cat objective_check.txt
fi
