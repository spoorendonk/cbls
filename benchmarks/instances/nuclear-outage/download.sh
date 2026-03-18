#!/usr/bin/env bash
# Download ROADEF 2010 nuclear outage scheduling instances.
# Source: https://www.roadef.org/challenge/2010/en/
set -euo pipefail

cd "$(dirname "$0")"

BASE_URL="https://www.roadef.org/challenge/2010/files"

echo "=== Downloading ROADEF 2010 Nuclear Outage instances ==="

# Instance set A (A1-A5, qualification)
if [ ! -f data_release_13102009.zip ]; then
    echo "Downloading instance set A..."
    wget -q "${BASE_URL}/data_release_13102009.zip" || \
        curl -sLO "${BASE_URL}/data_release_13102009.zip"
fi

# Instance set B (B1-B10, final round)
if [ ! -f dataB-chall.zip ]; then
    echo "Downloading instance set B..."
    wget -q "${BASE_URL}/dataB-chall.zip" || \
        curl -sLO "${BASE_URL}/dataB-chall.zip"
fi

# Official checker
if [ ! -f CHECKER.zip ]; then
    echo "Downloading official checker..."
    wget -q "${BASE_URL}/CHECKER.zip" || \
        curl -sLO "${BASE_URL}/CHECKER.zip"
fi

# Problem specification PDF
if [ ! -f sujetEDFv22.pdf ]; then
    echo "Downloading problem specification..."
    wget -q "${BASE_URL}/sujetEDFv22.pdf" || \
        curl -sLO "${BASE_URL}/sujetEDFv22.pdf"
fi

# Extract
echo "Extracting archives..."
for f in data_release_13102009.zip dataB-chall.zip CHECKER.zip; do
    if [ -f "$f" ]; then
        unzip -qo "$f" -d "${f%.zip}" 2>/dev/null || true
    fi
done

echo "Done. Instance files are in:"
ls -d data_release_13102009/ dataB-chall/ CHECKER/ 2>/dev/null || true
