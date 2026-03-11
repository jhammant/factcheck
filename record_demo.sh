#!/bin/bash
# Record demo output for README
echo "🔍 FactCheck Demo"
echo "================="
echo ""

echo "$ factcheck verify 'The capital of Australia is Sydney' -v"
echo ""
DIR="$(dirname "$0")"
cd "$DIR"
PYTHON="$DIR/.venv/bin/python"

$PYTHON -m factcheck.cli verify "The capital of Australia is Sydney" -v 2>&1

echo ""
echo "---"
echo ""

echo "$ factcheck verify 'Marie Curie won two Nobel Prizes' -v"
echo ""
$PYTHON -m factcheck.cli verify "Marie Curie won two Nobel Prizes" -v 2>&1

echo ""
echo "---"
echo ""

echo "$ factcheck verify 'Shakespeare wrote War and Peace' -v"
echo ""
$PYTHON -m factcheck.cli verify "Shakespeare wrote War and Peace" -v 2>&1
