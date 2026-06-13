#!/usr/bin/env bash
# arianna2arianna integration test runner.
# Usage: bash tests/run.sh   or: make test

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/lib.sh"

echo "arianna2arianna tests"
echo "  bin:  $A2A_BIN"
echo "  root: $A2A_ROOT"
echo ""

a2a_ensure_built

# source (not bash) so pass/fail counters accumulate in lib.sh
source "$DIR/test_cli.sh"
source "$DIR/test_inference.sh"
source "$DIR/test_field.sh"
source "$DIR/test_portable.sh"

a2a_summary