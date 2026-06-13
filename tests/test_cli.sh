#!/usr/bin/env bash
# CLI and flag parsing (no model required for usage errors).

set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_cli ---"

a2a_assert_exit 1 "\"$A2A_BIN\"" "no args prints usage and exits 1"

out="$("$A2A_BIN" 2>&1 || true)"
a2a_assert_grep "usage:" "$out" "usage banner present"

out="$("$A2A_BIN" --nope 2>&1 || true)"
a2a_assert_grep "unknown flag" "$out" "unknown flag rejected"

out="$("$A2A_BIN" --theta-lo 0.1 --theta-hi 0.9 2>&1 || true)"
a2a_assert_grep "usage:" "$out" "flags without model still need model path"