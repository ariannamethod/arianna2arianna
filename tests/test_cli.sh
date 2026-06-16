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
a2a_assert_grep "field \\[cells\\].*\\[qloop\\].*\\[kvpos\\]" "$out" "usage shows field qloop/kvpos arguments"

out="$("$A2A_BIN" /tmp/arianna2arianna-missing.gguf 2>&1 || true)"
a2a_assert_grep "gguf: cannot open" "$out" "missing model path is rejected"
