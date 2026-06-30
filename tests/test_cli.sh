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
a2a_assert_grep "repl \\[cells\\]" "$out" "usage shows repl arguments"
a2a_assert_grep "field \\[cells\\].*\\[qloop\\].*\\[kvpos\\]" "$out" "usage shows field qloop/kvpos arguments"

out="$("$A2A_BIN" /tmp/arianna2arianna-missing.gguf 2>&1 || true)"
a2a_assert_grep "gguf: cannot open" "$out" "missing model path is rejected"

set +e
out="$(env -u OPENAI_API_KEY -u OPENAI_API_KEY_FILE bash "$A2A_ROOT/tools/openai_repl_probe.sh" 2>&1)"
got=$?
set -e
if [[ "$got" -eq 2 ]]; then
    a2a_ok "openai repl probe refuses to run without API key"
else
    a2a_fail "openai repl probe missing-key exit (wanted 2, got $got)"
fi
a2a_assert_grep "OPENAI_API_KEY" "$out" "openai repl probe names key env without printing a key"
