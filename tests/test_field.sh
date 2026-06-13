#!/usr/bin/env bash
# δ-field: JSON telemetry, sweep CSV, leap modes.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_field ---"

if ! a2a_have_f16; then
    a2a_skip "field tests need f16 weights"
    return 0 2>/dev/null || exit 0
fi

# --json: one JSON object per round on stdout
json_out="$("$A2A_BIN" --16 --quiet --json "resonance" field 3 6 2 0 0 0 2>&1)"
a2a_assert_grep '"round":1' "$json_out" "field --json emits round 1"
a2a_assert_grep '"deltaR":' "$json_out" "field --json includes deltaR"
a2a_assert_grep '"dR":' "$json_out" "field --json includes dR"

json_lines="$(echo "$json_out" | grep -c '^{' || true)"
if [[ "$json_lines" -ge 2 ]]; then
    a2a_ok "field --json prints >=2 round lines ($json_lines)"
else
    a2a_fail "field --json expected 2 round lines, got $json_lines"
fi

# sweep: header + seeds × leaps rows
sweep_out="$("$A2A_BIN" --16 --quiet "resonance" sweep 2 3 6 2 0 0 2>&1)"
a2a_assert_grep '^seed,leap,kv_beta' "$sweep_out" "sweep CSV header"
sweep_rows="$(echo "$sweep_out" | grep -cE '^[0-9]+,[0-3],' || true)"
if [[ "$sweep_rows" -eq 8 ]]; then
    a2a_ok "sweep 2 seeds × 4 leaps = 8 rows"
else
    a2a_fail "sweep expected 8 data rows, got $sweep_rows"
fi

# leap mode 2 should not crash with kv_beta
leap_out="$("$A2A_BIN" --16 --quiet "resonance" field 3 6 1 0 2 0.25 2>&1)"
a2a_assert_grep "f16-packed" "$leap_out" "leap=2 kv_beta=0.25 runs"