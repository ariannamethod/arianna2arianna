#!/usr/bin/env bash
# δ-field, resonance control, and δ-life smoke tests.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_field ---"

if ! a2a_have_f16; then
    a2a_skip "field tests need f16 weights"
    return 0 2>/dev/null || exit 0
fi

field_out="$("$A2A_BIN" "$A2A_MODEL_F16" "resonance" field 3 4 2 0 2 0.25 2>&1)"
a2a_assert_grep "δ-field: 3 cells × 2 rounds" "$field_out" "field starts with requested shape"
a2a_assert_grep "floor \\(sampling noise" "$field_out" "field reports sampling floor"
a2a_assert_grep "round 1:" "$field_out" "field reports round 1 metrics"
a2a_assert_grep "d_R" "$field_out" "field reports d_R"
a2a_assert_grep "Δ_R\\(text n/a\\)" "$field_out" "chorus marks text delta as n/a"
a2a_assert_grep "Δ_R\\^kv" "$field_out" "chorus reports KV-order delta"
a2a_assert_grep "margin" "$field_out" "KV-order delta reports floor margin"
a2a_assert_grep "I_N\\^kv" "$field_out" "chorus reports neighbour influence"
a2a_assert_grep "Dpos" "$field_out" "field reports positional dissonance"
a2a_assert_grep "δ-field done" "$field_out" "field completes"

if [[ -f "$A2A_ROOT/FIELDLOG.md" ]] && grep -q "resonance" "$A2A_ROOT/FIELDLOG.md"; then
    a2a_ok "field appends FIELDLOG.md"
else
    a2a_fail "field did not append FIELDLOG.md"
fi

restest_out="$("$A2A_BIN" "$A2A_MODEL_F16" "resonance" restest 3 4 2 2>&1)"
a2a_assert_grep "resonance test: coherent vs same-length SHUFFLED" "$restest_out" "restest starts"
a2a_assert_grep "COHERENT" "$restest_out" "restest includes coherent arm"
a2a_assert_grep "SHUFFLED" "$restest_out" "restest includes shuffled arm"
a2a_assert_grep "resonance-beyond-length" "$restest_out" "restest reports control delta"

sweep_prompts="$(mktemp)"
printf "Let the cells remember each other.\n" > "$sweep_prompts"
sweep_out="$(A2A_CELLS=3 A2A_FRAG=4 A2A_ROUNDS=1 bash "$A2A_ROOT/tools/kv_influence_sweep.sh" "$sweep_prompts" 2>&1)"
rm -f "$sweep_prompts"
a2a_assert_grep "^prompt[[:space:]]+mode[[:space:]]+avg_entropy" "$sweep_out" "influence sweep reports TSV header"
a2a_assert_grep "Let the cells remember each other\\.[[:space:]]+sem[[:space:]]" "$sweep_out" "influence sweep reports semantic row"
if printf "%s\n" "$sweep_out" | awk -F '\t' '$1 == "Let the cells remember each other." && ($7 + 0) > 0 { ok = 1 } END { exit ok ? 0 : 1 }'; then
    a2a_ok "memory prompt sharpens under semantic neighbour"
else
    a2a_fail "memory prompt did not sharpen under semantic neighbour"
fi

life_out="$("$A2A_BIN" "$A2A_MODEL_F16" "resonance" life 2 4 3 2>&1)"
a2a_assert_grep "δ-life: Game of Life" "$life_out" "life starts"
a2a_assert_grep "births" "$life_out" "life reports births"
a2a_assert_grep "deaths" "$life_out" "life reports deaths"
a2a_assert_grep "pop ended" "$life_out" "life completes"
