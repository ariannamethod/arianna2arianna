#!/usr/bin/env bash
# Packed f16/q8 inference smoke.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_inference ---"

if a2a_have_f16; then
    out="$("$A2A_BIN" "$A2A_MODEL_F16" "ping" 4 0 2>&1)"
    a2a_assert_grep "f16-packed" "$out" "f16 model loads with packed banner"
    a2a_assert_grep "decode:" "$out" "f16 decode completes"
    a2a_assert_grep "t/s" "$out" "f16 reports tokens/sec"
else
    a2a_skip "f16 weights missing ($A2A_MODEL_F16)"
fi

if a2a_have_q8; then
    out="$("$A2A_BIN" "$A2A_MODEL_Q8" "ping" 4 0 2>&1)"
    a2a_assert_grep "q8-packed" "$out" "q8 model loads with packed banner"
    a2a_assert_grep "decode:" "$out" "q8 decode completes"
else
    a2a_skip "q8 weights missing ($A2A_MODEL_Q8)"
fi

if a2a_have_f16; then
    out="$("$A2A_BIN" --16 "hi" 3 0 2>&1)"
    a2a_assert_grep "f16-packed" "$out" "--16 HF path works"
else
    a2a_skip "--16 (no local f16 weights; run make weights)"
fi