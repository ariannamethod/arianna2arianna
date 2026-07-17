#!/usr/bin/env bash
# Packed f16/q8 inference smoke.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_inference ---"

if a2a_have_f16; then
    out="$("$A2A_BIN" "$A2A_MODEL_F16" "ping" 4 0 2>&1)"
    a2a_assert_grep "model: arch=(llama|nlama)" "$out" "f16 model loads"
    if echo "$out" | grep -q "model: arch=nlama"; then
        a2a_assert_grep "NEOX rope" "$out" "nlama f16 uses nanollama RoPE pairing"
    fi
    a2a_assert_grep "packed linear" "$out" "f16 uses packed linear weights"
    a2a_assert_grep "0 dense fallback" "$out" "f16 has no dense linear fallback"
    a2a_assert_grep "loaded in" "$out" "f16 reports load time"
    a2a_assert_grep 'prompt: "ping"' "$out" "f16 prompt path runs"
    a2a_assert_grep "decode:" "$out" "f16 decode completes"
    a2a_assert_grep "t/s" "$out" "f16 reports tokens/sec"
else
    a2a_skip "f16 weights missing ($A2A_MODEL_F16)"
fi

if a2a_have_q8; then
    out="$("$A2A_BIN" "$A2A_MODEL_Q8" "ping" 4 0 2>&1)"
    a2a_assert_grep "model: arch=(llama|nlama)" "$out" "q8 model loads"
    if echo "$out" | grep -q "model: arch=nlama"; then
        a2a_assert_grep "NEOX rope" "$out" "nlama q8 uses nanollama RoPE pairing"
    fi
    a2a_assert_grep "packed linear" "$out" "q8 uses packed linear weights"
    a2a_assert_grep "decode:" "$out" "q8 decode completes"
else
    a2a_skip "q8 weights missing ($A2A_MODEL_Q8)"
fi
