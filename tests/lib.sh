#!/usr/bin/env bash
# Shared helpers for arianna2arianna integration tests.

set -euo pipefail

A2A_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A2A_BIN="${A2A_BIN:-$A2A_ROOT/arianna2arianna}"
A2A_MODEL_F16="${A2A_MODEL_F16:-$A2A_ROOT/weights/nano_arianna_resft_2026_07_09_f16.gguf}"
A2A_MODEL_Q8="${A2A_MODEL_Q8:-$A2A_ROOT/weights/nano_arianna_resft_2026_07_09_q8_0.gguf}"

if [[ -z "${A2A_LIB_LOADED:-}" ]]; then
    A2A_PASS=0
    A2A_FAIL=0
    A2A_SKIP=0
    A2A_LIB_LOADED=1
fi

a2a_ensure_built() {
    if [[ ! -x "$A2A_BIN" ]]; then
        echo "building $A2A_BIN ..."
        make -C "$A2A_ROOT" >/dev/null
    fi
}

a2a_have_f16() {
    [[ -f "$A2A_MODEL_F16" ]] && [[ "$(stat -f%z "$A2A_MODEL_F16" 2>/dev/null || stat -c%s "$A2A_MODEL_F16")" -gt 1000000 ]]
}

a2a_have_q8() {
    [[ -f "$A2A_MODEL_Q8" ]] && [[ "$(stat -f%z "$A2A_MODEL_Q8" 2>/dev/null || stat -c%s "$A2A_MODEL_Q8")" -gt 1000000 ]]
}

a2a_ok() {
    echo "  OK   $1"
    A2A_PASS=$((A2A_PASS + 1))
}

a2a_fail() {
    echo "  FAIL $1" >&2
    A2A_FAIL=$((A2A_FAIL + 1))
}

a2a_skip() {
    echo "  SKIP $1"
    A2A_SKIP=$((A2A_SKIP + 1))
}

a2a_assert_exit() {
    local want="$1" cmd="$2" msg="$3"
    set +e
    eval "$cmd" >/dev/null 2>&1
    local got=$?
    set -e
    if [[ "$got" -eq "$want" ]]; then
        a2a_ok "$msg (exit $got)"
    else
        a2a_fail "$msg (wanted exit $want, got $got)"
    fi
}

a2a_assert_grep() {
    local pattern="$1" hay="$2" msg="$3"
    if echo "$hay" | grep -qE "$pattern"; then
        a2a_ok "$msg"
    else
        a2a_fail "$msg (pattern /$pattern/ not found)"
    fi
}

a2a_summary() {
    echo ""
    echo "=== summary: $A2A_PASS passed, $A2A_FAIL failed, $A2A_SKIP skipped ==="
    [[ "$A2A_FAIL" -eq 0 ]]
}
