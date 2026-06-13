#!/usr/bin/env bash
# Scalar-only portable build smoke.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

echo "--- test_portable ---"

if ! a2a_have_f16; then
    a2a_skip "portable inference needs f16 weights"
    return 0 2>/dev/null || exit 0
fi

make -C "$A2A_ROOT" portable >/dev/null
PORT_BIN="$A2A_ROOT/arianna2arianna"
if [[ ! -x "$PORT_BIN" ]]; then
    a2a_fail "portable build did not produce binary"
    return 0 2>/dev/null || exit 0
fi

out="$("$PORT_BIN" "$A2A_MODEL_F16" "ok" 2 0 2>&1)"
a2a_assert_grep "f16-packed" "$out" "portable binary runs inference"
a2a_assert_grep "decode:" "$out" "portable decode completes"

# restore SIMD build for other tests
make -C "$A2A_ROOT" >/dev/null