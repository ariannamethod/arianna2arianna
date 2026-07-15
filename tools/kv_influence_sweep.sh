#!/usr/bin/env bash
# Sweep prompts through the semantic neighbour diagnostic and print one TSV row per prompt.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${A2A_BIN:-$ROOT/arianna2arianna}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"
PROMPTS="${1:-$ROOT/prompts/kv_influence.txt}"

CELLS="${A2A_CELLS:-4}"
FRAG="${A2A_FRAG:-12}"
ROUNDS="${A2A_ROUNDS:-1}"
ALPHA="${A2A_ALPHA:-0}"
LEAP="${A2A_LEAP:-2}"
XCELL="${A2A_XCELL:-0.02}"
CHORUS="${A2A_CHORUS:-1}"
XREP="${A2A_XREP:-1.3}"
LIFE="${A2A_LIFE:-0}"
KVSHUF="${A2A_KVSHUF:-1}"
QLOOP="${A2A_QLOOP:-2}"
KVPOS="${A2A_KVPOS:-0}"

if [[ ! -x "$BIN" ]]; then
    make -C "$ROOT" >/dev/null
fi

if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

if [[ ! -f "$PROMPTS" ]]; then
    echo "missing prompts file: $PROMPTS" >&2
    exit 1
fi

printf "prompt\tmode\tavg_entropy\tkv_delta\tkv_floor\tkv_margin\tkv_influence\tdisso\tdpos\n"

while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    [[ -z "$prompt" || "${prompt:0:1}" == "#" ]] && continue

    out="$("$BIN" "$MODEL" "$prompt" field "$CELLS" "$FRAG" "$ROUNDS" "$ALPHA" "$LEAP" "$XCELL" "$CHORUS" "$XREP" "$LIFE" "$KVSHUF" "$QLOOP" "$KVPOS" 2>&1)"
    line="$(printf "%s\n" "$out" | grep "I_N\\^kv" | grep "round" | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
        safe_prompt="${prompt//$'\t'/ }"
        printf "%s\tERROR\t\t\t\t\t\t\t\n" "$safe_prompt"
        continue
    fi

    safe_prompt="${prompt//$'\t'/ }"
    printf "%s\n" "$line" | awk -v prompt="$safe_prompt" '
        BEGIN { FS = "|" }
        {
            avg = $1
            sub(/.*avg entropy /, "", avg)
            sub(/ .*/, "", avg)

            kv = $4
            mode = kv
            sub(/.*kv\[/, "", mode)
            sub(/\].*/, "", mode)

            delta = kv
            sub(/.*\] /, "", delta)
            sub(/ .*/, "", delta)

            floor = kv
            sub(/.*floor /, "", floor)
            sub(/ .*/, "", floor)

            margin = kv
            sub(/.*margin /, "", margin)
            sub(/\).*/, "", margin)

            infl = $5
            sub(/.*\] /, "", infl)
            sub(/ .*/, "", infl)

            disso = $6
            sub(/.*D_R /, "", disso)
            sub(/ .*/, "", disso)

            dpos = $7
            sub(/.*Dpos /, "", dpos)
            sub(/ .*/, "", dpos)

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", prompt, mode, avg, delta, floor, margin, infl, disso, dpos
        }
    '
done < "$PROMPTS"
