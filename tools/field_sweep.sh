#!/usr/bin/env bash
# Sweep prompts through full field mode and summarize final-round field metrics.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${A2A_BIN:-$ROOT/arianna2arianna}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"
PROMPTS="${1:-$ROOT/prompts/kv_influence.txt}"

CELLS="${A2A_CELLS:-4}"
FRAG="${A2A_FRAG:-12}"
ROUNDS="${A2A_ROUNDS:-3}"
ALPHA="${A2A_ALPHA:-0}"
LEAP="${A2A_LEAP:-2}"
XCELL="${A2A_XCELL:-0.05}"
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

printf "prompt\tmode\tcells\tfrag\trounds\tavg_entropy\td_r\td_floor\td_margin\tkv_delta\tkv_floor\tkv_margin\tkv_influence\tdisso\tdpos\tqloop_routes\tqloop_kv_routes\tqloop_triggers\n"

while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    [[ -z "$prompt" || "${prompt:0:1}" == "#" ]] && continue

    out="$("$BIN" "$MODEL" "$prompt" field "$CELLS" "$FRAG" "$ROUNDS" "$ALPHA" "$LEAP" "$XCELL" "$CHORUS" "$XREP" "$LIFE" "$KVSHUF" "$QLOOP" "$KVPOS" 2>&1)"
    line="$(printf "%s\n" "$out" | grep "→ round" | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
        safe_prompt="${prompt//$'\t'/ }"
        printf "%s\tERROR\t%s\t%s\t%s\t\t\t\t\t\t\t\t\t\t\t\t\t\n" "$safe_prompt" "$CELLS" "$FRAG" "$ROUNDS"
        continue
    fi

    qloop_routes="$(printf "%s\n" "$out" | grep -Ec "↳ qloop (c[0-9]|user)" || true)"
    qloop_kv_routes="$(printf "%s\n" "$out" | grep -Ec "↳ qloop (c[0-9]|user).*\\[(user-)?kv\\]" || true)"
    qloop_triggers="$(printf "%s\n" "$out" | grep -c "↳ qloop trigger" || true)"

    safe_prompt="${prompt//$'\t'/ }"
    printf "%s\n" "$line" | awk -v prompt="$safe_prompt" -v cells="$CELLS" -v frag="$FRAG" -v rounds="$ROUNDS" \
        -v qroutes="$qloop_routes" -v qkv="$qloop_kv_routes" -v qtrig="$qloop_triggers" '
        BEGIN { FS = "|" }
        {
            avg = $1
            sub(/.*avg entropy /, "", avg)
            sub(/ .*/, "", avg)

            dr = $2
            d_floor = dr
            sub(/.*d_R[[:space:]]+/, "", dr)
            sub(/[[:space:]]+.*/, "", dr)
            if (dr == "—" || dr == "-") dr = "nan"
            sub(/.*floor /, "", d_floor)
            sub(/[)].*/, "", d_floor)
            d_margin = (dr == "nan" || d_floor == "") ? "nan" : sprintf("%+.3f", dr - d_floor)

            kv = $4
            if (kv ~ /off/) {
                mode = "off"
                delta = "nan"
                kfloor = "nan"
                kmargin = "nan"
                infl = "nan"
                disso = $5
                dpos = $6
            } else {
                mode = kv
                sub(/.*kv\[/, "", mode)
                sub(/\].*/, "", mode)

                delta = kv
                sub(/.*\] /, "", delta)
                sub(/ .*/, "", delta)

                kfloor = kv
                sub(/.*floor /, "", kfloor)
                sub(/ .*/, "", kfloor)

                kmargin = kv
                sub(/.*margin /, "", kmargin)
                sub(/\).*/, "", kmargin)

                infl = $5
                sub(/.*\] /, "", infl)
                sub(/ .*/, "", infl)

                disso = $6
                dpos = $7
            }
            sub(/.*D_R /, "", disso)
            sub(/ .*/, "", disso)

            sub(/.*Dpos /, "", dpos)
            sub(/ .*/, "", dpos)

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                prompt, mode, cells, frag, rounds, avg, dr, d_floor, d_margin,
                delta, kfloor, kmargin, infl, disso, dpos, qroutes, qkv, qtrig
        }
    '
done < "$PROMPTS"
