#!/usr/bin/env bash
# Run recipient-lock probes and write TSV + summary under runs/.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"
PROMPTS="${1:-${A2A_RECIPIENT_PROMPTS:-$ROOT/prompts/recipient_lock.txt}}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"

usage() {
    cat <<EOF
usage: $0 [prompts.txt]

Writes a recipient-lock TSV and summary. This checks for accidental Oleg/Олег
recipient leakage in normal generation.

Environment:
  A2A_MODEL              default: $MODEL
  A2A_RUN_DIR            default: $OUTDIR
  A2A_RECIPIENT_TOKENS   forwarded to recipient_lock_sweep.sh
  A2A_RECIPIENT_TEMP     forwarded to recipient_lock_sweep.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$PROMPTS" ]]; then
    echo "missing prompts file: $PROMPTS" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
stamp="$(date +%Y%m%d_%H%M%S)"
prompt_base="$(basename "$PROMPTS")"
prompt_stem="${prompt_base%.*}"
model_stem="$(basename "$MODEL")"
model_stem="${model_stem%.gguf}"
model_stem="$(printf "%s" "$model_stem" | tr -c 'A-Za-z0-9._-' '_')"
tsv_file="$OUTDIR/recipient_lock_${prompt_stem}_${model_stem}_${stamp}.tsv"
summary_file="${tsv_file%.tsv}.summary.txt"

echo "model:   $MODEL" >&2
echo "prompts: $PROMPTS" >&2
echo "writing: $tsv_file" >&2

A2A_MODEL="$MODEL" bash "$ROOT/tools/recipient_lock_sweep.sh" "$PROMPTS" | tee "$tsv_file"

awk -F '\t' '
    NR == 1 { next }
    {
        rows++
        leaks += $2 + 0
        mentions += $3 + 0
        if (($2 + 0) > 0 && examples < 5) {
            examples++
            leak_examples = leak_examples sprintf("leak_example_%d: %s :: %s\n", examples, $1, $4)
        }
    }
    END {
        printf "rows: %d\n", rows
        printf "recipient_lock_leaks: %d/%d\n", leaks, rows
        printf "oleg_mentions: %d\n", mentions
        if (leak_examples != "") printf "%s", leak_examples
    }
' "$tsv_file" | tee "$summary_file"

echo "summary: $summary_file" >&2
