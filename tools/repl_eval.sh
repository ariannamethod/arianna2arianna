#!/usr/bin/env bash
# Run the offline REPL probe corpus, write TSV results, then summarize them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"
PROMPTS="${1:-${A2A_EVAL_PROMPTS:-$ROOT/prompts/repl_probe_regression.txt}}"

CELLS="${A2A_EVAL_CELLS:-${A2A_CELLS:-3}}"
FRAG="${A2A_EVAL_FRAG:-${A2A_FRAG:-4}}"
ROUNDS="${A2A_EVAL_ROUNDS:-${A2A_ROUNDS:-1}}"

if [[ ! -f "$PROMPTS" ]]; then
    echo "missing prompts file: $PROMPTS" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
stamp="$(date +%Y%m%d_%H%M%S)"
prompt_base="$(basename "$PROMPTS")"
prompt_stem="${prompt_base%.*}"
tsv_file="${A2A_EVAL_TSV:-$OUTDIR/repl_eval_${prompt_stem}_${stamp}.tsv}"
summary_file="${A2A_EVAL_SUMMARY:-${tsv_file%.tsv}.summary.txt}"

mkdir -p "$(dirname "$tsv_file")"
mkdir -p "$(dirname "$summary_file")"

echo "sweeping $PROMPTS with cells=$CELLS frag=$FRAG rounds=$ROUNDS -> $tsv_file" >&2
A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$ROUNDS" \
    bash "$ROOT/tools/repl_question_sweep.sh" "$PROMPTS" | tee "$tsv_file"

echo "summarizing -> $summary_file" >&2
if [[ -n "${A2A_BASELINE_TSV:-}" ]]; then
    bash "$ROOT/tools/repl_tsv_summary.sh" "$tsv_file" "$A2A_BASELINE_TSV" | tee "$summary_file"
else
    bash "$ROOT/tools/repl_tsv_summary.sh" "$tsv_file" | tee "$summary_file"
fi

echo "results: $tsv_file" >&2
echo "summary: $summary_file" >&2
