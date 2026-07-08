#!/usr/bin/env bash
# Compare the same offline REPL probe corpus across two GGUF substrates.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"
PROMPTS="${1:-${A2A_SUBSTRATE_PROMPTS:-$ROOT/prompts/repl_probe_regression.txt}}"

BASE_MODEL="${A2A_BASE_MODEL:-$ROOT/weights/nanollama-arianna-full-v4-step2750-f16.gguf}"
CANDIDATE_MODEL="${A2A_CANDIDATE_MODEL:-}"

CELLS="${A2A_SUBSTRATE_CELLS:-${A2A_EVAL_CELLS:-${A2A_CELLS:-3}}}"
FRAG="${A2A_SUBSTRATE_FRAG:-${A2A_EVAL_FRAG:-${A2A_FRAG:-4}}}"
ROUNDS="${A2A_SUBSTRATE_ROUNDS:-${A2A_EVAL_ROUNDS:-${A2A_ROUNDS:-1}}}"

usage() {
    cat <<EOF
usage: A2A_CANDIDATE_MODEL=/path/to/new.gguf $0 [prompts.txt]

Runs tools/repl_question_sweep.sh against BASE and CANDIDATE GGUF bodies with
the same prompt corpus, then compares candidate TSV against base TSV.

Environment:
  A2A_BASE_MODEL          default: $BASE_MODEL
  A2A_CANDIDATE_MODEL     required unless passed by Makefile CANDIDATE_MODEL
  A2A_RUN_DIR             default: $OUTDIR
  A2A_SUBSTRATE_CELLS     default: $CELLS
  A2A_SUBSTRATE_FRAG      default: $FRAG
  A2A_SUBSTRATE_ROUNDS    default: $ROUNDS
EOF
}

safe_stem() {
    local name
    name="$(basename "$1")"
    name="${name%.gguf}"
    printf "%s" "$name" | tr -c 'A-Za-z0-9._-' '_'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "$CANDIDATE_MODEL" ]]; then
    echo "missing A2A_CANDIDATE_MODEL; set it to the new SFT GGUF path" >&2
    exit 2
fi

if [[ ! -f "$PROMPTS" ]]; then
    echo "missing prompts file: $PROMPTS" >&2
    exit 1
fi

if [[ ! -f "$BASE_MODEL" ]]; then
    echo "missing base model: $BASE_MODEL" >&2
    exit 1
fi

if [[ ! -f "$CANDIDATE_MODEL" ]]; then
    echo "missing candidate model: $CANDIDATE_MODEL" >&2
    exit 1
fi

if [[ ! -x "$ROOT/arianna2arianna" ]]; then
    make -C "$ROOT" >/dev/null
fi

mkdir -p "$OUTDIR"
stamp="$(date +%Y%m%d_%H%M%S)"
prompt_base="$(basename "$PROMPTS")"
prompt_stem="${prompt_base%.*}"
base_stem="$(safe_stem "$BASE_MODEL")"
candidate_stem="$(safe_stem "$CANDIDATE_MODEL")"
run_stem="substrate_compare_${prompt_stem}_${base_stem}_vs_${candidate_stem}_${stamp}"

base_tsv="$OUTDIR/${run_stem}.base.tsv"
candidate_tsv="$OUTDIR/${run_stem}.candidate.tsv"
summary_file="$OUTDIR/${run_stem}.summary.txt"

echo "base:      $BASE_MODEL" >&2
echo "candidate: $CANDIDATE_MODEL" >&2
echo "prompts:   $PROMPTS" >&2
echo "shape:     cells=$CELLS frag=$FRAG rounds=$ROUNDS" >&2

echo "sweeping base -> $base_tsv" >&2
A2A_MODEL="$BASE_MODEL" A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$ROUNDS" \
    bash "$ROOT/tools/repl_question_sweep.sh" "$PROMPTS" | tee "$base_tsv"

echo "sweeping candidate -> $candidate_tsv" >&2
A2A_MODEL="$CANDIDATE_MODEL" A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$ROUNDS" \
    bash "$ROOT/tools/repl_question_sweep.sh" "$PROMPTS" | tee "$candidate_tsv"

{
    echo "base_model: $BASE_MODEL"
    echo "candidate_model: $CANDIDATE_MODEL"
    echo "prompts: $PROMPTS"
    echo "shape: cells=$CELLS frag=$FRAG rounds=$ROUNDS"
    echo ""
    bash "$ROOT/tools/repl_tsv_summary.sh" "$candidate_tsv" "$base_tsv"
} | tee "$summary_file"

echo "base TSV:      $base_tsv" >&2
echo "candidate TSV: $candidate_tsv" >&2
echo "summary:       $summary_file" >&2
