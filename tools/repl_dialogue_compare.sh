#!/usr/bin/env bash
# Compare default and candidate REPL dialogue policies on the same prompt corpus.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"
PROMPTS="${1:-${A2A_DIALOGUE_PROMPTS:-$ROOT/prompts/repl_probe_regression.txt}}"

CELLS="${A2A_DIALOGUE_CELLS:-${A2A_EVAL_CELLS:-${A2A_CELLS:-3}}}"
FRAG="${A2A_DIALOGUE_FRAG:-${A2A_EVAL_FRAG:-${A2A_FRAG:-4}}}"
ROUNDS="${A2A_DIALOGUE_ROUNDS:-${A2A_EVAL_ROUNDS:-${A2A_ROUNDS:-1}}}"

BASE_QLOOP="${A2A_DIALOGUE_BASE_QLOOP:-1}"
BASE_ADAPT="${A2A_DIALOGUE_BASE_ADAPT:-0}"
BASE_TCONF="${A2A_DIALOGUE_BASE_TCONF:-0.20}"
CAND_QLOOP="${A2A_DIALOGUE_CANDIDATE_QLOOP:-2}"
CAND_ADAPT="${A2A_DIALOGUE_CANDIDATE_ADAPT:-1}"
CAND_TCONF="${A2A_DIALOGUE_CANDIDATE_TCONF:-0.20}"

usage() {
    cat <<EOF
usage: $0 [prompts.txt]

Runs the same REPL dialogue probe corpus twice, then compares candidate against
baseline with repl_tsv_summary.sh.

Defaults:
  baseline:  A2A_REPL_QLOOP=$BASE_QLOOP A2A_QLOOP_TCONF_ADAPT=$BASE_ADAPT A2A_QLOOP_TCONF_WEIGHT=$BASE_TCONF
  candidate: A2A_REPL_QLOOP=$CAND_QLOOP A2A_QLOOP_TCONF_ADAPT=$CAND_ADAPT A2A_QLOOP_TCONF_WEIGHT=$CAND_TCONF

Environment:
  A2A_MODEL                         default: $MODEL
  A2A_RUN_DIR                       default: $OUTDIR
  A2A_DIALOGUE_PROMPTS              default: $PROMPTS
  A2A_DIALOGUE_CELLS                default: $CELLS
  A2A_DIALOGUE_FRAG                 default: $FRAG
  A2A_DIALOGUE_ROUNDS               default: $ROUNDS
  A2A_DIALOGUE_BASE_QLOOP           default: $BASE_QLOOP
  A2A_DIALOGUE_BASE_ADAPT           default: $BASE_ADAPT
  A2A_DIALOGUE_BASE_TCONF           default: $BASE_TCONF
  A2A_DIALOGUE_CANDIDATE_QLOOP      default: $CAND_QLOOP
  A2A_DIALOGUE_CANDIDATE_ADAPT      default: $CAND_ADAPT
  A2A_DIALOGUE_CANDIDATE_TCONF      default: $CAND_TCONF
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "$ROOT/arianna2arianna" ]]; then
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

mkdir -p "$OUTDIR"
stamp="$(date +%Y%m%d_%H%M%S)"
prompt_base="$(basename "$PROMPTS")"
prompt_stem="${prompt_base%.*}"
model_stem="$(basename "$MODEL")"
model_stem="${model_stem%.gguf}"
model_stem="$(printf "%s" "$model_stem" | tr -c 'A-Za-z0-9._-' '_')"

base_tag="base_q${BASE_QLOOP}_adapt${BASE_ADAPT}_tconf${BASE_TCONF}"
cand_tag="cand_q${CAND_QLOOP}_adapt${CAND_ADAPT}_tconf${CAND_TCONF}"
base_tag="$(printf "%s" "$base_tag" | tr '+-.' 'pmp')"
cand_tag="$(printf "%s" "$cand_tag" | tr '+-.' 'pmp')"

baseline_tsv="$OUTDIR/repl_dialogue_${prompt_stem}_${model_stem}_${base_tag}_${stamp}.tsv"
candidate_tsv="$OUTDIR/repl_dialogue_${prompt_stem}_${model_stem}_${cand_tag}_${stamp}.tsv"
baseline_summary="${baseline_tsv%.tsv}.summary.txt"
candidate_summary="${candidate_tsv%.tsv}.summary.txt"
compare_summary="$OUTDIR/repl_dialogue_${prompt_stem}_${model_stem}_${base_tag}_vs_${cand_tag}_${stamp}.compare.txt"

echo "model:     $MODEL" >&2
echo "prompts:   $PROMPTS" >&2
echo "cells/frg: $CELLS/$FRAG rounds=$ROUNDS" >&2
echo "baseline:  qloop=$BASE_QLOOP adapt=$BASE_ADAPT tconf=$BASE_TCONF -> $baseline_tsv" >&2
A2A_MODEL="$MODEL" A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$ROUNDS" \
A2A_REPL_QLOOP="$BASE_QLOOP" A2A_QLOOP_TCONF_ADAPT="$BASE_ADAPT" \
A2A_QLOOP_TCONF_WEIGHT="$BASE_TCONF" \
    bash "$ROOT/tools/repl_question_sweep.sh" "$PROMPTS" > "$baseline_tsv"
bash "$ROOT/tools/repl_tsv_summary.sh" "$baseline_tsv" > "$baseline_summary"

echo "candidate: qloop=$CAND_QLOOP adapt=$CAND_ADAPT tconf=$CAND_TCONF -> $candidate_tsv" >&2
A2A_MODEL="$MODEL" A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$ROUNDS" \
A2A_REPL_QLOOP="$CAND_QLOOP" A2A_QLOOP_TCONF_ADAPT="$CAND_ADAPT" \
A2A_QLOOP_TCONF_WEIGHT="$CAND_TCONF" \
    bash "$ROOT/tools/repl_question_sweep.sh" "$PROMPTS" > "$candidate_tsv"
bash "$ROOT/tools/repl_tsv_summary.sh" "$candidate_tsv" "$baseline_tsv" > "$candidate_summary"

{
    printf "model: %s\n" "$MODEL"
    printf "prompts: %s\n" "$PROMPTS"
    printf "cells: %s\nfrag: %s\nrounds: %s\n" "$CELLS" "$FRAG" "$ROUNDS"
    printf "baseline_env: A2A_REPL_QLOOP=%s A2A_QLOOP_TCONF_ADAPT=%s A2A_QLOOP_TCONF_WEIGHT=%s\n" "$BASE_QLOOP" "$BASE_ADAPT" "$BASE_TCONF"
    printf "candidate_env: A2A_REPL_QLOOP=%s A2A_QLOOP_TCONF_ADAPT=%s A2A_QLOOP_TCONF_WEIGHT=%s\n" "$CAND_QLOOP" "$CAND_ADAPT" "$CAND_TCONF"
    printf "baseline_tsv: %s\ncandidate_tsv: %s\n\n" "$baseline_tsv" "$candidate_tsv"
    printf "## baseline\n"
    cat "$baseline_summary"
    printf "\n## candidate_vs_baseline\n"
    cat "$candidate_summary"
} | tee "$compare_summary"

echo "baseline_summary:  $baseline_summary" >&2
echo "candidate_summary: $candidate_summary" >&2
echo "compare_summary:   $compare_summary" >&2
