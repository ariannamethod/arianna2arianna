#!/usr/bin/env bash
# Sweep direct-user REPL bridge sampler temperature/filter settings.

set -euo pipefail

usage() {
    cat <<EOF
usage: $0 [prompts.txt]

Runs repl_question_sweep.sh across direct-user bridge sampler settings and
writes per-setting TSV/summary files under runs/.

Knobs:
  A2A_TEMP_BASES="0.35 0.45 0.55"   direct-user temp bases
  A2A_TEMP_SPANS="0.10"             direct-user cell-fraction spans
  A2A_TEMP_TOP_KS="40"              direct-user top_k values
  A2A_TEMP_TOP_PS="1.00"            direct-user top_p nucleus values
  A2A_TEMP_REPS="1.30"              direct-user repetition penalties
  A2A_TEMP_USER_KVS="0.05"          direct-user cross-KV weights
  A2A_TEMP_FORMATS="qa"             direct-user context formats
  A2A_TEMP_REPL_FORMATS="user_arianna qa" outer REPL prompt formats

These map to the runtime env knobs consumed by arianna2arianna:
  A2A_USER_QTEMP_BASE, A2A_USER_QTEMP_SPAN, A2A_USER_TOP_K,
  A2A_USER_TOP_P, A2A_USER_REP, A2A_USER_KV_WEIGHT,
  A2A_USER_CTX_FORMAT, A2A_REPL_PROMPT_FORMAT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"
PROMPTS="${1:-${A2A_TEMP_PROMPTS:-$ROOT/prompts/repl_probe_regression.txt}}"

CELLS="${A2A_TEMP_CELLS:-${A2A_EVAL_CELLS:-${A2A_CELLS:-3}}}"
FRAG="${A2A_TEMP_FRAG:-${A2A_EVAL_FRAG:-${A2A_FRAG:-4}}}"
ROUNDS="${A2A_TEMP_ROUNDS:-${A2A_EVAL_ROUNDS:-${A2A_ROUNDS:-1}}}"

BASES="${A2A_TEMP_BASES:-0.30 0.35 0.45 0.55 0.70}"
SPANS="${A2A_TEMP_SPANS:-0.10}"
TOP_KS="${A2A_TEMP_TOP_KS:-40}"
TOP_PS="${A2A_TEMP_TOP_PS:-1.00}"
REPS="${A2A_TEMP_REPS:-1.30}"
USER_KVS="${A2A_TEMP_USER_KVS:-0.05}"
FORMATS="${A2A_TEMP_FORMATS:-qa}"
REPL_FORMATS="${A2A_TEMP_REPL_FORMATS:-user_arianna}"

if [[ ! -f "$PROMPTS" ]]; then
    echo "missing prompts file: $PROMPTS" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
stamp="$(date +%Y%m%d_%H%M%S)"
prompt_base="$(basename "$PROMPTS")"
prompt_stem="${prompt_base%.*}"

safe_num() {
    printf "%s" "$1" | tr '+-' 'pm' | tr '.' 'p'
}

compact_line() {
    local fmt="$1" repl_fmt="$2" base="$3" span="$4" top_k="$5" top_p="$6" rep="$7" user_kv="$8" tsv_file="$9" summary_file="${10}"
    awk -v fmt="$fmt" -v repl_fmt="$repl_fmt" -v base="$base" -v span="$span" -v top_k="$top_k" -v top_p="$top_p" -v rep="$rep" -v user_kv="$user_kv" \
        -v tsv="$tsv_file" -v summary="$summary_file" '
        BEGIN {
            rows = bridge = iu = in_kv = route = any = short = question = label = notation = morph = yesno = repeat = kv = "-"
        }
        /^rows:/ { rows = $2 }
        /^user_bridge:/ { bridge = $2 }
        /^I_U\^kv:/ { iu = $3; gsub(/,/, "", iu) }
        /^I_N\^kv:/ { in_kv = $3; gsub(/,/, "", in_kv) }
        /^route_score:/ { route = $3; gsub(/,/, "", route) }
        /^answer_quality:/ {
            any = $3
            gsub(/,/, "", any)
            for (i = 4; i + 1 <= NF; i += 2) {
                key = $i
                val = $(i + 1)
                gsub(/,/, "", val)
                if (key == "short") short = val
                else if (key == "question_like") question = val
                else if (key == "label_artifact") label = val
                else if (key == "notation_artifact") notation = val
                else if (key == "morph_artifact") morph = val
                else if (key == "yes_no_start") yesno = val
                else if (key == "repetition") repeat = val
            }
        }
        /^answer_kv_changed:/ { kv = $2 }
        END {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                fmt, repl_fmt, base, span, top_k, top_p, rep, user_kv, rows, bridge, iu, in_kv, route,
                any, short, question, label, notation, morph, yesno, repeat, kv, tsv, summary
        }
    ' "$summary_file"
}

printf "user_format\trepl_format\ttemp_base\ttemp_span\ttop_k\ttop_p\trep\tuser_kv\trows\tbridge\ti_u_avg\ti_n_avg\troute_score\tquality_any\tshort\tquestion_like\tlabel\tnotation\tmorph\tyes_no\trepetition\tkv_changed\ttsv\tsummary\n"

for fmt in $FORMATS; do
    for repl_fmt in $REPL_FORMATS; do
        for base in $BASES; do
            for span in $SPANS; do
                for top_k in $TOP_KS; do
                    for top_p in $TOP_PS; do
                        for rep in $REPS; do
                            for user_kv in $USER_KVS; do
                                tag="fmt${fmt}_repl${repl_fmt}_base$(safe_num "$base")_span$(safe_num "$span")_topk$(safe_num "$top_k")_topp$(safe_num "$top_p")_rep$(safe_num "$rep")_userkv$(safe_num "$user_kv")"
                                tsv_file="$OUTDIR/repl_temp_${prompt_stem}_${tag}_${stamp}.tsv"
                                summary_file="${tsv_file%.tsv}.summary.txt"

                                echo "sweeping user_format=$fmt repl_format=$repl_fmt base=$base span=$span top_k=$top_k top_p=$top_p rep=$rep user_kv=$user_kv cells=$CELLS frag=$FRAG rounds=$ROUNDS -> $tsv_file" >&2
                                A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$ROUNDS" \
                                A2A_USER_QTEMP_BASE="$base" A2A_USER_QTEMP_SPAN="$span" \
                                A2A_USER_TOP_K="$top_k" A2A_USER_TOP_P="$top_p" A2A_USER_REP="$rep" A2A_USER_CTX_FORMAT="$fmt" \
                                A2A_USER_KV_WEIGHT="$user_kv" A2A_REPL_PROMPT_FORMAT="$repl_fmt" \
                                    bash "$ROOT/tools/repl_question_sweep.sh" "$PROMPTS" > "$tsv_file"

                                bash "$ROOT/tools/repl_tsv_summary.sh" "$tsv_file" > "$summary_file"
                                compact_line "$fmt" "$repl_fmt" "$base" "$span" "$top_k" "$top_p" "$rep" "$user_kv" "$tsv_file" "$summary_file"
                            done
                        done
                    done
                done
            done
        done
    done
done
