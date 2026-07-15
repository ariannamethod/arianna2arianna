#!/usr/bin/env bash
# Sweep field-level coupling/routing settings and print compact TSV summaries.

set -euo pipefail
export LC_ALL=C

usage() {
    cat <<EOF
usage: $0 [prompts.txt]

Runs field_sweep.sh across field-level settings and writes per-setting TSV plus
summary files under runs/.

Knobs:
  A2A_FIELD_XCELLS="0 0.02 0.05"   field neighbour KV weights
  A2A_FIELD_QLOOPS="2"             qloop route limits
  A2A_FIELD_ROUNDS_LIST="2"        round counts to compare
  A2A_FIELD_CELLS=4                field cells
  A2A_FIELD_FRAG=12                tokens per cell fragment

These map to field_sweep.sh / runtime knobs:
  A2A_XCELL, A2A_QLOOP, A2A_ROUNDS, A2A_CELLS, A2A_FRAG
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"
PROMPTS="${1:-${A2A_FIELD_PROMPTS:-$ROOT/prompts/kv_influence.txt}}"

CELLS="${A2A_FIELD_CELLS:-${A2A_CELLS:-4}}"
FRAG="${A2A_FIELD_FRAG:-${A2A_FRAG:-12}}"
XCELLS="${A2A_FIELD_XCELLS:-0 0.02 0.05}"
QLOOPS="${A2A_FIELD_QLOOPS:-2}"
ROUNDS_LIST="${A2A_FIELD_ROUNDS_LIST:-${A2A_ROUNDS:-2}}"

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
    local xcell="$1" qloop="$2" rounds="$3" cells="$4" frag="$5" tsv_file="$6" summary_file="$7"
    awk -F '\t' -v xcell="$xcell" -v qloop="$qloop" -v rounds="$rounds" -v cells="$cells" -v frag="$frag" \
        -v tsv="$tsv_file" -v summary="$summary_file" '
        function numeric(x) { return x ~ /^[-+]?[0-9]+([.][0-9]+)?$/ }
        function col(name) { return idx[name] }
        NR == 1 {
            for (i = 1; i <= NF; i++) idx[$i] = i
            next
        }
        {
            rows++
            qroutes = $(col("qloop_routes")) + 0
            qkv = $(col("qloop_kv_routes")) + 0
            qroutes_sum += qroutes
            qkv_sum += qkv
            qtrig_sum += $(col("qloop_triggers")) + 0
            if (qroutes > 0) qprompt_rows++
            qquality_sum += $(col("qloop_quality")) + 0
            cfrag_sum += $(col("cell_fragments")) + 0
            cquality_sum += $(col("cell_quality")) + 0

            v = $(col("kv_influence"))
            if (numeric(v)) { in_sum += v + 0; in_n++ }
            v = $(col("qloop_iq_avg"))
            if (numeric(v) && qkv > 0) { iq_sum += (v + 0) * qkv; iq_n += qkv }
            v = $(col("d_r"))
            if (numeric(v)) { dr_sum += v + 0; dr_n++ }
            v = $(col("d_margin"))
            if (numeric(v)) { dm_sum += v + 0; dm_n++ }
            v = $(col("disso"))
            if (numeric(v)) { dsum += v + 0; d_n++ }
            v = $(col("dpos"))
            if (numeric(v)) { dpos_sum += v + 0; dpos_n++ }
        }
        END {
            printf "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d/%d\t%d/%d\t%d/%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                xcell, qloop, rounds, cells, frag, rows, qroutes_sum, qkv_sum,
                qprompt_rows, rows, qquality_sum, qroutes_sum, cquality_sum, cfrag_sum,
                in_n ? sprintf("%+.3f", in_sum / in_n) : "nan",
                iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan",
                dr_n ? sprintf("%.3f", dr_sum / dr_n) : "nan",
                dm_n ? sprintf("%+.3f", dm_sum / dm_n) : "nan",
                d_n ? sprintf("%.3f", dsum / d_n) : "nan",
                dpos_n ? sprintf("%.2f", dpos_sum / dpos_n) : "nan",
                tsv, summary
        }
    ' "$tsv_file"
}

printf "xcell\tqloop\trounds\tcells\tfrag\trows\tqloop_routes\tqloop_kv\tqloop_prompts\tqloop_quality\tcell_quality\tavg_i_n_kv\tavg_i_q_kv\tavg_d_r\tavg_d_margin\tavg_disso\tavg_dpos\ttsv\tsummary\n"

for xcell in $XCELLS; do
    for qloop in $QLOOPS; do
        for rounds in $ROUNDS_LIST; do
            tag="x$(safe_num "$xcell")_qloop$(safe_num "$qloop")_rounds$(safe_num "$rounds")_cells$(safe_num "$CELLS")_frag$(safe_num "$FRAG")"
            tsv_file="$OUTDIR/field_grid_${prompt_stem}_${tag}_${stamp}.tsv"
            summary_file="${tsv_file%.tsv}.summary.txt"

            echo "sweeping xcell=$xcell qloop=$qloop rounds=$rounds cells=$CELLS frag=$FRAG -> $tsv_file" >&2
            A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$rounds" \
            A2A_XCELL="$xcell" A2A_QLOOP="$qloop" \
                bash "$ROOT/tools/field_sweep.sh" "$PROMPTS" > "$tsv_file"

            bash "$ROOT/tools/field_tsv_summary.sh" "$tsv_file" > "$summary_file"
            compact_line "$xcell" "$qloop" "$rounds" "$CELLS" "$FRAG" "$tsv_file" "$summary_file"
        done
    done
done
