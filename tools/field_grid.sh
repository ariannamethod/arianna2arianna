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
  A2A_FIELD_XCELLS="0 0.01 0.02 0.05" field neighbour KV weights
  A2A_FIELD_QLOOPS="1 2"           qloop route limits
  A2A_FIELD_QLOOP_TCONFS="0.20"    qloop target-confidence route weights
  A2A_FIELD_QLOOP_TCONF_ADAPTS="0" qloop adaptive target-confidence policy flags
  A2A_FIELD_QLOOP_TCONF_ADAPT_WEIGHTS="-0.10" adaptive tconf weights when adapt=1 and qloop>1
  A2A_FIELD_ROUNDS_LIST="3"        round counts to compare
  A2A_FIELD_CELLS=4                field cells
  A2A_FIELD_FRAG=12                tokens per cell fragment
  A2A_FIELD_KEEP_RAW=0             save raw per-prompt field outputs next to TSVs
  A2A_QLOOP_MIN_IQ=0.0             reject KV-backed qloop answers below this I_Q^kv

These map to field_sweep.sh / runtime knobs:
  A2A_XCELL, A2A_QLOOP, A2A_QLOOP_TCONF_WEIGHT, A2A_QLOOP_TCONF_ADAPT,
  A2A_QLOOP_TCONF_ADAPT_WEIGHT, A2A_ROUNDS, A2A_CELLS, A2A_FRAG, A2A_QLOOP_MIN_IQ
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
XCELLS="${A2A_FIELD_XCELLS:-0 0.01 0.02 0.05}"
QLOOPS="${A2A_FIELD_QLOOPS:-1 2}"
QLOOP_TCONFS="${A2A_FIELD_QLOOP_TCONFS:-${A2A_QLOOP_TCONF_WEIGHT:-0.20}}"
QLOOP_TCONF_ADAPTS="${A2A_FIELD_QLOOP_TCONF_ADAPTS:-${A2A_QLOOP_TCONF_ADAPT:-0}}"
QLOOP_TCONF_ADAPT_WEIGHTS="${A2A_FIELD_QLOOP_TCONF_ADAPT_WEIGHTS:-${A2A_QLOOP_TCONF_ADAPT_WEIGHT:--0.10}}"
ROUNDS_LIST="${A2A_FIELD_ROUNDS_LIST:-${A2A_ROUNDS:-3}}"
KEEP_RAW="${A2A_FIELD_KEEP_RAW:-0}"

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
    local xcell="$1" qloop="$2" tconf="$3" adapt="$4" adapt_weight="$5" rounds="$6" cells="$7" frag="$8" tsv_file="$9" summary_file="${10}" raw_dir="${11}"
    awk -F '\t' -v xcell="$xcell" -v qloop="$qloop" -v tconf="$tconf" -v adapt="$adapt" -v adapt_weight="$adapt_weight" -v rounds="$rounds" -v cells="$cells" -v frag="$frag" \
        -v tsv="$tsv_file" -v summary="$summary_file" -v raw="$raw_dir" '
        function numeric(x) { return x ~ /^[-+]?[0-9]+([.][0-9]+)?$/ }
        function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x) }
        function pospart(x) { return x > 0 ? x : 0 }
        function col(name) { return idx[name] }
        function add_weighted(name, weight, key,     v) {
            v = $(col(name))
            if (!numeric(v) || weight <= 0) return
            wsum[key] += (v + 0) * weight
            wn[key] += weight
        }
        function avg_text(key) { return wn[key] ? sprintf("%.3f", wsum[key] / wn[key]) : "nan" }
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
            qgate = $(col("qloop_gated")) + 0
            qgate_sum += qgate
            if (qroutes > 0) qprompt_rows++
            v = $(col("qloop_score_avg"))
            if (numeric(v) && qroutes > 0) { qscore_sum += (v + 0) * qroutes; qscore_n += qroutes }
            v = $(col("qloop_gate_score_avg"))
            if (numeric(v) && qgate > 0) {
                qgate_score_sum += (v + 0) * qgate
                qgate_score_n += qgate
            }
            add_weighted("qloop_dist_avg", qroutes, "dist")
            add_weighted("qloop_qopen_avg", qroutes, "qopen")
            add_weighted("qloop_tconf_avg", qroutes, "tconf")
            add_weighted("qloop_qmarks_avg", qroutes, "qmarks")
            add_weighted("qloop_gate_dist_avg", qgate, "gate_dist")
            add_weighted("qloop_gate_qopen_avg", qgate, "gate_qopen")
            add_weighted("qloop_gate_tconf_avg", qgate, "gate_tconf")
            add_weighted("qloop_gate_qmarks_avg", qgate, "gate_qmarks")
            add_weighted("qloop_words_avg", qroutes, "qwords")
            qquality_sum += $(col("qloop_quality")) + 0
            iq_pos += $(col("qloop_iq_pos")) + 0
            iq_neg += $(col("qloop_iq_neg")) + 0
            iq_zero += $(col("qloop_iq_zero")) + 0
            cfrags = $(col("cell_fragments")) + 0
            cfrag_sum += cfrags
            add_weighted("cell_words_avg", cfrags, "cwords")
            cquality_sum += $(col("cell_quality")) + 0

            v = $(col("kv_influence"))
            if (numeric(v)) {
                in_sum += v + 0
                in_n++
                if (v + 0 > 0.0005) in_pos++
                else if (v + 0 < -0.0005) in_neg++
                else in_zero++
            }
            v = $(col("qloop_iq_avg"))
            if (numeric(v) && qkv > 0) { iq_sum += (v + 0) * qkv; iq_n += qkv }
            v = $(col("d_r"))
            if (numeric(v)) { dr_sum += v + 0; dr_n++ }
            v = $(col("d_margin"))
            if (numeric(v)) {
                dm_sum += v + 0
                dm_n++
                if (v + 0 > 0.0005) dm_pos++
                else if (v + 0 < -0.0005) dm_neg++
                else dm_zero++
            }
            v = $(col("disso"))
            if (numeric(v)) { dsum += v + 0; d_n++ }
            v = $(col("dpos"))
            if (numeric(v)) { dpos_sum += v + 0; dpos_n++ }
        }
        END {
            qprompt_rate = rows ? qprompt_rows / rows : 0
            qkv_rate = qroutes_sum ? qkv_sum / qroutes_sum : 0
            qdebt_rate = qroutes_sum ? qquality_sum / qroutes_sum : 0
            qgate_rate = (qroutes_sum + qgate_sum) ? qgate_sum / (qroutes_sum + qgate_sum) : 0
            qeff_rate = (qroutes_sum + qgate_sum) ? qroutes_sum / (qroutes_sum + qgate_sum) : 0
            qscore_avg = qscore_n ? sprintf("%.3f", qscore_sum / qscore_n) : "nan"
            qgate_score_avg = qgate_score_n ? sprintf("%.3f", qgate_score_sum / qgate_score_n) : "nan"
            qprofile = sprintf("%s/%s/%s/%s", avg_text("dist"), avg_text("qopen"), avg_text("tconf"), avg_text("qmarks"))
            qgate_profile = sprintf("%s/%s/%s/%s", avg_text("gate_dist"), avg_text("gate_qopen"), avg_text("gate_tconf"), avg_text("gate_qmarks"))
            cdebt_rate = cfrag_sum ? cquality_sum / cfrag_sum : 0
            in_avg = in_n ? in_sum / in_n : 0
            iq_avg = iq_n ? iq_sum / iq_n : 0
            dm_avg = dm_n ? dm_sum / dm_n : 0
            d_avg = d_n ? dsum / d_n : 0
            dpos_avg = dpos_n ? dpos_sum / dpos_n : 0
            in_neg_rate = in_n ? in_neg / in_n : 0
            iq_neg_rate = iq_n ? iq_neg / iq_n : 0
            dm_pos_rate = dm_n ? dm_pos / dm_n : 0
            field_score = 2.0 * qprompt_rate + 0.5 * qkv_rate + 0.5 * clamp(iq_avg, -1, 1) + 0.2 * clamp(in_avg, -1, 1) \
                        - 2.0 * qdebt_rate - cdebt_rate - 0.5 * dpos_avg - 0.5 * d_avg - 0.25 * pospart(dm_avg) \
                        - 0.2 * in_neg_rate - 0.4 * iq_neg_rate - 0.2 * dm_pos_rate - 0.15 * qgate_rate

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%.3f\t%s\t%s\t%s\t%s\t%d/%d\t%s\t%d/%d\t%s\t%d/%d\t%.3f\t%.3f\t%.3f\t%d/%d/%d\t%s\t%d/%d/%d\t%s\t%s\t%s\t%d/%d/%d\t%s\t%s\t%+.3f\t%s\t%s\t%s\n",
                xcell, qloop, tconf, adapt, adapt_weight, rounds, cells, frag, rows, qroutes_sum, qkv_sum,
                qgate_sum, qeff_rate, qscore_avg, qgate_score_avg, qprofile, qgate_profile,
                qprompt_rows, rows, avg_text("qwords"), qquality_sum, qroutes_sum,
                avg_text("cwords"), cquality_sum, cfrag_sum,
                qprompt_rate, qdebt_rate, cdebt_rate, in_pos, in_neg, in_zero,
                in_n ? sprintf("%+.3f", in_sum / in_n) : "nan",
                iq_pos, iq_neg, iq_zero,
                iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan",
                dr_n ? sprintf("%.3f", dr_sum / dr_n) : "nan",
                dm_n ? sprintf("%+.3f", dm_sum / dm_n) : "nan",
                dm_pos, dm_neg, dm_zero,
                d_n ? sprintf("%.3f", dsum / d_n) : "nan",
                dpos_n ? sprintf("%.2f", dpos_sum / dpos_n) : "nan",
                field_score, raw, tsv, summary
        }
    ' "$tsv_file"
}

printf "xcell\tqloop\tqloop_tconf_weight\tqloop_tconf_adapt\tqloop_tconf_adapt_weight\trounds\tcells\tfrag\trows\tqloop_routes\tqloop_kv\tqloop_gated\tqloop_efficiency\tqloop_score_avg\tqloop_gate_score_avg\tqloop_profile\tqloop_gate_profile\tqloop_prompts\tqloop_words_avg\tqloop_quality\tcell_words_avg\tcell_quality\tqloop_prompt_rate\tqloop_debt_rate\tcell_debt_rate\ti_n_signs\tavg_i_n_kv\ti_q_signs\tavg_i_q_kv\tavg_d_r\tavg_d_margin\td_margin_signs\tavg_disso\tavg_dpos\tfield_score\traw_dir\ttsv\tsummary\n"

for xcell in $XCELLS; do
    for qloop in $QLOOPS; do
        for tconf in $QLOOP_TCONFS; do
            for adapt in $QLOOP_TCONF_ADAPTS; do
                for adapt_weight in $QLOOP_TCONF_ADAPT_WEIGHTS; do
                    for rounds in $ROUNDS_LIST; do
                        tag="x$(safe_num "$xcell")_qloop$(safe_num "$qloop")_tconf$(safe_num "$tconf")_adapt$(safe_num "$adapt")_adaptw$(safe_num "$adapt_weight")_rounds$(safe_num "$rounds")_cells$(safe_num "$CELLS")_frag$(safe_num "$FRAG")"
                        tsv_file="$OUTDIR/field_grid_${prompt_stem}_${tag}_${stamp}.tsv"
                        summary_file="${tsv_file%.tsv}.summary.txt"
                        raw_dir="-"
                        if [[ "$KEEP_RAW" != "0" ]]; then
                            raw_dir="${tsv_file%.tsv}.raw"
                        fi
                        raw_env="$raw_dir"
                        [[ "$raw_env" == "-" ]] && raw_env=""

                        echo "sweeping xcell=$xcell qloop=$qloop tconf=$tconf adapt=$adapt adapt_weight=$adapt_weight rounds=$rounds cells=$CELLS frag=$FRAG -> $tsv_file" >&2
                        A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$rounds" \
                        A2A_XCELL="$xcell" A2A_QLOOP="$qloop" \
                        A2A_QLOOP_TCONF_WEIGHT="$tconf" \
                        A2A_QLOOP_TCONF_ADAPT="$adapt" \
                        A2A_QLOOP_TCONF_ADAPT_WEIGHT="$adapt_weight" \
                        A2A_FIELD_RAW_DIR="$raw_env" \
                            bash "$ROOT/tools/field_sweep.sh" "$PROMPTS" > "$tsv_file"

                        bash "$ROOT/tools/field_tsv_summary.sh" "$tsv_file" > "$summary_file"
                        compact_line "$xcell" "$qloop" "$tconf" "$adapt" "$adapt_weight" "$rounds" "$CELLS" "$FRAG" "$tsv_file" "$summary_file" "$raw_dir"
                    done
                done
            done
        done
    done
done
