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
  A2A_FIELD_QLOOP_MIN_IQS="0.0"    qloop I_Q^kv admission thresholds
  A2A_FIELD_QLOOP_UNIQUE_ASKERS="0" qloop same-asker fan-out policy flags
  A2A_FIELD_QLOOP_CANDIDATE_POOLS="0" qloop pre-generation candidate pools (0=auto)
  A2A_FIELD_QLOOP_STATEMENT_POOLS="0" qloop statement fallback candidate caps (0=inherit pool)
  A2A_FIELD_QLOOP_STATEMENT_ROUTES="0" qloop clean non-question fallback flags when question routes are silent
  A2A_FIELD_CELL_RETRY_MAXS="4" base-cell surface retry caps (1 disables retries)
  A2A_FIELD_PROMPT_FORMATS="raw" base-cell prompt frames (raw, qa, auto, user_arianna)
  A2A_FIELD_TEMP_BASES="0.60" base-cell temperature bases
  A2A_FIELD_TEMP_SPANS="0.70" base-cell temperature spans across cells
  A2A_FIELD_LANG_BIASES="0" base-cell language-match retry biases (0=measure only)
  A2A_FIELD_ROUNDS_LIST="3"        round counts to compare
  A2A_FIELD_CELLS=4                field cells
  A2A_FIELD_FRAG=12                tokens per cell fragment
  A2A_FIELD_KEEP_RAW=0             save raw per-prompt field outputs next to TSVs
  A2A_QLOOP_MIN_IQ=0.0             reject KV-backed qloop answers below this I_Q^kv

These map to field_sweep.sh / runtime knobs:
  A2A_XCELL, A2A_QLOOP, A2A_QLOOP_TCONF_WEIGHT, A2A_QLOOP_TCONF_ADAPT,
  A2A_QLOOP_TCONF_ADAPT_WEIGHT, A2A_QLOOP_MIN_IQ, A2A_QLOOP_UNIQUE_ASKER,
  A2A_QLOOP_CANDIDATE_POOL, A2A_QLOOP_STATEMENT_POOL,
  A2A_QLOOP_STATEMENT_ROUTES, A2A_CELL_RETRY_MAX,
  A2A_FIELD_PROMPT_FORMAT, A2A_ROUNDS, A2A_CELLS, A2A_FRAG
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
QLOOP_MIN_IQS="${A2A_FIELD_QLOOP_MIN_IQS:-${A2A_QLOOP_MIN_IQ:-0.0}}"
QLOOP_UNIQUE_ASKERS="${A2A_FIELD_QLOOP_UNIQUE_ASKERS:-${A2A_QLOOP_UNIQUE_ASKER:-0}}"
QLOOP_CANDIDATE_POOLS="${A2A_FIELD_QLOOP_CANDIDATE_POOLS:-${A2A_QLOOP_CANDIDATE_POOL:-0}}"
QLOOP_STATEMENT_POOLS="${A2A_FIELD_QLOOP_STATEMENT_POOLS:-${A2A_QLOOP_STATEMENT_POOL:-0}}"
QLOOP_STATEMENT_ROUTES="${A2A_FIELD_QLOOP_STATEMENT_ROUTES:-${A2A_QLOOP_STATEMENT_ROUTES:-0}}"
CELL_RETRY_MAXS="${A2A_FIELD_CELL_RETRY_MAXS:-${A2A_CELL_RETRY_MAX:-4}}"
FIELD_PROMPT_FORMATS="${A2A_FIELD_PROMPT_FORMATS:-${A2A_FIELD_PROMPT_FORMAT:-raw}}"
FIELD_TEMP_BASES="${A2A_FIELD_TEMP_BASES:-${A2A_FIELD_TEMP_BASE:-0.60}}"
FIELD_TEMP_SPANS="${A2A_FIELD_TEMP_SPANS:-${A2A_FIELD_TEMP_SPAN:-0.70}}"
FIELD_LANG_BIASES="${A2A_FIELD_LANG_BIASES:-${A2A_FIELD_LANG_BIAS:-0}}"
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
    local xcell="$1" qloop="$2" tconf="$3" adapt="$4" adapt_weight="$5" min_iq="$6" unique_asker="$7" candidate_pool="$8" statement_pool="$9" statement_routes="${10}" cell_retry_max="${11}" field_prompt_format="${12}" field_temp_base="${13}" field_temp_span="${14}" field_lang_bias="${15}" rounds="${16}" cells="${17}" frag="${18}" tsv_file="${19}" summary_file="${20}" raw_dir="${21}"
    awk -F '\t' -v xcell="$xcell" -v qloop="$qloop" -v tconf="$tconf" -v adapt="$adapt" -v adapt_weight="$adapt_weight" -v min_iq="$min_iq" -v unique_asker="$unique_asker" -v candidate_pool="$candidate_pool" -v statement_pool="$statement_pool" -v statement_routes="$statement_routes" -v cell_retry_max="$cell_retry_max" -v field_prompt_format="$field_prompt_format" -v field_temp_base="$field_temp_base" -v field_temp_span="$field_temp_span" -v field_lang_bias="$field_lang_bias" -v rounds="$rounds" -v cells="$cells" -v frag="$frag" \
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
            have_base_rescue = ("base_rescue" in idx)
            have_base_fail = ("base_fail" in idx)
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
            qstmt_sum += $(col("qloop_stmt_routes")) + 0
            qstmt_gate_sum += $(col("qloop_stmt_gated")) + 0
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
            qlang_sum += $(col("qloop_lang_mismatch")) + 0
            iq_pos += $(col("qloop_iq_pos")) + 0
            iq_neg += $(col("qloop_iq_neg")) + 0
            iq_zero += $(col("qloop_iq_zero")) + 0
            iq_low += $(col("qloop_iq_low")) + 0
            iq_strong += $(col("qloop_iq_strong")) + 0
            cfrags = $(col("cell_fragments")) + 0
            cfrag_sum += cfrags
            add_weighted("cell_words_avg", cfrags, "cwords")
            cquality_sum += $(col("cell_quality")) + 0
            clang_sum += $(col("cell_lang_mismatch")) + 0

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
            v = $(col("base_ms"))
            if (numeric(v)) {
                base_ms_sum += v + 0
                base_ms_n++
                if (!base_ms_seen || v + 0 > base_ms_max) {
                    base_ms_max = v + 0
                    base_ms_seen = 1
                }
            }
            v = $(col("qloop_ms"))
            if (numeric(v)) {
                qloop_ms_sum += v + 0
                qloop_ms_n++
                if (!qloop_ms_seen || v + 0 > qloop_ms_max) {
                    qloop_ms_max = v + 0
                    qloop_ms_seen = 1
                }
            }
            base_gen_sum += $(col("base_gen")) + 0
            base_retry_sum += $(col("base_retry")) + 0
            base_probe_sum += $(col("base_probe")) + 0
            base_rescue_sum += have_base_rescue ? $(col("base_rescue")) + 0 : 0
            base_fail_sum += have_base_fail ? $(col("base_fail")) + 0 : 0
            qloop_gen_sum += $(col("qloop_gen")) + 0
            qloop_retry_sum += $(col("qloop_retry")) + 0
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
            v = $(col("elapsed_sec"))
            if (numeric(v)) {
                elapsed_sum += v + 0
                elapsed_n++
                if (!elapsed_seen || v + 0 > elapsed_max) {
                    elapsed_max = v + 0
                    elapsed_seen = 1
                }
            }
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

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%s\t%s\t%s\t%s\t%d/%d\t%s\t%d/%d\t%d\t%s\t%d/%d\t%d\t%.3f\t%.3f\t%.3f\t%d/%d/%d\t%s\t%d/%d/%d\t%s\t%s\t%s\t%s\t%d/%d/%d\t%s\t%s\t%+.3f\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n",
                xcell, qloop, tconf, adapt, adapt_weight, min_iq, unique_asker, candidate_pool, statement_pool, statement_routes, cell_retry_max, field_prompt_format, field_temp_base, field_temp_span, field_lang_bias, rounds, cells, frag, rows, qroutes_sum, qkv_sum,
                qgate_sum, qstmt_sum, qstmt_gate_sum, qeff_rate, qscore_avg, qgate_score_avg, qprofile, qgate_profile,
                qprompt_rows, rows, avg_text("qwords"), qquality_sum, qroutes_sum, qlang_sum,
                avg_text("cwords"), cquality_sum, cfrag_sum, clang_sum,
                qprompt_rate, qdebt_rate, cdebt_rate, in_pos, in_neg, in_zero,
                in_n ? sprintf("%+.3f", in_sum / in_n) : "nan",
                iq_pos, iq_neg, iq_zero,
                sprintf("%d/%d", iq_low, iq_strong),
                iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan",
                dr_n ? sprintf("%.3f", dr_sum / dr_n) : "nan",
                dm_n ? sprintf("%+.3f", dm_sum / dm_n) : "nan",
                dm_pos, dm_neg, dm_zero,
                d_n ? sprintf("%.3f", dsum / d_n) : "nan",
                dpos_n ? sprintf("%.2f", dpos_sum / dpos_n) : "nan",
                field_score,
                base_ms_n ? sprintf("%.0f", base_ms_sum / base_ms_n) : "nan",
                base_ms_seen ? sprintf("%.0f", base_ms_max) : "nan",
                base_gen_sum, base_retry_sum, base_probe_sum, base_rescue_sum, base_fail_sum,
                qloop_ms_n ? sprintf("%.0f", qloop_ms_sum / qloop_ms_n) : "nan",
                qloop_ms_seen ? sprintf("%.0f", qloop_ms_max) : "nan",
                qloop_gen_sum, qloop_retry_sum,
                elapsed_n ? sprintf("%.3f", elapsed_sum / elapsed_n) : "nan",
                elapsed_seen ? sprintf("%.3f", elapsed_max) : "nan",
                raw, tsv, summary
        }
    ' "$tsv_file"
}

printf "xcell\tqloop\tqloop_tconf_weight\tqloop_tconf_adapt\tqloop_tconf_adapt_weight\tqloop_min_iq\tqloop_unique_asker\tqloop_candidate_pool\tqloop_statement_pool\tqloop_statement_routes\tcell_retry_max\tfield_prompt_format\tfield_temp_base\tfield_temp_span\tfield_lang_bias\trounds\tcells\tfrag\trows\tqloop_routes\tqloop_kv\tqloop_gated\tqloop_stmt_routes\tqloop_stmt_gated\tqloop_efficiency\tqloop_score_avg\tqloop_gate_score_avg\tqloop_profile\tqloop_gate_profile\tqloop_prompts\tqloop_words_avg\tqloop_quality\tqloop_lang_mismatch\tcell_words_avg\tcell_quality\tcell_lang_mismatch\tqloop_prompt_rate\tqloop_debt_rate\tcell_debt_rate\ti_n_signs\tavg_i_n_kv\ti_q_signs\ti_q_bands\tavg_i_q_kv\tavg_d_r\tavg_d_margin\td_margin_signs\tavg_disso\tavg_dpos\tfield_score\tbase_ms_avg\tbase_ms_max\tbase_gen\tbase_retry\tbase_probe\tbase_rescue\tbase_fail\tqloop_ms_avg\tqloop_ms_max\tqloop_gen\tqloop_retry\telapsed_avg\telapsed_max\traw_dir\ttsv\tsummary\n"

for xcell in $XCELLS; do
    for qloop in $QLOOPS; do
        for tconf in $QLOOP_TCONFS; do
            for adapt in $QLOOP_TCONF_ADAPTS; do
                for adapt_weight in $QLOOP_TCONF_ADAPT_WEIGHTS; do
                    for min_iq in $QLOOP_MIN_IQS; do
                        for unique_asker in $QLOOP_UNIQUE_ASKERS; do
                            for candidate_pool in $QLOOP_CANDIDATE_POOLS; do
                                for statement_pool in $QLOOP_STATEMENT_POOLS; do
                                    for statement_routes in $QLOOP_STATEMENT_ROUTES; do
                                        for cell_retry_max in $CELL_RETRY_MAXS; do
                                            for field_prompt_format in $FIELD_PROMPT_FORMATS; do
                                                for field_temp_base in $FIELD_TEMP_BASES; do
                                                    for field_temp_span in $FIELD_TEMP_SPANS; do
                                                        for field_lang_bias in $FIELD_LANG_BIASES; do
                                                            for rounds in $ROUNDS_LIST; do
                                                                tag="x$(safe_num "$xcell")_qloop$(safe_num "$qloop")_tconf$(safe_num "$tconf")_adapt$(safe_num "$adapt")_adaptw$(safe_num "$adapt_weight")_miniq$(safe_num "$min_iq")_uniqueq$(safe_num "$unique_asker")_pool$(safe_num "$candidate_pool")_stmtpool$(safe_num "$statement_pool")_stmt$(safe_num "$statement_routes")_retry$(safe_num "$cell_retry_max")_fmt${field_prompt_format}_tbase$(safe_num "$field_temp_base")_tspan$(safe_num "$field_temp_span")_lbias$(safe_num "$field_lang_bias")_rounds$(safe_num "$rounds")_cells$(safe_num "$CELLS")_frag$(safe_num "$FRAG")"
                                                                tsv_file="$OUTDIR/field_grid_${prompt_stem}_${tag}_${stamp}.tsv"
                                                                summary_file="${tsv_file%.tsv}.summary.txt"
                                                                raw_dir="-"
                                                                if [[ "$KEEP_RAW" != "0" ]]; then
                                                                    raw_dir="${tsv_file%.tsv}.raw"
                                                                fi
                                                                raw_env="$raw_dir"
                                                                [[ "$raw_env" == "-" ]] && raw_env=""

                                                                echo "sweeping xcell=$xcell qloop=$qloop tconf=$tconf adapt=$adapt adapt_weight=$adapt_weight min_iq=$min_iq unique_asker=$unique_asker candidate_pool=$candidate_pool statement_pool=$statement_pool statement_routes=$statement_routes cell_retry_max=$cell_retry_max field_prompt_format=$field_prompt_format field_temp_base=$field_temp_base field_temp_span=$field_temp_span field_lang_bias=$field_lang_bias rounds=$rounds cells=$CELLS frag=$FRAG -> $tsv_file" >&2
                                                                A2A_CELLS="$CELLS" A2A_FRAG="$FRAG" A2A_ROUNDS="$rounds" \
                                                                A2A_XCELL="$xcell" A2A_QLOOP="$qloop" \
                                                                A2A_QLOOP_TCONF_WEIGHT="$tconf" \
                                                                A2A_QLOOP_TCONF_ADAPT="$adapt" \
                                                                A2A_QLOOP_TCONF_ADAPT_WEIGHT="$adapt_weight" \
                                                                A2A_QLOOP_MIN_IQ="$min_iq" \
                                                                A2A_QLOOP_UNIQUE_ASKER="$unique_asker" \
                                                                A2A_QLOOP_CANDIDATE_POOL="$candidate_pool" \
                                                                A2A_QLOOP_STATEMENT_POOL="$statement_pool" \
                                                                A2A_QLOOP_STATEMENT_ROUTES="$statement_routes" \
                                                                A2A_CELL_RETRY_MAX="$cell_retry_max" \
                                                                A2A_FIELD_PROMPT_FORMAT="$field_prompt_format" \
                                                                A2A_FIELD_TEMP_BASE="$field_temp_base" \
                                                                A2A_FIELD_TEMP_SPAN="$field_temp_span" \
                                                                A2A_FIELD_LANG_BIAS="$field_lang_bias" \
                                                                A2A_FIELD_RAW_DIR="$raw_env" \
                                                                    bash "$ROOT/tools/field_sweep.sh" "$PROMPTS" > "$tsv_file"

                                                                bash "$ROOT/tools/field_tsv_summary.sh" "$tsv_file" > "$summary_file"
                                                                compact_line "$xcell" "$qloop" "$tconf" "$adapt" "$adapt_weight" "$min_iq" "$unique_asker" "$candidate_pool" "$statement_pool" "$statement_routes" "$cell_retry_max" "$field_prompt_format" "$field_temp_base" "$field_temp_span" "$field_lang_bias" "$rounds" "$CELLS" "$FRAG" "$tsv_file" "$summary_file" "$raw_dir"
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
