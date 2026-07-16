#!/usr/bin/env bash
# Summarize field_sweep.sh TSV output into stable tuning counters.

set -euo pipefail
export LC_ALL=C

usage() {
    cat <<EOF
usage: $0 field_sweep.tsv

Reads tools/field_sweep.sh TSV output and prints aggregate field/qloop/cell
metrics for tuning xcell, qloop, and rounds.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

tsv="$1"
if [[ ! -f "$tsv" ]]; then
    echo "missing TSV: $tsv" >&2
    exit 1
fi

awk -F '\t' '
    function numeric(x) { return x ~ /^[-+]?[0-9]+([.][0-9]+)?$/ }
    function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x) }
    function pospart(x) { return x > 0 ? x : 0 }
    function col(name) {
        if (!(name in idx)) {
            printf "missing column: %s\n", name > "/dev/stderr"
            exit 2
        }
        return idx[name]
    }
    function add_numeric(name, arr_sum, arr_n,     v) {
        v = $(col(name))
        if (!numeric(v)) return
        arr_sum[name] += v + 0
        arr_n[name]++
    }
    function add_weighted(name, weight, key,     v) {
        v = $(col(name))
        if (!numeric(v) || weight <= 0) return
        wsum[key] += (v + 0) * weight
        wn[key] += weight
    }
    function avg_text(key) { return wn[key] ? sprintf("%.3f", wsum[key] / wn[key]) : "nan" }
    NR == 1 {
        for (i = 1; i <= NF; i++) idx[$i] = i
        col("prompt"); col("kv_influence"); col("d_r"); col("d_margin")
        col("disso"); col("dpos"); col("qloop_routes"); col("qloop_kv_routes")
        col("qloop_triggers"); col("qloop_gated")
        col("qloop_stmt_routes"); col("qloop_stmt_gated")
        col("qloop_iq_avg"); col("qloop_quality")
        col("qloop_score_avg"); col("qloop_gate_score_avg")
        col("qloop_dist_avg"); col("qloop_gate_dist_avg")
        col("qloop_qopen_avg"); col("qloop_gate_qopen_avg")
        col("qloop_tconf_avg"); col("qloop_gate_tconf_avg")
        col("qloop_qmarks_avg"); col("qloop_gate_qmarks_avg")
        col("qloop_iq_pos"); col("qloop_iq_neg"); col("qloop_iq_zero")
        col("qloop_iq_low"); col("qloop_iq_strong")
        col("qloop_tail"); col("qloop_morph"); col("qloop_label")
        col("qloop_short"); col("qloop_question"); col("qloop_recipient")
        col("qloop_words_avg")
        col("cell_fragments"); col("cell_words_avg")
        col("cell_quality"); col("cell_tail"); col("cell_morph")
        col("cell_label"); col("cell_short"); col("cell_question")
        col("base_ms"); col("qloop_ms"); col("qloop_gen"); col("qloop_retry")
        col("elapsed_sec")
        next
    }
    {
        rows++
        qroutes = $(col("qloop_routes")) + 0
        qkv = $(col("qloop_kv_routes")) + 0
        qloop_routes += qroutes
        qloop_kv_routes += qkv
        qloop_triggers += $(col("qloop_triggers")) + 0
        qgate = $(col("qloop_gated")) + 0
        qloop_gated += qgate
        qloop_stmt_routes += $(col("qloop_stmt_routes")) + 0
        qloop_stmt_gated += $(col("qloop_stmt_gated")) + 0
        if (qroutes > 0) qloop_prompt_rows++

        v = $(col("qloop_score_avg"))
        if (numeric(v) && qroutes > 0) {
            qscore_sum += (v + 0) * qroutes
            qscore_n += qroutes
        }
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

        qloop_quality += $(col("qloop_quality")) + 0
        qloop_iq_pos += $(col("qloop_iq_pos")) + 0
        qloop_iq_neg += $(col("qloop_iq_neg")) + 0
        qloop_iq_zero += $(col("qloop_iq_zero")) + 0
        qloop_iq_low += $(col("qloop_iq_low")) + 0
        qloop_iq_strong += $(col("qloop_iq_strong")) + 0
        qloop_tail += $(col("qloop_tail")) + 0
        qloop_morph += $(col("qloop_morph")) + 0
        qloop_label += $(col("qloop_label")) + 0
        qloop_short += $(col("qloop_short")) + 0
        qloop_question += $(col("qloop_question")) + 0
        qloop_recipient += $(col("qloop_recipient")) + 0
        add_weighted("qloop_words_avg", qroutes, "qwords")

        cell_fragments += $(col("cell_fragments")) + 0
        add_weighted("cell_words_avg", $(col("cell_fragments")) + 0, "cwords")
        cell_quality += $(col("cell_quality")) + 0
        cell_tail += $(col("cell_tail")) + 0
        cell_morph += $(col("cell_morph")) + 0
        cell_label += $(col("cell_label")) + 0
        cell_short += $(col("cell_short")) + 0
        cell_question += $(col("cell_question")) + 0

        v = $(col("qloop_iq_avg"))
        if (numeric(v) && qkv > 0) {
            iq_sum += (v + 0) * qkv
            iq_n += qkv
            iq_rows++
        }

        v = $(col("base_ms"))
        if (numeric(v)) {
            base_ms_sum += v + 0
            base_ms_n++
            if (!base_ms_seen || v + 0 > base_ms_max) {
                base_ms_max = v + 0
                base_ms_max_prompt = $(col("prompt"))
                base_ms_seen = 1
            }
        }
        v = $(col("qloop_ms"))
        if (numeric(v)) {
            qloop_ms_sum += v + 0
            qloop_ms_n++
            if (!qloop_ms_seen || v + 0 > qloop_ms_max) {
                qloop_ms_max = v + 0
                qloop_ms_max_prompt = $(col("prompt"))
                qloop_ms_seen = 1
            }
        }
        qloop_gen += $(col("qloop_gen")) + 0
        qloop_retry += $(col("qloop_retry")) + 0

        v = $(col("kv_influence"))
        if (numeric(v)) {
            in_sum += v + 0
            in_n++
            if (v + 0 > 0.0005) in_pos++
            else if (v + 0 < -0.0005) in_neg++
            else in_zero++
        }

        add_numeric("d_r", num_sum, num_n)
        add_numeric("d_margin", num_sum, num_n)
        v = $(col("d_margin"))
        if (numeric(v)) {
            if (v + 0 > 0.0005) dm_pos++
            else if (v + 0 < -0.0005) dm_neg++
            else dm_zero++
        }
        add_numeric("disso", num_sum, num_n)
        add_numeric("dpos", num_sum, num_n)
        v = $(col("elapsed_sec"))
        if (numeric(v)) {
            elapsed_sum += v + 0
            elapsed_n++
            if (!elapsed_seen || v + 0 > elapsed_max) {
                elapsed_max = v + 0
                elapsed_max_prompt = $(col("prompt"))
                elapsed_seen = 1
            }
        }
    }
    END {
        qprompt_rate = rows ? qloop_prompt_rows / rows : 0
        qkv_rate = qloop_routes ? qloop_kv_routes / qloop_routes : 0
        qdebt_rate = qloop_routes ? qloop_quality / qloop_routes : 0
        qgate_rate = (qloop_routes + qloop_gated) ? qloop_gated / (qloop_routes + qloop_gated) : 0
        qeff_rate = (qloop_routes + qloop_gated) ? qloop_routes / (qloop_routes + qloop_gated) : 0
        cdebt_rate = cell_fragments ? cell_quality / cell_fragments : 0
        iq_avg = iq_n ? iq_sum / iq_n : 0
        iq_neg_rate = iq_n ? qloop_iq_neg / iq_n : 0
        in_avg = in_n ? in_sum / in_n : 0
        in_neg_rate = in_n ? in_neg / in_n : 0
        dm_avg = num_n["d_margin"] ? num_sum["d_margin"] / num_n["d_margin"] : 0
        dm_pos_rate = num_n["d_margin"] ? dm_pos / num_n["d_margin"] : 0
        disso_avg = num_n["disso"] ? num_sum["disso"] / num_n["disso"] : 0
        dpos_avg = num_n["dpos"] ? num_sum["dpos"] / num_n["dpos"] : 0
        field_score = 2.0 * qprompt_rate + 0.5 * qkv_rate + 0.5 * clamp(iq_avg, -1, 1) + 0.2 * clamp(in_avg, -1, 1) \
                    - 2.0 * qdebt_rate - cdebt_rate - 0.5 * dpos_avg - 0.5 * disso_avg - 0.25 * pospart(dm_avg) \
                    - 0.2 * in_neg_rate - 0.4 * iq_neg_rate - 0.2 * dm_pos_rate - 0.15 * qgate_rate

        printf "rows: %d\n", rows
        printf "qloop: routes %d, kv %d, triggers %d, gated %d, prompts %d/%d\n",
            qloop_routes, qloop_kv_routes, qloop_triggers, qloop_gated, qloop_prompt_rows, rows
        printf "qloop_statement: accepted %d/%d, gated %d/%d\n",
            qloop_stmt_routes, qloop_routes, qloop_stmt_gated, qloop_gated
        printf "qloop_score: accepted avg %s, gated avg %s\n",
            qscore_n ? sprintf("%.3f", qscore_sum / qscore_n) : "nan",
            qgate_score_n ? sprintf("%.3f", qgate_score_sum / qgate_score_n) : "nan"
        printf "qloop_route_profile: accepted d %s, qopen %s, tconf %s, qmarks %s; gated d %s, qopen %s, tconf %s, qmarks %s\n",
            avg_text("dist"), avg_text("qopen"), avg_text("tconf"), avg_text("qmarks"),
            avg_text("gate_dist"), avg_text("gate_qopen"), avg_text("gate_tconf"), avg_text("gate_qmarks")
        printf "qloop_quality: any %d/%d, tail %d, morph %d, label %d, short %d, question %d, recipient %d\n",
            qloop_quality, qloop_routes, qloop_tail, qloop_morph, qloop_label, qloop_short, qloop_question, qloop_recipient
        printf "cell_quality: any %d/%d, tail %d, morph %d, label %d, short %d, question %d\n",
            cell_quality, cell_fragments, cell_tail, cell_morph, cell_label, cell_short, cell_question
        printf "density: qloop_words_avg %s, cell_words_avg %s\n",
            avg_text("qwords"), avg_text("cwords")
        printf "I_N^kv: avg %s, pos %d, neg %d, zero %d\n",
            in_n ? sprintf("%+.3f", in_sum / in_n) : "nan", in_pos, in_neg, in_zero
        printf "I_Q^kv: avg %s, pos %d, neg %d, zero %d, low %d, strong %d, rows %d\n",
            iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan",
            qloop_iq_pos, qloop_iq_neg, qloop_iq_zero, qloop_iq_low, qloop_iq_strong, iq_rows
        printf "rates: qloop_prompt_rate %.3f, qloop_kv_rate %.3f, qloop_debt_rate %.3f, qloop_gate_rate %.3f, qloop_efficiency %.3f, cell_debt_rate %.3f\n",
            qprompt_rate, qkv_rate, qdebt_rate, qgate_rate, qeff_rate, cdebt_rate
        printf "field: avg_d_r %s, avg_d_margin %s, avg_D_R %s, avg_Dpos %s\n",
            num_n["d_r"] ? sprintf("%.3f", num_sum["d_r"] / num_n["d_r"]) : "nan",
            num_n["d_margin"] ? sprintf("%+.3f", num_sum["d_margin"] / num_n["d_margin"]) : "nan",
            num_n["disso"] ? sprintf("%.3f", num_sum["disso"] / num_n["disso"]) : "nan",
            num_n["dpos"] ? sprintf("%.2f", num_sum["dpos"] / num_n["dpos"]) : "nan"
        printf "settling: d_margin pos %d, neg %d, zero %d\n", dm_pos, dm_neg, dm_zero
        printf "timing: base_ms_avg %.0f, base_ms_max %.0f :: %s, qloop_ms_avg %.0f, qloop_ms_max %.0f :: %s, qloop_gen %d, qloop_retry %d\n",
            base_ms_n ? base_ms_sum / base_ms_n : 0,
            base_ms_seen ? base_ms_max : 0,
            base_ms_seen ? base_ms_max_prompt : "n/a",
            qloop_ms_n ? qloop_ms_sum / qloop_ms_n : 0,
            qloop_ms_seen ? qloop_ms_max : 0,
            qloop_ms_seen ? qloop_ms_max_prompt : "n/a",
            qloop_gen, qloop_retry
        printf "latency: avg_sec %.3f, max_sec %.3f :: %s\n",
            elapsed_n ? elapsed_sum / elapsed_n : 0,
            elapsed_seen ? elapsed_max : 0,
            elapsed_seen ? elapsed_max_prompt : "n/a"
        printf "field_score: %+.3f (rough rank: coverage + influence - debt - disagreement)\n", field_score
    }
' "$tsv"
