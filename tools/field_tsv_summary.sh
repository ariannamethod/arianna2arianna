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
    NR == 1 {
        for (i = 1; i <= NF; i++) idx[$i] = i
        col("prompt"); col("kv_influence"); col("d_r"); col("d_margin")
        col("disso"); col("dpos"); col("qloop_routes"); col("qloop_kv_routes")
        col("qloop_triggers"); col("qloop_gated"); col("qloop_iq_avg"); col("qloop_quality")
        col("qloop_iq_pos"); col("qloop_iq_neg"); col("qloop_iq_zero")
        col("qloop_tail"); col("qloop_morph"); col("qloop_label")
        col("qloop_short"); col("qloop_question"); col("cell_fragments")
        col("cell_quality"); col("cell_tail"); col("cell_morph")
        col("cell_label"); col("cell_short"); col("cell_question")
        next
    }
    {
        rows++
        qroutes = $(col("qloop_routes")) + 0
        qkv = $(col("qloop_kv_routes")) + 0
        qloop_routes += qroutes
        qloop_kv_routes += qkv
        qloop_triggers += $(col("qloop_triggers")) + 0
        qloop_gated += $(col("qloop_gated")) + 0
        if (qroutes > 0) qloop_prompt_rows++

        qloop_quality += $(col("qloop_quality")) + 0
        qloop_iq_pos += $(col("qloop_iq_pos")) + 0
        qloop_iq_neg += $(col("qloop_iq_neg")) + 0
        qloop_iq_zero += $(col("qloop_iq_zero")) + 0
        qloop_tail += $(col("qloop_tail")) + 0
        qloop_morph += $(col("qloop_morph")) + 0
        qloop_label += $(col("qloop_label")) + 0
        qloop_short += $(col("qloop_short")) + 0
        qloop_question += $(col("qloop_question")) + 0

        cell_fragments += $(col("cell_fragments")) + 0
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
        printf "qloop_quality: any %d/%d, tail %d, morph %d, label %d, short %d, question %d\n",
            qloop_quality, qloop_routes, qloop_tail, qloop_morph, qloop_label, qloop_short, qloop_question
        printf "cell_quality: any %d/%d, tail %d, morph %d, label %d, short %d, question %d\n",
            cell_quality, cell_fragments, cell_tail, cell_morph, cell_label, cell_short, cell_question
        printf "I_N^kv: avg %s, pos %d, neg %d, zero %d\n",
            in_n ? sprintf("%+.3f", in_sum / in_n) : "nan", in_pos, in_neg, in_zero
        printf "I_Q^kv: avg %s, pos %d, neg %d, zero %d, rows %d\n",
            iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan", qloop_iq_pos, qloop_iq_neg, qloop_iq_zero, iq_rows
        printf "rates: qloop_prompt_rate %.3f, qloop_kv_rate %.3f, qloop_debt_rate %.3f, qloop_gate_rate %.3f, qloop_efficiency %.3f, cell_debt_rate %.3f\n",
            qprompt_rate, qkv_rate, qdebt_rate, qgate_rate, qeff_rate, cdebt_rate
        printf "field: avg_d_r %s, avg_d_margin %s, avg_D_R %s, avg_Dpos %s\n",
            num_n["d_r"] ? sprintf("%.3f", num_sum["d_r"] / num_n["d_r"]) : "nan",
            num_n["d_margin"] ? sprintf("%+.3f", num_sum["d_margin"] / num_n["d_margin"]) : "nan",
            num_n["disso"] ? sprintf("%.3f", num_sum["disso"] / num_n["disso"]) : "nan",
            num_n["dpos"] ? sprintf("%.2f", num_sum["dpos"] / num_n["dpos"]) : "nan"
        printf "settling: d_margin pos %d, neg %d, zero %d\n", dm_pos, dm_neg, dm_zero
        printf "field_score: %+.3f (rough rank: coverage + influence - debt - disagreement)\n", field_score
    }
' "$tsv"
