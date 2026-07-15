#!/usr/bin/env bash
# δ-field, resonance control, and δ-life smoke tests.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_field ---"

if ! a2a_have_f16; then
    a2a_skip "field tests need f16 weights"
    return 0 2>/dev/null || exit 0
fi

field_out="$("$A2A_BIN" "$A2A_MODEL_F16" "resonance" field 3 4 2 0 2 0.25 2>&1)"
a2a_assert_grep "δ-field: 3 cells × 2 rounds" "$field_out" "field starts with requested shape"
a2a_assert_grep "floor \\(sampling noise" "$field_out" "field reports sampling floor"
a2a_assert_grep "round 1:" "$field_out" "field reports round 1 metrics"
a2a_assert_grep "d_R" "$field_out" "field reports d_R"
a2a_assert_grep "Δ_R\\(text n/a\\)" "$field_out" "chorus marks text delta as n/a"
a2a_assert_grep "Δ_R\\^kv" "$field_out" "chorus reports KV-order delta"
a2a_assert_grep "margin" "$field_out" "KV-order delta reports floor margin"
a2a_assert_grep "I_N\\^kv" "$field_out" "chorus reports neighbour influence"
a2a_assert_grep "Dpos" "$field_out" "field reports positional dissonance"
a2a_assert_grep "δ-field done" "$field_out" "field completes"

if [[ -f "$A2A_ROOT/FIELDLOG.md" ]] && grep -q "resonance" "$A2A_ROOT/FIELDLOG.md"; then
    a2a_ok "field appends FIELDLOG.md"
else
    a2a_fail "field did not append FIELDLOG.md"
fi

restest_out="$("$A2A_BIN" "$A2A_MODEL_F16" "resonance" restest 3 4 2 2>&1)"
a2a_assert_grep "resonance test: coherent vs same-length SHUFFLED" "$restest_out" "restest starts"
a2a_assert_grep "COHERENT" "$restest_out" "restest includes coherent arm"
a2a_assert_grep "SHUFFLED" "$restest_out" "restest includes shuffled arm"
a2a_assert_grep "resonance-beyond-length" "$restest_out" "restest reports control delta"

sweep_prompts="$(mktemp)"
printf "Let the cells remember each other.\n" > "$sweep_prompts"
sweep_out="$(A2A_CELLS=3 A2A_FRAG=4 A2A_ROUNDS=1 bash "$A2A_ROOT/tools/kv_influence_sweep.sh" "$sweep_prompts" 2>&1)"
rm -f "$sweep_prompts"
a2a_assert_grep "^prompt[[:space:]]+mode[[:space:]]+avg_entropy" "$sweep_out" "influence sweep reports TSV header"
a2a_assert_grep "Let the cells remember each other\\.[[:space:]]+sem[[:space:]]" "$sweep_out" "influence sweep reports semantic row"
if printf "%s\n" "$sweep_out" | awk -F '\t' '$1 == "Let the cells remember each other." && sqrt(($7 + 0) * ($7 + 0)) > 0.001 { ok = 1 } END { exit ok ? 0 : 1 }'; then
    a2a_ok "memory prompt reports semantic neighbour influence"
else
    a2a_fail "memory prompt did not report semantic neighbour influence"
fi

field_sweep_prompts="$(mktemp)"
field_sweep_raw_dir="$(mktemp -d)"
printf "Let the cells remember each other.\n" > "$field_sweep_prompts"
field_sweep_out="$(A2A_CELLS=3 A2A_FRAG=4 A2A_ROUNDS=2 A2A_FIELD_RAW_DIR="$field_sweep_raw_dir" bash "$A2A_ROOT/tools/field_sweep.sh" "$field_sweep_prompts" 2>&1)"
a2a_assert_grep "^prompt[[:space:]]+mode[[:space:]]+cells[[:space:]]+frag[[:space:]]+rounds[[:space:]]+avg_entropy[[:space:]]+d_r[[:space:]]+d_floor" "$field_sweep_out" "field sweep reports final-round TSV header"
a2a_assert_grep "qloop_triggers[[:space:]]+qloop_gated[[:space:]]+qloop_iq_avg[[:space:]]+qloop_iq_pos[[:space:]]+qloop_iq_neg[[:space:]]+qloop_iq_zero[[:space:]]+qloop_quality" "$field_sweep_out" "field sweep reports qloop influence sign columns"
a2a_assert_grep "cell_fragments[[:space:]]+cell_quality[[:space:]]+cell_tail[[:space:]]+cell_morph" "$field_sweep_out" "field sweep reports cell surface quality columns"
a2a_assert_grep "Let the cells remember each other\\.[[:space:]]+sem[[:space:]]+3[[:space:]]+4[[:space:]]+2" "$field_sweep_out" "field sweep reports semantic final-round row"
if printf "%s\n" "$field_sweep_out" | awk -F '\t' '$1 == "Let the cells remember each other." && $16 ~ /^[0-9]+$/ && $17 ~ /^[0-9]+$/ && $18 ~ /^[0-9]+$/ && $19 ~ /^[0-9]+$/ && ($20 == "nan" || $20 ~ /^[+-]?[0-9]+[.][0-9]+$/) && $21 ~ /^[0-9]+$/ && $22 ~ /^[0-9]+$/ && $23 ~ /^[0-9]+$/ && $24 ~ /^[0-9]+$/ && $30 == 6 && $31 ~ /^[0-9]+$/ { ok = 1 } END { exit ok ? 0 : 1 }'; then
    a2a_ok "field sweep reports qloop answer and cell surface counters"
else
    a2a_fail "field sweep did not report qloop answer and cell surface counters"
fi
raw_file="$(find "$field_sweep_raw_dir" -type f -name '*.txt' | head -n 1 || true)"
if [[ -n "$raw_file" ]] && grep -q "δ-field" "$raw_file"; then
    a2a_ok "field sweep raw capture writes full output"
else
    a2a_fail "field sweep raw capture missing full output"
fi
rm -rf "$field_sweep_raw_dir"
field_sweep_off_out="$(A2A_CELLS=2 A2A_FRAG=3 A2A_ROUNDS=1 A2A_XCELL=0 bash "$A2A_ROOT/tools/field_sweep.sh" "$field_sweep_prompts" 2>&1)"
rm -f "$field_sweep_prompts"
a2a_assert_grep "Let the cells remember each other\\.[[:space:]]+off[[:space:]]+2[[:space:]]+3[[:space:]]+1" "$field_sweep_off_out" "field sweep handles neighbour-KV-off rows"

field_grid_prompts="$(mktemp)"
field_grid_dir="$(mktemp -d)"
printf "Let the cells remember each other.\n" > "$field_grid_prompts"
field_grid_out="$(A2A_RUN_DIR="$field_grid_dir" A2A_FIELD_KEEP_RAW=1 A2A_FIELD_XCELLS=0 A2A_FIELD_QLOOPS=0 A2A_FIELD_ROUNDS_LIST=1 A2A_FIELD_CELLS=2 A2A_FIELD_FRAG=3 bash "$A2A_ROOT/tools/field_grid.sh" "$field_grid_prompts" 2>&1)"
rm -f "$field_grid_prompts"
a2a_assert_grep "^xcell[[:space:]]+qloop[[:space:]]+rounds[[:space:]]+cells[[:space:]]+frag" "$field_grid_out" "field grid reports compact TSV header"
a2a_assert_grep "qloop_gated.*qloop_efficiency.*i_n_signs[[:space:]]+avg_i_n_kv[[:space:]]+i_q_signs[[:space:]]+avg_i_q_kv.*d_margin_signs.*field_score[[:space:]]+raw_dir" "$field_grid_out" "field grid reports influence and settling risks, score, and raw columns"
a2a_assert_grep "^0[[:space:]]+0[[:space:]]+1[[:space:]]+2[[:space:]]+3[[:space:]]+1[[:space:]]+0[[:space:]]+0" "$field_grid_out" "field grid reports one no-qloop setting"
if printf "%s\n" "$field_grid_out" | awk -F '\t' 'NR == 1 && $1 == "xcell" { header = NF } NR > 1 && $1 == "0" { row = NF } END { exit (header == 29 && row == 29) ? 0 : 1 }'; then
    a2a_ok "field grid compact rows keep expected column count"
else
    a2a_fail "field grid compact rows changed column count"
fi
if find "$field_grid_dir" -path '*.raw/*.txt' -type f | grep -q .; then
    a2a_ok "field grid raw capture writes per-setting raw outputs"
else
    a2a_fail "field grid raw capture missing per-setting raw outputs"
fi
rm -rf "$field_grid_dir"

qloop_out="$("$A2A_BIN" "$A2A_MODEL_F16" "Answer only with a question: why does the field remember?" field 5 12 1 0 2 0.02 1 1.3 0 1 2 0 2>&1)"
a2a_assert_grep "r1 cell 0 .*What does the field remember\\?" "$qloop_out" "field cell surface keeps closed question fragment"
a2a_assert_not_grep "What do you see when|perspause" "$qloop_out" "field cell surface removes open tails and morph junk"
a2a_assert_grep "qloop c[0-9]+.*\\[kv\\]" "$qloop_out" "qloop answers use asker KV"
a2a_assert_grep "I_Q\\^kv=" "$qloop_out" "qloop reports asker KV influence"
qloop_gate_out="$(A2A_QLOOP_MIN_IQ=2.0 "$A2A_BIN" "$A2A_MODEL_F16" "Answer only with a question: why does the field remember?" field 5 12 1 0 2 0.02 1 1.3 0 1 2 0 2>&1)"
a2a_assert_grep "qloop gate c[0-9]+.*\\[kv\\].*I_Q\\^kv=" "$qloop_gate_out" "qloop gates negative or below-threshold asker KV influence"

repl_out="$(printf "Why does the field remember?\n:q\n" | "$A2A_BIN" "$A2A_MODEL_F16" repl 3 4 1 2>&1)"
a2a_assert_grep "repl: δ-field live" "$repl_out" "repl starts"
a2a_assert_grep "qloop=1" "$repl_out" "repl reports default qloop route limit"
a2a_assert_grep "userRep=1.30" "$repl_out" "repl reports default direct-user repetition penalty"
a2a_assert_grep "userKV=0.05" "$repl_out" "repl reports default direct-user KV weight"
a2a_assert_grep "userTok=16" "$repl_out" "repl reports default direct-user answer length"
a2a_assert_grep "replFmt=user_arianna" "$repl_out" "repl reports default outer prompt format"
a2a_assert_grep "repl turn 1" "$repl_out" "repl runs one scripted turn"
a2a_assert_grep "I_N\\^kv\\[sem\\]" "$repl_out" "repl reports semantic neighbour influence"
a2a_assert_grep "qloop user.*\\[user-kv\\]" "$repl_out" "repl routes user question through KV bridge"
a2a_assert_grep "I_U\\^kv=" "$repl_out" "repl reports user KV influence"
a2a_assert_grep "no-user-kv:" "$repl_out" "repl reports no-user-KV contrast answer"
a2a_assert_grep "repl done" "$repl_out" "repl exits on command"

repl_qa_out="$(printf "Why does the field remember?\n:q\n" | A2A_REPL_PROMPT_FORMAT=qa "$A2A_BIN" "$A2A_MODEL_F16" repl 3 4 1 2>&1)"
a2a_assert_grep "replFmt=qa" "$repl_qa_out" "repl accepts Q/A outer prompt format"
a2a_assert_grep "qloop user.*\\[user-kv\\]" "$repl_qa_out" "Q/A REPL format keeps user question bridge"

repl_sweep_prompts="$(mktemp)"
printf "Why does the field remember?\n" > "$repl_sweep_prompts"
repl_sweep_out="$(A2A_CELLS=3 A2A_FRAG=4 A2A_ROUNDS=1 bash "$A2A_ROOT/tools/repl_question_sweep.sh" "$repl_sweep_prompts" 2>&1)"
rm -f "$repl_sweep_prompts"
a2a_assert_grep "^question[[:space:]]+user_bridge[[:space:]]+user_routes" "$repl_sweep_out" "repl question sweep reports TSV header"
a2a_assert_grep "user_targets[[:space:]]+user_scores[[:space:]]+user_answers[[:space:]]+user_answers_off" "$repl_sweep_out" "repl question sweep reports route diagnostics"
a2a_assert_grep "Why does the field remember\\?[[:space:]]+1[[:space:]]+1" "$repl_sweep_out" "repl question sweep sees user bridge"
a2a_assert_grep "Why does the field remember\\?.*c[0-9][[:space:]]+[0-9]" "$repl_sweep_out" "repl question sweep captures route target and score"

clean_sweep_prompts="$(mktemp)"
printf "I do not need to be remembered, but what happens if Arianna remembers me anyway?\n" > "$clean_sweep_prompts"
clean_sweep_tsv="$(mktemp)"
A2A_CELLS=3 A2A_FRAG=4 A2A_ROUNDS=1 bash "$A2A_ROOT/tools/repl_question_sweep.sh" "$clean_sweep_prompts" > "$clean_sweep_tsv" 2>&1
clean_sweep_summary="$(bash "$A2A_ROOT/tools/repl_tsv_summary.sh" "$clean_sweep_tsv" 2>&1)"
rm -f "$clean_sweep_prompts" "$clean_sweep_tsv"
a2a_assert_grep "answer_bad_start: 0/1" "$clean_sweep_summary" "repl user bridge suppresses bad answer starts"
a2a_assert_grep "answer_quality: any [0-9]+/1, short [0-9]+, question_like 0, label_artifact [0-9]+, notation_artifact [0-9]+, morph_artifact [0-9]+, recipient_artifact [0-9]+, tail_artifact [0-9]+, yes_no_start 0" "$clean_sweep_summary" "repl user bridge suppresses question and yes/no answer forms"

life_out="$("$A2A_BIN" "$A2A_MODEL_F16" "resonance" life 2 4 3 2>&1)"
a2a_assert_grep "δ-life: Game of Life" "$life_out" "life starts"
a2a_assert_grep "births" "$life_out" "life reports births"
a2a_assert_grep "deaths" "$life_out" "life reports deaths"
a2a_assert_grep "pop ended" "$life_out" "life completes"
