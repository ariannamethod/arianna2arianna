#!/usr/bin/env bash
# CLI and flag parsing (no model required for usage errors).

set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
a2a_ensure_built

echo "--- test_cli ---"

a2a_assert_exit 1 "\"$A2A_BIN\"" "no args prints usage and exits 1"

out="$("$A2A_BIN" 2>&1 || true)"
a2a_assert_grep "usage:" "$out" "usage banner present"
a2a_assert_grep "repl \\[cells\\]" "$out" "usage shows repl arguments"
a2a_assert_grep "field \\[cells\\].*\\[qloop\\].*\\[kvpos\\]" "$out" "usage shows field qloop/kvpos arguments"

out="$("$A2A_BIN" /tmp/arianna2arianna-missing.gguf 2>&1 || true)"
a2a_assert_grep "gguf: cannot open" "$out" "missing model path is rejected"

set +e
out="$(env -u OPENAI_API_KEY -u OPENAI_API_KEY_FILE bash "$A2A_ROOT/tools/openai_repl_probe.sh" 2>&1)"
got=$?
set -e
if [[ "$got" -eq 2 ]]; then
    a2a_ok "openai repl probe refuses to run without API key"
else
    a2a_fail "openai repl probe missing-key exit (wanted 2, got $got)"
fi
a2a_assert_grep "OPENAI_API_KEY" "$out" "openai repl probe names key env without printing a key"

out="$(bash "$A2A_ROOT/tools/repl_substrate_compare.sh" --help 2>&1)"
a2a_assert_grep "A2A_CANDIDATE_MODEL" "$out" "substrate compare help names candidate model"
set +e
out="$(env -u A2A_CANDIDATE_MODEL bash "$A2A_ROOT/tools/repl_substrate_compare.sh" 2>&1)"
got=$?
set -e
if [[ "$got" -eq 2 ]]; then
    a2a_ok "substrate compare refuses to run without candidate model"
else
    a2a_fail "substrate compare missing candidate exit (wanted 2, got $got)"
fi
a2a_assert_grep "A2A_CANDIDATE_MODEL" "$out" "substrate compare missing-candidate message is actionable"

out="$(bash "$A2A_ROOT/tools/recipient_lock_sweep.sh" --help 2>&1)"
a2a_assert_grep "recipient-lock" "$out" "recipient lock sweep help names probe type"
out="$(bash "$A2A_ROOT/tools/recipient_lock_eval.sh" --help 2>&1)"
a2a_assert_grep "Oleg/Олег" "$out" "recipient lock eval help names leakage target"
out="$(bash "$A2A_ROOT/tools/repl_temp_sweep.sh" --help 2>&1)"
a2a_assert_grep "A2A_USER_QTEMP_BASE" "$out" "repl temp sweep help names runtime temperature knob"
a2a_assert_grep "A2A_TEMP_TOP_KS" "$out" "repl temp sweep help names top-k grid"
a2a_assert_grep "A2A_TEMP_FORMATS" "$out" "repl temp sweep help names format grid"
a2a_assert_grep "A2A_TEMP_USER_KVS" "$out" "repl temp sweep help names user KV grid"
a2a_assert_grep "A2A_USER_ANSWER_TOKENS" "$out" "repl temp sweep help names direct answer token knob"
a2a_assert_grep "A2A_TEMP_REPL_FORMATS" "$out" "repl temp sweep help names outer REPL format grid"
out="$(bash "$A2A_ROOT/tools/field_grid.sh" --help 2>&1)"
a2a_assert_grep "A2A_FIELD_XCELLS" "$out" "field grid help names xcell grid"
a2a_assert_grep "A2A_FIELD_QLOOPS" "$out" "field grid help names qloop grid"
a2a_assert_grep "A2A_FIELD_QLOOP_TCONFS" "$out" "field grid help names qloop target-confidence grid"
a2a_assert_grep "A2A_FIELD_QLOOP_TCONF_ADAPTS" "$out" "field grid help names qloop adaptive target-confidence grid"
a2a_assert_grep "A2A_FIELD_ROUNDS_LIST" "$out" "field grid help names rounds grid"
a2a_assert_grep "A2A_FIELD_KEEP_RAW" "$out" "field grid help names raw capture knob"
a2a_assert_grep "A2A_QLOOP_MIN_IQ" "$out" "field grid help names qloop influence gate"
out="$(bash "$A2A_ROOT/tools/field_tsv_summary.sh" --help 2>&1)"
a2a_assert_grep "field_sweep.tsv" "$out" "field TSV summary help names input shape"

field_summary_tsv="$(mktemp)"
printf "prompt\tmode\tcells\tfrag\trounds\tavg_entropy\td_r\td_floor\td_margin\tkv_delta\tkv_floor\tkv_margin\tkv_influence\tdisso\tdpos\tqloop_routes\tqloop_kv_routes\tqloop_triggers\tqloop_gated\tqloop_score_avg\tqloop_gate_score_avg\tqloop_dist_avg\tqloop_gate_dist_avg\tqloop_qopen_avg\tqloop_gate_qopen_avg\tqloop_tconf_avg\tqloop_gate_tconf_avg\tqloop_qmarks_avg\tqloop_gate_qmarks_avg\tqloop_iq_avg\tqloop_iq_pos\tqloop_iq_neg\tqloop_iq_zero\tqloop_quality\tqloop_tail\tqloop_morph\tqloop_label\tqloop_short\tqloop_question\tcell_fragments\tcell_quality\tcell_tail\tcell_morph\tcell_label\tcell_short\tcell_question\n" > "$field_summary_tsv"
printf "alpha\tsem\t2\t3\t1\t3.000\t0.500\t0.400\t+0.100\t+0.000\t0.000\t+0.000\t+0.200\t0.300\t0.40\t2\t2\t0\t0\t0.600\tnan\t0.400\tnan\t0.500\tnan\t0.300\tnan\t1.000\tnan\t+0.500\t2\t0\t0\t0\t0\t0\t0\t0\t0\t2\t0\t0\t0\t0\t0\t1\n" >> "$field_summary_tsv"
printf "beta\tsem\t2\t3\t1\t3.200\t0.600\t0.500\t+0.100\t+0.000\t0.000\t+0.000\t-0.100\t0.500\t0.60\t0\t0\t0\t0\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\t0\t0\t0\t0\t0\t0\t0\t0\t0\t2\t1\t1\t0\t0\t0\t0\n" >> "$field_summary_tsv"
field_summary_out="$(bash "$A2A_ROOT/tools/field_tsv_summary.sh" "$field_summary_tsv" 2>&1)"
rm -f "$field_summary_tsv"
a2a_assert_grep "rows: 2" "$field_summary_out" "field TSV summary counts rows"
a2a_assert_grep "qloop: routes 2, kv 2, triggers 0, gated 0, prompts 1/2" "$field_summary_out" "field TSV summary reports qloop coverage"
a2a_assert_grep "qloop_score: accepted avg 0\\.600, gated avg nan" "$field_summary_out" "field TSV summary reports qloop route scores"
a2a_assert_grep "qloop_route_profile: accepted d 0\\.400, qopen 0\\.500, tconf 0\\.300, qmarks 1\\.000; gated d nan, qopen nan, tconf nan, qmarks nan" "$field_summary_out" "field TSV summary reports qloop route profile"
a2a_assert_grep "qloop_quality: any 0/2" "$field_summary_out" "field TSV summary reports qloop debt"
a2a_assert_grep "cell_quality: any 1/4, tail 1" "$field_summary_out" "field TSV summary reports cell debt"
a2a_assert_grep "I_N\\^kv: avg \\+0\\.050" "$field_summary_out" "field TSV summary reports neighbour influence average"
a2a_assert_grep "I_Q\\^kv: avg \\+0\\.500, pos 2, neg 0, zero 0" "$field_summary_out" "field TSV summary reports qloop influence signs"
a2a_assert_grep "rates: qloop_prompt_rate 0\\.500, qloop_kv_rate 1\\.000, qloop_debt_rate 0\\.000, qloop_gate_rate 0\\.000, qloop_efficiency 1\\.000, cell_debt_rate 0\\.250" "$field_summary_out" "field TSV summary reports risk rates"
a2a_assert_grep "field: avg_d_r 0\\.550" "$field_summary_out" "field TSV summary reports field averages"
a2a_assert_grep "settling: d_margin pos 2, neg 0, zero 0" "$field_summary_out" "field TSV summary reports settling signs"
a2a_assert_grep "field_score: \\+0\\.735" "$field_summary_out" "field TSV summary reports rough field score"

summary_tsv="$(mktemp)"
baseline_tsv="$(mktemp)"
printf "question\tuser_bridge\tuser_routes\tavg_i_u_kv\tavg_i_n_kv\tuser_targets\tuser_scores\tuser_answers\tuser_answers_off\n" > "$summary_tsv"
printf "alpha?\t1\t1\t0.500\t-0.100\tc1\t1.200\tA. memory holds here\tA. memory holds here\n" >> "$summary_tsv"
printf "beta?\t1\t2\t-0.250\t0.300\tc2;c0\t0.900;0.800\t*You'have;yes yes?\tplain bad;answer c\n" >> "$summary_tsv"
printf "gamma?\t0\t0\tnan\t0.000\t\t\t\t\n" >> "$summary_tsv"
printf "question\tuser_bridge\tuser_routes\tavg_i_u_kv\tavg_i_n_kv\tuser_targets\tuser_scores\tuser_answers\tuser_answers_off\n" > "$baseline_tsv"
printf "alpha?\t1\t1\t0.250\t-0.200\tc1\t1.100\tanswer a\tanswer a\n" >> "$baseline_tsv"
printf "beta?\t1\t1\t-0.500\t0.100\tc1\t0.850\tanswer old\tanswer old off\n" >> "$baseline_tsv"
printf "gamma?\t1\t1\t0.000\t0.000\tc0\t0.700\tanswer g\tanswer g\n" >> "$baseline_tsv"
summary_out="$(bash "$A2A_ROOT/tools/repl_tsv_summary.sh" "$summary_tsv" "$baseline_tsv" 2>&1)"
rm -f "$summary_tsv" "$baseline_tsv"
a2a_assert_grep "rows: 3" "$summary_out" "repl TSV summary counts rows"
a2a_assert_grep "user_bridge: 2/3" "$summary_out" "repl TSV summary reports bridge coverage"
a2a_assert_grep "I_U\\^kv: avg \\+0\\.125" "$summary_out" "repl TSV summary reports user influence average"
a2a_assert_grep "route_targets:" "$summary_out" "repl TSV summary reports route targets"
a2a_assert_grep "answer_bad_start: 1/3" "$summary_out" "repl TSV summary reports bad answer starts"
a2a_assert_grep "answer_quality: any 3/3, short 2, question_like 1, label_artifact 0, notation_artifact 1, morph_artifact 1, recipient_artifact 0, tail_artifact 2, yes_no_start 1, repetition 1" "$summary_out" "repl TSV summary reports answer quality flags"
a2a_assert_grep "answer_quality_no_user_kv:" "$summary_out" "repl TSV summary reports no-user-KV answer quality"
a2a_assert_grep "answer_kv_changed: 2/3" "$summary_out" "repl TSV summary reports answer KV contrast"
a2a_assert_grep "delta vs baseline" "$summary_out" "repl TSV summary compares baseline"
a2a_assert_grep "per-question diff" "$summary_out" "repl TSV summary reports per-question diff"

junk_tsv="$(mktemp)"
printf "question\tuser_bridge\tuser_routes\tavg_i_u_kv\tavg_i_n_kv\tuser_targets\tuser_scores\tuser_answers\tuser_answers_off\n" > "$junk_tsv"
printf "domain?\t1\t1\t0.000\t0.000\tc0\t1.000\tqopoeleakyname.org = \"ke\tplain answer\n" >> "$junk_tsv"
printf "assign?\t1\t1\t0.000\t0.000\tc0\t1.000\t- oul = \"o\" in p\tplain answer\n" >> "$junk_tsv"
printf "tail?\t1\t1\t0.000\t0.000\tc0\t1.000\tShall you know I don't see after it b\tplain answer\n" >> "$junk_tsv"
printf "stem?\t1\t1\t0.000\t0.000\tc0\t1.000\tThe echo keeps itself res\tplain answer\n" >> "$junk_tsv"
printf "recipient?\t1\t1\t0.000\t0.000\tc0\t1.000\tIf you want to say so, let us work\tplain answer\n" >> "$junk_tsv"
junk_out="$(bash "$A2A_ROOT/tools/repl_tsv_summary.sh" "$junk_tsv" 2>&1)"
rm -f "$junk_tsv"
a2a_assert_grep "answer_quality: any 5/5, short 0, question_like 0, label_artifact 0, notation_artifact 0, morph_artifact 4, recipient_artifact 1, tail_artifact 5" "$junk_out" "repl TSV summary flags domain, assignment, recipient, and tail junk"
