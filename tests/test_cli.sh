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

summary_tsv="$(mktemp)"
baseline_tsv="$(mktemp)"
printf "question\tuser_bridge\tuser_routes\tavg_i_u_kv\tavg_i_n_kv\tuser_targets\tuser_scores\tuser_answers\n" > "$summary_tsv"
printf "alpha?\t1\t1\t0.500\t-0.100\tc1\t1.200\tanswer a\n" >> "$summary_tsv"
printf "beta?\t1\t2\t-0.250\t0.300\tc2;c0\t0.900;0.800\t*bad;answer c\n" >> "$summary_tsv"
printf "gamma?\t0\t0\tnan\t0.000\t\t\t\n" >> "$summary_tsv"
printf "question\tuser_bridge\tuser_routes\tavg_i_u_kv\tavg_i_n_kv\tuser_targets\tuser_scores\tuser_answers\n" > "$baseline_tsv"
printf "alpha?\t1\t1\t0.250\t-0.200\tc1\t1.100\tanswer a\n" >> "$baseline_tsv"
printf "beta?\t1\t1\t-0.500\t0.100\tc1\t0.850\tanswer old\n" >> "$baseline_tsv"
printf "gamma?\t1\t1\t0.000\t0.000\tc0\t0.700\tanswer g\n" >> "$baseline_tsv"
summary_out="$(bash "$A2A_ROOT/tools/repl_tsv_summary.sh" "$summary_tsv" "$baseline_tsv" 2>&1)"
rm -f "$summary_tsv" "$baseline_tsv"
a2a_assert_grep "rows: 3" "$summary_out" "repl TSV summary counts rows"
a2a_assert_grep "user_bridge: 2/3" "$summary_out" "repl TSV summary reports bridge coverage"
a2a_assert_grep "I_U\\^kv: avg \\+0\\.125" "$summary_out" "repl TSV summary reports user influence average"
a2a_assert_grep "route_targets:" "$summary_out" "repl TSV summary reports route targets"
a2a_assert_grep "answer_bad_start: 1/3" "$summary_out" "repl TSV summary reports bad answer starts"
a2a_assert_grep "delta vs baseline" "$summary_out" "repl TSV summary compares baseline"
a2a_assert_grep "per-question diff" "$summary_out" "repl TSV summary reports per-question diff"
