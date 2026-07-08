#!/usr/bin/env bash
# Sweep direct user questions through the live REPL bridge and print TSV metrics.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${A2A_BIN:-$ROOT/arianna2arianna}"
MODEL="${A2A_MODEL:-$ROOT/weights/nanollama-arianna-full-v4-step2750-f16.gguf}"
PROMPTS="${1:-$ROOT/prompts/repl_questions.txt}"

CELLS="${A2A_CELLS:-5}"
FRAG="${A2A_FRAG:-12}"
ROUNDS="${A2A_ROUNDS:-1}"

if [[ ! -x "$BIN" ]]; then
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

avg_metric() {
    local label="$1"
    awk -v label="$label" '
        {
            while ((p = index($0, label)) > 0) {
                rest = substr($0, p + length(label))
                if (match(rest, /^[+-]?[0-9]+(\.[0-9]+)?/)) {
                    v = substr(rest, RSTART, RLENGTH) + 0
                    sum += v; n++
                }
                $0 = substr(rest, RLENGTH + 1)
            }
        }
        END { if (n) printf "%.3f", sum / n; else printf "nan" }
    '
}

route_fields() {
    awk '
        function clean(s) {
            gsub(/\t/, " ", s)
            gsub(/\r/, " ", s)
            gsub(/\n/, " ", s)
            gsub(/;/, ",", s)
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }
        function append(list, val) {
            return list == "" ? val : list ";" val
        }
        /qloop user/ {
            line = $0

            if (match(line, /qloop user[^ ]*c[0-9][0-9]*/)) {
                target = substr(line, RSTART, RLENGTH)
                sub(/^.*c/, "c", target)
                targets = append(targets, target)
            }

            if (match(line, / score [-+]?[0-9][0-9.]*:/)) {
                score = substr(line, RSTART + 7, RLENGTH - 8)
                scores = append(scores, score)
            }

            answer = line
            sub(/^.* score [-+]?[0-9][0-9.]*:/, "", answer)
            sub(/[[:space:]]+\[entropy=.*/, "", answer)
            answer = clean(answer)
            if (answer != "") answers = append(answers, answer)
        }
        END {
            printf "%s\t%s\t%s\n", targets, scores, answers
        }
    '
}

printf "question\tuser_bridge\tuser_routes\tavg_i_u_kv\tavg_i_n_kv\tuser_targets\tuser_scores\tuser_answers\n"

while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    [[ -z "$prompt" || "${prompt:0:1}" == "#" ]] && continue

    out="$(printf "%s\n:q\n" "$prompt" | "$BIN" "$MODEL" repl "$CELLS" "$FRAG" "$ROUNDS" 2>&1)"
    routes="$(printf "%s\n" "$out" | grep -c "qloop user" || true)"
    bridge=0
    if [[ "$routes" -gt 0 ]]; then bridge=1; fi
    avg_iu="$(printf "%s\n" "$out" | avg_metric "I_U^kv=")"
    avg_in="$(printf "%s\n" "$out" | avg_metric "I_N^kv[sem] ")"
    route_diag="$(printf "%s\n" "$out" | route_fields)"
    IFS=$'\t' read -r user_targets user_scores user_answers <<< "$route_diag"

    safe_prompt="${prompt//$'\t'/ }"
    printf "%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n" \
        "$safe_prompt" "$bridge" "$routes" "$avg_iu" "$avg_in" \
        "$user_targets" "$user_scores" "$user_answers"
done < "$PROMPTS"
