#!/usr/bin/env bash
# Sweep prompts through full field mode and summarize final-round field metrics.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${A2A_BIN:-$ROOT/arianna2arianna}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"
PROMPTS="${1:-$ROOT/prompts/kv_influence.txt}"

CELLS="${A2A_CELLS:-4}"
FRAG="${A2A_FRAG:-12}"
ROUNDS="${A2A_ROUNDS:-3}"
ALPHA="${A2A_ALPHA:-0}"
LEAP="${A2A_LEAP:-2}"
XCELL="${A2A_XCELL:-0.02}"
CHORUS="${A2A_CHORUS:-1}"
XREP="${A2A_XREP:-1.3}"
LIFE="${A2A_LIFE:-0}"
KVSHUF="${A2A_KVSHUF:-1}"
QLOOP="${A2A_QLOOP:-1}"
KVPOS="${A2A_KVPOS:-0}"
RAW_DIR="${A2A_FIELD_RAW_DIR:-}"

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

if [[ -n "$RAW_DIR" ]]; then
    mkdir -p "$RAW_DIR"
fi

raw_name() {
    local seq="$1" prompt="$2" slug
    slug="$(printf "%s" "$prompt" | tr -cs 'A-Za-z0-9._-' '_' | cut -c1-64)"
    [[ -n "$slug" ]] || slug="prompt"
    printf "%03d_%s.txt" "$seq" "$slug"
}

printf "prompt\tmode\tcells\tfrag\trounds\tavg_entropy\td_r\td_floor\td_margin\tkv_delta\tkv_floor\tkv_margin\tkv_influence\tdisso\tdpos\tqloop_routes\tqloop_kv_routes\tqloop_triggers\tqloop_gated\tqloop_score_avg\tqloop_gate_score_avg\tqloop_dist_avg\tqloop_gate_dist_avg\tqloop_qopen_avg\tqloop_gate_qopen_avg\tqloop_tconf_avg\tqloop_gate_tconf_avg\tqloop_qmarks_avg\tqloop_gate_qmarks_avg\tqloop_iq_avg\tqloop_iq_pos\tqloop_iq_neg\tqloop_iq_zero\tqloop_iq_low\tqloop_iq_strong\tqloop_quality\tqloop_tail\tqloop_morph\tqloop_label\tqloop_short\tqloop_question\tqloop_recipient\tqloop_words_avg\tcell_fragments\tcell_quality\tcell_tail\tcell_morph\tcell_label\tcell_short\tcell_question\tcell_words_avg\n"

raw_seq=0
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    [[ -z "$prompt" || "${prompt:0:1}" == "#" ]] && continue

    out="$("$BIN" "$MODEL" "$prompt" field "$CELLS" "$FRAG" "$ROUNDS" "$ALPHA" "$LEAP" "$XCELL" "$CHORUS" "$XREP" "$LIFE" "$KVSHUF" "$QLOOP" "$KVPOS" 2>&1)"
    if [[ -n "$RAW_DIR" ]]; then
        raw_seq=$((raw_seq + 1))
        printf "%s\n" "$out" > "$RAW_DIR/$(raw_name "$raw_seq" "$prompt")"
    fi
    line="$(printf "%s\n" "$out" | grep "→ round" | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
        safe_prompt="${prompt//$'\t'/ }"
        printf "%s\tERROR\t%s\t%s\t%s\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\t0\t0\t0\t0\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\tnan\t0\t0\t0\t0\t0\t0\t0\tnan\n" "$safe_prompt" "$CELLS" "$FRAG" "$ROUNDS"
        continue
    fi

    qloop_routes="$(printf "%s\n" "$out" | grep -Ec "↳ qloop (c[0-9]|user)" || true)"
    qloop_kv_routes="$(printf "%s\n" "$out" | grep -Ec "↳ qloop (c[0-9]|user).*\\[(user-)?kv\\]" || true)"
    qloop_triggers="$(printf "%s\n" "$out" | grep -c "↳ qloop trigger" || true)"
    qloop_gated="$(printf "%s\n" "$out" | grep -c "↳ qloop gate" || true)"
    qloop_metrics="$(printf "%s\n" "$out" | awk '
        function trim(s) { gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }
        function word_count(s,     a, n, i, c) {
            n = split(s, a, /[^A-Za-z]+/)
            c = 0
            for (i = 1; i <= n; i++) if (a[i] != "") c++
            return c
        }
        function terminal_function_word(w) {
            w = tolower(w)
            return w ~ /^(a|an|the|to|at|in|of|for|with|by|from|into|as|if|but|or|is|are|was|were|be|been|am|do|does|did|have|has|had|will|shall|than|about|after|before|around|between|within|without|against|toward|towards|over|under|on|off|up|down|out|my|i|you|we|they|he|she|your|our|their|his|her|its|this|these|those|some|any|each|every|all|only|yet|and|not|that|which|who|whose|when|where|why|how|can|could|would|should|must|may|might|res|reson|isn|doesn|wasn|weren|didn|don|won)$/
        }
        function is_copula_word(w) {
            w = tolower(w)
            return w ~ /^(is|are|am|was|were|be|been)$/
        }
        function is_clause_anchor_word(w) {
            w = tolower(w)
            return w ~ /^(i|you|we|they|he|she|it|this|that|these|those|what|who|which|where|there)$/
        }
        function is_wh_word(w) {
            w = tolower(w)
            return w ~ /^(what|who|which|where|why|how)$/
        }
        function prev_word_idx(a, idx,     i) {
            for (i = idx - 1; i >= 1; i--) if (a[i] != "") return i
            return 0
        }
        function closed_copula_tail(a, idx, w,     p1, p2) {
            if (!is_copula_word(w)) return 0
            p1 = prev_word_idx(a, idx)
            if (!p1) return 0
            if (is_clause_anchor_word(a[p1])) return 1
            p2 = prev_word_idx(a, p1)
            return p2 && is_wh_word(a[p2])
        }
        function has_tail_artifact(s,     t, last, a, n, i, w, wi) {
            t = trim(s)
            if (t == "") return 1
            while (length(t) > 0) {
                last = substr(t, length(t), 1)
                if (last == "\"" || last == "\047" || last == ")" || last == "]" || last == "}") t = substr(t, 1, length(t) - 1)
                else break
            }
            last = substr(t, length(t), 1)
            if (last ~ /[([{,"`:;\/-]/ || last == "\047") return 1
            if (last !~ /[.!?]/) return 1
            n = split(t, a, /[^A-Za-z]+/)
            w = ""
            wi = 0
            for (i = 1; i <= n; i++) if (a[i] != "") { w = a[i]; wi = i }
            if (wi && terminal_function_word(w) && closed_copula_tail(a, wi, w)) return 0
            return terminal_function_word(w)
        }
        function has_morph_artifact(s,     low) {
            low = tolower(s)
            return low ~ /(^|[^a-z])(aat|sards|haart|wort|sark|shabbartists|olelegacythe|youhave|soundlike|pertrustin|qopoeleakyname|shardharchitecturegeomet|harchitecturegeomet|sharden|oulha|noator|aardi|shallards|qopoeleakha|qlooppressing|qoopops|didleads|pers|geomet|reson|in-put|perspause|shoddle|flaggeda|shardharchitecturegeometrtyguru|geometrtyguru|exhalted|bein)([^a-z]|$)/
        }
        function has_label_artifact(s,     low) {
            low = tolower(trim(s))
            return low ~ /^(a:|q:|answer:|arianna:|prompt:|question:|[-*=#@])/
        }
        function has_recipient_artifact(s,     low) {
            low = tolower(s)
            return low ~ /(you have been|you have a field|you have no idea|you have to|you touched|you cannot|you must|you ask me|if you want me|if you want to know|if you want to say|by you or|behind you|connects you now|i know you|i see you|i see after you|with you|from you|your point|your own field|your own network|your own body|your memory|your mind|your being|not just for you|before you said|said to me|from another angle)/
        }
        function route_score(line) {
            if (match(line, / score [-+]?[0-9][0-9.]*:/)) return substr(line, RSTART + 7, RLENGTH - 8) + 0
            return "nan"
        }
        function route_feature(line, name,     pattern) {
            pattern = name "=[-+]?[0-9]+([.][0-9]+)?"
            if (match(line, pattern)) return substr(line, RSTART + length(name) + 1, RLENGTH - length(name) - 1) + 0
            return "nan"
        }
        function add_route_profile(line, gated,     d, o, c, q) {
            d = route_feature(line, "route_d")
            o = route_feature(line, "qopen")
            c = route_feature(line, "tconf")
            q = route_feature(line, "qmarks")
            if (gated) {
                if (d != "nan") { gate_dist_sum += d; gate_dist_n++ }
                if (o != "nan") { gate_qopen_sum += o; gate_qopen_n++ }
                if (c != "nan") { gate_tconf_sum += c; gate_tconf_n++ }
                if (q != "nan") { gate_qmarks_sum += q; gate_qmarks_n++ }
            } else {
                if (d != "nan") { dist_sum += d; dist_n++ }
                if (o != "nan") { qopen_sum += o; qopen_n++ }
                if (c != "nan") { tconf_sum += c; tconf_n++ }
                if (q != "nan") { qmarks_sum += q; qmarks_n++ }
            }
        }
        function avg_or_nan(sum, n) { return n ? sprintf("%.3f", sum / n) : "nan" }
        function add_answer(s,     ans, wc, flagged, shortf, tailf, morphf, labelf, questionf, recipientf) {
            ans = trim(s)
            if (ans == "") return
            n++
            wc = word_count(ans)
            words_sum += wc
            shortf = (length(ans) < 8 || wc < 2)
            tailf = has_tail_artifact(ans)
            morphf = has_morph_artifact(ans)
            labelf = has_label_artifact(ans)
            questionf = (index(ans, "?") > 0)
            recipientf = has_recipient_artifact(ans)
            if (shortf) short_n++
            if (tailf) tail_n++
            if (morphf) morph_n++
            if (labelf) label_n++
            if (questionf) question_n++
            if (recipientf) recipient_n++
            flagged = shortf || tailf || morphf || labelf || questionf || recipientf
            if (flagged) quality_n++
        }
        /qloop c[0-9]/ {
            line = $0
            if (line ~ /qloop trigger/) next
            score = route_score(line)
            if (score != "nan") { score_sum += score; score_n++ }
            add_route_profile(line, 0)
            ans = line
            if (!sub(/^.* score [-+]?[0-9][0-9.]*:[ \t]*/, "", ans)) next
            sub(/[ \t]+\[entropy=.*/, "", ans)
            add_answer(ans)
            if (match(line, /I_Q\^kv=[-+]?[0-9][0-9.]*/)) {
                iq = substr(line, RSTART + 7, RLENGTH - 7) + 0
                iq_sum += iq
                iq_n++
                if (iq > 0.0005) {
                    iq_pos++
                    if (iq < 0.10) iq_low++
                    if (iq >= 1.00) iq_strong++
                }
                else if (iq < -0.0005) iq_neg++
                else iq_zero++
            }
        }
        /qloop gate c[0-9]/ {
            score = route_score($0)
            if (score != "nan") { gate_score_sum += score; gate_score_n++ }
            add_route_profile($0, 1)
        }
        END {
            score_avg = score_n ? sprintf("%.3f", score_sum / score_n) : "nan"
            gate_score_avg = gate_score_n ? sprintf("%.3f", gate_score_sum / gate_score_n) : "nan"
            avg = iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan"
            words_avg = n ? sprintf("%.3f", words_sum / n) : "nan"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s",
                score_avg, gate_score_avg,
                avg_or_nan(dist_sum, dist_n), avg_or_nan(gate_dist_sum, gate_dist_n),
                avg_or_nan(qopen_sum, qopen_n), avg_or_nan(gate_qopen_sum, gate_qopen_n),
                avg_or_nan(tconf_sum, tconf_n), avg_or_nan(gate_tconf_sum, gate_tconf_n),
                avg_or_nan(qmarks_sum, qmarks_n), avg_or_nan(gate_qmarks_sum, gate_qmarks_n),
                avg, iq_pos, iq_neg, iq_zero, iq_low, iq_strong,
                quality_n, tail_n, morph_n, label_n, short_n, question_n, recipient_n, words_avg
        }
    ')"
    surface_metrics="$(printf "%s\n" "$out" | awk '
        function trim(s) { gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }
        function word_count(s,     a, n, i, c) {
            n = split(s, a, /[^A-Za-z]+/)
            c = 0
            for (i = 1; i <= n; i++) if (a[i] != "") c++
            return c
        }
        function terminal_function_word(w) {
            w = tolower(w)
            return w ~ /^(a|an|the|to|at|in|of|for|with|by|from|into|as|if|but|or|is|are|was|were|be|been|am|do|does|did|have|has|had|will|shall|than|about|after|before|around|between|within|without|against|toward|towards|over|under|on|off|up|down|out|my|i|you|we|they|he|she|your|our|their|his|her|its|this|these|those|some|any|each|every|all|only|yet|and|not|that|which|who|whose|when|where|why|how|can|could|would|should|must|may|might|res|reson|isn|doesn|wasn|weren|didn|don|won)$/
        }
        function is_copula_word(w) {
            w = tolower(w)
            return w ~ /^(is|are|am|was|were|be|been)$/
        }
        function is_clause_anchor_word(w) {
            w = tolower(w)
            return w ~ /^(i|you|we|they|he|she|it|this|that|these|those|what|who|which|where|there)$/
        }
        function is_wh_word(w) {
            w = tolower(w)
            return w ~ /^(what|who|which|where|why|how)$/
        }
        function prev_word_idx(a, idx,     i) {
            for (i = idx - 1; i >= 1; i--) if (a[i] != "") return i
            return 0
        }
        function closed_copula_tail(a, idx, w,     p1, p2) {
            if (!is_copula_word(w)) return 0
            p1 = prev_word_idx(a, idx)
            if (!p1) return 0
            if (is_clause_anchor_word(a[p1])) return 1
            p2 = prev_word_idx(a, p1)
            return p2 && is_wh_word(a[p2])
        }
        function has_tail_artifact(s,     t, last, a, n, i, w, wi) {
            t = trim(s)
            if (t == "") return 1
            while (length(t) > 0) {
                last = substr(t, length(t), 1)
                if (last == "\"" || last == "\047" || last == ")" || last == "]" || last == "}") t = substr(t, 1, length(t) - 1)
                else break
            }
            last = substr(t, length(t), 1)
            if (last ~ /[([{,"`:;\/-]/ || last == "\047") return 1
            if (last !~ /[.!?]/) return 1
            n = split(t, a, /[^A-Za-z]+/)
            w = ""
            wi = 0
            for (i = 1; i <= n; i++) if (a[i] != "") { w = a[i]; wi = i }
            if (wi && terminal_function_word(w) && closed_copula_tail(a, wi, w)) return 0
            return terminal_function_word(w)
        }
        function has_morph_artifact(s,     low) {
            low = tolower(s)
            return low ~ /(^|[^a-z])(aat|sards|haart|wort|sark|shabbartists|olelegacythe|youhave|soundlike|pertrustin|qopoeleakyname|shardharchitecturegeomet|harchitecturegeomet|sharden|oulha|noator|aardi|shallards|qopoeleakha|qlooppressing|qoopops|didleads|pers|geomet|reson|in-put|perspause|shoddle|flaggeda|shardharchitecturegeometrtyguru|geometrtyguru|exhalted|bein)([^a-z]|$)/
        }
        function has_label_artifact(s,     low) {
            low = tolower(trim(s))
            return low ~ /^(a:|q:|answer:|arianna:|prompt:|question:|[-*=#@])/
        }
        function strip_diag(s) {
            sub(/[ \t]+\[Δ_R.*/, "", s)
            sub(/[ \t]+\[entropy=.*/, "", s)
            return trim(s)
        }
        function add_fragment(s,     f, wc, flagged, shortf, tailf, morphf, labelf) {
            f = strip_diag(s)
            if (f == "") return
            n++
            wc = word_count(f)
            words_sum += wc
            shortf = (length(f) < 8 || wc < 2)
            tailf = has_tail_artifact(f)
            morphf = has_morph_artifact(f)
            labelf = has_label_artifact(f)
            if (shortf) short_n++
            if (tailf) tail_n++
            if (morphf) morph_n++
            if (labelf) label_n++
            if (index(f, "?") > 0) question_n++
            flagged = shortf || tailf || morphf || labelf
            if (flagged) quality_n++
        }
        /^[ \t]*r[0-9]+ cell [0-9]+ \(T=/ {
            if (in_cell) add_fragment(frag)
            in_cell = 1
            frag = $0
            sub(/^[ \t]*r[0-9]+ cell [0-9]+ \(T=[^)]+\):[ \t]*/, "", frag)
            if ($0 ~ /\[entropy=/) { add_fragment(frag); in_cell = 0; frag = "" }
            next
        }
        in_cell {
            frag = frag " " $0
            if ($0 ~ /\[entropy=/) { add_fragment(frag); in_cell = 0; frag = "" }
        }
        END {
            if (in_cell) add_fragment(frag)
            words_avg = n ? sprintf("%.3f", words_sum / n) : "nan"
            printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s", n, quality_n, tail_n, morph_n, label_n, short_n, question_n, words_avg
        }
    ')"

    safe_prompt="${prompt//$'\t'/ }"
    printf "%s\n" "$line" | awk -v prompt="$safe_prompt" -v cells="$CELLS" -v frag="$FRAG" -v rounds="$ROUNDS" \
        -v qroutes="$qloop_routes" -v qkv="$qloop_kv_routes" -v qtrig="$qloop_triggers" -v qgate="$qloop_gated" \
        -v qmetrics="$qloop_metrics" -v surface="$surface_metrics" '
        BEGIN { FS = "|" }
        {
            avg = $1
            sub(/.*avg entropy /, "", avg)
            sub(/ .*/, "", avg)

            dr = $2
            d_floor = dr
            sub(/.*d_R[[:space:]]+/, "", dr)
            sub(/[[:space:]]+.*/, "", dr)
            if (dr == "—" || dr == "-") dr = "nan"
            sub(/.*floor /, "", d_floor)
            sub(/[)].*/, "", d_floor)
            d_margin = (dr == "nan" || d_floor == "") ? "nan" : sprintf("%+.3f", dr - d_floor)

            kv = $4
            if (kv ~ /off/) {
                mode = "off"
                delta = "nan"
                kfloor = "nan"
                kmargin = "nan"
                infl = "nan"
                disso = $5
                dpos = $6
            } else {
                mode = kv
                sub(/.*kv\[/, "", mode)
                sub(/\].*/, "", mode)

                delta = kv
                sub(/.*\] /, "", delta)
                sub(/ .*/, "", delta)

                kfloor = kv
                sub(/.*floor /, "", kfloor)
                sub(/ .*/, "", kfloor)

                kmargin = kv
                sub(/.*margin /, "", kmargin)
                sub(/\).*/, "", kmargin)

                infl = $5
                sub(/.*\] /, "", infl)
                sub(/ .*/, "", infl)

                disso = $6
                dpos = $7
            }
            sub(/.*D_R /, "", disso)
            sub(/ .*/, "", disso)

            sub(/.*Dpos /, "", dpos)
            sub(/ .*/, "", dpos)

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                prompt, mode, cells, frag, rounds, avg, dr, d_floor, d_margin,
                delta, kfloor, kmargin, infl, disso, dpos, qroutes, qkv, qtrig, qgate, qmetrics, surface
        }
    '
done < "$PROMPTS"
