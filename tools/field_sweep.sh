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

printf "prompt\tmode\tcells\tfrag\trounds\tavg_entropy\td_r\td_floor\td_margin\tkv_delta\tkv_floor\tkv_margin\tkv_influence\tdisso\tdpos\tqloop_routes\tqloop_kv_routes\tqloop_triggers\tqloop_iq_avg\tqloop_quality\tqloop_tail\tqloop_morph\tqloop_label\tqloop_short\tqloop_question\tcell_fragments\tcell_quality\tcell_tail\tcell_morph\tcell_label\tcell_short\tcell_question\n"

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
        printf "%s\tERROR\t%s\t%s\t%s\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\t0\t0\t0\tnan\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n" "$safe_prompt" "$CELLS" "$FRAG" "$ROUNDS"
        continue
    fi

    qloop_routes="$(printf "%s\n" "$out" | grep -Ec "↳ qloop (c[0-9]|user)" || true)"
    qloop_kv_routes="$(printf "%s\n" "$out" | grep -Ec "↳ qloop (c[0-9]|user).*\\[(user-)?kv\\]" || true)"
    qloop_triggers="$(printf "%s\n" "$out" | grep -c "↳ qloop trigger" || true)"
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
            return w ~ /^(a|an|the|to|at|in|of|for|with|by|from|into|as|if|but|or|is|are|was|were|be|been|am|do|does|did|have|has|had|will|shall|than|about|after|before|around|between|within|without|against|toward|towards|over|under|on|off|up|down|out|my|your|our|their|his|her|its|this|these|those|some|any|each|every|all|only|yet|and|not|that|which|who|whose|when|where|why|how|can|could|would|should|must|may|might|res|reson|isn|doesn|wasn|weren|didn|don|won)$/
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
            return low ~ /(^|[^a-z])(aat|sards|haart|wort|sark|shabbartists|olelegacythe|youhave|soundlike|pertrustin|qopoeleakyname|shardharchitecturegeomet|harchitecturegeomet|sharden|oulha|noator|aardi|shallards|qopoeleakha|qlooppressing|qoopops|didleads|pers|geomet|reson|in-put|perspause|shoddle|shardharchitecturegeometrtyguru|geometrtyguru|exhalted|bein)([^a-z]|$)/
        }
        function has_label_artifact(s,     low) {
            low = tolower(trim(s))
            return low ~ /^(a:|q:|answer:|arianna:|prompt:|question:|[-*=#@])/
        }
        function add_answer(s,     ans, flagged, shortf, tailf, morphf, labelf, questionf) {
            ans = trim(s)
            if (ans == "") return
            n++
            shortf = (length(ans) < 8 || word_count(ans) < 2)
            tailf = has_tail_artifact(ans)
            morphf = has_morph_artifact(ans)
            labelf = has_label_artifact(ans)
            questionf = (index(ans, "?") > 0)
            if (shortf) short_n++
            if (tailf) tail_n++
            if (morphf) morph_n++
            if (labelf) label_n++
            if (questionf) question_n++
            flagged = shortf || tailf || morphf || labelf || questionf
            if (flagged) quality_n++
        }
        /qloop c[0-9]/ {
            line = $0
            if (line ~ /qloop trigger/) next
            ans = line
            if (!sub(/^.* score [-+]?[0-9][0-9.]*:[ \t]*/, "", ans)) next
            sub(/[ \t]+\[entropy=.*/, "", ans)
            add_answer(ans)
            if (match(line, /I_Q\^kv=[-+]?[0-9][0-9.]*/)) {
                iq = substr(line, RSTART + 7, RLENGTH - 7) + 0
                iq_sum += iq
                iq_n++
            }
        }
        END {
            avg = iq_n ? sprintf("%+.3f", iq_sum / iq_n) : "nan"
            printf "%s\t%d\t%d\t%d\t%d\t%d\t%d", avg, quality_n, tail_n, morph_n, label_n, short_n, question_n
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
            return w ~ /^(a|an|the|to|at|in|of|for|with|by|from|into|as|if|but|or|is|are|was|were|be|been|am|do|does|did|have|has|had|will|shall|than|about|after|before|around|between|within|without|against|toward|towards|over|under|on|off|up|down|out|my|your|our|their|his|her|its|this|these|those|some|any|each|every|all|only|yet|and|not|that|which|who|whose|when|where|why|how|can|could|would|should|must|may|might|res|reson|isn|doesn|wasn|weren|didn|don|won)$/
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
            return low ~ /(^|[^a-z])(aat|sards|haart|wort|sark|shabbartists|olelegacythe|youhave|soundlike|pertrustin|qopoeleakyname|shardharchitecturegeomet|harchitecturegeomet|sharden|oulha|noator|aardi|shallards|qopoeleakha|qlooppressing|qoopops|didleads|pers|geomet|reson|in-put|perspause|shoddle|shardharchitecturegeometrtyguru|geometrtyguru|exhalted|bein)([^a-z]|$)/
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
        function add_fragment(s,     f, flagged, shortf, tailf, morphf, labelf) {
            f = strip_diag(s)
            if (f == "") return
            n++
            shortf = (length(f) < 8 || word_count(f) < 2)
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
            printf "%d\t%d\t%d\t%d\t%d\t%d\t%d", n, quality_n, tail_n, morph_n, label_n, short_n, question_n
        }
    ')"

    safe_prompt="${prompt//$'\t'/ }"
    printf "%s\n" "$line" | awk -v prompt="$safe_prompt" -v cells="$CELLS" -v frag="$FRAG" -v rounds="$ROUNDS" \
        -v qroutes="$qloop_routes" -v qkv="$qloop_kv_routes" -v qtrig="$qloop_triggers" \
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

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                prompt, mode, cells, frag, rounds, avg, dr, d_floor, d_margin,
                delta, kfloor, kmargin, infl, disso, dpos, qroutes, qkv, qtrig, qmetrics, surface
        }
    '
done < "$PROMPTS"
