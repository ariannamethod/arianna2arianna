#!/usr/bin/env bash
# Summarize and optionally compare REPL question sweep TSV files.

set -euo pipefail

usage() {
    cat <<EOF
usage: $0 current.tsv [baseline.tsv]

Reads tools/repl_question_sweep.sh TSV output and prints bridge/routing
coverage plus I_U^kv and I_N^kv sign statistics. Newer TSVs may also include
route targets, route scores, answer snippets, quality flags, and no-user-KV contrast snippets. If a baseline TSV is
provided, also prints aggregate and per-question deltas.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

current="$1"
baseline="${2:-}"

if [[ ! -f "$current" ]]; then
    echo "missing TSV: $current" >&2
    exit 1
fi

if [[ -n "$baseline" && ! -f "$baseline" ]]; then
    echo "missing baseline TSV: $baseline" >&2
    exit 1
fi

summarize() {
    local label="$1"
    local file="$2"
    awk -F '\t' -v label="$label" '
        function numeric(x) { return x ~ /^[-+]?[0-9]+([.][0-9]+)?$/ }
        function clean_route(x) { return x == "" ? "-" : x }
        function add_targets(s,     i, a, n) {
            if (s == "") return
            n = split(s, a, ";")
            for (i = 1; i <= n; i++) if (a[i] != "") target_count[a[i]]++
        }
        function add_scores(s, q,     i, a, n, v) {
            if (s == "") return
            n = split(s, a, ";")
            for (i = 1; i <= n; i++) {
                if (!numeric(a[i])) continue
                v = a[i] + 0
                score_sum += v
                score_n++
                if (!score_max_seen || v > score_max) {
                    score_max = v
                    score_max_q = q
                    score_max_seen = 1
                }
                if (!score_min_seen || v < score_min) {
                    score_min = v
                    score_min_q = q
                    score_min_seen = 1
                }
            }
        }
        function bad_answer_start(s,     first) {
            sub(/^[[:space:]]+/, "", s)
            first = substr(s, 1, 1)
            return first ~ /^[0-9]$/ || index("*\"`#:/@-\\", first) > 0 ||
                   first == sprintf("%c", 39) || s ~ /^(http|HTTP|www|WWW)/
        }
        function trim(s) {
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }
        function word_count(s,     i, a, n, w) {
            gsub(/[^[:alnum:]_]+/, " ", s)
            s = trim(s)
            if (s == "") return 0
            n = split(s, a, /[[:space:]]+/)
            w = 0
            for (i = 1; i <= n; i++) if (a[i] != "") w++
            return w
        }
        function has_repetition(s,     i, a, n, prev, run, w) {
            s = tolower(s)
            gsub(/[^[:alnum:]_]+/, " ", s)
            n = split(s, a, /[[:space:]]+/)
            prev = ""; run = 0
            for (i = 1; i <= n; i++) {
                w = a[i]
                if (length(w) < 2) continue
                if (w == prev) run++
                else { prev = w; run = 1 }
                if (run >= 2) return 1
            }
            return 0
        }
        function add_answer_quality(ans,     low, words, flagged) {
            ans = trim(ans)
            if (ans == "") return
            low = tolower(ans)
            words = word_count(ans)
            flagged = 0
            if (length(ans) < 12 || words < 3) { answer_short_n++; flagged = 1 }
            if (index(ans, "?") > 0) { answer_question_n++; flagged = 1 }
            if (low ~ /(^|[[:space:][:punct:]])(answer|reply|prompt|cell|thread|qloop|question)([[:space:][:punct:]]|$)/) {
                answer_label_n++; flagged = 1
            }
            if (low ~ /^(yes|no)([[:space:][:punct:]]|$)/) { answer_yesno_n++; flagged = 1 }
            if (has_repetition(ans)) { answer_repeat_n++; flagged = 1 }
            if (flagged) answer_quality_any_n++
        }
        function add_answers(s,     i, a, n, ans) {
            if (s == "") return
            n = split(s, a, ";")
            for (i = 1; i <= n; i++) {
                ans = trim(a[i])
                if (ans == "") continue
                answer_n++
                if (bad_answer_start(ans)) bad_answer_n++
                add_answer_quality(ans)
            }
        }
        function add_answer_contrasts(on, off, q,     i, a, b, n, m, ao, bo) {
            if (on == "" || off == "") return
            n = split(on, a, ";")
            m = split(off, b, ";")
            for (i = 1; i <= n && i <= m; i++) {
                ao = a[i]
                bo = b[i]
                sub(/^[[:space:]]+/, "", ao)
                sub(/[[:space:]]+$/, "", ao)
                sub(/^[[:space:]]+/, "", bo)
                sub(/[[:space:]]+$/, "", bo)
                if (ao == "" && bo == "") continue
                answer_contrast_n++
                if (ao != bo) {
                    answer_contrast_changed++
                    if (!first_contrast_seen) {
                        first_contrast_q = q
                        first_contrast_on = ao
                        first_contrast_off = bo
                        first_contrast_seen = 1
                    }
                }
            }
        }
        NR == 1 {
            if ($1 != "question" || $2 != "user_bridge" || $3 != "user_routes" ||
                $4 != "avg_i_u_kv" || $5 != "avg_i_n_kv") {
                bad_header = 1
            }
            route_cols = ($6 == "user_targets" && $7 == "user_scores" && $8 == "user_answers")
            contrast_cols = (route_cols && $9 == "user_answers_off")
            next
        }
        NF >= 5 {
            n++
            bridge += $2 + 0
            routes += $3 + 0

            if (numeric($4)) {
                v = $4 + 0
                iusum += v
                iun++
                if (v > 0) iupos++
                else if (v < 0) iuneg++
                else iuzero++
                if (!iumax_seen || v > iumax) {
                    iumax = v
                    iumax_q = $1
                    iumax_seen = 1
                }
                if (!iumin_seen || v < iumin) {
                    iumin = v
                    iumin_q = $1
                    iumin_seen = 1
                }
            } else {
                iunan++
            }

            if (numeric($5)) {
                w = $5 + 0
                insum += w
                inn++
                if (w > 0) inpos++
                else if (w < 0) inneg++
                else inzero++
                if (!inmax_seen || w > inmax) {
                    inmax = w
                    inmax_q = $1
                    inmax_seen = 1
                }
                if (!inmin_seen || w < inmin) {
                    inmin = w
                    inmin_q = $1
                    inmin_seen = 1
                }
            } else {
                innan++
            }

            if (route_cols && NF >= 8) {
                add_targets($6)
                add_scores($7, $1)
                add_answers($8)
                if ($8 != "" && !first_answer_seen) {
                    first_answer = $8
                    first_answer_q = $1
                    first_answer_seen = 1
                }
                if (contrast_cols) add_answer_contrasts($8, $9, $1)
            }
        }
        END {
            if (bad_header) {
                print "bad TSV header in " FILENAME > "/dev/stderr"
                exit 1
            }
            if (!n) {
                print "no data rows in " FILENAME > "/dev/stderr"
                exit 1
            }

            brate = bridge / n
            ravg = routes / n
            iuavg = iun ? iusum / iun : 0
            inavg = inn ? insum / inn : 0

            printf "%s: %s\n", label, FILENAME
            printf "rows: %d\n", n
            printf "user_bridge: %d/%d (%.3f), avg_routes %.3f\n", bridge, n, brate, ravg
            if (iun) {
                printf "I_U^kv: avg %+.3f, pos %d, neg %d, zero %d, nan %d\n", iuavg, iupos, iuneg, iuzero, iunan
                printf "I_U^kv max: %+.3f :: %s\n", iumax, iumax_q
                printf "I_U^kv min: %+.3f :: %s\n", iumin, iumin_q
            } else {
                printf "I_U^kv: no numeric rows, nan %d\n", iunan
            }
            if (inn) {
                printf "I_N^kv: avg %+.3f, pos %d, neg %d, zero %d, nan %d\n", inavg, inpos, inneg, inzero, innan
                printf "I_N^kv max: %+.3f :: %s\n", inmax, inmax_q
                printf "I_N^kv min: %+.3f :: %s\n", inmin, inmin_q
            } else {
                printf "I_N^kv: no numeric rows, nan %d\n", innan
            }
            if (route_cols) {
                route_line = ""
                for (target in target_count) {
                    route_line = route_line sprintf("%s%s:%d", route_line == "" ? "" : " ", target, target_count[target])
                }
                printf "route_targets: %s\n", clean_route(route_line)
                if (score_n) {
                    printf "route_score: avg %.3f, min %.3f, max %.3f\n", score_sum / score_n, score_min, score_max
                    printf "route_score max q: %.3f :: %s\n", score_max, score_max_q
                    printf "route_score min q: %.3f :: %s\n", score_min, score_min_q
                }
                if (first_answer_seen) {
                    printf "answer_sample: %s :: %s\n", first_answer_q, first_answer
                }
                printf "answer_bad_start: %d/%d\n", bad_answer_n, answer_n
                printf "answer_quality: any %d/%d, short %d, question_like %d, label_artifact %d, yes_no_start %d, repetition %d\n",
                    answer_quality_any_n, answer_n, answer_short_n, answer_question_n,
                    answer_label_n, answer_yesno_n, answer_repeat_n
                if (contrast_cols) {
                    printf "answer_kv_changed: %d/%d\n", answer_contrast_changed, answer_contrast_n
                    if (first_contrast_seen) {
                        printf "answer_kv_sample: %s :: with=%s :: without=%s\n",
                            first_contrast_q, first_contrast_on, first_contrast_off
                    }
                }
            }
        }
    ' "$file"
}

stats_line() {
    local file="$1"
    awk -F '\t' '
        function numeric(x) { return x ~ /^[-+]?[0-9]+([.][0-9]+)?$/ }
        NR == 1 { next }
        NF >= 5 {
            n++
            bridge += $2 + 0
            routes += $3 + 0
            if (numeric($4)) {
                iu = $4 + 0
                iusum += iu
                iun++
                if (iu > 0) iupos++
                else if (iu < 0) iuneg++
            }
            if (numeric($5)) {
                inv = $5 + 0
                insum += inv
                inn++
                if (inv > 0) inpos++
                else if (inv < 0) inneg++
            }
        }
        END {
            if (!n) exit 1
            printf "%d\t%.12g\t%.12g\t%.12g\t%.12g\t%d\t%d\t%.12g\t%d\t%d\n",
                n, bridge, bridge / n, routes / n,
                (iun ? iusum / iun : 0), iupos, iuneg,
                (inn ? insum / inn : 0), inpos, inneg
        }
    ' "$file"
}

compare_rows() {
    local base_file="$1"
    local cur_file="$2"
    awk -F '\t' '
        function numeric(x) { return x ~ /^[-+]?[0-9]+([.][0-9]+)?$/ }
        function sign(x) { return x > 0 ? 1 : (x < 0 ? -1 : 0) }
        function abs(x) { return x < 0 ? -x : x }
        function first_score(s,     a) {
            split(s, a, ";")
            return numeric(a[1]) ? a[1] + 0 : "nan"
        }
        FNR == 1 {
            if (NR == 1) base_routes = ($6 == "user_targets" && $7 == "user_scores" && $8 == "user_answers")
            else cur_routes = ($6 == "user_targets" && $7 == "user_scores" && $8 == "user_answers")
            next
        }
        NR == FNR {
            q = $1
            base_seen[q] = 1
            base_rows++
            b_iu[q] = $4
            b_in[q] = $5
            b_targets[q] = $6
            b_score[q] = first_score($7)
            b_answer[q] = $8
            next
        }
        {
            q = $1
            cur_seen[q] = 1
            cur_rows++
            if (!(q in base_seen)) {
                cur_only++
                next
            }
            matched++

            if (numeric($4) && numeric(b_iu[q])) {
                d = ($4 + 0) - (b_iu[q] + 0)
                if (sign($4 + 0) != sign(b_iu[q] + 0)) iu_flips++
                if (abs(d) >= 0.250) iu_big++
                if (!iu_seen || abs(d) > max_iu_abs) {
                    max_iu_abs = abs(d)
                    max_iu_delta = d
                    max_iu_q = q
                    iu_seen = 1
                }
            }

            if (numeric($5) && numeric(b_in[q])) {
                e = ($5 + 0) - (b_in[q] + 0)
                if (sign($5 + 0) != sign(b_in[q] + 0)) in_flips++
                if (abs(e) >= 0.100) in_big++
                if (!in_seen || abs(e) > max_in_abs) {
                    max_in_abs = abs(e)
                    max_in_delta = e
                    max_in_q = q
                    in_seen = 1
                }
            }

            if (base_routes && cur_routes) {
                route_comparable++
                if ($6 != b_targets[q]) target_changed++
                if ($8 != b_answer[q]) answer_changed++
                cs = first_score($7)
                bs = b_score[q]
                if (numeric(cs) && numeric(bs)) {
                    sd = cs - bs
                    score_delta_sum += sd
                    score_delta_n++
                    if (!score_delta_seen || abs(sd) > max_score_abs) {
                        max_score_abs = abs(sd)
                        max_score_delta = sd
                        max_score_q = q
                        score_delta_seen = 1
                    }
                }
            }
        }
        END {
            for (q in base_seen) if (!(q in cur_seen)) base_only++

            printf "per-question diff:\n"
            printf "matched: %d, current_only: %d, baseline_only: %d\n", matched, cur_only, base_only
            if (iu_seen) {
                printf "I_U^kv: sign_flips %d, |delta|>=0.250 %d, largest %+.3f :: %s\n",
                    iu_flips, iu_big, max_iu_delta, max_iu_q
            }
            if (in_seen) {
                printf "I_N^kv: sign_flips %d, |delta|>=0.100 %d, largest %+.3f :: %s\n",
                    in_flips, in_big, max_in_delta, max_in_q
            }
            if (base_routes && cur_routes) {
                printf "routes: comparable %d, target_changed %d, answer_changed %d\n",
                    route_comparable, target_changed, answer_changed
                if (score_delta_n) {
                    printf "route_score: avg_delta %+.3f, largest %+.3f :: %s\n",
                        score_delta_sum / score_delta_n, max_score_delta, max_score_q
                }
            } else {
                printf "routes: target/score/snippet comparison unavailable for old TSV shape\n"
            }
        }
    ' "$base_file" "$cur_file"
}

summarize "current" "$current"

if [[ -n "$baseline" ]]; then
    echo ""
    summarize "baseline" "$baseline"

    IFS=$'\t' read -r c_rows c_bridge c_bridge_rate c_routes_avg c_iu_avg c_iu_pos c_iu_neg c_in_avg c_in_pos c_in_neg < <(stats_line "$current")
    IFS=$'\t' read -r b_rows b_bridge b_bridge_rate b_routes_avg b_iu_avg b_iu_pos b_iu_neg b_in_avg b_in_pos b_in_neg < <(stats_line "$baseline")

    echo ""
    awk \
        -v c_rows="$c_rows" -v c_bridge="$c_bridge" -v c_bridge_rate="$c_bridge_rate" \
        -v c_routes_avg="$c_routes_avg" -v c_iu_avg="$c_iu_avg" \
        -v c_iu_pos="$c_iu_pos" -v c_iu_neg="$c_iu_neg" \
        -v c_in_avg="$c_in_avg" -v c_in_pos="$c_in_pos" -v c_in_neg="$c_in_neg" \
        -v b_rows="$b_rows" -v b_bridge="$b_bridge" -v b_bridge_rate="$b_bridge_rate" \
        -v b_routes_avg="$b_routes_avg" -v b_iu_avg="$b_iu_avg" \
        -v b_iu_pos="$b_iu_pos" -v b_iu_neg="$b_iu_neg" \
        -v b_in_avg="$b_in_avg" -v b_in_pos="$b_in_pos" -v b_in_neg="$b_in_neg" '
        BEGIN {
            printf "delta vs baseline:\n"
            printf "rows: %+d\n", c_rows - b_rows
            printf "user_bridge: %+g, bridge_rate %+.3f, avg_routes %+.3f\n",
                c_bridge - b_bridge, c_bridge_rate - b_bridge_rate, c_routes_avg - b_routes_avg
            printf "I_U^kv: avg %+.3f, pos %+d, neg %+d\n",
                c_iu_avg - b_iu_avg, c_iu_pos - b_iu_pos, c_iu_neg - b_iu_neg
            printf "I_N^kv: avg %+.3f, pos %+d, neg %+d\n",
                c_in_avg - b_in_avg, c_in_pos - b_in_pos, c_in_neg - b_in_neg
        }
    '

    echo ""
    compare_rows "$baseline" "$current"
fi
