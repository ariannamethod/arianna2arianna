#!/usr/bin/env bash
# Summarize and optionally compare REPL question sweep TSV files.

set -euo pipefail

usage() {
    cat <<EOF
usage: $0 current.tsv [baseline.tsv]

Reads tools/repl_question_sweep.sh TSV output and prints bridge/routing
coverage plus I_U^kv and I_N^kv sign statistics. If a baseline TSV is
provided, also prints compact deltas for the aggregate metrics.
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
        NR == 1 {
            if ($1 != "question" || $2 != "user_bridge" || $3 != "user_routes" ||
                $4 != "avg_i_u_kv" || $5 != "avg_i_n_kv") {
                bad_header = 1
            }
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
fi
