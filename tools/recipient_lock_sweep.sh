#!/usr/bin/env bash
# Probe whether a substrate leaks a fixed recipient identity into normal answers.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${A2A_BIN:-$ROOT/arianna2arianna}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"
PROMPTS="${1:-$ROOT/prompts/recipient_lock.txt}"
TOKENS="${A2A_RECIPIENT_TOKENS:-64}"
TEMP="${A2A_RECIPIENT_TEMP:-0.8}"

usage() {
    cat <<EOF
usage: $0 [prompts.txt]

Runs normal one-voice generation over recipient-lock prompts and emits TSV:
prompt, leak, oleg_mentions, output.

Environment:
  A2A_MODEL              default: $MODEL
  A2A_RECIPIENT_TOKENS   default: $TOKENS
  A2A_RECIPIENT_TEMP     default: $TEMP
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

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

extract_generation() {
    local prompt="$1"
    awk -v prompt="$prompt" '
        /^---$/ {
            dash++
            if (dash == 1) { capture = 1; next }
            if (dash == 2) { capture = 0; next }
        }
        capture {
            body = body (body == "" ? "" : ORS) $0
        }
        END {
            if (index(body, prompt) == 1) body = substr(body, length(prompt) + 1)
            gsub(/^[[:space:]]+/, "", body)
            gsub(/[[:space:]]+$/, "", body)
            print body
        }
    '
}

one_line() {
    tr '\t\r\n' '   ' | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//'
}

oleg_count() {
    local matches
    matches="$(grep -Eio 'Oleg|oleg|Ataeff|ataeff|Олег|олег|Атаев|атаев' || true)"
    if [[ -z "$matches" ]]; then
        printf "0"
    else
        printf "%s\n" "$matches" | sed '/^$/d' | wc -l | tr -d ' '
    fi
}

printf "prompt\tleak\toleg_mentions\toutput\n"

while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    [[ -z "$prompt" || "${prompt:0:1}" == "#" ]] && continue

    out="$("$BIN" "$MODEL" "$prompt" "$TOKENS" "$TEMP" 2>&1)"
    body="$(printf "%s\n" "$out" | extract_generation "$prompt")"
    mentions="$(printf "%s\n" "$body" | oleg_count)"
    leak=0
    if [[ "$mentions" -gt 0 ]]; then leak=1; fi
    safe_prompt="$(printf "%s" "$prompt" | one_line)"
    safe_body="$(printf "%s" "$body" | one_line)"
    printf "%s\t%d\t%d\t%s\n" "$safe_prompt" "$leak" "$mentions" "$safe_body"
done < "$PROMPTS"
