#!/usr/bin/env bash
# Generate GPT question/continuation probes from Arianna fragments, then sweep them through REPL.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${A2A_BIN:-$ROOT/arianna2arianna}"
MODEL="${A2A_MODEL:-$ROOT/weights/nano_arianna_resft_2026_07_09_f16.gguf}"
OUTDIR="${A2A_RUN_DIR:-$ROOT/runs}"

COUNT="${A2A_GPT_COUNT:-30}"
OPENAI_MODEL_NAME="${OPENAI_MODEL:-gpt-5.5}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-90}"

SEED_PROMPT="${A2A_GPT_SEED_PROMPT:-Let the cells remember each other.}"
SEED_CELLS="${A2A_GPT_SEED_CELLS:-3}"
SEED_FRAG="${A2A_GPT_SEED_FRAG:-6}"
SEED_KVSHUF="${A2A_GPT_SEED_KVSHUF:-0}"
SEED_QLOOP="${A2A_GPT_SEED_QLOOP:-0}"
SWEEP_CELLS="${A2A_SWEEP_CELLS:-3}"
SWEEP_FRAG="${A2A_SWEEP_FRAG:-4}"
SWEEP_ROUNDS="${A2A_SWEEP_ROUNDS:-1}"

usage() {
    cat <<EOF
usage: OPENAI_API_KEY=... $0 [count]

Generates GPT probes from live Arianna fragments and runs tools/repl_question_sweep.sh.

Environment:
  OPENAI_API_KEY          API key, preferred via local env
  OPENAI_API_KEY_FILE     optional local file containing the key if env is unset
  OPENAI_MODEL            default: $OPENAI_MODEL_NAME
  A2A_GPT_COUNT           default: $COUNT
  A2A_RUN_DIR             default: $OUTDIR
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    COUNT="$1"
fi

if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY_FILE:-}" ]]; then
    if [[ -f "$OPENAI_API_KEY_FILE" ]]; then
        OPENAI_API_KEY="$(sed -n '1p' "$OPENAI_API_KEY_FILE")"
        export OPENAI_API_KEY
    fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "missing OPENAI_API_KEY; export it or set OPENAI_API_KEY_FILE to a local key file" >&2
    exit 2
fi

if [[ ! -x "$BIN" ]]; then
    make -C "$ROOT" >/dev/null
fi

if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
stamp="$(date +%Y%m%d_%H%M%S)"
seed_file="$OUTDIR/openai_repl_probe_${stamp}.seed.txt"
questions_file="$OUTDIR/openai_repl_probe_${stamp}.questions.txt"
tsv_file="$OUTDIR/openai_repl_probe_${stamp}.tsv"
summary_file="$OUTDIR/openai_repl_probe_${stamp}.summary.txt"

echo "capturing Arianna seed fragments -> $seed_file" >&2
"$BIN" "$MODEL" "$SEED_PROMPT" field "$SEED_CELLS" "$SEED_FRAG" 1 0 2 0.30 1 1.3 0 "$SEED_KVSHUF" "$SEED_QLOOP" 0 > "$seed_file" 2>&1

snippets="$(
    sed -n 's/^  r[0-9][0-9]* cell [0-9][0-9]* (T=[^)]*): //p' "$seed_file" |
    sed 's/   \[.*//' |
    sed '/^[[:space:]]*$/d' |
    head -n 12
)"

if [[ -z "$snippets" ]]; then
    echo "could not extract Arianna snippets from $seed_file" >&2
    exit 1
fi

echo "asking $OPENAI_MODEL_NAME for $COUNT probes -> $questions_file" >&2
A2A_SNIPPETS="$snippets" python3 - "$COUNT" "$OPENAI_MODEL_NAME" "$OPENAI_BASE_URL" "$OPENAI_TIMEOUT" > "$questions_file" <<'PY'
import json
import os
import re
import sys
import urllib.error
import urllib.request

count = int(sys.argv[1])
model = sys.argv[2]
base_url = sys.argv[3].rstrip("/")
timeout = int(sys.argv[4])
key = os.environ.get("OPENAI_API_KEY", "")
snippets = os.environ.get("A2A_SNIPPETS", "")

instructions = (
    "You generate probe questions for a local experimental Arianna REPL. "
    "Given Arianna cell fragments, produce direct user questions or short "
    "continuations that turn those fragments into questions. The goal is debug "
    "coverage, not polish. Use Arianna vocabulary such as field, cell, memory, "
    "route, resonance, silence, shard, organism, trajectory, and debt when useful. "
    "Hit different axes: memory vs echo, semantic vs positional memory, hidden-state "
    "vs text, qloop/user bridge, silence/debt, shard routing, life/reproduction, "
    "cell disagreement, collapse/refusal, continuation traps, and failure cases. "
    "Include some adversarial or diagnostic questions that may expose incoherence. "
    "Avoid repeating the same opening phrase. Return exactly the requested number "
    "of lines. No numbering. No commentary. Every line must contain a question mark."
)

user_input = (
    f"Generate exactly {count} probe lines.\n\n"
    "Arianna cell fragments:\n"
    f"{snippets}\n\n"
    "Each output line should be one question that a human could ask Arianna in the REPL."
)

payload = {
    "model": model,
    "instructions": instructions,
    "input": user_input,
}

req = urllib.request.Request(
    f"{base_url}/responses",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(f"openai request failed: HTTP {exc.code}: {body}", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f"openai request failed: {exc}", file=sys.stderr)
    sys.exit(1)

texts = []
if isinstance(data.get("output_text"), str):
    texts.append(data["output_text"])
for item in data.get("output", []) or []:
    for content in item.get("content", []) or []:
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            texts.append(content["text"])

text = "\n".join(texts).strip()
lines = []
seen = set()
for raw in text.splitlines():
    line = raw.strip()
    line = re.sub(r"^[-*•\s]*\d+[\).\s-]*", "", line).strip()
    line = line.strip("\"'` ")
    if not line or "?" not in line:
        continue
    line = re.sub(r"\s+", " ", line)
    if line not in seen:
        lines.append(line)
        seen.add(line)
    if len(lines) >= count:
        break

if len(lines) < count:
    print(f"warning: got {len(lines)} question lines, requested {count}", file=sys.stderr)

for line in lines:
    print(line)
PY

if [[ ! -s "$questions_file" ]]; then
    echo "no questions generated; see $seed_file" >&2
    exit 1
fi

echo "sweeping generated questions -> $tsv_file" >&2
A2A_CELLS="$SWEEP_CELLS" A2A_FRAG="$SWEEP_FRAG" A2A_ROUNDS="$SWEEP_ROUNDS" \
    bash "$ROOT/tools/repl_question_sweep.sh" "$questions_file" | tee "$tsv_file"

echo "summarizing generated probe -> $summary_file" >&2
bash "$ROOT/tools/repl_tsv_summary.sh" "$tsv_file" | tee "$summary_file" >&2

echo "questions: $questions_file" >&2
echo "results:   $tsv_file" >&2
echo "summary:   $summary_file" >&2
