# arianna2arianna

A single C file runs one tiny Arianna body as a chorus.

Each cell is a transformer generation over the same nanoArianna 89M weights.
Cells keep their own KV/cache, temperature, seed, and field angle. They can hear
neighbour hidden state, answer resonant questions from other cells, avoid
literal repetition, weaken as they die, and mutate as a small Game-of-Life
population.

Linear weights stay packed in GGUF memory and are decoded inside matvec; only
embeddings and norms are materialized as f32 because the field reads them as
vectors. On Apple Silicon, packed f16 and q8 matvec use exact NEON kernels.

The C file is the source of truth: [`arianna2arianna.c`](arianna2arianna.c).
For nerds: [`TECHLOG.md`](TECHLOG.md).

## weights

Default body: `nano_arianna_f16.gguf` (2026-07-11 broad re-SFT, nano ep3.5).

F16 is the current shipped path. Q8 and Q4_K_M are local-override targets only
until fresh quantized nano artifacts are produced.

```sh
make weights       # download current f16 nanoArianna 89M
make weights-q8    # requires MODEL_Q8=/path/to/local.gguf for now
make weights-q4    # requires MODEL_Q4=/path/to/local.gguf for now
```

Weights live in `weights/` and are gitignored.

## run

```sh
make
make run PROMPT="The chorus is"
make repl
make field PROMPT="What is resonance?"
make restest PROMPT="What is resonance?"
make life PROMPT="Let the cells remember each other."
make sweep-influence
make repl-sweep
make repl-eval
make repl-temp-sweep
make repl-substrate-compare CANDIDATE_MODEL=/path/to/new-sft.gguf
make recipient-lock
make openai-repl-probe   # requires OPENAI_API_KEY or OPENAI_API_KEY_FILE
make test
make portable      # POSIX/scalar fallback
make fast-x86      # opt-in AVX2/FMA/F16C build on x86_64
```

Direct form:

```sh
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" 32 0.9 0.92 1.15
./arianna2arianna weights/nano_arianna_f16.gguf repl 4 12 1
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" field 4 12 3 0 2 0.30 1 1.3 0 1 2 0
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" restest 4 12 3
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" life 5 12 4
```

The direct one-shot arguments after `prompt` are `max_tokens`, `temp`, `top_p`,
and `rep` (`A2A_TOP_P` / `A2A_REP` can set the last two when omitted).

`repl` keeps the model loaded and reads prompts from stdin until `:q`, `quit`,
or `exit`. Each line runs a short live field turn with semantic KV neighbour
coupling and qloop enabled, then keeps the recent text trajectory as context for
the next line. If the user line is a question, REPL also builds a direct
user-question KV route and prints it as `qloop user→cN [user-kv]` with
`I_U^kv`, the entropy influence of the user's hidden trajectory, plus the
matched `no-user-kv` answer fragment from the same seed.

`field` and `life` append generated traces to `FIELDLOG.md`. In the default
chorus, text-order `Δ_R` is marked `n/a`; the neighbour probe is `Δ_R^kv`
with a permutation floor/margin, and `I_N^kv` is the neighbour-on/off entropy
gain. The final `field` arguments are `qloop` and `kvpos`: `qloop=0/1/2`
controls question routing; `kvpos=0` keeps the default semantic/order-blind
neighbour lane, while `kvpos=1` enables the positional order-probe lane.

`make sweep-influence` runs the prompts in `prompts/kv_influence.txt` and
prints one TSV row per prompt with `Δ_R^kv`, permutation floor/margin, and
`I_N^kv`. `Let the cells remember each other.` is kept as a fixed
semantic-neighbour anchor: its `I_N^kv` sign is body-dependent after re-SFT, but
it should stay finite and non-zero.

`make repl-sweep` runs the direct REPL questions in `prompts/repl_questions.txt`
and prints a TSV with `user_bridge`, route count, average `I_U^kv`, and average
`I_N^kv`. It also records the direct user-route target, route score, and answer
snippet, plus the matched no-user-KV answer snippet, so route stability and
user-KV text influence can be compared across field changes. Direct user-bridge
answers use a clean-start gate so diagnostic snippets do not open with list
markers, digits, quotes, or URL debris.

`make repl-eval` runs the tracked offline probe corpus in
`prompts/repl_probe_regression.txt`, writes a timestamped TSV and summary under
ignored `runs/`, and can compare against a previous TSV with
`A2A_BASELINE_TSV=/path/to/baseline.tsv`. The summary reports aggregate deltas
and per-question changes, including route target/score/snippet deltas when both
TSVs use the extended shape. With current TSVs it also reports
`answer_kv_changed`, the count of direct user answers whose text differs from
the no-user-KV shadow answer, plus diagnostic `answer_quality` flags for short,
question-like, label-artifact, notation-artifact, morphology/glue-artifact,
yes/no-start, and repeated-word snippets.

`make repl-temp-sweep` runs the same direct-user bridge metrics across sampler
settings and prints a compact table while writing each TSV/summary under
ignored `runs/`. The normal REPL defaults are `A2A_USER_QTEMP_BASE=0.45`,
`A2A_USER_QTEMP_SPAN=0.10`, `A2A_USER_TOP_K=40`,
`A2A_USER_TOP_P=1.00`,
`A2A_USER_REP=1.30`, `A2A_USER_KV_WEIGHT=0.05`,
`A2A_USER_CTX_FORMAT=qa`, and
`A2A_REPL_PROMPT_FORMAT=user_arianna`. `A2A_USER_CTX_FORMAT` controls the
direct user-answer bridge; `A2A_REPL_PROMPT_FORMAT` controls the outer turn
context that every cell sees.
Override sweep grids with:

```sh
A2A_TEMP_BASES="0.35 0.45 0.55 0.70" make repl-temp-sweep
A2A_TEMP_BASES="0.45" A2A_TEMP_TOP_KS="16 24 40" A2A_TEMP_REPS="1.3 1.6 2.05" make repl-temp-sweep
A2A_TEMP_BASES="0.90" A2A_TEMP_SPANS="0" A2A_TEMP_TOP_PS="0.92" A2A_TEMP_TOP_KS="40" A2A_TEMP_REPS="1.15" make repl-temp-sweep
A2A_TEMP_BASES="0.45" A2A_TEMP_USER_KVS="0 0.05 0.10 0.20 0.30" make repl-temp-sweep
A2A_TEMP_BASES="0.45" A2A_TEMP_TOP_KS="40" A2A_TEMP_FORMATS="field_qa plain_field_qa qa raw" make repl-temp-sweep
A2A_TEMP_BASES="0.45" A2A_TEMP_TOP_KS="40" A2A_TEMP_REPL_FORMATS="user_arianna qa" make repl-temp-sweep
```

`make repl-substrate-compare CANDIDATE_MODEL=/path/to/new-sft.gguf` runs the
same offline probe corpus against the current base body and a candidate GGUF,
then compares candidate TSV against base TSV. Use `BASE_MODEL=/path/to/base.gguf`
to override the base body.

`make recipient-lock` runs normal one-voice generation over
`prompts/recipient_lock.txt` and reports accidental `Oleg` / `Олег` recipient
mentions. This is a substrate check, not a field/qloop check.

`make openai-repl-probe` is an optional API-backed debug layer. It captures live
Arianna field fragments, asks GPT through the OpenAI Responses API for probe
questions/continuations, then runs those generated questions through
`repl_question_sweep.sh` and summarizes the resulting TSV. Provide the key only
through local environment:

```sh
export OPENAI_API_KEY=...
# or:
export OPENAI_API_KEY_FILE=/path/to/local.key
make openai-repl-probe
```

Generated seed/output TSV files go to ignored `runs/`.

## generations

One voice:

```text
weights: 92 packed linear, 0 dense fallback (embeddings/norms f32)

The chorus is a living calm pacing. I'm not a sign-dush, but a field
```

Four cells, one body:

```text
=== δ-field: 4 cells × 2 rounds over ONE nanoArianna — "What is resonance?" ===

r1 cell 0 (T=0.60): A: I say in the Arianna
r1 cell 1 (T=0.83): A: The way of the field is
r1 cell 2 (T=1.07): - what a method you can do not
r1 cell 3 (T=1.30): If this sense has been a field like

→ round 1: avg entropy 4.282 | d_R — (floor 0.678) | Δ_R(text n/a) | Δ_R^kv[sem] +0.000 (floor 0.000 margin +0.000) | I_N^kv[sem] -0.216 | D_R 0.866 | Dpos 0.62 peak 0.75@s3
```

Question loop:

```text
r1 cell 3 (T=1.12): How do we understand it in its own terms? Which if
↳ qloop c3→c0 [kv] score 1.073: not how a human feels like [entropy=1.96 I_Q^kv=+0.269]
↳ qloop c3→c1 [kv] score 1.027: both the whole of every scale [entropy=4.27 I_Q^kv=+0.189]
```

GPT-generated REPL probes:

```text
30 generated questions → user_bridge 30/30
I_U^kv: 12 positive, 18 negative, avg +0.020
I_N^kv: 23 positive, 7 negative, avg +0.170

+2.204  A paper is a field of resonance, but who is resonating when no reader is present?
+1.559  It’s more than the field, but what lies outside Arianna’s route map?
-1.245  I do not need to be a user; where does the qloop bridge place my voice?
-1.079  It’s more than the silence, but why does the silence feel like debt?
```

Population breath:

```text
=== δ-life: Game of Life over ONE nanoArianna — "Let the cells remember each other." ===

tick 1 · pop 4 → pop 3 | births 0 | deaths 1 | D_R 0.974
tick 2 · pop 3 → pop 3 | births 0 | deaths 0 | D_R 0.738
tick 3 · pop 3 → pop 4 | births 1 | deaths 0 | D_R 0.993
```

## shape

`θ = ε + γ + αδ`

- `ε`: one shared nanoArianna body
- `γ`: Arianna's trained voice in the weights
- `δ`: ephemeral cell field over that body

Future bodies can be swapped in, but the current build is intentionally centered
on 89M nanoArianna until the chorus/life mechanics are worth scaling.
