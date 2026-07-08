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

Default body: `nanollama-arianna-full-v4-step2750-f16.gguf`.

F16 is the main path for now. The model is small enough that quantizing lower is
not the point; q8 is kept as an alternate local fallback.

```sh
make weights       # download f16 nanoArianna 89M
make weights-q8    # optional q8
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
make openai-repl-probe   # requires OPENAI_API_KEY or OPENAI_API_KEY_FILE
make test
make portable      # POSIX/scalar fallback
make fast-x86      # opt-in AVX2/FMA/F16C build on x86_64
```

Direct form:

```sh
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" 48 0.8
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf repl 4 12 1
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" field 4 12 3 0 2 0.30 1 1.3 0 1 2 0
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" restest 4 12 3
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" life 5 12 4
```

`repl` keeps the model loaded and reads prompts from stdin until `:q`, `quit`,
or `exit`. Each line runs a short live field turn with semantic KV neighbour
coupling and qloop enabled, then keeps the recent text trajectory as context for
the next line. If the user line is a question, REPL also builds a direct
user-question KV route and prints it as `qloop user→cN [user-kv]` with
`I_U^kv`, the entropy influence of the user's hidden trajectory.

`field` and `life` append generated traces to `FIELDLOG.md`. In the default
chorus, text-order `Δ_R` is marked `n/a`; the neighbour probe is `Δ_R^kv`
with a permutation floor/margin, and `I_N^kv` is the neighbour-on/off entropy
gain. The final `field` arguments are `qloop` and `kvpos`: `qloop=0/1/2`
controls question routing; `kvpos=0` keeps the default semantic/order-blind
neighbour lane, while `kvpos=1` enables the positional order-probe lane.

`make sweep-influence` runs the prompts in `prompts/kv_influence.txt` and
prints one TSV row per prompt with `Δ_R^kv`, permutation floor/margin, and
`I_N^kv`. `Let the cells remember each other.` is kept as the first positive
semantic-neighbour anchor: its `I_N^kv` should stay above zero.

`make repl-sweep` runs the direct REPL questions in `prompts/repl_questions.txt`
and prints a TSV with `user_bridge`, route count, average `I_U^kv`, and average
`I_N^kv`.

`make repl-eval` runs the tracked offline probe corpus in
`prompts/repl_probe_regression.txt`, writes a timestamped TSV and summary under
ignored `runs/`, and can compare against a previous TSV with
`A2A_BASELINE_TSV=/path/to/baseline.tsv`.

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
