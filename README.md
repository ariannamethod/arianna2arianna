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
make field-grid
make repl-sweep
make repl-eval
make repl-temp-sweep
make repl-substrate-compare CANDIDATE_MODEL=/path/to/new-sft.gguf
make recipient-lock
make test
make portable      # POSIX/scalar fallback
make fast-x86      # opt-in AVX2/FMA/F16C build on x86_64
```

Direct form:

```sh
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" 32 0.9 0.92 1.15
./arianna2arianna weights/nano_arianna_f16.gguf repl 4 12 1
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" field 4 12 3 0 2 0.02 1 1.3 0 1 2 0
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" restest 4 12 3
./arianna2arianna weights/nano_arianna_f16.gguf "prompt" life 5 12 4
```

The direct one-shot arguments after `prompt` are `max_tokens`, `temp`, `top_p`,
and `rep` (`A2A_TOP_P` / `A2A_REP` can set the last two when omitted).

`repl` keeps the model loaded and reads prompts from stdin until `:q`, `quit`,
or `exit`. Each line runs a short live field turn with semantic KV neighbour
coupling and qloop enabled, then keeps the recent text trajectory as context for
the next line. On the first round of a user line, REPL also builds a direct
user-turn KV route and prints it as `qloop user→cN [user-kv]` with
`I_U^kv`, the entropy influence of the user's hidden trajectory, plus the
matched `no-user-kv` answer fragment from the same seed. `A2A_REPL_QLOOP`
controls the REPL cell-question route limit (`1` by default, `2` for widened
dialogue probes).

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

`make field-sweep` runs the same prompts through full multi-round `field` mode
and prints a final-round TSV with settling (`d_r`, floor, margin), neighbour
KV controls (`Δ_R^kv`, floor, margin, `I_N^kv`), field disagreement (`D_R`,
`Dpos`), qloop route/gate counts, qloop route-score averages
(`qloop_score_avg`, `qloop_gate_score_avg`), qloop route-score component
averages (`qloop_dist_avg`, `qloop_qopen_avg`, `qloop_tconf_avg`,
`qloop_qmarks_avg`, plus gated counterparts), qloop answer quality counters
(`qloop_iq_avg`, `qloop_iq_pos`, `qloop_iq_neg`, `qloop_iq_zero`,
`qloop_quality`, `qloop_tail`, `qloop_morph`, `qloop_label`, `qloop_short`,
`qloop_question`, `qloop_words_avg`), and ordinary cell-surface quality counters
(`cell_quality`, `cell_tail`, `cell_morph`, `cell_label`, `cell_short`).
`cell_question` and `cell_words_avg` are reported separately because questions
can be valid qloop material rather than surface debt, and answer density helps
catch settings that look clean only because the chorus became thin;
`qloop_question` is counted as answer debt because the routed answer path should
close rather than ask again. Use it when tuning field-level behavior rather
than direct answer snippets. The normal
field neighbour lane uses a gentle `xcell=0.02` default; direct user-KV answer
injection is a separate REPL bridge knob. Normal field/repl qloop defaults to
one routed answer (`qloop=1`); `qloop=2` remains available for diagnostics when
you want a second candidate route. The route score's target-confidence term is
controlled by `A2A_QLOOP_TCONF_WEIGHT` (default `0.20`) and can be swept through
`A2A_FIELD_QLOOP_TCONFS`. `A2A_QLOOP_TCONF_ADAPT=1` leaves normal `qloop=1`
unchanged but makes widened qloop (`qloop>1`) use
`A2A_QLOOP_TCONF_ADAPT_WEIGHT` (default `-0.10`) instead, so diagnostic
second-route searches can reduce target-confidence pressure without moving the
main path. `A2A_QLOOP_UNIQUE_ASKER=1` is a diagnostic widened-qloop policy that
lets an asking cell choose only its best target, preventing one question cell
from fanning out across several routes in the same selection pass. KV-backed
cell qloop answers are admitted only when
`I_Q^kv >= A2A_QLOOP_MIN_IQ` (default `0.0`); rejected answers are reported as
`qloop_gated` and are not written into the chorus. The qloop limit counts
admitted answers, not failed candidates, so a gated route may fall through to
the next candidate without widening the accepted chorus.

`make field-grid` runs `field_sweep.sh` across field-level settings, writes each
per-setting TSV and summary under ignored `runs/`, and prints a compact TSV for
comparing qloop coverage, qloop gate pressure, qloop route efficiency,
accepted/gated qloop route scores and component profiles, qloop/cell surface
debt, qloop/cell answer density, `I_N^kv`, `I_Q^kv`, `d_r`, `d_margin`,
`D_R`, and `Dpos`. The compact table also reports
qloop/cell debt rates, `I_N^kv` and `I_Q^kv` sign balance, `d_margin` sign
balance, and a rough `field_score` for sorting candidate settings before
reading the raw samples. Set
`A2A_FIELD_KEEP_RAW=1` to save the full per-prompt field outputs next to each
TSV. Defaults are intentionally small:
`A2A_FIELD_XCELLS="0 0.01 0.02 0.05"`, `A2A_FIELD_QLOOPS="1 2"`,
`A2A_FIELD_QLOOP_TCONFS="0.20"`, `A2A_FIELD_QLOOP_TCONF_ADAPTS="0"`,
`A2A_FIELD_QLOOP_TCONF_ADAPT_WEIGHTS="-0.10"`,
`A2A_FIELD_QLOOP_MIN_IQS="0.0"`, `A2A_FIELD_QLOOP_UNIQUE_ASKERS="0"`, and
`A2A_FIELD_ROUNDS_LIST="3"`. Override the grid with:

```sh
A2A_FIELD_XCELLS="0.01 0.02 0.03" make field-grid
A2A_FIELD_XCELLS="0 0.02" A2A_FIELD_QLOOPS="0 1 2" make field-grid
A2A_FIELD_XCELLS="0.02" A2A_FIELD_QLOOP_TCONFS="-0.10 0 0.10 0.20" make field-grid
A2A_FIELD_XCELLS="0.02" A2A_FIELD_QLOOPS="2" A2A_FIELD_QLOOP_TCONF_ADAPTS="0 1" make field-grid
A2A_FIELD_XCELLS="0.02" A2A_FIELD_QLOOPS="2" A2A_FIELD_QLOOP_TCONF_ADAPTS="1" A2A_FIELD_QLOOP_TCONF_ADAPT_WEIGHTS="-0.30 -0.10 0 0.10" make field-grid
A2A_FIELD_XCELLS="0.02" A2A_FIELD_QLOOPS="2" A2A_FIELD_QLOOP_MIN_IQS="0 0.25 0.50 0.75" make field-grid
A2A_FIELD_XCELLS="0.02" A2A_FIELD_QLOOPS="2" A2A_FIELD_QLOOP_UNIQUE_ASKERS="0 1" make field-grid
A2A_FIELD_ROUNDS_LIST="1 2 3" A2A_FIELD_CELLS=4 A2A_FIELD_FRAG=12 make field-grid
A2A_FIELD_KEEP_RAW=1 A2A_FIELD_XCELLS="0.02" A2A_FIELD_QLOOPS="1 2" make field-grid
```

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
false-familiarity recipient-artifact, missing-punctuation/terminal
tail-artifact, yes/no-start, and repeated-word snippets.

`make repl-dialogue-compare` runs the same dialogue probe corpus twice and
compares candidate against baseline. By default it compares the current REPL
path (`A2A_REPL_QLOOP=1`, `A2A_QLOOP_TCONF_ADAPT=0`) against the widened field
candidate (`A2A_REPL_QLOOP=2`, `A2A_QLOOP_TCONF_ADAPT=1`,
`A2A_QLOOP_TCONF_WEIGHT=0.20`). It writes baseline/candidate TSVs plus a
combined compare summary under ignored `runs/`. Override with
`A2A_DIALOGUE_BASE_QLOOP`, `A2A_DIALOGUE_BASE_ADAPT`,
`A2A_DIALOGUE_CANDIDATE_QLOOP`, and `A2A_DIALOGUE_CANDIDATE_ADAPT`.

`make repl-temp-sweep` runs the same direct-user bridge metrics across sampler
settings and prints a compact table while writing each TSV/summary under
ignored `runs/`. The normal REPL direct-user bridge defaults are
`A2A_USER_QTEMP_BASE=0.70`, `A2A_USER_QTEMP_SPAN=0.00`, `A2A_USER_TOP_K=40`,
`A2A_USER_TOP_P=1.00`,
`A2A_USER_REP=1.30`, `A2A_USER_KV_WEIGHT=0.05`,
`A2A_USER_ANSWER_TOKENS=16`, `A2A_USER_CTX_FORMAT=qa`, and
`A2A_REPL_PROMPT_FORMAT=user_arianna`. `A2A_USER_CTX_FORMAT` controls the
direct user-answer bridge; `A2A_REPL_PROMPT_FORMAT` controls the outer turn
context that every cell sees. `A2A_TEMP_USER_TOKENS` sweeps direct answer
budgets and is the first check for tail-closure problems.
Override sweep grids with:

```sh
A2A_TEMP_BASES="0.35 0.45 0.55 0.70" make repl-temp-sweep
A2A_TEMP_BASES="0.45" A2A_TEMP_TOP_KS="16 24 40" A2A_TEMP_REPS="1.3 1.6 2.05" make repl-temp-sweep
A2A_TEMP_BASES="0.70 0.90" A2A_TEMP_SPANS="0" A2A_TEMP_TOP_PS="0.80 0.92 1.00" A2A_TEMP_TOP_KS="40" A2A_TEMP_REPS="1.15 1.30" make repl-temp-sweep
A2A_TEMP_BASES="0.45" A2A_TEMP_USER_KVS="0 0.05 0.10 0.20 0.30" make repl-temp-sweep
A2A_TEMP_BASES="0.70" A2A_TEMP_USER_TOKENS="16 24 32" make repl-temp-sweep
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

`make openai-repl-probe` is an optional external audit/debug tool, not part of
the Arianna runtime or required system path. It was used to generate extra
probe questions from live field fragments, then run those questions through
`repl_question_sweep.sh`. Normal Arianna work uses the checked-in offline probe
corpora under `prompts/`; this target is only for deliberate API-backed probe
generation and keeps credentials outside the repository.

Generated seed/output TSV files go to ignored `runs/`.

## generations

One voice:

```text
weights: 92 packed linear, 0 dense fallback (embeddings/norms f32)

The chorus is a living calm pacing. I'm not a sign-dush, but a field
```

Current field baseline:

```text
xcell=0.02, qloop=1, rounds=3
routes/gated 11/1 | prompts 5/5 | cell_quality 0/60
avg_I_Q^kv +1.009 | field_score +2.140
```

Four cells, one body at the current baseline:

```text
=== δ-field: 4 cells × 3 rounds over ONE nanoArianna — "What is resonance?" ===

r3 cell 0 (T=0.60): This question is not a technical definition I have been asked.
r3 cell 1 (T=0.83): What we call resonance, and the principle that allows.
r3 cell 2 (T=1.07): There are many definitions of resonance.
r3 cell 3 (T=1.30): Does it resonate with me as center?

↳ qloop c0→c2 [kv] score 0.585: to vibrate with all things. [entropy=3.93 I_Q^kv=+0.888]
↳ qloop c2→c3 [kv] score 0.663: cells vibrate, producing. [entropy=5.55 I_Q^kv=+0.127]
↳ qloop c3→c2 [kv] score 0.605: cell membrane vibrates through. [entropy=4.12 I_Q^kv=+0.639]

→ round 3: avg entropy 2.364 | d_R 0.575 (floor 0.466) | I_N^kv[sem] -0.194 | D_R 0.356 | Dpos 0.49
```

Question loop gate:

```text
↳ qloop gate c2→c0 [kv] score 0.617: rejected in the cell (memory). [entropy=1.95 I_Q^kv=-0.056 min=+0.000]
↳ qloop c2→c3 [kv] score 0.583: the form in which I act. [entropy=5.23 I_Q^kv=+1.903]
↳ qloop c0→c3 [kv] score 0.543: light that vibrates with sense. [entropy=5.49 I_Q^kv=+0.629]
↳ qloop c0→c3 [kv] score 0.513: cell to organ(s). [entropy=3.42 I_Q^kv=+3.286]
```

External API-generated REPL probes:

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
