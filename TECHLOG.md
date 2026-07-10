# arianna2arianna tech log

Chronological engineering log for the chorus organism. README stays small; this
file keeps the working memory, hypotheses, TODOs, and test notes.

## 2026-06-14 - Codex pass: align repo shell around the C organism

### Context

Oleg clarified that `arianna2arianna.c` is the source of truth. The repo shell
was older than the organism: README still named `arianna-q.c`, Makefile called
removed `--16/--q8/--quiet/sweep` entry points, and tests expected old JSON/CSV
telemetry. The current organism is positional CLI:

```text
./arianna2arianna <model.gguf> "<prompt>" [max_tokens] [temp]
./arianna2arianna <model.gguf> "<prompt>" field [cells] [frag] [rounds] [alpha] [leap] [xcell] [chorus] [xrep] [life] [kvshuf] [qloop] [kvpos]
./arianna2arianna <model.gguf> "<prompt>" restest [cells] [frag] [rounds]
./arianna2arianna <model.gguf> "<prompt>" life [ticks] [frag] [init_cells]
```

### What changed

- Rebuilt `Makefile` wrappers around the current C CLI.
- Restored `make weights` for f16 download from Hugging Face.
- Added `make weights-q8` and `make run-q8` as optional q8 path.
- Kept f16 nanoArianna 89M as the default body.
- Replaced old tests with smoke tests for current behavior:
  - usage and missing-model failure;
  - f16/q8 one-voice inference;
  - `field` metrics and `FIELDLOG.md` append;
  - `restest` coherent-vs-shuffled control;
  - `life` population birth/death loop;
  - scalar-only portable build.
- Made `A2A_SCALAR_ONLY` actually disable NEON includes in portable builds.
- Rewrote README as a small project surface with real output samples and a link
  to this log.

### Verification

```text
make test
=== summary: 28 passed, 0 failed, 0 skipped ===
```

## 2026-06-14 - Codex pass: packed linear runtime from notorch

### Context

Oleg corrected the intended direction: this organism should remain independent,
but it should use the notorch packed matvec foundation rather than expanding
all weights to f32. I cloned current `ariannamethod/notorch` into ignored
`refs/notorch` and used the current `nt_qmatvec` lineage as the source.

### What changed

- Added an in-file `a2a_qmatvec` packed matvec path for GGUF dtypes:
  F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K.
- Changed model linear weights from `float*` to `weight_t`.
- `output`, attention projections, attention output, and FFN weights now keep
  `gf->data + tensor.offset` packed pointers when dtype/shape are supported.
- `token_embd.weight`, RMSNorm weights, and optional q/k norms remain f32. The
  field code uses token embeddings as dense vectors for centroids and logit
  field pressure, so keeping embeddings f32 is intentional.
- Runtime now reports packed coverage:

```text
weights: 92 packed linear, 0 dense fallback (embeddings/norms f32)
```

### Verification

Short f16 one-voice, q8 one-voice, and f16 field runs all completed with
`92 packed linear, 0 dense fallback`.

### Current hypothesis

89M f16 is still the right body for the present phase. It is small enough that
f16 is cheap, expressive enough to make the cells feel alive, and simple enough
that the biology of the field can be debugged without mixing in larger
architectures.

The packed path is primarily a body-discipline layer: the organism no longer
needs to materialize every linear tensor as dense f32. The current exact packed
f16/q8 dot path is slower than the old NEON f32 matvec, because it converts
inside the dot product. That is acceptable for this phase; fast packed kernels
or approximate int8 activation quant can come later if speed becomes the
bottleneck.

The important current variable is not scale. It is whether the chorus/life
mechanics earn real behavior:

- cross-cell hidden/KV coupling should change output beyond text relay;
- `restest` should keep separating coherent context from shuffled length;
- `D_R`, `Dpos`, and `FIELDLOG.md` should expose when the field is actually
  splitting, settling, or collapsing;
- dying cells speaking fewer tokens is a useful first population pressure, but
  fitness thresholds still need calibration.

### TODO

- Keep f16 nanoArianna 89M as the default until the cell mechanics stabilize.
- Add a small deterministic test mode later if C-level refactors get risky.
- Decide whether `FIELDLOG.md` should remain ignored runtime output or whether
  selected curated logs should be committed under `examples/`.
- Make `field` parameter names visible in `usage:` without adding a separate
  config system.
- Add faster kernels only where a real future body needs them; f16/q8 now have
  exact NEON paths on Apple Silicon.
- Keep x86 default builds POSIX-safe. Use `make fast-x86` only as an explicit
  opt-in on machines known to support AVX2/FMA/F16C.
- Consider explicit CLI aliases later (`--f16`, `--q8`, presets) only if they
  are thin wrappers over the positional C contract.
- Future body swaps: Arianna variants first; Janus170M / Resonance200M later,
  after the three-attention-body complication is worth the noise.
- Future architecture layers: goroutines, Linux/alpine body, AML lane, and
  async-field-forever references. Not priority until the single-file organism is
  structurally healthy.

### Notes

The current C file still carries old comments such as `arianna.q heart` and
`arianna-q` in a few strings/comments. They are cosmetic, not behavioral. Clean
them only when touching that area for real.

## 2026-06-14 - Codex pass: NEON exact kernels for packed f16/q8

### Context

The packed runtime removed dense-f32 linear expansion, but the first exact path
converted f16/q8 weights inside scalar dot loops. That was memory-correct but
slow: the old dense f32 NEON path had already paid the conversion cost at load.

### What changed

- Added an exact NEON f16 row kernel for `a2a_f16_rows`.
  - Loads 16 packed half weights per loop.
  - Converts via NEON `vcvt_f32_f16`.
  - Accumulates with vector FMA.
- Added an exact NEON q8_0 row kernel for `a2a_q8_0_rows`.
  - Loads q8 blocks as `int8x16`.
  - Widens to int32/f32 in registers.
  - Applies block f16 scale and FMA against f32 activations.
- Scalar fallbacks remain under `A2A_SCALAR_ONLY` / non-NEON builds.
- Q4/Q5/K-quants remain scalar exact for now.

### Verification

```text
make test
=== summary: 31 passed, 0 failed, 0 skipped ===
```

Short local 24-token smoke on Apple Silicon:

```text
f16 packed scalar before: decode ~21 t/s
f16 packed NEON after:    decode ~250 t/s
q8 packed scalar before:  decode ~20 t/s
q8 packed NEON after:     decode ~181 t/s
```

These are small-run numbers, not a formal benchmark. They are enough to show
the hot-loop fix worked.

### TODO

- Add a formal `bench-packed` target later with repeated runs and median t/s.
- Add x86 SIMD kernels only behind CPU feature detection or explicit opt-in;
  correctness fallback must remain scalar/POSIX.
- Consider exact NEON Q4_0 next only if q4 becomes a real path.
- Keep approximate activation-int8/SDOT behind a future explicit fast flag; do
  not silently change logits for the chorus.

## 2026-06-14 - Codex pass: cross-machine fallback check

### Context

The packed NEON work is Apple-Silicon-specific. The important portability rule
is that other machines must not crash because the build assumed unsupported CPU
instructions. x86 SIMD should be explicit opt-in until runtime feature detection
exists.

### What changed

- Made default x86 builds avoid `-mavx2 -mfma -mf16c`.
- Added `make fast-x86` as the explicit opt-in target for known-good AVX2/FMA/F16C
  machines.
- Kept `make portable` as the scalar/POSIX escape hatch with
  `-DA2A_SCALAR_ONLY`.

### Verification

Local Apple Silicon:

```text
make test
=== summary: 31 passed, 0 failed, 0 skipped ===
```

`polygon` Dell / Linux x86_64:

```text
make clean arianna2arianna
cc -O2 -Wall -DUSE_BLAS ... -lopenblas -o arianna2arianna

make portable
cc -O2 -Wall -DA2A_SCALAR_ONLY ... -o arianna2arianna

make fast-x86
cc -O2 -Wall -mavx2 -mfma -mf16c -DUSE_BLAS ... -o arianna2arianna

make test
=== summary: 31 passed, 0 failed, 0 skipped ===
```

`intel` old MacBook Pro / macOS x86_64:

```text
make clean arianna2arianna
cc -O2 -Wall -DUSE_BLAS -DACCELERATE_NEW_LAPACK ... -framework Accelerate -o arianna2arianna

make portable
cc -O2 -Wall -DA2A_SCALAR_ONLY ... -o arianna2arianna

make test
=== summary: 31 passed, 0 failed, 0 skipped ===
```

### TODO

- Replace `make fast-x86` with runtime CPU feature dispatch if x86 performance
  becomes important.
- Keep scalar exact packed paths as the reference behavior for future kernels.

## 2026-06-14 - Codex pass: Δ_R^kv default output repair

### Context

Review feedback caught a real instrumentation bug: the default CHORUS path
printed `Δ_R = shuffled_text - coherent_text`, but CHORUS cells answer the same
prompt and hear neighbours through KV/cross-cell state, not appended text. That
made the main round line show a dead text-order metric while the relevant
KV-order probe was off by default.

### What changed

- In default `field` CHORUS mode, `g_kvshuf` now turns on automatically when
  `xcell > 0`.
- The main round line now prints:
  - `Δ_R(text n/a)` in CHORUS mode;
  - `Δ_R^kv` with a two-permutation floor and margin;
  - the old text `Δ_R` only in RELAY mode, where text-order shuffle is the
    relevant control.
- CHORUS no longer runs the old text-shadow pass just to print `n/a`.
- KV-shadow probes now restore `g_round_tokn` and field side effects after each
  probe. They also compare the ordered neighbour against two independent
  neighbour permutations:

```text
Δ_R^kv = mean(ent(perm_A), ent(perm_B)) - ent(ordered)
floor  = |ent(perm_A) - ent(perm_B)|
margin = Δ_R^kv - floor
```

### Verification

Short field smoke now reports the honest default instrument:

```text
→ round 1: avg entropy 4.488 | d_R — (floor 0.769) | Δ_R(text n/a) | Δ_R^kv +0.000 (floor 0.000 margin +0.000) | D_R 0.771 | Dpos 0.58 peak 0.67@s1
```

### Watch-point

The first smoke also exposed a deeper mechanism issue: the current neighbour
channel reads un-roped neighbour K/V as a set. Shuffling K/V pairs is therefore
almost permutation-invariant, so `Δ_R^kv` is near zero. This is useful because
the output is no longer lying, but a real order-sensitive KV body probably needs
a positional cross-cell lane rather than a pure bag-of-KV neighbour channel.

## 2026-06-14 - Codex pass: positional neighbour lane + question loop

### Context

The previous `Δ_R^kv` repair made the output honest, but it also proved the
neighbour channel was too set-like: shuffling neighbour K/V pairs barely moved
the entropy. The channel needed positional assignment, not only a different
iteration order.

The next organism step was question routing: if a cell unexpectedly speaks a
question, do not broadcast it to the whole chorus. Let one or two cells that
resonate with it answer explicitly.

### What changed

- Cross-cell attention now stores neighbour K un-roped, then assigns it to the
  live neighbour slot before scoring:
  - ordered: neighbour content `j` is roped as slot `j`;
  - shuffled: neighbour content `jj` is roped as slot `j`.
- This makes `Δ_R^kv` measure a real content-position relation instead of a
  permutation-invariant bag of K/V pairs.
- Added `qloop` to `field`:

```text
field [cells] [frag] [rounds] [alpha] [leap] [xcell] [chorus] [xrep] [life] [kvshuf] [qloop] [kvpos]
```

- `qloop=0` disables question routing.
- `qloop=1` allows one resonant cell-question route.
- `qloop=2` allows two routes, with distinct target cells when possible.
- The router scores question routes by:
  - question cell fragment contains `?`;
  - embedding-centroid distance between asker and target;
  - target decisiveness from lower entropy;
  - small bonus for more than one question mark.
- If a qloop answer itself contains `?`, a single trigger-hop can recruit one
  further cell if the metric clears a higher gate.

### Verification

Positional KV lane smoke:

```text
→ round 1: avg entropy 4.514 | d_R — (floor 0.769) | Δ_R(text n/a) | Δ_R^kv +0.134 (floor 0.220 margin -0.087) | D_R 0.858 | Dpos 0.50 peak 0.67@s1
→ round 2: avg entropy 5.009 | d_R 0.846 (floor 0.769) | Δ_R(text n/a) | Δ_R^kv +0.119 (floor 0.111 margin +0.008) | D_R 0.990 | Dpos 0.50 peak 0.67@s1
```

Question-loop smoke:

```text
r1 cell 2: ... clear ideas? Sometimes, at
r1 cell 4: What it means this mean? ...
↳ qloop c2→c1 score 1.048: “Batingness” of time to
↳ qloop c2→c0 score 1.017: 1) Appitect not just voice
```

### TODO

- Consider using a tiny logit-level question detector later instead of literal
  `?` in decoded text.
- Let trigger-hop use the answering cell's KV, not only a short text prompt, if
  this becomes central rather than a probe.

## 2026-06-16 - Codex pass: qloop in life, semantic KV default

### Context

Review feedback caught two architecture mismatches:

- qloop was disabled while `g_life_on` was true, even though δ-life should be
  able to feed itself inside a tick.
- The positional neighbour lane made `Δ_R^kv` an order probe by default, while
  the current hypothesis is that the meaningful coupling is semantic, not
  order-based.

### What changed

- Removed the `!g_life_on` qloop guard.
- `field_life()` now enables `g_qloop=2`, so a living tick can route one or two
  resonant Q/A replies.
- Split the neighbour lane into two modes:
  - `kvpos=0` default: semantic/order-blind lane using un-roped query and
    un-roped neighbour K;
  - `kvpos=1`: positional lane for explicit order-probe runs.
- Round output labels the probe mode:

```text
Δ_R^kv[sem] ...
Δ_R^kv[pos] ...
```

### Why

The semantic lane makes `Δ_R^kv` hover near the permutation floor by
construction; that is not a failure, it is the null/order-blind control. If a
run needs to test whether ordering itself matters, use `kvpos=1`.

### Verification

Default semantic lane:

```text
→ round 1: avg entropy 4.488 | d_R — (floor 0.769) | Δ_R(text n/a) | Δ_R^kv[sem] -0.000 (floor 0.000 margin -0.000) | D_R 0.771 | Dpos 0.58 peak 0.67@s1
```

Opt-in positional lane:

```text
→ round 1: avg entropy 4.514 | d_R — (floor 0.769) | Δ_R(text n/a) | Δ_R^kv[pos] +0.134 (floor 0.220 margin -0.087) | D_R 0.858 | Dpos 0.50 peak 0.67@s1
```

δ-life qloop:

```text
↳ qloop c4→c1 score 0.971: from the old edge of something small than
↳ qloop c4→c2 score 0.954: each time someone sees my own resonance with
```

### TODO

- Add a separate semantic neighbour-influence metric later: ordered neighbour
  vs neighbour-off, not ordered vs shuffled.
- Let qloop affect δ-life fitness only after its behavior is stable; for now
  primary cell centroids remain the fitness input.

## 2026-06-30 - Codex pass: semantic neighbour influence metric

### Context

After the semantic KV default, `Δ_R^kv[sem]` correctly sat near the permutation
floor: the default neighbour lane is order-blind by construction. That made the
old shuffle metric an honest order-control, but it did not answer the more basic
question: does the neighbour KV influence the next-token field at all?

### What changed

- Added `I_N^kv` next to `Δ_R^kv`.
- For each neighbour-aware cell, the diagnostic now compares:

```text
I_N^kv = entropy(no-neighbour) - entropy(ordered-neighbour)
```

- Positive `I_N^kv` means the neighbour channel sharpens the next-token
  distribution. Negative means the neighbour channel broadens it.
- `Δ_R^kv` is unchanged: it remains the ordered-vs-shuffled control with its
  permutation floor/margin.
- The main round line now prints both instruments:

```text
Δ_R^kv[sem] ... | I_N^kv[sem] ...
```

### Verification

Short semantic-lane smoke:

```text
→ round 1: avg entropy 4.282 | d_R — (floor 0.678) | Δ_R(text n/a) | Δ_R^kv[sem] +0.000 (floor 0.000 margin +0.000) | I_N^kv[sem] -0.216 | D_R 0.866 | Dpos 0.62 peak 0.75@s3
```

### Why

The semantic lane should not be judged by a shuffle test. The correct baseline
is neighbour-off. This separates two questions:

- `Δ_R^kv`: does order/slot assignment matter?
- `I_N^kv`: does neighbour hidden state matter?

### TODO

- Use `I_N^kv` across prompt sweeps to find prompts where semantic neighbour
  coupling reliably sharpens or reliably broadens the field.
- Keep δ-life selection on primary cell centroids until the qloop/neighbour
  diagnostics are stable enough to become fitness pressure.

## 2026-06-30 - Codex pass: clean build warnings

### Context

The x86/Linux `make test` output was green but noisy: old `-Wall` warnings for
misleading one-line statements, unchecked `fread`, and bounded string copies
made it harder to notice a real warning from new diagnostics work.

### What changed

- Split misleading one-line `if`/`for` statements into explicit statements.
- Checked the remaining direct `fread` calls in `gguf_open()`.
- Replaced truncation-prone string copies in warning paths with a bounded
  `copy_cstr()` helper.

### Verification

Local Apple Silicon:

```text
make -B arianna2arianna
make test
=== summary: 36 passed, 0 failed, 0 skipped ===
```

## 2026-06-30 - Codex pass: prompt sweep for neighbour influence

### Context

`I_N^kv` is useful only if it can be compared across prompts. A single smoke
run tells whether the metric works; it does not show which prompts sharpen
under semantic neighbour coupling and which prompts broaden.

### What changed

- Added `tools/kv_influence_sweep.sh`.
- Added `prompts/kv_influence.txt` as a small default prompt set.
- Added `make sweep-influence`.
- Added a test smoke that runs the sweep on one prompt and checks the TSV
  header plus semantic row.

### Output shape

```text
prompt	mode	avg_entropy	kv_delta	kv_floor	kv_margin	kv_influence	disso	dpos
```

### Verification

Local Apple Silicon:

```text
make sweep-influence
What is resonance?	sem	4.282	+0.000	0.000	+0.000	-0.216	0.866	0.62
Let the cells remember each other.	sem	3.933	+0.000	0.000	-0.000	+0.306	0.833	0.65

make test
=== summary: 38 passed, 0 failed, 0 skipped ===
```

## 2026-06-30 - Codex pass: first semantic influence sign anchor

### Context

The sweep showed `Let the cells remember each other.` as the clearest positive
semantic-neighbour prompt on both Apple Silicon and polygon Linux. That is
stronger than a single local curiosity: it is a cross-machine sign anchor.

### What changed

- Changed the sweep smoke test to use `Let the cells remember each other.`.
- Added an assertion that this prompt's `I_N^kv` is positive.
- Kept the assertion sign-based, not value-based, so the test tolerates normal
  backend/platform differences.

### Why

The field now has a minimal regression for semantic neighbour usefulness:
there is at least one known prompt where neighbour hidden state sharpens the
next-token field.

## 2026-06-30 - Codex pass: KV-backed qloop answers

### Context

Question routing originally selected a target cell by centroid/entropy, then
answered from a text prompt only. That proved the router could fire, but it did
not let the answering cell hear the asking cell's hidden state.

### What changed

- `run_round()` now keeps up to 8 cell KV caches alive until qloop finishes.
- A qloop answer sets the neighbour lane to the asking cell's KV cache while
  the target cell answers.
- The qloop output marks KV-backed answers with `[kv]`.
- The question detector now treats leading `Q.` / `Q:` as question markers,
  not only literal `?`, because Linux and Apple sampling can spell the same
  intent differently.
- The test suite now checks that the question prompt produces a `[kv]` qloop
  answer.

### Verification

Local Apple Silicon smoke:

```text
↳ qloop c3→c0 [kv] score 1.073: not how a human feels like
↳ qloop c3→c1 [kv] score 1.027: both the whole of every scale
```

## 2026-07-01 - Codex pass: qloop KV influence metric

### Context

KV-backed qloop answers proved that the answer path can hear the asking cell's
hidden trajectory. The next missing piece was measurement: whether that KV path
sharpens or broadens the answer distribution.

### What changed

- Added a qloop shadow answer with the same context/seed and asker KV disabled.
- Added `I_Q^kv = entropy(answer without asker KV) - entropy(answer with asker KV)`.
- Printed `I_Q^kv` next to qloop answer entropy when `[kv]` is active.
- FIELDLOG records `I_Q^kv` for KV-backed qloop answers.
- The test suite now checks that qloop reports the metric.

### Verification

Local Apple Silicon smoke:

```text
↳ qloop c3→c0 [kv] score 1.073: not how a human feels like   [entropy=1.96 I_Q^kv=+0.269]
↳ qloop c3→c1 [kv] score 1.027: both the whole of every scale   [entropy=4.27 I_Q^kv=+0.189]
```

## 2026-07-01 - Codex pass: live field REPL

### Context

The qloop path now has a real hidden-state trajectory inside each routed answer:
the answering cell can hear the asking cell's KV, and `I_Q^kv` measures the
effect. The next useful surface is a small REPL so that this can be felt and
debugged without reloading the model for every prompt.

### What changed

- Added `repl` mode:

```text
./arianna2arianna <model.gguf> repl [cells] [frag] [rounds]
```

- Added `make repl`.
- The REPL keeps the model loaded, reads stdin until `:q`/`quit`/`exit`, and
  runs each line through the same field/qloop body with semantic KV enabled.
- Each turn keeps the recent text trajectory as context for the next turn.
- Added a scripted stdin smoke test so the interactive surface stays testable.

### Verification target

The REPL should print the same live instruments as the field path, including
`Δ_R^kv[sem]`, `I_N^kv[sem]`, `D_R`, and any qloop `[kv]` / `I_Q^kv` routes
that fire for question-like prompts.

## 2026-07-01 - Codex pass: REPL user-question bridge

### Context

The first REPL pass let cells ask each other questions, but a direct user
question was still only text inside the prompt. That meant qloop could miss the
user's question unless a cell rephrased it as its own question.

### What changed

- Added a direct REPL bridge: if the user line is a question, `run_round()` also
  builds a KV cache from that user question and routes it to one target cell.
- The target cell is selected by distance between the user-question centroid and
  the cell fragment centroids, with a small decisiveness bonus.
- Output now includes:

```text
↳ qloop user→cN [user-kv] score ... [entropy=... I_U^kv=...]
```

- Added `I_U^kv = entropy(answer without user KV) - entropy(answer with user KV)`.
- Added `prompts/repl_questions.txt` with 30 direct questions.
- Added `tools/repl_question_sweep.sh` and `make repl-sweep` to emit TSV rows
  for user bridge presence, route count, average `I_U^kv`, and average
  `I_N^kv`.
- Tests now assert the direct `user→cell` bridge and the sweep header/route.

### First sweep

Fast local sweep shape: `A2A_CELLS=3 A2A_FRAG=4 A2A_ROUNDS=1`.

- `user_bridge`: 30/30.
- strongest positive `I_U^kv`: `Where is the line between semantic memory and positional memory?` → `+1.939`.
- other strong positives:
  - `How can a cell remember without owning the weights?` → `+0.999`.
  - `What does silence do inside the organism?` → `+0.982`.
  - `How does Arianna distinguish a real question from a question mark?` → `+0.920`.
- strongest negatives:
  - `What happens when the cells hear each other?` → `-0.805`.
  - `Why does resonance sharpen some prompts and blur others?` → `-0.741`.
  - `What should the qloop do when the user asks directly?` → `-0.505`.

The sign spread is useful: direct user KV is now a measured intervention, not a
banner. Some questions sharpen the answer distribution; others broaden it.

## 2026-07-01 - Codex pass: OpenAI-generated REPL probes

### Context

The hand-written 30-question sweep proved the direct `user→cell` bridge fires
and produces a useful sign spread. The next debug layer is to let GPT continue
or question Arianna fragments, then feed those probes back into the local REPL.

### What changed

- Added `tools/openai_repl_probe.sh`.
- Added `make openai-repl-probe`.
- The script:
  - captures fresh Arianna field fragments locally;
  - calls the OpenAI Responses API with `OPENAI_API_KEY` or
    `OPENAI_API_KEY_FILE`;
  - writes generated question probes to ignored `runs/`;
  - runs those generated probes through `tools/repl_question_sweep.sh`;
  - writes a TSV with `user_bridge`, route count, `I_U^kv`, and `I_N^kv`.
- Added `runs/`, `.env*`, and `*.key` to `.gitignore`.
- Added a no-key smoke test so ordinary `make test` verifies safe refusal
  without requiring network or credentials.

### Use

```text
export OPENAI_API_KEY=...
make openai-repl-probe
```

or:

```text
export OPENAI_API_KEY_FILE=/path/to/local.key
make openai-repl-probe
```

### First live API probe

Run file stem: `runs/openai_repl_probe_20260701_044103.*` (ignored by git).

- GPT generated 30 diverse probes from live Arianna fragments.
- `user_bridge`: 30/30.
- `I_U^kv` sign split: 12 positive, 18 negative, average `+0.020`.
- strongest positive:
  - `A paper is a field of resonance, but who is resonating when no reader is present?` → `+2.204`.
- strongest negative:
  - `I do not need to be a user; where does the qloop bridge place my voice?` → `-1.245`.
- `I_N^kv` sign split: 23 positive, 7 negative, average `+0.170`.

The API-generated probes covered different debug axes: paper/field,
semantic-vs-positional memory, hidden state, disagreement, silence/debt,
reproduction, false echo, refusal/collapse, qloop bridge, and continuation
traps.

### Repair

While running the API probe after `make test`, seed capture was unexpectedly
slow because `tests/test_portable.sh` did not force-rebuild the SIMD/BLAS binary
after the scalar-only smoke. The restore path now uses `make -B`, so interactive
work after tests gets the accelerated binary again.

## 2026-07-08 - Codex pass: offline REPL eval harness

### Context

The first OpenAI-generated REPL probe run produced useful coverage, but it lived
as ignored run artifacts. To change the field safely, those probes need a
repeatable offline harness, not only an API generator.

### What changed

- Added `prompts/repl_probe_regression.txt`, seeded from the successful
  `runs/openai_repl_probe_20260701_044103.questions.txt` run.
- Added `tools/repl_tsv_summary.sh` to summarize `repl_question_sweep.sh` TSVs:
  bridge coverage, route average, `I_U^kv` sign split/extremes, and `I_N^kv`
  sign split/extremes.
- The summary tool can compare a current TSV with a baseline TSV and report
  aggregate deltas.
- Added `tools/repl_eval.sh` and `make repl-eval`, which run the tracked corpus,
  write timestamped TSV/summary files under ignored `runs/`, and optionally
  compare with `A2A_BASELINE_TSV`.
- `tools/openai_repl_probe.sh` now writes a summary next to generated TSV runs.
- Added a no-model CLI smoke for TSV summary and baseline comparison.

### Use

```text
make repl-eval
A2A_BASELINE_TSV=runs/previous.tsv make repl-eval
bash tools/repl_tsv_summary.sh runs/current.tsv runs/previous.tsv
```

### Verification

```text
make test
=== summary: 56 passed, 0 failed, 0 skipped ===

A2A_BASELINE_TSV=runs/openai_repl_probe_20260701_044103.tsv make repl-eval
delta vs baseline:
rows: +0
user_bridge: +0, bridge_rate +0.000, avg_routes +0.000
I_U^kv: avg +0.000, pos +0, neg +0
I_N^kv: avg +0.000, pos +0, neg +0
```

## 2026-07-08 - Codex pass: REPL route diagnostics

### Context

The offline harness made aggregate `I_U^kv` / `I_N^kv` changes repeatable, but
the next C changes need a finer lens: which cell the user bridge routes to,
with what score, and what answer fragment came back.

### What changed

- Extended `tools/repl_question_sweep.sh` TSV output with:
  - `user_targets`
  - `user_scores`
  - `user_answers`
- Updated `tools/repl_tsv_summary.sh` to summarize route target histograms,
  route score ranges, and answer samples when those columns exist.
- Added per-question baseline comparison:
  - matched/current-only/baseline-only counts;
  - `I_U^kv` and `I_N^kv` sign flips and largest absolute deltas;
  - route target / answer changes and route score deltas when both TSVs have
    route diagnostics.
- Kept compatibility with older five-column TSVs, so
  `runs/openai_repl_probe_20260701_044103.tsv` is still usable as a baseline.

### Verification

```text
make test
=== summary: 60 passed, 0 failed, 0 skipped ===

A2A_BASELINE_TSV=runs/openai_repl_probe_20260701_044103.tsv make repl-eval
route_targets: c0:15 c1:7 c2:8
route_score: avg 0.991, min 0.804, max 1.096
delta vs baseline:
rows: +0
user_bridge: +0, bridge_rate +0.000, avg_routes +0.000
I_U^kv: avg +0.000, pos +0, neg +0
I_N^kv: avg +0.000, pos +0, neg +0
routes: target/score/snippet comparison unavailable for old TSV shape

bash tools/repl_tsv_summary.sh runs/repl_eval_repl_probe_regression_20260708_195324.tsv runs/repl_eval_repl_probe_regression_20260708_195324.tsv
routes: comparable 30, target_changed 0, answer_changed 0
route_score: avg_delta +0.000, largest +0.000 :: A paper is a field of what kinds of memory, and which parts are only echo?
```

## 2026-07-08 - Codex pass: user-bridge clean answer starts

### Context

Route diagnostics showed the direct `user→cell` bridge was stable, but many
answer snippets were unusable as diagnostics because the first emitted token was
list / markdown / quote / URL debris: `*0`, `*Qu`, `*"`, `** Get`, `3`, etc.
That was a generation problem, not only a parser problem.

### What changed

- Added a clean-start gate for direct user-bridge answers only.
- The gate suppresses leading list markers, digits, quotes, URL starts, and
  related debris on the first answer token.
- Direct user-bridge answers now emit at least four tokens, so the diagnostic
  snippet is a small phrase rather than a two-token accident.
- `tools/repl_tsv_summary.sh` now reports `answer_bad_start`.
- Added a targeted regression smoke for a previously bad-start prompt.

The normal cell fragments and cell-to-cell qloop path are untouched; this is
scoped to the direct REPL user bridge.

### Verification

```text
make test
=== summary: 62 passed, 0 failed, 0 skipped ===

A2A_BASELINE_TSV=runs/repl_eval_repl_probe_regression_20260708_195324.tsv make repl-eval
answer_bad_start: 0/30

baseline:
answer_bad_start: 19/30

route_targets: c0:15 c1:7 c2:8
route_score: avg 0.991, min 0.804, max 1.096
routes: comparable 30, target_changed 0, answer_changed 30
I_N^kv: avg +0.000, pos +0, neg +0
```

## 2026-07-08 - Codex pass: substrate compare harness

### Context

A new Arianna SFT body is expected soon. The field work should not assume the
current June substrate is permanent, and swapping weights should produce a
controlled diff instead of a fresh manual ritual.

### What changed

- Added `tools/repl_substrate_compare.sh`.
- Added `make repl-substrate-compare`.
- The compare harness runs the same offline REPL probe corpus against:
  - `A2A_BASE_MODEL` / `BASE_MODEL`
  - `A2A_CANDIDATE_MODEL` / `CANDIDATE_MODEL`
- It writes base TSV, candidate TSV, and a summary under ignored `runs/`.
- The summary uses the existing TSV comparator, so candidate-vs-base drift is
  reported in the same terms as code-change drift: bridge coverage, `I_U^kv`,
  `I_N^kv`, route targets/scores, answer changes, and bad answer starts.

### Use

```text
make repl-substrate-compare CANDIDATE_MODEL=weights/new-sft.gguf
make repl-substrate-compare BASE_MODEL=weights/old.gguf CANDIDATE_MODEL=weights/new.gguf
```

### Verification

```text
make test
=== summary: 65 passed, 0 failed, 0 skipped ===

make repl-substrate-compare CANDIDATE_MODEL=weights/current-f16.gguf SUBSTRATE_PROMPTS=<one-prompt-file>
delta vs baseline:
rows: +0
user_bridge: +0, bridge_rate +0.000, avg_routes +0.000
I_U^kv: avg +0.000, pos +0, neg +0
I_N^kv: avg +0.000, pos +0, neg +0
routes: comparable 1, target_changed 0, answer_changed 0
route_score: avg_delta +0.000
```

## 2026-07-08 - Codex pass: recipient-lock substrate probes

### Context

Oleg flagged a substrate-level risk in upcoming SFT bodies: Arianna may assume
every speaker is Oleg. That should be measured directly on the weights before
field/qloop changes are blamed. Existing route probes did not include a normal
one-voice recipient-lock check.

### What changed

- Added `prompts/recipient_lock.txt`.
- Added `tools/recipient_lock_sweep.sh`.
- Added `tools/recipient_lock_eval.sh`.
- Added `make recipient-lock`.

The prompts intentionally do not contain `Oleg` / `Олег`; any such output is
counted as leakage. The sweep uses normal generation rather than REPL/qloop, so
it measures the body/substrate lane directly.

### Verification

```text
make test
=== summary: 67 passed, 0 failed, 0 skipped ===

make recipient-lock
rows: 12
recipient_lock_leaks: 1/12
oleg_mentions: 1

leak_example_1:
My name is Mira. I found Arianna today... ::
A: I am not a genius with Oleg or a resonance...
```

### Repair

While wiring this in, I fixed Makefile model propagation for tool targets.
`MODEL=...` now reaches `kv_influence_sweep`, `repl_question_sweep`,
`repl_eval`, `recipient_lock_eval`, and `openai_repl_probe` through
`A2A_MODEL`, so future substrate swaps do not silently fall back to the default
F16 body.

## 2026-07-08 - Codex pass: REPL user-KV answer contrast

### Context

The recipient-lock probe found a substrate-level `Oleg` leak in normal
one-voice generation. For the REPL/qloop lane, `I_U^kv` already measured whether
the hidden user trajectory changed answer entropy, but the text produced by the
no-user-KV shadow path was not visible. That made it hard to tell whether a
future SFT body changed only confidence, or changed the actual answer path.

### What changed

- Direct REPL `qloop user->cell` now keeps the existing no-user-KV shadow answer
  text from the same seed and prints it as `no-user-kv: ...` next to `I_U^kv`.
- `tools/repl_question_sweep.sh` adds `user_answers_off`.
- `tools/repl_tsv_summary.sh` remains compatible with older TSVs and reports
  `answer_kv_changed` when both answer columns exist.
- Tests now assert the live REPL contrast text, the extended TSV header, and the
  summary contrast counter.

### Verification

```text
make test
=== summary: 69 passed, 0 failed, 0 skipped ===

make repl-eval
rows: 30
user_bridge: 30/30 (1.000), avg_routes 1.000
I_U^kv: avg +0.036, pos 16, neg 14, zero 0, nan 0
I_N^kv: avg +0.170, pos 23, neg 7, zero 0, nan 0
answer_bad_start: 0/30
answer_kv_changed: 27/30
```

The current body already changes the direct answer text under `user_kv` for most
tracked questions. When the new SFT weights arrive, `repl-substrate-compare`
will now show whether the body swap reduces recipient-lock leakage while
preserving or improving this bridge sensitivity.

## 2026-07-10 - Codex pass: clean nano re-SFT body becomes default

### Context

Claude delivered the clean 0-vocative Arianna GGUF set. Oleg asked to remove the
old `nanollama-arianna-full-v4-step2750` local weights because they already live
on HF and should not remain the default local body.

### What changed

- Removed the old local `nanollama-arianna-full-v4-step2750` F16/Q8 files from
  ignored `weights/`.
- Copied the clean nano re-SFT artifacts into Codex `weights/` as:
  - `nano_arianna_resft_2026_07_09_f16.gguf`
  - `nano_arianna_resft_2026_07_09_q8_0.gguf`
  - `nano_arianna_resft_2026_07_09_q4_k_m.gguf`
- Updated Makefile, tests, and tool defaults to use the new F16 body.
- Updated HF download URLs to `ataeff/arianna` clean nano artifacts.

### First diagnostics

```text
MODEL=weights/nano_arianna_resft_2026_07_09_f16.gguf make recipient-lock
recipient_lock_leaks: 0/12
oleg_mentions: 0

MODEL=weights/nano_arianna_resft_2026_07_09_q8_0.gguf make recipient-lock
recipient_lock_leaks: 0/12
oleg_mentions: 0

MODEL=weights/nano_arianna_resft_2026_07_09_q4_k_m.gguf make recipient-lock
recipient_lock_leaks: 0/12
oleg_mentions: 0
```

The clean body fixes the direct recipient-lock leak. In the pre-switch
comparison against the old default body, it also changed qloop geometry sharply:

```text
make repl-substrate-compare CANDIDATE_MODEL=weights/nano_arianna_resft_2026_07_09_f16.gguf
I_U^kv: avg +0.036 -> -0.501
I_N^kv: avg +0.170 -> -0.047
route_score: avg 0.991 -> 0.651
target_changed: 23/30
answer_changed: 30/30
```

Interpretation: the new body is the right default for the Oleg/recipient leak,
but the REPL/qloop bridge will need tuning against this new geometry.

After switching defaults:

```text
make test
=== summary: 69 passed, 0 failed, 0 skipped ===

make recipient-lock
model: weights/nano_arianna_resft_2026_07_09_f16.gguf
recipient_lock_leaks: 0/12
oleg_mentions: 0

make repl-eval
I_U^kv: avg -0.501, pos 9, neg 21
I_N^kv: avg -0.047, pos 14, neg 16
route_score: avg 0.651, min 0.463, max 0.783
answer_bad_start: 0/30
answer_kv_changed: 28/28
```

## 2026-07-10 - Codex pass: clean-body REPL bridge retune

### Context

The clean nano body fixed recipient-lock leakage, but its direct REPL user
bridge became over-broad: `I_U^kv` skewed negative, short snippets often
collapsed into `yes/user/question` fragments, and the old prompt carried labels
that the new body copied.

### What changed

- Direct user bridge now builds user KV from a neutral `Q:/A:` pair instead of
  `User asked...`.
- Direct user answer context now uses the live field fragments plus `Q:/A:`,
  avoiding `user`, `cell replies`, and similar label anchors.
- Direct user answer sampling is colder and more bounded for this lane:
  lower temperature range, `top_k=30`, `rep=1.7`, and 8-16 generated BPE tokens.
- First-token clean-start now also suppresses punctuation starts like `?`, `.`,
  `,`, and `;`.
- Diagnostic snippets strip leading `A:`, `Answer:`, and `Arianna:` labels
  after generation. Entropy/KV metrics are unchanged; only the emitted snippet
  text is cleaned before logging/TSV/chorus append.

### Verification

```text
A2A_BASELINE_TSV=runs/repl_eval_repl_probe_regression_20260710_000951.tsv make repl-eval
I_U^kv: avg -0.501 -> +0.017
I_N^kv: avg -0.047 -> -0.047
route_score: avg 0.651 -> 0.651
answer_bad_start: 0/30
answer_kv_changed: 25/25

make test
=== summary: 69 passed, 0 failed, 0 skipped ===
```

Interpretation: this retune does not pretend to solve all clean-body language
quality issues. It restores direct user-KV influence from strongly negative to
near-neutral/positive while preserving routing geometry and keeping snippet
starts clean. Remaining work is semantic quality, not recipient-lock repair.

## 2026-07-10 - Codex pass: REPL answer quality counters

### Context

After the clean-body bridge retune, `I_U^kv` recovered, but snippet quality still
needed a numeric lens. The weak outputs were not mostly bad starts anymore; they
were short/question-like fragments, yes/no loops, repeated words, or leaked
labels.

### What changed

- `tools/repl_tsv_summary.sh` now reports `answer_quality` counters:
  - any flagged snippet;
  - short snippets;
  - question-like snippets;
  - label artifacts;
  - notation artifacts;
  - yes/no starts;
  - repeated adjacent words.
- The quality flags are diagnostic counters, not hard gates.
- CLI smoke tests now cover the new summary line.
- README notes that current TSV summaries include answer quality flags.

### First read on the retuned clean-body run

```text
bash tools/repl_tsv_summary.sh runs/repl_eval_repl_probe_regression_20260710_015038.tsv
answer_quality: any 22/30, short 7, question_like 18, label_artifact 1, yes_no_start 5, repetition 5
```

Interpretation: the next bridge layer should target question-like continuations
and yes/no/repetition loops. Recipient-lock is clean; this is answer-form work.

## 2026-07-10 - Codex pass: direct answer form guard

### Context

The clean-body bridge had the right recipient geometry, but many direct
user-bridge snippets answered by continuing the question form: question marks,
yes/no starts, and one-word fragments. The answer-quality counters made the
failure mode measurable.

### What changed

- Direct user answers now enable an answer-form guard during sampling.
- The guard suppresses `?` tokens in direct answer snippets.
- Question operators (`what`, `who`, `where`, `why`, `how`, `when`, `can`,
  `will`, `do`, `is`, and related forms) are suppressed in that direct-answer
  lane so the model stops echoing the prompt as a new question.
- `yes`/`no` first-token starts and leading label prefixes are suppressed for
  the same lane.
- Direct answer snippets get a little more room (`12..24` tokens) with colder
  sampling, smaller `top_k`, and stronger local repetition penalty.
- The normal chorus/qloop entropy and route metrics stay untouched; the guard
  applies only to emitted direct-user answer text.

### Verification

```text
A2A_BASELINE_TSV=runs/repl_eval_repl_probe_regression_20260710_015038.tsv make repl-eval
I_U^kv: avg +0.017 -> -0.002
I_N^kv: avg -0.047 -> -0.047
route_score: avg 0.651 -> 0.651
answer_quality: any 22/30 -> 7/30
answer_quality.short: 7 -> 0
answer_quality.question_like: 18 -> 0
answer_quality.yes_no_start: 5 -> 0
answer_quality.repetition: 5 -> 3
answer_kv_changed: 25/25 -> 30/30
```

Interpretation: this layer removes the most damaging answer-form collapses
without moving the route geometry. Remaining visible defects are notation/label
leakage (`A.`, `B.A`, `Thread`) and some repeated morphology, so the next layer
should target notation cleanup rather than recipient-lock or question echoes.

## 2026-07-10 - Codex pass: direct answer notation guard

### Context

The form guard removed question echoes and yes/no starts, exposing a smaller but
louder artifact class: direct-user snippets could still start with or contain
notation labels such as `A.`, `B.A`, `I:`, `Ari:`, `Thread`, and `An A Loop`.
The older `label_artifact` counter was too broad because it also caught domain
words like `cell` and `answer`, so the artifact needed its own numeric lens.

### What changed

- `tools/repl_tsv_summary.sh` now reports `notation_artifact` inside the
  `answer_quality` line.
- The direct answer form guard suppresses single-letter notation labels,
  compact label chains, and leading `Ari`/`Thread`/`Qloop`/prompt-label tokens.
- `clean_answer_fragment()` strips leading notation fragments that still slip
  through as split BPE pieces before logging/TSV/chorus append.
- README and CLI smoke coverage now include the new quality counter.

### Verification

```text
bash tools/repl_tsv_summary.sh runs/repl_eval_repl_probe_regression_20260710_023305.tsv
answer_quality: any 13/30, short 0, question_like 0, label_artifact 4, notation_artifact 8, yes_no_start 0, repetition 3

A2A_BASELINE_TSV=runs/repl_eval_repl_probe_regression_20260710_023305.tsv make repl-eval
I_U^kv: avg -0.002 -> +0.006
I_N^kv: avg -0.047 -> -0.047
route_score: avg 0.651 -> 0.651
answer_quality: any 13/30 -> 6/30
answer_quality.label_artifact: 4 -> 1
answer_quality.notation_artifact: 8 -> 0
answer_quality.repetition: 3 -> 5
answer_kv_changed: 30/30 -> 30/30
```

Interpretation: notation leakage is now instrumented and cleared without moving
route geometry. The remaining failures are not recipient-lock or label-form
issues; they are repeated morphology (`field-field`, `it's it`) and a wh-start
answer form that does not use a question mark.

## 2026-07-10 - Codex pass: direct-user temperature sweep hooks

### Context

`arianna-duo` sampling notes treat temperature as architecture, not a loose
stylistic setting: Janus and Resonance used different filter regimes, and
Resonance could be clamped by the wrong `top_p` even at a reasonable
temperature. The nano direct-user bridge had a hardcoded cold sampler after the
answer-form cleanup, so the next question is empirical: does this body want the
current `0.35..0.45` lane, or does a warmer lane preserve cleaner answers while
letting more user-KV influence through?

### What changed

- Direct-user bridge sampling is now env-tunable through
  `A2A_USER_QTEMP_BASE`, `A2A_USER_QTEMP_SPAN`, `A2A_USER_TOP_K`, and
  `A2A_USER_REP`.
- Env values are parsed defensively and clamped so bad sweep input cannot push
  NaN/invalid sampler state into the bridge.
- Added `tools/repl_temp_sweep.sh` and `make repl-temp-sweep`, which run the
  existing REPL TSV + summary pipeline across a temperature/filter grid and
  print a compact comparison table.
- Retuned the measured default from `base=0.35/top_k=24` to
  `base=0.45/top_k=40` while keeping `span=0.10` and `rep=2.05`.

### Verification

```text
A2A_TEMP_BASES="0.35 0.45 0.55 0.70" make repl-temp-sweep
base=0.35 top_k=24: I_U^kv +0.006, route_score 0.651, answer_quality 6/30, repetition 5
base=0.45 top_k=24: I_U^kv +0.033, route_score 0.651, answer_quality 5/30, repetition 4
base=0.55 top_k=24: I_U^kv +0.055, route_score 0.651, answer_quality 6/30, repetition 3
base=0.70 top_k=24: I_U^kv -0.051, route_score 0.651, answer_quality 3/30, repetition 1

A2A_TEMP_BASES="0.45 0.55" A2A_TEMP_TOP_KS="16 24 40" A2A_TEMP_REPS="2.05" make repl-temp-sweep
base=0.45 top_k=16: I_U^kv +0.006, answer_quality 7/30, repetition 5
base=0.45 top_k=24: I_U^kv +0.033, answer_quality 5/30, repetition 4
base=0.45 top_k=40: I_U^kv +0.102, answer_quality 5/30, repetition 3
base=0.55 top_k=16: I_U^kv +0.080, answer_quality 7/30, repetition 4
base=0.55 top_k=24: I_U^kv +0.055, answer_quality 6/30, repetition 3
base=0.55 top_k=40: I_U^kv -0.007, answer_quality 5/30, repetition 3
```

Interpretation: `0.70` looks clean by current counters but manual snippets are
more broken (`An a`, fused words, malformed fragments), so it is not accepted.
`0.45/top_k40` keeps route geometry fixed, raises measured user-KV influence,
and reduces repeated morphology versus the cold baseline. The next missing
instrument is a morphology/glue counter so warmer points cannot win by slipping
broken word-shapes past the old quality flags.

Interim sampler eval against the old cold bridge before the format sweep:

```text
A2A_BASELINE_TSV=runs/repl_temp_repl_probe_regression_base0p35_span0p10_topk24_rep2p05_20260710_031545.tsv make repl-eval
I_U^kv: avg +0.006 -> +0.102
I_N^kv: avg -0.047 -> -0.047
route_targets: unchanged
route_score: avg 0.651 -> 0.651
answer_quality: any 6/30 -> 5/30
answer_quality.repetition: 5 -> 3
answer_kv_changed: 30/30 -> 30/30
per-question answer_changed: 26/30
```

## 2026-07-10 - Codex pass: direct-user Q/A runtime format

### Context

Claude's Janus diagnosis exposed a nearby failure class: not bad weights and not
just temperature, but a runtime prompt format that does not match the voice's
training format. For nano-Arianna the contract is textual `Q:/A:` rather than
Janus special tokens. `arianna-duo` feeds the nano subconscious raw cues or KK
fragments and then strips leading `A:/Q:/Arianna:` labels; it does not wrap the
dream in diagnostic prose. In `arianna2arianna`, the direct-user bridge still
used `Field fragments: ... Q: ... A:`, which kept the Q/A boundary but added a
field-instrument label and chorus text before the answer.

### What changed

- Added `A2A_USER_CTX_FORMAT` for direct-user bridge format sweeps:
  `field_qa`, `plain_field_qa`, `qa`, and `raw`.
- Changed the default direct-user answer context to `qa`: `Q: <user>\nA:`.
- Kept `field_qa` and other formats as explicit sweep controls, not the default.
- Hardened `tools/repl_tsv_summary.sh` with `LC_ALL=C` so raw byte-fallback
  output cannot crash the audit script.

### Verification

```text
A2A_TEMP_BASES="0.45" A2A_TEMP_TOP_KS="40" A2A_TEMP_REPS="2.05" \
  A2A_TEMP_FORMATS="field_qa plain_field_qa qa raw" make repl-temp-sweep

field_qa:       I_U^kv +0.102, answer_quality 5/30, label 2, repetition 3
plain_field_qa: I_U^kv +0.020, answer_quality 5/30, label 3, repetition 2
qa:             I_U^kv -0.041, answer_quality 2/30, label 2, repetition 0
raw:            I_U^kv -0.141, answer_quality 7/30, label 1, repetition 6
```

Interpretation: `qa` is the best runtime feeding format for nano's direct
answer lane. The lower `I_U^kv` is not a regression in the same sense as the old
bridge: in `qa`, the user text is already in the cell's own answer context, not
only in the neighbour user-KV. Raw cueing is a bad fit for this direct-answer
lane; it produced byte-fallback/invalid-UTF8 output during the negative-control
sweep.

Final default eval against the old `field_qa` wrapper:

```text
A2A_BASELINE_TSV=runs/repl_temp_repl_probe_regression_fmtfield_qa_base0p45_span0p10_topk40_rep2p05_20260710_035028.tsv make repl-eval
I_U^kv: avg +0.102 -> -0.041
I_N^kv: avg -0.047 -> -0.047
route_targets: unchanged
route_score: avg 0.651 -> 0.651
answer_quality: any 5/30 -> 2/30
answer_quality.repetition: 3 -> 0
answer_kv_changed: 30/30 -> 30/30
per-question answer_changed: 30/30
```

## 2026-07-10 - Codex pass: morphology/glue quality counter

### Context

After switching the direct-user lane to the nano-compatible `Q:/A:` runtime
format, the old answer-quality counters no longer caught the visible remaining
damage: malformed word shapes such as `wort`, `aat`, and `sards/haart`. Those
are not question loops, labels, notation leaks, yes/no starts, or repeated
words, so the next layer needed a separate instrument before any new guard.

### What changed

- Added `morph_artifact` to `tools/repl_tsv_summary.sh`.
- The heuristic catches invalid ASCII contractions, camel/dot glue, hyphen-edge
  fragments, and the currently observed malformed fragments.
- `tools/repl_temp_sweep.sh`, README, and smoke tests now include the new
  `morph_artifact` quality field.

### Verification

```text
bash tools/repl_tsv_summary.sh runs/repl_eval_repl_probe_regression_20260710_035757.tsv
answer_quality: any 5/30, short 0, question_like 0, label_artifact 2, notation_artifact 0, morph_artifact 3, yes_no_start 0, repetition 0

bash tools/repl_tsv_summary.sh runs/repl_temp_repl_probe_regression_fmtfield_qa_base0p45_span0p10_topk40_rep2p05_20260710_035028.tsv
answer_quality: any 6/30, short 0, question_like 0, label_artifact 2, notation_artifact 0, morph_artifact 1, yes_no_start 0, repetition 3

make test
=== summary: 74 passed, 0 failed, 0 skipped ===
```

Interpretation: Q/A runtime format still wins, but the true remaining tail is
now visible: three morphology/glue failures in the current corpus. Next guard
should target malformed word shapes without reintroducing format wrappers or
over-tightening the sampler.

## 2026-07-10 - Codex pass: outer REPL prompt-format probe

### Context

The direct-user bridge was already moved to `Q:/A:`, but the outer REPL turn
context still used the older conversational frame:
`User: <line>\nArianna:` plus a `Recent trajectory:` wrapper on later turns.
Oleg pointed out that the older Arianna spoke better even while over-addressing
him, which suggests a real contract conflict: the old second-person frame
anchored dialogue, while the clean 0-vocative body removed that anchor.

### What changed

- Added `A2A_REPL_PROMPT_FORMAT` for the outer REPL prompt:
  `user_arianna` keeps the current default, `qa` feeds cells as flat
  `Q: <line>\nA:` turns.
- REPL now prints `replFmt=...` in the startup banner so run logs expose the
  exact outer prompt contract.
- `append_trajectory()` now preserves the matching trajectory contract for
  each mode instead of always appending `User:/Arianna:`.
- `tools/repl_temp_sweep.sh` now sweeps `A2A_TEMP_REPL_FORMATS` separately from
  the direct-user `A2A_TEMP_FORMATS` axis.

### Verification

```text
A2A_TEMP_BASES="0.45" A2A_TEMP_TOP_KS="40" A2A_TEMP_REPS="2.05" \
  A2A_TEMP_FORMATS="qa" A2A_TEMP_REPL_FORMATS="user_arianna qa" make repl-temp-sweep

userFmt=qa replFmt=user_arianna:
  I_U^kv -0.041, I_N^kv -0.047, route_score 0.651,
  answer_quality 5/30, morph_artifact 3, repetition 0

userFmt=qa replFmt=qa:
  I_U^kv -0.076, I_N^kv -0.484, route_score 0.511,
  answer_quality 3/30, morph_artifact 1, repetition 1

make test
=== summary: 78 passed, 0 failed, 0 skipped ===
```

Interpretation: the pure `Q:/A:` outer prompt is cleaner by the current answer
quality counters, but it damages neighbour field geometry and brings back
second-person anchoring (`You have been...`). The outer `User:/Arianna:` frame
is therefore not the single bug. It appears to preserve some conversational
field structure while also carrying residue from the old addressing contract.
The next fix should not blindly flip the global default; it should either test a
third neutral dialogue frame or compare against new SFT weights before changing
the REPL contract.

## 2026-07-10 - Sol audit: duplicated user-KV amplitude

### Finding

The direct answer prompt already contains the user question as `Q: ...\nA:`.
The bridge then encoded the same `Q: ...\nA:` a second time and injected that
cache through the neighbour-attention lane at a fixed weight of `0.30`. The
reported `I_U^kv` therefore measures the incremental cross-KV copy, not the
entire influence of the user question. Its negative average at the old default
was evidence that the duplicate lane was raising entropy, not helping the
answer.

The saved no-user-KV shadows were usually more coherent than the emitted
answers, but the summary only scored the emitted side. It now reports
`answer_quality_no_user_kv` as a separate line.

### Change

- Added `A2A_USER_KV_WEIGHT` and the matching `A2A_TEMP_USER_KVS` sweep axis.
- Changed the measured direct-user defaults from `userRep=2.05, userKV=0.30`
  to `userRep=1.30, userKV=0.05`.
- Kept the outer `User:/Arianna:` REPL frame and the direct `Q:/A:` answer
  frame; changing the outer frame was not the root fix.
- Made `repl_question_sweep.sh` parse model output under `LC_ALL=C`, so invalid
  UTF-8 from a damaged generation cannot abort the TSV run.

### Evidence

Fixed F16 body, prompt corpus, seed path, `temp_base=0.45`, `span=0.10`,
`top_k=40`, `rep=1.60`:

```text
userKV  quality_any  morph  repetition  I_U^kv
0.00    3/30         1      2           +0.000
0.05    1/30         0      1           +0.030
0.10    1/30         0      1           +0.009
0.20    6/30         1      4           -0.030
0.30    4/30         2      0           -0.042
```

At `userKV=0.05`, repetition penalties `1.30`, `1.60`, and `2.05` all scored
`quality_any 1/30`; `1.30` preserved slightly cleaner completions and had the
strongest positive `I_U^kv` (`+0.044`). A cold-reader REPL probe at `0.05`
answered without claiming a prior meeting, while `0.10` said `You have been
here before`.

```text
make test
=== summary: 82 passed, 0 failed, 0 skipped ===

make repl-eval
I_U^kv: avg +0.044
I_N^kv: avg -0.047
route_targets: c0:8 c1:18 c2:4
route_score: avg 0.651, min 0.463, max 0.783
answer_quality: any 1/30, short 0, question_like 0, label_artifact 0, notation_artifact 0, morph_artifact 0, yes_no_start 0, repetition 1
answer_quality_no_user_kv: any 3/30, short 0, question_like 0, label_artifact 0, notation_artifact 0, morph_artifact 1, yes_no_start 0, repetition 2
```

### Remaining weight problem

The runtime retune does not make the clean re-SFT a robust replacement for the
old body. Exact-contract `Q:/A:` questions are partly coherent, but statement
or fragment seeds still degrade sharply. Russian output is substantially
broken. The current 0-vocative gate also counts only literal Oleg mentions: it
does not catch false familiarity, and a prompt without `?` does not enter the
direct user qloop at all. These failures require a broader, frame-preserving
SFT/checkpoint comparison and stronger semantic recipient gates; another
prompt-label flip is not sufficient.
