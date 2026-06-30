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
