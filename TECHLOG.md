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
./arianna2arianna <model.gguf> "<prompt>" field [cells] [frag] [rounds] [alpha] [leap] [xcell] [chorus] [xrep] [life] [kvshuf]
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
