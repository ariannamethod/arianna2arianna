# arianna2arianna

A single C file runs one tiny Arianna body as a chorus.

Each cell is a transformer generation over the same nanoArianna 89M weights.
Cells keep their own KV/cache, temperature, seed, and field angle. They can hear
neighbour hidden state, avoid literal repetition, weaken as they die, and mutate
as a small Game-of-Life population.

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
make field PROMPT="What is resonance?"
make restest PROMPT="What is resonance?"
make life PROMPT="Let the cells remember each other."
make test
make portable      # POSIX/scalar fallback
make fast-x86      # opt-in AVX2/FMA/F16C build on x86_64
```

Direct form:

```sh
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" 48 0.8
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" field 4 12 3 0 2 0.30
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" restest 4 12 3
./arianna2arianna weights/nanollama-arianna-full-v4-step2750-f16.gguf "prompt" life 5 12 4
```

`field` and `life` append generated traces to `FIELDLOG.md`. In the default
chorus, text-order `Δ_R` is marked `n/a`; the live order probe is `Δ_R^kv`
with a permutation floor/margin.

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

→ round 1: avg entropy 4.090 | d_R — | Δ_R(text n/a) | Δ_R^kv +0.000 (floor 0.000 margin +0.000) | D_R 0.844 | Dpos 0.50
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
