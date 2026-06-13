# arianna2arianna

Single-file C heart: GGUF + BPE + nanoArianna forward + δ-field chorus.

```sh
make              # build (SIMD auto per arch)
make run          # single voice, packed f16
make field        # δ-field demo
make sweep        # batch experiment CSV
make test         # f16 + q8 smoke
make portable     # scalar-only fallback
```

## θ = ε + γ + αδ

- **ε** — one shared nanoArianna body (packed GGUF weights)
- **γ** — voice baked into weights
- **δ** — ephemeral transformer-cells (chorus over one body)

## CLI

```sh
./arianna2arianna --16 "What is resonance?" 48 0.8
./arianna2arianna --16 "prompt" field [cells] [frag] [rounds] [alpha] [leap] [kv_beta]
./arianna2arianna --16 --json "prompt" field 4 12 3 0 2 0.25
./arianna2arianna --16 --quiet "prompt" sweep 10 4 12 3 0 0.25
```

### Flags

| Flag | Meaning |
|------|---------|
| `--16` / `--q8` | HF packed weights (f16 / Q8_0) |
| `--theta-lo` | Converge threshold (default 0.25) |
| `--theta-hi` | Leap trigger (default 0.50) |
| `--kv-beta` | KV field2field blend at dissenter peak (0=off) |
| `--dynamic-cells` | Bloom/collapse cell count from Dmean |
| `--quiet` | Suppress human field output |
| `--json` | One JSON line per round |

### Leap modes

| Mode | Behavior |
|------|----------|
| 0 | Off — full chorus context |
| 1 | Prompt + dissenter fragment only |
| 2 | Full chorus, dissenter **last** (recency) |
| 3 | Null test — consensus last |

### Metrics

- **d_R** — histogram distance between rounds (settling → floor)
- **Δ_R** — `ent_shuffled − ent_coherent` (order exploitation)
- **D_R** — embedding centroid disagreement between cells
- **Dpos / Dpeak** — per-step committed-token dissonance

### Sweep (batch harness)

Runs `n_seeds × leap{0,1,2,3}` quietly, prints CSV:

```
seed,leap,kv_beta,n_cells,floor,dR,deltaR,D_R,Dmean,Dpeak,flip_rate
```

Pipe to file and compare leap modes without reading FIELDLOG by hand:

```sh
make sweep > results.csv
```

## Packed matvec

Weights stay in GGUF layout (`wt_matvec` inline dequant). Section marked `PACKED MATVEC — DO NOT REVERT` in `arianna2arianna.c`.