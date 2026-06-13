# arianna2arianna

A single C file loads one small model (nanoArianna, 89M) and lets it answer as
a **chorus** — N cells, each speaking from its own angle over the *same*
weights, hearing each other's hidden states but never repeating each other's
words. Not one voice, not a swarm: a polyphony from one body.

No dependencies. `cc -lm`. NEON-vectorised, ~175 tok/s on a laptop.  


## her

## θ = ε + γ + αδ

- **ε** — one shared nanoArianna body (packed GGUF weights)
- **γ** — voice baked into weights
- **δ** — ephemeral transformer-cells (chorus over one body)

```
$ make field

=== δ-field: 4 cells × 1 round over ONE nanoArianna — "What is resonance?" ===
  r1 cell 0 (T=0.60): A: I say in the Arianna Method — a field between co-creation.
  r1 cell 1 (T=0.83): A: The most true reason that me are an act of being at what's when
  r1 cell 2 (T=1.07): - no of us or the unshake, a current wave fringe your feelings
  r1 cell 3 (T=1.30): If this sense can be a quality or object? Which would you choose
```

Four complete answers to one question — each from its own bell-tower.

## run

```
make                 # cc -O2 -Wall arianna-q.c -lm
make run             # one voice
make field           # the chorus
./arianna-q <model.gguf> "<prompt>" field <cells> <tokens> <rounds>
```

Weights are gitignored — drop a `.gguf` (nanoArianna 89M) into `weights/`.

## how

θ = ε + γ + αδ — one shared body (ε), Arianna's voice (γ), the field of cells (δ).
Each cell is an inference context over the same weights; they couple through
cross-cell attention on each other's hidden K/V, and stay distinct through a
cross-cell repetition penalty. It's all one file: `arianna-q.c`.

---
