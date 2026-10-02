---
title: "recurrentgemma (google-deepmind/recurrentgemma)"
type: codebase
tags: [recurrentgemma, griffin, deepmind, pallas, mosaic-tpu, lru, linear-recurrence, ssm]
repo: google-deepmind/recurrentgemma
commit: 2efa84d
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [recurrentgemma overview](recurrentgemma/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Small but load-bearing: ships **the canonical public Mosaic-TPU Pallas LRU scan** (`h_t = a_t · h_{t-1} + x_t`), backing the Griffin / RecurrentGemma RG-LRU block. Single kernel, fwd + bwd, supports **real and complex accumulators** (rare), with a `multi_shard_correction` for sequence-parallel training. **Direct ancestor of axlearn's Mamba Pallas kernel** — [axlearn](axlearn.md) imports the LRU pattern from here. Per-kernel detail in [§4.2 of the Pallas kernel directory](../analyses/pallas-kernel-directory/04-research-labs.md#42-google-deepmindrecurrentgemma).

## Architecture

```
recurrentgemma/
  jax/
    pallas.py          ← the LRU Pallas TPU scan (fwd + bwd)
    scan.py            ← thin module exposing lru_pallas_scan = pallas.lru_pallas_scan
    layers.py          ← RecurrentBlock uses lru_pallas_scan
    griffin.py         ← Griffin model assembly
  ...
```

## Key abstractions

| Kernel | File | Backend | Stability | Notes |
|---|---|---|---|---|
| `linear_rnn_pallas_kernel` / `lru_pallas_scan` / `pallas_lru` | `recurrentgemma/jax/pallas.py` | `mosaic_tpu` | stable | LRU scan fwd + bwd. `complex_lib.Complex` accumulators supported. `multi_shard_correction` runs local scan per shard, broadcasts final `h` + `a_prod`, recomposes |

Configurable knobs:
- `pallas_max_seq_block_size` (default 256)
- `pallas_min_seq_block_size` (16)
- `pallas_pad_seq_to_min_block_size`
- `pallas_pad_last_dim_to_128`

## Entry points

```bash
cd raw/code/recurrentgemma

pip install -e .

# import path
python -c "from recurrentgemma.jax.scan import lru_pallas_scan"
python -c "from recurrentgemma.jax.pallas import pallas_lru"
```

## Dependencies

- `jax` + `jaxlib` + `libtpu` (TPU runtime).
- `flax` (model layers).

## Performance-relevant surfaces

### 1. Only public TPU Pallas LRU scan from first principles
No equivalent in `jax.experimental.pallas.ops`, tokamax, or elsewhere (pre-axlearn). This is the canonical implementation for `h_t = a_t · h_{t-1} + x_t`-style recurrences. Any Griffin-family or linear-RNN hypothesis starts here.

### 2. `multi_shard_correction` for sequence-parallel training
Run local scan per shard, broadcast final `h` and `a_prod` across `sequence_shard_index`, recompose. **Concept-page candidate** — applies to any linear-recurrence Pallas kernel. Template for this wiki's sequence-parallel work.

### 3. Complex-number accumulator support
Most Pallas TPU kernels are real-only. This one supports `complex_lib.Complex` accumulators — relevant if a hypothesis touches complex-valued scans (FFT-like recurrences, complex-linear attention).

### 4. Ancestor of axlearn's Mamba kernel
[axlearn's `ssm_kernels/mamba_kernels.py`](axlearn.md) is adapted from this. Lineage-informative when reading axlearn's SSM code — the LRU is the simpler parent of Mamba's selective scan.

## Connections

- **[Pallas kernel directory §4.2](../analyses/pallas-kernel-directory/04-research-labs.md#42-google-deepmindrecurrentgemma)** — per-kernel detail.
- [axlearn](axlearn.md) — adapts this LRU scan for Mamba1.
- [marin](marin.md) — has XLA-only SSD / Mamba3 paths; if a Pallas TPU SSD path is ever added, this is one of two existing Pallas recurrence references (the other is axlearn).

## See also

- [scan-over-layers](../concepts/scan-over-layers.md) — related but distinct (scan-over-*layers* is compile-time; LRU is sequence-axis recurrence).
- [pallas-kernel](../concepts/pallas-kernel.md)
- [sequence-parallelism](../concepts/sequence-parallelism.md)

## Sources

- `raw/code/recurrentgemma/recurrentgemma/jax/pallas.py`
- `raw/code/recurrentgemma/recurrentgemma/jax/scan.py`
- Upstream: <https://github.com/google-deepmind/recurrentgemma>
