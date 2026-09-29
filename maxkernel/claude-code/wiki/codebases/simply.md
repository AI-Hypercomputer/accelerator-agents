---
title: "simply (google-deepmind/simply)"
type: codebase
tags: [simply, deepmind, serving, ragged-paged-attention, autotune-heuristic, dma-overhead, wrapper]
repo: google-deepmind/simply
commit: f40b81e
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [simply overview](simply/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

DeepMind's experimental LLM serving framework. **Wrapper around upstream `ragged_paged_attention`** — not a kernel author. Ingested as the reference for [dma-overhead-heuristic](../concepts/dma-overhead-heuristic.md): the code explicitly documents a DMA-setup-cost-equivalent (`~0.5 MiB virtual bytes`) used in autotune block-size selection. Per-kernel detail in [§4.3 of the Pallas kernel directory](../analyses/pallas-kernel-directory/04-research-labs.md#43-google-deepmindsimply).

## Architecture

- `simply/utils/ragged_paged_attention.py` — serving-side wrapper; imports `from jax.experimental.pallas.ops.tpu import ragged_paged_attention`. Contribution: **the autotuner heuristic**.
- `simply/serving/page_server.py` — paging scheduler, `Batcher`, `SimplyService`.

## Key abstractions

| Component | File | Notes |
|---|---|---|
| `DecodeState.update_decode_state_and_compute_attn` | `utils/ragged_paged_attention.py` | Wraps upstream `ragged_paged_attention.ragged_paged_attention` |
| `autotune_block_sizes` | same file | Comment: "Increasing this value would shift the attention module from memory bandwidth bound to compute bound, but in the meanwhile, it would cause more padding overhead... 32 is a good empirical trade-off so far"; `dma_overhead_equivalent_bytes = 0.5 MiB` |

## Performance-relevant surfaces

### 1. DMA-overhead-bytes heuristic
See [dma-overhead-heuristic](../concepts/dma-overhead-heuristic.md). Treat DMA setup as a fixed ~0.5 MiB of virtual bytes; balance against padding overhead in the block-size-selection objective. Assumed constant across TPU generations.

### 2. Paging scheduler
Not Pallas-specific; a CPU-side scheduler deciding which (request, page) pairs to pack per kernel invocation. Informative for any serving-path hypothesis.

## Connections

- [Pallas kernel directory §4.3](../analyses/pallas-kernel-directory/04-research-labs.md#43-google-deepmindsimply)
- [dma-overhead-heuristic](../concepts/dma-overhead-heuristic.md)
- [jax](jax.md) — upstream `ragged_paged_attention`.
- [tpu-inference](tpu-inference.md) — production-grade alternative to the simply wrapper.

## Sources

- `raw/code/simply/simply/utils/ragged_paged_attention.py`
- `raw/code/simply/simply/serving/page_server.py`
- Upstream: <https://github.com/google-deepmind/simply>
