---
title: "custom splash masks for non-LLM domains"
type: concept
tags: [pallas, splash-attention, block-sparse, graphcast, weather, stub]
created: 2026-04-23
updated: 2026-04-23
---

Any structured adjacency pattern can become a banded / block-sparse splash-attention mask by subclassing `splash_attention_mask.Mask`. First-party non-LLM reference: GraphCast's `WeatherMeshMask` for graph-based weather forecasting. *Stub — expand when more sources are available.*

## Definition

`splash_attention`'s `_ComputableMask` base class defines a mask by a computable predicate over `(q_block_idx, kv_block_idx)` or `(q_pos, kv_pos)`. Any structured sparsity — sliding window (standard), chunked-causal (Gemma4), graph-adjacency (weather models), hierarchical (multi-resolution transformers) — can be expressed as a subclass. Splash attention then compiles the mask into its block-sparse schedule.

## Why it matters for TPU perf

Splash's machinery for sparse-block skipping is application-general. A weather forecasting model (GraphCast) attending over an icosahedral mesh gets the same block-sparse-load benefit as a chunked-causal LLM — no need to hand-roll a separate kernel. Applies to any domain where the attention mask is structured and computable.

## Mechanism

1. Subclass `splash_attention_mask._ComputableMask` (or a higher-level base like `FullMask`, `LazyMask`).
2. Implement the mask predicate as a pure function of block / position indices.
3. Pass the subclass instance to `splash_attention.make_splash_mha(mask=...)`.
4. Splash uses the mask to emit `MaskInfo` (block-sparse metadata) feeding into the kernel's `sparse_q_kv_load` path.

## When it applies / when it doesn't

- **Applies** to any structured adjacency that can be expressed as a predicate on block/position indices.
- **Does not apply** to data-dependent masks (computed at runtime from activations) — splash needs the mask at compile time to emit `MaskInfo`.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `WeatherMeshMask(splash_attention_mask.Mask)` | `google-deepmind/graphcast` (not ingested as its own codebase — wrapper) | Banded mask for icosahedral-mesh graph attention; used by GraphCast + GenCast + WeatherMesh |
| `ChunkedCausalMask` | [maxtext](../codebases/maxtext.md) | Gemma4 local-sliding chunks; subclasses `_ComputableMask` |
| `SlidingWindow` / `SegmentId` / truncated-key masks | [axlearn](../codebases/axlearn.md) | Training-time attention bias system |

## Connections

- [splash-attention](splash-attention.md)
- [flash-attention](flash-attention.md)
- [block-sparse-offset-masks](block-sparse-offset-masks.md)
- [attention-block-sizes](attention-block-sizes.md)

## Sources

- [Pallas kernel directory §4.4](../analyses/pallas-kernel-directory/04-research-labs.md#44-google-deepmindgraphcast).
- [maxtext codebase](../codebases/maxtext.md) `ChunkedCausalMask` reference.
