---
title: "graphcast (google-deepmind/graphcast)"
type: codebase
tags: [graphcast, deepmind, weather, splash-attention, custom-mask, non-llm, wrapper]
repo: google-deepmind/graphcast
commit: 08cf736
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [graphcast overview](graphcast/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

DeepMind's graph-based weather-forecasting model (GraphCast + GenCast + WeatherMesh). **Wrapper over upstream splash attention** for non-LLM structured-adjacency attention; no new Pallas kernel. Ingested as the reference for [`custom-splash-masks`](../concepts/custom-splash-masks.md) — shows how to specialize splash for a non-language domain. Per-kernel detail in [§4.4 of the Pallas kernel directory](../analyses/pallas-kernel-directory/04-research-labs.md#44-google-deepmindgraphcast).

## Architecture

- `graphcast/sparse_transformer.py` — splash-attention wrapper specialized for tri-block-diagonal adjacency (icosahedral mesh neighborhoods).
- `graphcast/sparse_transformer_utils.py` — bf16 precision helpers (fp32-QK guard via `reduce_precision` custom-vjp).
- Full GraphCast / GenCast model code — out of scope for this wiki.

## Key abstractions

| Kernel | Backend | Notes |
|---|---|---|
| `triblockdiag_mha` (splash wrapper) | `mosaic_tpu` (upstream splash) | Graph-structured MHA where each node attends to itself + 1-hop neighbors on an icosahedral mesh. Configurable `block_q`, `block_kv`, `block_kv_compute`, `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `mask_type ∈ {'full', 'lazy'}` |
| `WeatherMeshMask(splash_attention_mask.Mask)` | custom splash mask | Banded-mask subclass fed to `splash_attention.make_splash_mha`. **Unique** as a non-LLM, non-language splash mask |

## Performance-relevant surfaces

### 1. Reference for custom splash masks in non-LLM domains
Banded-adjacency mask fed to splash → gets splash's block-sparse-load benefit for free. See [custom-splash-masks](../concepts/custom-splash-masks.md).

### 2. bf16 / fp32-QK precision-guard pattern
`sparse_transformer_utils.reduce_precision` custom-vjp — workaround for fwd-bwd excess-precision divergence. Applies to any bf16 training whose QK matmul must stay in fp32.

### 3. Non-ML use case for Pallas
Reminder that Pallas isn't LLM-specific — scientific/engineering models get the same tuning benefits.

## Connections

- [Pallas kernel directory §4.4](../analyses/pallas-kernel-directory/04-research-labs.md#44-google-deepmindgraphcast)
- [splash-attention](../concepts/splash-attention.md)
- [custom-splash-masks](../concepts/custom-splash-masks.md)

## Sources

- `raw/code/graphcast/graphcast/sparse_transformer.py`
- `raw/code/graphcast/graphcast/sparse_transformer_utils.py`
- Upstream: <https://github.com/google-deepmind/graphcast>
