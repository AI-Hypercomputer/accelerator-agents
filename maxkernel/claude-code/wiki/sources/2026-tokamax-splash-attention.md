---
title: "tokamax docs — splash attention (TPU)"
type: source
tags: [docs, kernels, pallas, attention, splash-attention, flash-attention, tpu, stub]
created: 2026-04-22
updated: 2026-04-22
---

The splash-attention doc in the tokamax repo is a placeholder at the time of ingest — the full kernel deep-dive is "coming soon". The authoritative description of the kernel lives in the code itself (`tokamax/_src/ops/experimental/tpu/splash_attention/`) and in the [tokamax codebase page](../codebases/tokamax.md). This source page captures what the doc *does* say, records that splash attention is currently tokamax's headline TPU attention kernel, and points downstream `splash-attention` concept work at the code rather than the doc.

## Overview

Splash attention is tokamax's experimental **TPU-native Pallas attention kernel**. It is reached by calling `tokamax.dot_product_attention(...)` with the default `implementation=None` (or explicitly `implementation="mosaic"` on TPU), which resolves via the Pallas-Mosaic-TPU backend (`_src/ops/attention/pallas_mosaic_tpu.py`) into the splash-attention kernel under `_src/ops/experimental/tpu/splash_attention/`. The tokamax implementation mirrors the upstream kernel in `jax.experimental.pallas.ops.tpu.splash_attention`, so much of what is true of that kernel carries over.

The doc file itself (`docs/splash_attention.md`) currently contains only an "experimental TPU op within Tokamax" one-liner and a promise of more info. For this wiki, the load-bearing facts about splash attention are in the source code and in the codebase page; the concept page `splash-attention.md` should cite those, not this doc, as its primary evidence.

## Key claims

Claims explicitly in the doc:

- Splash attention is an **experimental** op in tokamax.
- Splash attention is **TPU-only** — not available on GPU.
- Further capability and usage documentation is **pending** ("coming soon").

Claims implied by the doc's existence (a dedicated entry in `docs/` alongside `basic_usage` and `autotuning`):

- Splash attention is the **canonical TPU attention kernel** in tokamax; there is no competing TPU attention backend in the repo.
- It is expected to be the default choice when a user calls `tokamax.dot_product_attention` on TPU v5+.

Claims that **come from the code, not this doc** (recorded here because the concept page will need them; see [tokamax codebase page](../codebases/tokamax.md) for authoritative file/line references):

- **TPU generation gate**: `supported_on` in `pallas_mosaic_tpu.py` requires `pltpu.get_tpu_info().generation >= 5`. TPU v4 and earlier are unsupported.
- **Tile-shape constraints**: `block_q`, `block_kv`, `block_kv_compute` must all be multiples of 128 (`NUM_LANES`). `block_kv` must be a multiple of `block_kv_compute`.
- **Heuristic default** (when not autotuned): `block_q = block_kv = block_kv_compute = 128`, all QKV layouts `HEAD_DIM_MINOR`, experimental scheduler on, base-2 exp on.
- **Autotune search space**: tiles chosen from `{128, 256, 512, 1024, 2048, 4096}` for each of the three block sizes, pruned by `seq_len ≥ 1024 ⇒ block ≥ 1024` and `bkv_c ≤ 1024`. Code has a TODO noting that `8192` is excluded pending a fix.
- **Layout knobs**: `q_layout`, `k_layout`, `v_layout` each take `splash.QKVLayout.{HEAD_DIM_MINOR, SEQ_MINOR}`.
- **Scheduler knob**: `use_experimental_scheduler: bool`.
- **Softmax numerics knob**: `use_base2_exp: bool = True` — base-2 softmax to avoid `log(2)` multiplies.
- **Backward-pass tiling knobs are not exposed**: the underlying `SplashConfig` (in `splash_attention_kernel.py`) carries `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq` for the VJP, but they are hard-wired to 128 in `SplashConfig.get_default()` and **not part of the Tokamax-TPU autotuner's search space**. This is a concrete tuning gap.
- **Sparse-mask aware**: supports MHA/MQA/GQA, soft-cap, local-window / causal / arbitrary-mask via `base.Mask`, and separate fwd/bwd tiling.
- **Ring-attention sibling exists but is not wired in**: `_src/ops/experimental/tpu/splash_attention/ring_attention_kernel.py` contains a sequence-parallel TPU ring-attention kernel reachable only by direct import.
- **Microbench PDF**: `tokamax/_src/ops/experimental/tpu/splash_attention/microbenchmarks.pdf` contains per-shape figures that will eventually anchor "known results" on the concept page.

## Key data points

### Config knobs on the TPU attention backend (`pallas_mosaic_tpu.py`)

| Knob | Type / default | Notes |
|---|---|---|
| `block_q` | int, default 128 | multiple of 128 |
| `block_kv` | int, default 128 | multiple of 128; multiple of `block_kv_compute` |
| `block_kv_compute` | int, default 128 | multiple of 128 |
| `q_layout` | `QKVLayout`, default `HEAD_DIM_MINOR` | alt `SEQ_MINOR` |
| `k_layout` | `QKVLayout`, default `HEAD_DIM_MINOR` | alt `SEQ_MINOR` |
| `v_layout` | `QKVLayout`, default `HEAD_DIM_MINOR` | alt `SEQ_MINOR` |
| `use_experimental_scheduler` | bool | autotuned |
| `use_base2_exp` | bool, default `True` | base-2 softmax |

### Autotune search space (as implemented in `pallas_mosaic_tpu.py`)

- `block_q ∈ {128, 256, 512, 1024, 2048, 4096}`
- `block_kv ∈ {128, 256, 512, 1024, 2048, 4096}`
- `block_kv_compute ∈ {128, 256, 512, 1024, 2048, 4096}` with `block_kv_compute ≤ 1024`
- Pruning rule: `seq_len ≥ 1024` implies block ≥ 1024
- `{HEAD_DIM_MINOR, SEQ_MINOR}` × 3 tensors
- `use_experimental_scheduler ∈ {True, False}`
- `8192` is **deliberately excluded** pending a referenced autotuning bug fix (code TODO).

### Backward-pass block knobs (NOT autotuned, hard-wired to 128)

| Knob | Purpose |
|---|---|
| `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute` | dKV tiling |
| `block_q_dq`, `block_kv_dq` | dQ tiling |

Source: `splash_attention_kernel.py`, `SplashConfig.get_default()`.

### Hardware support

| TPU generation | Supported? | Source |
|---|---|---|
| v4 and earlier | No | `supported_on` gate |
| v5e | Yes (v5+) | `supported_on` gate |
| v5p | Yes | `supported_on` gate |
| v6e / Trillium | Yes | `supported_on` gate |
| v7 / future | Yes (v5+) | `supported_on` gate |

### Features supported vs. not

| Feature | Status |
|---|---|
| MHA / MQA / GQA | Supported |
| Causal mask | Supported |
| Local window | Supported |
| Arbitrary explicit mask | Supported (via `base.Mask`) |
| Soft-cap | Supported |
| Paged KV (inference) | Supported (per codebase page) |
| Quantized Q/K/V (e.g., int8 via `qwix`) | Supported |
| Backward-pass tile autotune | **Not supported** — knobs not exposed |
| Ring / seq-parallel attention | **Not via public API** — separate `ring_attention_kernel.py` |

## Techniques referenced

- **Splash Attention** — sparse-mask-aware FlashAttention-family kernel for TPU. Originally authored upstream in `jax.experimental.pallas.ops.tpu.splash_attention`; tokamax's copy tracks that kernel.
- **FlashAttention** (Dao 2022) — the tiled, SRAM-resident algorithmic ancestor. Splash attention inherits the online-softmax structure and adds mask-sparsity, soft-cap, and TPU-specific layout knobs.
- **Base-2 softmax** — rewrite `exp(x) = 2^(x · log2(e))`, trading one multiply for the ability to use `pow2` primitives. Controlled via `use_base2_exp`.
- **QKV layout selection** — `HEAD_DIM_MINOR` vs. `SEQ_MINOR` chooses which axis is contiguous in VMEM, which trades between load coalescing and head-wise parallelism.
- **Experimental scheduler** — a tokamax/Pallas scheduler flag that reorders DMAs relative to compute; autotuned.
- **Ring attention** (not via this doc's API) — sequence-parallel attention where KV blocks rotate around a ring of devices.

## Gaps & caveats

- **The doc is a stub.** One sentence plus a "coming soon" promise. Do not cite *this doc* for any mechanism claim — cite the code files. This source page reflects the doc's current contents; the concept page should be anchored on `pallas_mosaic_tpu.py`, `splash_attention_kernel.py`, and the microbench PDF.
- **No usage recipe in the doc.** Users have to read `basic_usage.md` and extrapolate (the examples there are GPU-only).
- **No performance numbers in the doc.** The microbenchmarks PDF in the repo is the current source of TPU perf figures; it is not transcribed here.
- **No TPU-generation-specific guidance.** The doc does not say "on v5e prefer block_q=N" or similar. The heuristic default is 128 across the board, but that is the *uniform* default — the autotuner exists precisely because it isn't universally optimal.
- **Backward-pass tuning gap is not acknowledged.** The doc does not mention that the VJP block sizes are fixed at 128. This is a real optimization surface for TPU training workloads.
- **Ring attention is not mentioned.** It is in the repo but sibling to splash attention, and not reachable via the public `tokamax.dot_product_attention` entry point.
- **Quantized KV and paged KV are not mentioned in the doc** — they are real features of the underlying kernel, documented only in the code.
- **Interaction with autotune cache** (which configs are pre-populated in `tokamax/data/`) is not discussed. A user who relies on the cache gets different behavior from a user who calls `tokamax.autotune(...)` explicitly.

## Connections

Concept slugs this source informs:

- `splash-attention` — this doc anchors the concept page; the concept page will need to draw on the code and the microbench PDF, not just the doc.
- `flash-attention` — splash attention is a TPU-targeted descendant of flash attention.
- `attention-block-sizes` — the Q/KV/KV-compute block-size autotune space is the key tuning surface.
- `attention-qkv-layout` — `HEAD_DIM_MINOR` vs. `SEQ_MINOR`.
- `base2-softmax` — `use_base2_exp` toggle.
- `ring-attention` — related TPU sequence-parallel kernel that lives next to splash attention but is not in the public API.
- `vmem-spill` — backward-pass tiling is currently fixed at 128; a separate concern and a known tuning gap.
- `tpu-generation-gate` — `pltpu.get_tpu_info().generation >= 5`.

### Open questions for the splash-attention concept page

When expanding the concept page beyond this stub-level doc, capture:

1. **Per-shape speedup over `xla_chunked`** from the microbench PDF — concrete numbers.
2. **Per-TPU-generation tile recommendation** — what does the autotuner pick on v5e vs v5p vs v6e for typical LLM shapes?
3. **Cost of exposing backward-pass tiles to the autotuner** — how much is on the table by tuning `block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`?
4. **Behavior on long-context** — the `seq_len ≥ 1024 ⇒ block ≥ 1024` pruning rule interacts with how many blocks are launched; at what seq_len does the kernel start to lose to chunked-XLA?
5. **`use_experimental_scheduler` regression surface** — the heuristic default turns it on; under what conditions does turning it off win?
6. **Interaction with paged KV / quantized KV** for inference shapes.

## See also

- [tokamax](../codebases/tokamax.md) — points at `pallas_mosaic_tpu.py` (the dispatcher) and `splash_attention_kernel.py` (the kernel).
- [tokamax basic usage](2026-tokamax-basic-usage.md) — how `implementation=` reaches this backend on TPU.
- [tokamax supported ops](2026-tokamax-supported-ops.md) — matrix-form; note that the supported-ops doc lists `dot_product_attention` as GPU-only, which contradicts the existence of this TPU backend.
- [tokamax autotuning](2026-tokamax-autotuning.md) — how the block/layout/scheduler knobs above get swept.
- [tokamax benchmarking](2026-tokamax-benchmarking.md) — how to measure splash attention vs. the XLA fallback.

## Sources

- [`raw/code/tokamax/docs/splash_attention.md`](../../raw/code/tokamax/docs/splash_attention.md)
