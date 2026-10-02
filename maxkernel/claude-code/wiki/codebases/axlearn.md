---
title: "axlearn (apple/axlearn)"
type: codebase
tags: [axlearn, apple, pallas, mosaic-tpu, triton, ssm, mamba, mamba2, rattention, splash, flash, paged-attention, megablock]
repo: apple/axlearn
commit: b479714
paper: "arXiv:2507.05411"
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [axlearn overview](axlearn/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Apple's public training framework and the **single largest source of non-upstream Pallas kernels** surveyed in this wiki. Ships side-by-side TPU (Mosaic) + GPU (Triton) Pallas implementations for flash attention (splash extensions with dropout + logit sink), paged attention (block-sparse offset masks), decoding, grouped matmul — plus **exclusive Pallas kernels for Mamba1, Mamba2 (SSD), and RAttention linear attention** that exist nowhere else publicly. Per-kernel detail in [§4.1 of the Pallas kernel directory](../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn).

## Overview

Ingestion is scoped to the Pallas-relevant subtrees (per the directory analysis's Wave 4 scoping recommendation) — the full axlearn framework includes extensive model code, data pipelines, and trainer machinery that is out of scope for this wiki.

Four directories to watch:
- `axlearn/common/flash_attention/` — attention dispatch + TPU splash / paged / decoding kernels + GPU (Triton) flash / decoding / paged kernels + AWS Neuron entry.
- `axlearn/common/ssm_kernels/` — Mamba1 (`mamba_kernels.py`) and Mamba2/SSD (`ssd_kernels.py`) — **unique in public Pallas ecosystem**.
- `axlearn/common/rattention/kernels/` — linear-attention kernels (`linear_attention_kernels.py`) with residual / timestep variants — **unique**.
- `axlearn/common/megablock/` — GPU-Triton `gmm_gpu.py` counterpart to upstream TPU `megablox.gmm`.

## Architecture

Three boundaries matter:

1. **TPU vs GPU vs Neuron three-way dispatch** in `flash_attention/layer.py` — unusual to see a single framework author kernels for all three.
2. **Splash extensions** (`tpu_splash_attention.py`) — adds dropout-inside-kernel (generated from prng_key + block indices), logit-sink online softmax (`m` init with sink; normalizer adds `exp(sink - m_final)`), and carries a known `key<pl>` Pallas lowering bug workaround (prefetch prng_key). Vendored from upstream splash at commit `b3c49b0`.
3. **Two-level chunk/subchunk recomputation** in `ssd_kernels.py` — Mamba2 paper §7.2 (arXiv:2405.21060). Store coarse chunk carries, recompute subchunk states. Template-pattern for any Pallas linear recurrence.

## Key abstractions

| Kernel | File | Backend | Stability | Why it matters |
|---|---|---|---|---|
| `tpu_splash_attention` (MHA fwd/bwd w/ dropout + logit sink) | `axlearn/common/flash_attention/tpu_splash_attention.py` | `mosaic_tpu` | stable | Splash extensions: dropout-in-kernel, logit-sink softmax |
| `tpu_decoding` | `axlearn/common/flash_attention/tpu_decoding.py` | `mosaic_tpu` | stable | Perf claim: up to `padded_kv_seq_len / kv_seq_len` speedup on sparse-mask decode |
| `tpu_paged_attention(_kernel)` | `axlearn/common/flash_attention/tpu_paged_attention*.py` | `mosaic_tpu` | stable | Block-sparse offset masks for sliding window (precompute `(n_kv_blocks, n_kv_blocks)` unmasked-offset table) |
| `paged_kv_cache_tpu_kernel.scatter_update_pages` | `axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py` | `mosaic_tpu` | experimental | Stopgap (10× slower than K/V projection); TODO to optimize. Cannot statically prove mult-of-8 alignment, no non-32-bit mask relayout |
| `mamba_kernels.compute_mamba_scan` | `axlearn/common/ssm_kernels/mamba_kernels.py` | `mosaic_tpu` | research | **UNIQUE.** Mamba1 selective scan fwd+VJP; two-level tiling via `MambaArgumentBlockSpecs`; adapted from recurrentgemma LRU scan |
| `ssd_kernels._ssd` | `axlearn/common/ssm_kernels/ssd_kernels.py` | `mosaic_tpu` | research | **UNIQUE.** Mamba2 / SSD chunked scan fwd+VJP; two-level chunk/subchunk recomputation |
| `linear_attention_kernels._linear_attention` | `axlearn/common/rattention/kernels/linear_attention_kernels.py` | `mosaic_tpu` | research | **UNIQUE.** RAttention-paper reference impl; exposes residual/timestep variants. Claim: "you can train RAttention with a 512-token window just as efficiently as Sliding Window Attention with a 4K window" |
| `megablock.gmm_gpu.gmm` | `axlearn/common/megablock/gmm_gpu.py` | `triton` (GPU) | stable | Public GPU-Triton counterpart of upstream TPU `megablox.gmm`. Default `(32, 32, 64)` tiling, `num_stages=4`, `num_warps=2` |
| `gpu_attention` (FA2 fwd/bwd) | `axlearn/common/flash_attention/gpu_attention.py` | `triton` | stable | Adds bias/dropout/FlexAttention-style mask_fn over upstream |
| `gpu_decoding` (Flash-Decoding) | `axlearn/common/flash_attention/gpu_decoding.py` | `triton` | stable | "FlashDecoding is faster than XLA across the board by some margin (5%~20%)"; removes double-vmap-transpose pattern |

## Entry points

```bash
# clone / enter submodule
cd raw/code/axlearn

# tests (much broader than Pallas — scope to kernel tests)
pytest axlearn/common/flash_attention/
pytest axlearn/common/ssm_kernels/
pytest axlearn/common/rattention/kernels/

# Framework import
python -c "from axlearn.common.ssm_kernels.mamba_kernels import compute_mamba_scan"
python -c "from axlearn.common.rattention.kernels.linear_attention_kernels import _linear_attention"
```

## Dependencies

- `jax` + `jaxlib` (Mosaic-TPU backend).
- `jax_triton` / `triton` (GPU kernels).
- `aws-neuronx-nki` (Neuron entry — not Pallas; only `flash_attention/neuron_attention.py`).
- Heavy framework deps (flax, tensorflow-datasets, etc.) — not needed just to import Pallas kernels.

## Performance-relevant surfaces

### 1. Only public TPU Pallas source for Mamba1 / Mamba2 / linear attention
`ssm_kernels/mamba_kernels.py` + `ssm_kernels/ssd_kernels.py` + `rattention/kernels/linear_attention_kernels.py`. No upstream JAX / tokamax equivalent. If any future hypothesis targets SSM / Mamba on TPU, this is where to start.

### 2. Two-level chunk/subchunk recomputation (SSD pattern)
`ssd_kernels.py`: store coarse-grained chunk-level hidden states, recompute subchunk states. Generalizable to any Pallas linear recurrence. **Concept-page candidate** per the directory analysis (not yet filed).

### 3. Block-sparse offset masks for paged attention
`tpu_paged_attention_kernel.py` precomputes `(n_kv_blocks, n_kv_blocks)` offsets into unmasked blocks → turns sliding-window into dense-load-over-sparse-offset. Directly applicable to any paged-attention hypothesis that has sparse masks.

### 4. Dropout-inside-splash-kernel from prng_key + block indices
`tpu_splash_attention.py`. Avoids host-materialized dropout mask. Paired with `key<pl>` Pallas lowering bug workaround (prefetch prng_key). Potential reuse in gemma4's splash path.

### 5. Online softmax with logit sink
Same file. `m` init with sink; normalizer adds `exp(sink - m_final)`. Applies to any flash-attention variant. **Concept-page candidate.**

### 6. `scatter_update_pages` honest-assessment doc-string
`paged_kv_cache_tpu_kernel.py`: "This kernel is quite slow: it has 10x latency compared to k/v projection at the same batch size." — the kind of honest "works but not optimal" disclosure this wiki values. Calls out Pallas/Mosaic alignment + relayout limitations.

### 7. Adapted-from-upstream attribution chain
Splash here came from upstream at commit `b3c49b0`; Mamba adapted from recurrentgemma LRU scan; megablox_gpu metadata-computation ported from upstream TPU `megablox.common.make_group_metadata`. Explicit chain preserved; useful for upstream-drift tracking.

### 8. Three-backend dispatch template
`flash_attention/layer.py` orchestrates TPU (splash / flash / paged / decoding) / GPU (Triton FA2 / decoding / paged) / Neuron (NKI) with unified bias system. Template if this wiki ever needs its own multi-backend kernel dispatch.

## Connections

- **[Pallas kernel directory §4.1](../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn)** — authoritative per-kernel detail.
- [recurrentgemma](recurrentgemma.md) — ancestor of axlearn's Mamba kernels (LRU scan → Mamba selective scan).
- [jax](jax.md) — upstream splash at `jax.experimental.pallas.ops.tpu.splash_attention` (the "adapted from" source).
- [tokamax](tokamax.md) — has no Mamba / SSM Pallas; axlearn fills that gap.
- Gemma4 program hypothesis anchors:
  - Splash extensions (dropout + logit sink) — candidate for exp ≥34 if Gemma adopts them.
  - Block-sparse offset masks — directly applicable if sliding-window attention enters the training loop.

## See also

- [splash-attention](../concepts/splash-attention.md)
- [flash-attention](../concepts/flash-attention.md)
- [pallas-kernel](../concepts/pallas-kernel.md)
- [mosaic-kernel](../concepts/mosaic-kernel.md)
- [ragged-dot](../concepts/ragged-dot.md) — `megablox.gmm` family.
- [kv-cache](../concepts/kv-cache.md)
- [attention-block-sizes](../concepts/attention-block-sizes.md)

## Sources

- `raw/code/axlearn/axlearn/common/flash_attention/*.py`
- `raw/code/axlearn/axlearn/common/ssm_kernels/*.py`
- `raw/code/axlearn/axlearn/common/rattention/kernels/*.py`
- `raw/code/axlearn/axlearn/common/megablock/*.py`
- `raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py`
- Paper: <https://arxiv.org/pdf/2507.05411>
- Upstream: <https://github.com/apple/axlearn>
