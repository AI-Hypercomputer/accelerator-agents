---
title: "axlearn — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, axlearn]
created: 2026-07-04
updated: 2026-07-04
---

AXLearn ships one of the most diverse *non-upstream* Pallas kernel collections in the open: a
dropout-and-logit-sink extension of Mosaic Splash attention, a family of Triton GPU
flash/decoding/paged kernels, a TPU paged-attention kernel with a **block-sparse offset mask** for
sliding-window long context, a fused streaming log-probs kernel, a GPU grouped-GEMM for MoE, and —
most distinctively — a set of **SSM / linear-attention** kernels (RAttention linear attention, plus
Mamba1 and Mamba2/SSD selective scans) for which no upstream JAX/tokamax Pallas equivalent exists.
Backend mix is `mosaic_tpu` (TPU) + `triton` (GPU). Roughly **11 locally-defined Pallas kernels**
are live in this checkout; **2 more (the Mamba/Mamba2 SSD kernels) were present until an ancestor
commit and have since been removed** — see [Notes](#notes) for the divergence.

> [!warning] Source diverged from the [2026-04-23 catalog](../../analyses/pallas-kernel-directory/04-research-labs.md) on 2026-07-04
> The catalog lists `axlearn/common/ssm_kernels/mamba_kernels.py` and `ssd_kernels.py` as present.
> In this checkout (submodule at `raw/code/axlearn`, HEAD `c8624c8`) the entire `ssm_kernels/`
> directory and `ssm.py` were **deleted** at ancestor commit `18b88dc4` ("Remove PyTorch and
> transformers from default test deps") — the Mamba parity tests depended on torch reference
> implementations. Those two kernels are documented below from git history (`18b88dc4~1`,
> `b18cb42`) and linked to GitHub rather than to a local file that no longer exists.

## Kernels

### Attention — flash / splash (TPU + GPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `tpu_splash_attention` (`flash_attention_kernel` + `_splash_attention_forward` / `_splash_attention_bwd_dq` / `_dkv`) | mosaic_tpu | Sparse/masked MHA fwd+bwd (block-sparse via upstream `MaskInfo`) with two AxLearn-only extensions: **in-kernel dropout** (Bernoulli mask generated inside the kernel from `prng_key` + block indices, then `s = where(mask,0,s)/(1-rate)`) and **logit-sink online softmax** (running max init'd to the per-head sink; denominator gains `exp(sink - m)` at normalize) | [tpu_splash_attention.py#L109](../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L109) | stable | novel-extension of upstream splash |
| `tpu_attention` (flash wrappers) | mosaic_tpu | Wraps upstream `jax.experimental.pallas.ops.tpu.flash_attention` fwd/bwd with AXLearn's bias system (`MaskFnAttentionBias`, `SlidingWindow`, `SegmentId`); the TPU path chosen when splash is not | [tpu_attention.py](../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention.py) | stable | vendored-wrapper (jax) |
| `gpu_attention.flash_attention` (`_mha_forward_kernel` + `_mha_backward_kernel_dkdv` / `_dq`) | triton | FlashAttention-2 fwd/bwd (arXiv:2205.14135 / 2307.08691) with AXLearn additions: 2D/4D logit bias, dropout, arbitrary `mask_fn` (FlexAttention-style), and `kv_seq_len ≠ q_seq_len`. Seq length must be a multiple of 128; tested A100/H100 | [gpu_attention.py#L114](../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L114) | stable | novel-extension of upstream FA2 |

### Attention — paged / decoding

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `tpu_paged_attention_kernel` (`_paged_flash_attention_kernel`, `_paged_flash_attention_sparse_kernel`, `prepare_block_sparse_map`) | mosaic_tpu | Long-context paged-KV MHA. Precomputes a **block-sparse offset map** of shape `(n_kv_blocks, n_kv_blocks)` and loads only the KV blocks the offsets point at — turning sliding-window / arbitrary `mask_fn` into dense-load-over-sparse-offset. Supports arbitrary logit bias | [tpu_paged_attention_kernel.py#L430](../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L430) (sparse: [#L211](../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L211), map: [#L90](../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L90)) | stable | novel-extension (port of jax paged-attn + block-sparse mask) |
| `tpu_decoding` | mosaic_tpu | Single-step decode; sequential over KV (no split-KV — TPU blocks run sequentially). Exploits padded-KV + `mask_fn` sparsity: speedup ≈ `padded_kv_len / kv_len` or masked/total block ratio; else within ~10% of XLA | [tpu_decoding.py](../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py) | stable | novel |
| `gpu_decoding` (FlashDecoding) | triton | Decode MHA parallelized over the KV-length dim for small-batch/long-context; supports sliding-window `mask_fn` via runtime thread-block voting. 5–20% over XLA no-sparsity; "few times faster" with sparsity | [gpu_decoding.py](../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py) | stable | novel-extension of upstream |
| `gpu_paged_attention` | triton | GPU paged-KV decode; port of upstream `jax-v0.6.0` paged-attention with logit bias + `MaskFn` added | [gpu_paged_attention.py](../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py) | stable | vendored-wrapper (jax) + bias/mask |
| `paged_kv_cache_tpu_kernel.scatter_update_pages` | mosaic_tpu | Scatter new K/V projections into paged-KV cache pages. ~10× latency vs k/v projection but faster than non-paged update — a documented stopgap (Mosaic can't prove multiple-of-8 alignment; no non-32-bit mask relayout) | [paged_kv_cache_tpu_kernel.py](../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_tpu_kernel.py) | experimental | novel (stopgap) |
| `paged_kv_cache_gpu_kernel.scatter_update_pages` | triton | GPU counterpart; uses `input_output_aliases` to avoid a full HBM round-trip | [paged_kv_cache_gpu_kernel.py](../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_gpu_kernel.py) | experimental | novel (stopgap) |

### SSM / linear attention — the rare content

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| **`residual_linear_attention`** (`_linear_attention_forward_kernel` + `_linear_attention_backward_kernel`) | mosaic_tpu | **Chunked linear attention** for RAttention's Residual Linear Attention (RLA) branch. Two-level tiling — grid `(batch, heads, chunk)`, inner `fori_loop` over 128-wide subchunks — with **flexible state-saving**: chunk-level hidden states carried across chunks (`mutable_final_state_ref`) and subchunk states recomputed. Parameter-free, feature-map configurable (`FeatureMap`). Enables a 512-token window to train as fast as a 4K SWA window | [linear_attention_kernels.py#L514](../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L514) (fwd [#L106](../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L106), bwd [#L281](../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L281)) | research | **novel — UNIQUE**, no upstream linear-attention Pallas kernel |
| **`ssd_kernels._ssd`** (Mamba2 / SSD) | mosaic_tpu | **Mamba2 state-space-duality scan**, fwd + VJP as a `jax.custom_vjp`. Two-level chunk/subchunk algorithm (Mamba2 paper §7.2, arXiv:2405.21060): chunk-level hidden states stored, subchunk states recomputed from them to trade memory for compute. q/k/v naming for `(b,c,x)`; optional fp32 matmul (`_matmul_fp32`) | GitHub [ssd_kernels.py](https://github.com/apple/axlearn/blob/b18cb42/axlearn/common/ssm_kernels/ssd_kernels.py#L46) — **removed from checkout at `18b88dc4`** | research (removed) | **novel — UNIQUE**, only public Mamba2 Pallas kernel |
| **`mamba_kernels.compute_mamba_scan`** (`_mamba_scan`) | mosaic_tpu | **Mamba1 selective scan**, fwd + VJP (`jax.custom_vjp`). Single kernel tiled over `(batch, inner_dim, seq)` via `MambaArgumentBlockSpecs`, with configurable `seq_tile_size × dim_tile_size`; carries per-tile boundary hidden states (`boundary_hs`) between sequence tiles. Adapted from RecurrentGemma's LRU scan, specialized to Mamba's `(state_dim, inner_dim, seq_len)` layout | GitHub [mamba_kernels.py](https://github.com/apple/axlearn/blob/b18cb42/axlearn/common/ssm_kernels/mamba_kernels.py#L772) — **removed from checkout at `18b88dc4`** | research (removed) | **novel — UNIQUE**, no upstream Mamba Pallas kernel |

### MoE / grouped matmul · fused elementwise

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `megablock.gmm_gpu.gmm` | triton | Grouped GEMM for MoE routing on GPU (public GPU-Triton counterpart of upstream TPU `megablox.gmm`). `make_group_metadata` ported verbatim from upstream; default tiling `(32,32,64)`, `num_stages=4`, `num_warps=2`. Group-offset aggregation not yet supported | [gmm_gpu.py#L374](../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L374) | stable | semi-novel (GPU counterpart of TPU megablox) |
| `online_softmax_log_probs_pallas` (`_online_softmax_log_probs_kernel`) | mosaic_tpu | Fused streaming log-probs + top-k on TPU: computes per-position log-normalizer and top-k logits **without materializing the full `[batch, seq, vocab]` logits** in HBM. Matmul, online log-sum-exp, and running top-k merge fused in VMEM over `[tile_s, tile_v]` tiles | [online_softmax_log_probs.py#L213](../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L213) (kernel [#L67](../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L67)) | stable | novel |

## Notes

**SSM kernels are the standout — and are the reason to keep this repo on the radar even after the
removal.** AXLearn was (per the RAttention README and the [directory catalog](../../analyses/pallas-kernel-directory/04-research-labs.md)) the *only* public source of Mamba1, Mamba2/SSD, and
generic linear-attention Pallas kernels — nothing equivalent exists in `jax.experimental.pallas.ops`,
tokamax, TorchTitan, MaxText, or Penzai. All three share one design idea worth stealing:
**two-level chunk/subchunk recomputation** — store coarse chunk-level carries, recompute fine-grained
subchunk states — which bounds the memory of a linear recurrence while keeping the arithmetic on-chip.
The RAttention `_linear_attention` kernel (still present) is the surviving, directly-runnable
instance of this pattern; `_ssd` and `compute_mamba_scan` are its Mamba-family siblings.

**Removal lineage.** `ssm_kernels/{mamba_kernels,ssd_kernels}.py`, `ssm.py`, and their `*_test.py` +
testdata `.npz` files were deleted in a single commit, `18b88dc4` ("Remove PyTorch and transformers
from default test deps"). The kernels themselves are pure JAX/Pallas, but their **parity tests**
compared against torch `mamba_ssm` reference implementations, so the whole subsystem went when torch
left the default test deps. To resurrect them, check out `b18cb42` (= `18b88dc4~1`) or fetch the
files from `apple/axlearn` history; the Mamba1 kernel derives from
[google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma)'s LRU scan
(the `## See also` catalog subpage traces that ancestry).

**RAttention scan fallbacks are not Pallas.** `residual_linear_attention_linear_scan` and
`residual_linear_attention_w_timestep`
([linear_attention_kernels.py#L611](../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L611),
[#L649](../../../raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py#L649))
are pure `jax.lax.scan` reference/inference implementations (vmapped over batch/head), not Pallas
kernels — the Pallas path is `residual_linear_attention` only. They are the numerical oracle for the
kernel and the prefill/decode-time path.

**Vendoring / autotune surface.**
- *Splash*: adapted from upstream Mosaic splash at commit `b3c49b0`. In-kernel dropout uses
  `jax._src.pallas.mosaic.random`; a known lowering bug (`AssertionError: key<pl>` when the
  `prng_key` is passed directly, as of jax 0.5.3) is worked around by **prefetching** the key.
  Tunable block sizes via `MaskInfo` / `BlockSizes`.
- *Paged (TPU)*: port of upstream `jax-v0.8.1` paged-attention; the block-sparse offset map is the
  AXLearn addition and generalizes beyond sliding window to any `mask_fn`.
- *GPU flash/decoding*: `gpu_decoding` removes upstream's double-vmap transpose (vmaps batch only +
  strided `BlockSpec` for K/V) and supports `mask_fn` by runtime thread-block voting instead of
  compile-time mask materialization.
- *Mamba*: `MATMUL_PREC`/`_matmul_fp32` toggles let the SSM scans opt into float32 matmul for
  accuracy; `seq_tile_size`/`dim_tile_size` (Mamba1) and chunk/subchunk sizes (SSD) are the tiling
  knobs.

**Hardware matrix.** `mosaic_tpu` kernels target TPU (splash/paged/decoding/log-probs/SSM);
`triton` kernels target NVIDIA A100/H100 (GPU flash/decoding/paged/gmm). A `neuron_attention`
flash path exists for AWS Trainium but is **NKI, not Pallas**, so it is out of scope here.

## See also
- [axlearn — overview](overview.md) — the repo's grounded wiki front door.
- [axlearn — flash_attention/layer](concepts/axlearn-common-flash_attention-layer.md) — the dispatch
  layer that selects among these attention kernels per platform/bias.
- [axlearn — flash_attention/common](concepts/axlearn-common-flash_attention-common.md) — shared base
  classes and bias plumbing the kernels consume.
- [axlearn — attention_bias](concepts/axlearn-common-attention_bias.md) — `MaskFn` / `SlidingWindow`
  / `SegmentId` bias system these kernels honor.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the
  cross-repo Pallas catalog; this repo appears in
  [subpage 04 — Research labs](../../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn).

## Sources
- `raw/code/axlearn/axlearn/common/flash_attention/` — `tpu_splash_attention.py`, `gpu_attention.py`,
  `tpu_paged_attention_kernel.py`, `tpu_decoding.py`, `gpu_decoding.py`, `gpu_paged_attention.py`,
  `tpu_attention.py` (read at HEAD `c8624c8`).
- `raw/code/axlearn/axlearn/common/rattention/kernels/linear_attention_kernels.py`,
  `.../rattention/README.md`.
- `raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_{tpu,gpu}_kernel.py`,
  `.../megablock/gmm_gpu.py`, `.../kernels/online_softmax_log_probs.py`.
- `apple/axlearn` git history at `b18cb42` (`18b88dc4~1`): `axlearn/common/ssm_kernels/mamba_kernels.py`,
  `ssd_kernels.py` (removed from checkout at `18b88dc4`).
