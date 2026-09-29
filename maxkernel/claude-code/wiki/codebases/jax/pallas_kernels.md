---
title: "jax — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, jax]
created: 2026-07-04
updated: 2026-07-04
---

The reference Pallas kernels shipped in-tree under `jax/experimental/pallas/ops/{tpu,gpu}/` — **27 kernels** spanning `mosaic_tpu` (11), Pallas-Triton (6), `mosaic_gpu_sm90` / Hopper (8), and `mosaic_gpu_sm100` / Blackwell (2). Because jax is the **root of the Pallas vendoring graph**, every kernel here is a *novel upstream original*: nothing is vendored in — instead tokamax, MaxText, vLLM-TPU, Tunix, Levanter, AxLearn and others copy or dispatch to these. All are `experimental` per JAX's stability policy (no stable-API guarantee); several self-describe as pedagogical/example. The TPU set (splash, flash, paged, ragged-paged, megablox grouped-matmul, PRNG) is the perf-relevant surface for this project; the GPU set is included for completeness and lineage.

## Kernels

### Attention — flash / splash (TPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `splash_attention` | mosaic_tpu | Sparse flash attention (MHA/MQA/GQA) with arbitrary block-sparse masks (causal, local, sliding-window, document, chunked, custom) via precomputed `MaskInfo`; custom VJP fwd+bwd; `SplashConfig`/`BlockSizes` tuning. The production TPU SDPA kernel. | [splash_attention_kernel.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py) | experimental | novel (upstream original) |
| `flash_attention` | mosaic_tpu | Fused flash attention fwd + `_dkv`/`_dq` bwd kernels; causal + `SegmentIds` masking; `BlockSizes` dataclass ("negligible effect on numerics, but affect performance greatly"). Legacy — superseded by splash for masked/sparse cases. | [flash_attention.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py) | experimental | novel (upstream original) |

### Attention — paged / ragged (TPU, decode/serving)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `paged_attention` | mosaic_tpu | Decode-phase paged-KV attention (vLLM-style); `MultiPageAsyncCopyDescriptor` streams KV pages HBM→VMEM; optional quantized-KV path via `quantization_utils.py`. Two grid variants (`paged_flash_attention_kernel` + inline-seq-dim). | [paged_attention/paged_attention_kernel.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py) | experimental | novel (upstream original) |
| `ragged_paged_attention` | mosaic_tpu | Fused mixed prefill + decode over paged KV — "highly optimized implementation…designed for TPU"; block sizes autotuned per shape via `tuned_block_sizes.py`. | [ragged_paged_attention/kernel.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py) | experimental | novel (upstream original) |

### MoE / grouped matmul (TPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `megablox.gmm` | mosaic_tpu | Grouped matmul for MoE expert routing (`group_sizes`/`group_offset`); bf16/fp32 (bf16·bf16 needs TPU v4+). Wrapped in custom VJP in `ops.py`. | [megablox/gmm.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py) | experimental | novel (upstream original) |
| `megablox.tgmm` | mosaic_tpu | Transposed grouped matmul — the backward-pass companion to `gmm` (`grad_rhs = tgmm(...)`). Same file. | [megablox/gmm.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py) | experimental | novel (upstream original) |

### PRNG (TPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `philox` (4×32) | mosaic_tpu | Philox counter-based PRNG fused with `shaped_iota` to avoid materializing counters in HBM. | [random/philox.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py) | experimental | novel (upstream original) |
| `threefry` (2×32) | mosaic_tpu | Threefry PRNG fused kernel (`jax._src.prng.threefry2x32_p`); same fusion pattern as Philox. | [random/threefry.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py) | experimental | novel (upstream original) |

### Collectives / examples (TPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `all_gather` | mosaic_tpu | Bidirectional ring all-gather over one mesh axis via remote DMA. Explicitly pedagogical — "doesn't have all possible performance optimizations". | [all_gather.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/all_gather.py) | pedagogical | novel (upstream original) |
| `matmul` | mosaic_tpu | Tutorial tiled matmul for the Pallas TPU docs. | [matmul.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/matmul.py) | example | novel (upstream original) |
| `double` | mosaic_tpu | Element-wise ×2 test fixture for `pallas_call` tests. | [example_kernel.py](../../../raw/code/jax/jax/experimental/pallas/ops/tpu/example_kernel.py) | example | novel (upstream original) |

### Attention (GPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `mha` / attention | triton | FlashAttention v1/v2 fused SDPA fwd+bwd; `BlockSizes` needs per-shape tuning. | [gpu/attention.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py) | experimental | novel (upstream original) |
| `attention_mgpu` | mosaic_gpu_sm90 | FlashAttention3 forward on Hopper — WGMMA + TMA pipeline, warp-specialized; `PipelineCallback` plug-in protocol. | [gpu/attention_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py) | experimental | novel (upstream original) |
| `decode_attention` | triton | Short-Q / batched-decode GPU attention; `block_h`/`block_k` tuning. | [gpu/decode_attention.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py) | experimental | novel (upstream original) |
| `paged_attention` (GPU) | triton | GPU paged-KV decode attention; supports quantized K/V via `k_scales_pages_ref`/`v_scales_pages_ref`. Docstring "decode attention" is stale-ish. | [gpu/paged_attention.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py) | experimental | novel (upstream original) |

### Norm / softmax (GPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `layer_norm` | triton | Fused LayerNorm fwd+bwd; uses `plgpu.load(eviction_policy="evict_last")`. | [gpu/layer_norm.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/layer_norm.py) | experimental | novel (upstream original) |
| `rms_norm` | triton | Fused RMSNorm fwd+bwd; structured like `layer_norm`, drops the mean. | [gpu/rms_norm.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/rms_norm.py) | experimental | novel (upstream original) |
| `softmax` | triton | Single-pass numerically-stable softmax; assumes `block_row ≥ row_len`. | [gpu/softmax.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/softmax.py) | experimental | novel (upstream original) |

### Matmul / MoE grouped-matmul (GPU, Mosaic-GPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `hopper_matmul_mgpu` | mosaic_gpu_sm90 | Dense GEMM on H100 — warp-specialized WGMMA + TMA, persistent grid. Base for the collective + ragged variants. | [gpu/hopper_matmul_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py) | experimental | novel (upstream original) |
| `hopper_mixed_type_matmul_mgpu` | mosaic_gpu_sm90 | bf16×int8/fp8 GEMM on H100; dequantizes before WGMMA. | [gpu/hopper_mixed_type_matmul_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py) | experimental | novel (upstream original) |
| `blackwell_matmul_mgpu` | mosaic_gpu_sm100 | Dense GEMM on B200 — TMEM / TCGEN05, 2-CTA clusters; `MatmulDimension` enum. | [gpu/blackwell_matmul_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py) | experimental | novel (upstream original) |
| `blackwell_ragged_dot_mgpu` | mosaic_gpu_sm100 | MoE grouped matmul on B200; composes with `blackwell_matmul_mgpu`. | [gpu/blackwell_ragged_dot_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py) | experimental | novel (upstream original) |
| `ragged_dot_mgpu` | mosaic_gpu_sm90 | MoE grouped matmul on H100. | [gpu/ragged_dot_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py) | experimental | novel (upstream original) |
| `transposed_ragged_dot_mgpu` | mosaic_gpu_sm90 | MoE backward grouped matmul on H100. | [gpu/transposed_ragged_dot_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py) | experimental | novel (upstream original) |

### Collectives (GPU, Mosaic-GPU)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `collective_matmul_mgpu` | mosaic_gpu_sm90 | AG/RS fused with GEMM across H100s (NVSHMEM-style multihost); builds on `hopper_matmul_mgpu`. | [gpu/collective_matmul_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py) | experimental | novel (upstream original) |
| `all_gather_mgpu` | mosaic_gpu_sm90 | Intra-node all-gather via NVLink. | [gpu/all_gather_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py) | experimental | novel (upstream original) |
| `reduce_scatter_mgpu` | mosaic_gpu_sm90 | Intra-node reduce-scatter via NVLink; parameterized by reduce op kind. | [gpu/reduce_scatter_mgpu.py](../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py) | experimental | novel (upstream original) |

## Notes

- **Vendoring lineage — jax is the source, not a sink.** These are the upstream originals other repos copy or dispatch to. Notably: tokamax **vendors** an extended fork of `splash_attention_kernel` at `_src/ops/experimental/tpu/splash_attention/` (own `SplashConfig` + ring integration) and builds its GPU MoE kernels on `ragged_dot_mgpu` / `blackwell_ragged_dot_mgpu`; MaxText, Tunix, Pax-ML, JetStream, vLLM-TPU dispatch to the TPU splash / paged / ragged-paged / megablox kernels; Gemma, Levanter, AxLearn use splash. See [subpage 01 §1.1](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) for the downstream-caller matrix and [§1.2](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) for how tokamax extends each.
- **Backend detection rule** (used to fill the Backend column): TPU files import `jax.experimental.pallas.tpu` → `mosaic_tpu`; GPU files import `jax.experimental.pallas.triton` → `triton`; `*_mgpu.py` files import Mosaic-GPU → `mosaic_gpu_sm90` (Hopper) or `mosaic_gpu_sm100` (Blackwell), read off the docstring ("Hopper"/"Blackwell").
- **Helper modules (not kernels, load-bearing for one).** `splash_attention/splash_attention_mask.py` (causal/local/chunked mask object library) and `splash_attention_mask_info.py` (precomputed block-sparse mask metadata enabling sparse-skip) are required by `splash_attention` but define no `pallas_call`. Similarly `ragged_paged_attention/tuned_block_sizes.py` (autotune LUT), `megablox/common.py`, `paged_attention/{util,quantization_utils}.py`, and `random/prng_utils.py` are support code, not kernels — excluded from the table.
- **Autotune surface.** `flash_attention.BlockSizes`, `splash` `SplashConfig`/`BlockSizes`, `ragged_paged_attention` `tuned_block_sizes.py` LUT (per (dtype, shape, TPU-gen)), `megablox` tiling args, and the GPU `BlockSizes`/`block_h`/`block_k` dataclasses are the tunable knobs for TPU-perf experiments.
- **Hardware matrix.** mosaic_tpu kernels target TPU v4+; `megablox.gmm` bf16·bf16 requires v4+. GPU: triton = pre-Hopper/generic; `mosaic_gpu_sm90` = H100; `mosaic_gpu_sm100` = B200.
- **Perf claims quoted from source.** `flash_attention` `BlockSizes`: block sizes have "negligible effect on numerics, but affect performance greatly." `ragged_paged_attention`: "highly optimized implementation of ragged paged attention, specifically designed for TPU." No file carries a benchmarked number in-tree.
- **Reuse gotcha.** Everything is `experimental` — import paths and signatures move between JAX releases. For this project's TPU work, prefer `splash_attention` over `flash_attention` (splash is the actively-maintained masked/sparse path). Divergence check: source at commit `415e5b7007` (2026-07-02) still matches the [2026-04-23 directory](../../analyses/2026-04-23-pallas-kernel-directory.md) enumeration — no kernels added/removed since the catalog was built.

## See also
- [jax — overview](overview.md) — the repo's grounded wiki front door.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; jax appears in [subpage 01 — Upstream JAX + tokamax](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md).
- [jax — pallas mosaic lowering](concepts/jax-_src-pallas-mosaic-lowering.md) and [pallas core](concepts/jax-_src-pallas-core.md) — the framework/lowering internals these kernels compile through (out of scope here).

## Sources
- [raw/code/jax/jax/experimental/pallas/ops/](../../../raw/code/jax/jax/experimental/pallas/ops/) — kernel definitions, read at commit `415e5b7007` (2026-07-02).
- [wiki/analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md](../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) — cross-repo directory rows for jax.
