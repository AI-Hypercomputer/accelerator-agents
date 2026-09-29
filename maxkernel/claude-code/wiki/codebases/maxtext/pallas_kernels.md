---
title: "maxtext — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, maxtext]
created: 2026-07-04
updated: 2026-07-04
---

MaxText ships roughly a dozen locally-defined Pallas / Mosaic kernels across four families —
**training attention** (splash fwd+bwd), **decode attention** (ragged flash), **MoE grouped-matmul**
(two generations: qwix-quantized `gmm`/`tgmm` v1 and a tokamax-forked `gmm_v2`/`tgmm_v2`), and **MoE
token-routing on SparseCore** (gather / gather-reduce, plus a newer Pallas rewrite and a `ragged/`
subpackage). Every kernel targets the **`mosaic_tpu`** backend — attention and grouped-matmul run on
the TensorCore/MXU, the routing kernels run on **SparseCore**; there are no Mosaic-GPU or Triton
kernels (GPU paths fall through to XLA or `cudnn_flash_te`). The heavy compute kernels (splash, GMM)
are **vendored** — splash and ragged from `jax-ml/jax`, GMM v2 forked from `openxla/tokamax` — while
the SparseCore gather/reduce/routing kernels are **novel to MaxText**. Commit ingested:
`fcb7ebe`.

> **Divergence from the 2026-04-23 [directory catalog](../../analyses/pallas-kernel-directory/02-ai-hypercomputer.md):**
> The paged / ragged-paged inference kernels that the catalog listed under
> `inference/paged_attention_kernel_v2.py` and `inference/paged_attention.py` are **no longer in
> the source tree** — paged attention has migrated out to
> [tpu-inference](https://github.com/AI-Hypercomputer/tpu-inference) and is now *imported*, not
> defined here (`attention: vllm_rpa` → `sharded_ragged_paged_attention`, see
> [layers/attentions.py#L1016](../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L1016)).
> Conversely, two things the catalog did not have: the megablox **v2** GMM/TGMM kernels
> (forked from tokamax) and the **`kernels/ragged/`** SparseCore subpackage + `gather_reduce_pallas.py`
> Pallas rewrite are new.

## Kernels

### Attention — training (splash)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `flash_attention_kernel` (splash fwd) | mosaic_tpu | Block-**sparse** flash-attention forward: online-softmax (`m`,`l`,`logsumexp`) over only the KV blocks a scalar-prefetched mask schedule (`data_next`/`block_mask`) marks live; MQA/GQA, segment ids, logit soft-cap, causal + chunked masks. Never materializes the S×S score matrix. | [splash_attention_kernel.py#L673](../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L673) | stable | vendored-from-jax (splash) |
| `_flash_attention_dq_kernel` (splash bwd dQ) | mosaic_tpu | Gradient dQ pass; grid `(heads, q//bq, kv_width)` Q-outer, streams KV into `dq_scratch`, reuses forward `logsumexp` so softmax is not recomputed. See [concept page](concepts/maxtext-kernels-attention-splash_attention_kernel.md). | [splash_attention_kernel.py#L1206](../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L1206) | stable | vendored-from-jax (splash) |
| `_flash_attention_dkv_kernel` (splash bwd dK/dV) | mosaic_tpu | Gradient dK/dV pass; **transposed** grid `(kv//bkv, heads, q_width)` KV-outer; unrolls inner loop over `bkv_compute`; optional fused-bwd variant emits unreduced dq. See [concept page](concepts/maxtext-kernels-attention-splash_attention_kernel.md). | [splash_attention_kernel.py](../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py) | stable | vendored-from-jax (splash) |

Autotune surface (`BlockSizes`): `block_q`, `block_kv`, `block_kv_compute`, `block_q_dkv`,
`block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`, `use_fused_bwd_kernel`, and the
`QKVLayout` (`HEAD_DIM_MINOR` / `SEQ_MINOR`) physical-layout enum. The MaxText copy still `import`s
the upstream `splash_attention_mask` / `splash_attention_mask_info` libraries from jax
([L34-35](../../../raw/code/maxtext/src/maxtext/kernels/attention/splash_attention_kernel.py#L34)).

### Attention — decode (ragged)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `ragged_flash_attention_kernel` (`ragged_mha`/`ragged_gqa`/`ragged_mqa`) | mosaic_tpu | Single-token **ragged decode** flash attention with per-sequence `lengths`; grid `(batch, seq_len//block_size)`, `dimension_semantics=("parallel","arbitrary")`, block indices computed from ragged lengths, `pl.CostEstimate` hints. Inference only — no backward path. | [ragged_attention.py#L162](../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L162) (`pl.pallas_call` [#L254](../../../raw/code/maxtext/src/maxtext/kernels/attention/ragged_attention.py#L254)) | stable (decode) | vendored-from-jax |

### MoE — grouped matmul, v1 (qwix-quantized)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `gmm` (fwd) | mosaic_tpu | Grouped GEMM for MoE: token stack `lhs[m,k] @ rhs[g(m)]` over ragged `group_sizes`; uses `make_group_metadata` for the ragged-M tile map and `qwix.pallas.QArray` for int4/int8 weights (quant-aware `pallas_call`). | [megablox/backend.py#L304](../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L304) (`qpl.pallas_call` [#L525](../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L525)) | stable | novel (MaxText Megablocks port) |
| `tgmm` (bwd / transposed) | mosaic_tpu | Transposed grouped GEMM — the `d(rhs)` / accumulation direction of the MoE matmul; same ragged metadata + qwix quantization path. | [megablox/backend.py#L575](../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L575) (`qpl.pallas_call` [#L782](../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L782)) | stable | novel (MaxText Megablocks port) |

### MoE — grouped matmul, v2 (tokamax fork)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `gmm_v2` (`kernel_main`) | mosaic_tpu | Rewritten GMM on `pltpu.emit_pipeline` (manual software pipeline over a **data-dependent** grid `(num_n, num_gm, num_k)`); `fill_metadata` builds the ragged-M tile map, **triple-buffered** weight DMAs, int4/int8/fp8 per-block quant, fused SwiGLU/`swigluoai`, dynamic zero-fill of untouched rows. See [concept page](concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md). | [pallas_mosaic_tpu_v2_gmm_kernel.py#L765](../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L765) (`pl.pallas_call` [#L1344](../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L1344)) | experimental (opt-in `use_gmm_v2`) | vendored-from-tokamax |
| `tgmm_v2` (`tgmm_kernel_main`) | mosaic_tpu | Transposed v2 kernel (the `d(rhs)` companion of `gmm_v2`); grid `(num_n, num_k, num_gm)`, shares `gmm_v2`'s metadata/block-spec machinery. | [pallas_mosaic_tpu_v2_tgmm_kernel.py#L524](../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L524) (`pl.pallas_call` [#L741](../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L741)) | experimental (opt-in `use_gmm_v2`) | vendored-from-tokamax |

### MoE — token routing (SparseCore gather / reduce)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `sc_gather_reduce` (MLIR-dialect) | mosaic_tpu (SparseCore) | Gather-reduce for MoE unroute: groups operand rows by `idx`, sums in fp32, writes bf16. Lowered via `jax.experimental.mosaic` MLIR dialects (`arith`/`func`/`memref`/`scf`/`vector`) + `mosaic.as_tpu_kernel`, **not** a `pallas_call`. `dimension_semantics<core_parallel>`. | [gather_reduce_sc.py#L1617](../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_sc.py#L1617) | stable | novel |
| `sc_gather_reduce` (Pallas) | mosaic_tpu (SparseCore) | Pallas rewrite of the above — same gather-reduce semantics via `pltpu.emit_pipeline` over row/col chunks with subcore tiling. The newer, `pallas_call`-based path. | [gather_reduce_pallas.py#L32](../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py#L32) (`emit_pipeline` [#L117](../../../raw/code/maxtext/src/maxtext/kernels/gather_reduce_pallas.py#L117)) | experimental | novel |
| `ragged_gather` (`main_kernel`) | mosaic_tpu (SparseCore) | Ragged token gather for AG-RS expert-parallel routing; `pl.kernel` + `emit_pipeline`, grid `(num_blocks, num_cores, num_cols)`, `use_tc_tiling_on_sc`, HBM→VMEM staged with explicit semaphores. | [ragged/ragged_gather.py#L45](../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L45) (public `ragged_gather` [#L367](../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather.py#L367)) | experimental | novel |
| `ragged_gather_reduce` (`main_kernel`) | mosaic_tpu (SparseCore) | Ragged gather **+ reduce** in one SparseCore kernel (`reduce_group_size`); grid `(num_cores,)`, `dimension_semantics=(PARALLEL,)`, `pl.CostEstimate` from `get_cost_estimate`. | [ragged/ragged_gather_reduce.py#L122](../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L122) (public [#L435](../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_gather_reduce.py#L435)) | experimental | novel |

### Orchestration & fallbacks (not standalone Pallas kernels)

- **`sort_activations.py`** — `route` / `unroute` / `_sort_activations` custom-VJP wrappers that dispatch
  to `sc_gather_reduce` when `use_gather_mosaic_kernel: True`; token sorting/routing glue, no
  `pallas_call` of its own. [sort_activations.py](../../../raw/code/maxtext/src/maxtext/kernels/sort_activations.py)
- **`ragged/ragged_sort.py`** — `ring_ragged_sort` / `a2a_ragged_sort` (+ unsort) custom-VJP routing
  ops that **compose** `ragged_gather` and `ragged_gather_reduce`; orchestration, not a kernel.
  [ragged/ragged_sort.py](../../../raw/code/maxtext/src/maxtext/kernels/ragged/ragged_sort.py)
- **`jax_flash_attention.py`** (`flash_attention_block_masked`) — pure-JAX/XLA block-sparse flash
  reference, **not Pallas**; the GPU/debug fallback when splash is unavailable.
  [attention/jax_flash_attention.py](../../../raw/code/maxtext/src/maxtext/kernels/attention/jax_flash_attention.py)

### Imported (not defined here)

- **Paged / ragged-paged attention** — no longer in the source tree (was `inference/paged_attention*`
  in the 2026-04-23 catalog). Now imported from `tpu-inference`
  (`sharded_ragged_paged_attention`) under `attention: vllm_rpa`.
- **`tokamax.ragged_dot` / `ragged_dot_general`** — the `use_tokamax_backend=True` fall-through for
  MoE grouped-matmul in [megablox/ops.py](../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py);
  runs tokamax's own mosaic kernel, not a local copy.

## Notes

**Vendoring lineage.**
- *Splash / ragged attention*: copied from `jax-ml/jax`
  (`jax.experimental.pallas.ops.tpu.splash_attention`), keeping the upstream mask libraries as live
  imports rather than forking them. Canonical chain: **jax-ml/jax upstream → (tokamax/DeepMind) →
  MaxText copy**. A tokamax splash flavor is togglable via `config.use_splash_scheduler`.
- *GMM v2 / TGMM v2*: explicit **forks of `openxla/tokamax`** at commit `3f332fc`
  (`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_{gmm,tgmm}_kernel.py`) — the file headers cite the
  exact upstream URL. Copyright "2026 DeepMind Technologies Limited". They are opt-in behind
  `use_gmm_v2` in [megablox/ops.py](../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py#L192);
  the default MoE path is still the v1 qwix `gmm`/`tgmm` in `backend.py`, and a third path routes to
  `tokamax.ragged_dot(implementation="mosaic")` when `use_tokamax_backend=True`.
- *SparseCore gather/reduce/routing*: **novel to MaxText** — no upstream jax/tokamax analogue. Two
  co-existing implementations of the same gather-reduce (`gather_reduce_sc.py` via raw Mosaic MLIR
  dialects; `gather_reduce_pallas.py` via Pallas `emit_pipeline`) signal an in-progress migration off
  hand-written MLIR onto Pallas. The `kernels/ragged/` subpackage is the newest layer.

**Backend / hardware matrix.** All `mosaic_tpu`. Attention and grouped-matmul use the TensorCore/MXU;
the routing kernels target **SparseCore** (`plsc` / `jax.experimental.pallas.tpu_sc`,
`use_tc_tiling_on_sc`, `#tpu.dimension_semantics<core_parallel>`). No Mosaic-GPU (SM90/SM100) or
Triton — GPU attention falls to `jax_flash_attention.py` or `cudnn_flash_te`.

**Autotune surfaces for the perf loop.**
- Splash: the ten-field `BlockSizes` + `QKVLayout` (see the [splash concept page](concepts/maxtext-kernels-attention-splash_attention_kernel.md)).
- Ragged decode: `ragged_block_size` (default 256), toggled by `use_ragged_attention`.
- GMM v1: 9-tuple `tiling` (default 128), `use_tokamax_gmm`; GMM v2: `TileSizes(tile_m/tile_n/tile_k)`
  + `vmem_limit_bytes` (defaults 90% VMEM) with N-shrink-before-K tiling and triple-buffered weights
  (see the [GMM v2 concept page](concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)).
- SparseCore gather-reduce: `col_chunk_size` (default 3.5×1024), `row_chunk_size` (512),
  `reduce_group_size`, `single_sc`, `topk_wgt_zero_nan`, plus loop-unroll/parallel-access factors.

**Gotchas for reuse.**
- The `kernels/ragged/*` kernels branch on `jax.__version__` (≤ 0.10.0 uses `out_shape`/`scratch_shapes`
  vs newer `out_type`/`scratch_types` and a `needs_layout_passes` compiler param) — copying them out
  requires matching the Pallas `pl.kernel` API version.
- GMM v2/TGMM v2 pass operands as registered pytrees (`OperandRef`/`WeightsRef` with optional scale
  leaves) so the kernel arity stays fixed with or without quantization — reusing them means honoring
  that pytree contract.
- `sc_gather_reduce` writes bf16 output from fp32 accumulation by design (quality trade-off is upstream
  of the kernel); the two implementations are interchangeable but not bit-identical.

## See also
- [maxtext — overview](overview.md) — the repo's grounded wiki front door.
- [SplashAttention backward kernels](concepts/maxtext-kernels-attention-splash_attention_kernel.md) — deep-dive on the splash bwd dQ/dKV passes.
- [MegaBlox GMM v2 grouped-matmul](concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md) — deep-dive on the v2 pipelined MoE kernel.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 02 §2.1](../../analyses/pallas-kernel-directory/02-ai-hypercomputer.md).

## Sources
- [raw/code/maxtext/src/maxtext/kernels/](../../../raw/code/maxtext/src/maxtext/kernels/) — commit `fcb7ebe`, read 2026-07-04.
