---
title: "tpu-inference — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, tpu-inference]
created: 2026-07-04
updated: 2026-07-04
---

`vllm-project/tpu-inference` is the **authoritative Pallas-kernel author** for the TPU inference tier — the richest first-party collection among the serving engines, and the upstream that `sglang-jax` and (indirectly) `aphrodite` vendor from. Every kernel here targets the **`mosaic_tpu`** backend (`jax.experimental.pallas.tpu`, i.e. `pltpu`); there is **no GPU/Triton/mosaic-gpu code** in the tree. Roughly **35 distinct Pallas kernels** across attention (RPA v2/v3, MLA v1/v2, flash, batched-RPA, DeepSeek-V4 experimental), MoE grouped-matmul, quantized matmul, TP collectives, Gated-Delta-Net SSM, causal-conv1d, SparseCore gather/scatter, and structured-sparse matmul. The large majority are **novel to this repo**; the notable vendored exceptions are `flash_attention` (from `jax-ml/jax`) and the `megablox` GMMs (from JAX's megablox). Backing them is an **autotune surface of ~5,480 tuned block-size entries** (see Notes) — the RPA v2 "~1,200 entries" figure the [directory catalog subpage 03](../../analyses/pallas-kernel-directory/03-inference-engines.md) cited on 2026-04-23 is now **1,341**, and the total surface has grown ~4.5× since, driven mostly by RPA v3's full v5e/v6e/v7 tables.

Commit ingested: `8eb31ee`.

## Kernels

### Attention — paged / ragged / MLA
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `ragged_paged_attention` v3 | mosaic_tpu | Default attention path. Mixed prefill+decode over a paged KV cache; dispatches into three separate `pallas_call`s (DECODE / PREFILL / MIXED, `RpaCase` enum), per-`bq` `l`/`m`/`acc` reinit, precise sliding-window page skipping. | [v3/kernel.py#L291](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L291) | stable | **Novel** |
| `ragged_paged_attention` v3 hd64 | mosaic_tpu | RPA v3 specialized for `head_dim == 64` (Gemma-class); separate kernel + separate tuned table. | [v3/kernel_hd64.py#L254](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L254) | stable | **Novel** specialization |
| `ragged_paged_attention` v2 | mosaic_tpu | Prior-generation RPA (v5-era). Same mixed prefill/decode contract via `MultiPageAsyncCopyDescriptor`. Note: the kernel pulls its block sizes from **upstream JAX** `get_tuned_block_sizes`, not the local table (see Notes). | [v2/kernel.py#L843](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L843) | stable | **Novel** |
| `mla_ragged_paged_attention` v1 | mosaic_tpu | Multi-Head Latent Attention (DeepSeek-V2/V3 class). "TPU-Friendly and Data-Movement-Friendly" ragged paged MLA; supports mixed prefill/decode; heuristic block sizes (autotune TODO). 100 MiB VMEM limit. | [mla/v1/kernel.py#L489](../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L489) | stable | **Novel** |
| `mla_ragged_paged_attention` v2 | mosaic_tpu | MLA successor: better VMEM management + double/triple-buffered pipelining; paired `xpose_pipeline` transpose kernels and `kv_utils`; fp8 Q/KV, bf16 softmax input. Driven by `MLA_XPOSE_N_TILE_SIZE` env. | [mla/v2/kernel.py#L286](../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L286) | stable | **Novel** |
| MLA v2 transpose (`xpose_pipeline`) | mosaic_tpu | Helper Pallas transposes (3 `pallas_call`s incl. a prefetch pipeline) that reshape the latent KV for the MLA v2 einsums. | [mla/v2/transpose.py#L48](../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L48) | stable | **Novel** |
| `flash_attention` | mosaic_tpu | Dense flash attention for non-paged paths (prefill / training-style), `SegmentIds`-based causal masking, blocked `pallas_call`. | [flash_attention/kernel.py#L740](../../../raw/code/tpu-inference/tpu_inference/kernels/flash_attention/kernel.py#L740) | stable | **Vendored** from [`jax-ml/jax` pallas flash_attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py) — API identical, no attribution header. Dao et al. 2022 algorithm |
| `batched_rpa` | mosaic_tpu | Batches multiple sequences to replace per-request flash loops; triple-buffering via `emit_pipeline`; precomputes page-location/bounds metadata once (`schedule.py`) and amortizes across layers. Ships a repo-local `flash_attention.py` variant. Gated on `USE_BATCHED_RPA_KERNEL=1`. | [experimental/batched_rpa/kernel.py#L496](../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L496) | experimental — `"all of the code in this directory is experimental and not fully tested"` | **Novel**, "build on top / derived from RPA3" |

### Attention — DeepSeek-V4 (experimental)
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| DeepSeek-V4 MLA | mosaic_tpu | MLA ragged-paged-attention variant for DeepSeek-V4 (DECODE/PREFILL/MIXED cases). | [experimental/deepseek_v4/mla.py#L84](../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L84) | experimental (research) | **Novel** |
| DeepSeek-V4 MLA sliding-window | mosaic_tpu | MLA + sliding-window masking variant. | [experimental/deepseek_v4/mla_swa.py#L69](../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L69) | experimental (research) | **Novel** |
| `streamindex_topk` | mosaic_tpu | Streaming top-k index selection for DeepSeek-V4 sparse/native-sparse attention (per-`RpaCase`). | [experimental/deepseek_v4/streamindex_topk.py#L68](../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L68) | experimental (research) | **Novel** |
| DeepSeek-V4 compressor | mosaic_tpu | Scatters `[kv | score + ape]` into a `state_cache`, skipping `slot < 0` — KV-compression cache write. | [experimental/deepseek_v4/compressor.py#L79](../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compressor.py#L79) | experimental (research) | **Novel** |
| `compress_norm_rope` | mosaic_tpu | Fused compress + norm + RoPE prologue for the DeepSeek-V4 compressed-KV path. | [experimental/deepseek_v4/compress_norm_rope.py](../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py) | experimental (research) | **Novel** |

### KV-cache update
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `ragged_kv_cache_update` | mosaic_tpu | Packs ragged new-KV into the paged cache via async DMAs; paired with RPA v2. Carries a vLLM SPDX header. | [v2/ragged_kv_cache_update.py#L21](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py#L21) | stable | **Novel** (vLLM-authored) |

### MoE / grouped matmul
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `fused_moe` v1 (`_fused_ep_moe_kernel`) | mosaic_tpu | Full fused MoE: expert routing + two grouped matmuls in one `pallas_call`. Expert-parallel (`ep`) friendly. v7-primary tuning. | [fused_moe/v1/kernel.py#L197](../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L197) | stable | **Novel** |
| `megablox_gmm` | mosaic_tpu | Grouped matmul for MoE experts; `tiling`-based, `(tm,tk,tn)` blocked, bf16×fp8_e4m3fn. | [megablox/gmm.py#L603](../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm.py#L603) | stable | **Vendored** from JAX megablox (name + signature match, no attribution header); local tuned table |
| `megablox_gmm_v2` | mosaic_tpu | GMM v2, quantized-friendly; TPU-hardware-conditional lhs quant dtype. Shares the megablox tuned table. | [megablox/gmm_v2.py#L1330](../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L1330) | stable | **Vendored / adapted** |

### Quantized matmul
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `quantized_matmul` (per-tensor/channel) | mosaic_tpu | Weight-quantized linear; `matmul_kernel` with acc/x_q scratch, dynamic x-quant, `MXU_SIZE=256`. Covers int8×int8, fp8×fp8, w16a16 baseline. | [quantized_matmul/kernel.py#L229](../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/kernel.py#L229) | stable | **Novel** |
| `quantized_matmul` blockwise | mosaic_tpu | Sub-channel *blockwise* quantized matmul (`w_scale` per `[n_in//block_size]` block). Zero-point / sub-channel-zp path explicitly `NotImplementedError`. | [quantized_matmul/blockwise_kernel.py#L190](../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/blockwise_kernel.py#L190) | stable | **Novel** |

### Collectives (TP fusion)
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `all_gather_matmul` | mosaic_tpu | Fuses the TP all-gather with the matmul it feeds — pipelines remote-copy, HBM→VMEM DMA and MXU compute with semaphores to hide the collective. Constraints: k,n divisible by 128; m divisible by `tp_size*2*8`. | [collectives/all_gather_matmul.py#L698](../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L698) | stable | **Novel** |
| `hierarchical_reduce_scatter` | mosaic_tpu | Hierarchical recursive-doubling reduce-scatter; VMEM-buffered async accumulate across phases. | [collectives/hierarchical_reduce_scatter.py#L418](../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L418) | stable | **Novel** |

### SSM / linear attention (Gated Delta Net)
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| GDN v1 fused decode | mosaic_tpu | Fused Gated-Delta-Net decode (Qwen-Next / hybrid SSM-attention); `emit_pipeline` q/k/v/g/b tiling, bulk manual DMA for state load/store. | [gdn/v1/fused_gdn_decode_kernel.py#L478](../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py#L478) | stable | **Novel** |
| GDN v1 fused recurrent | mosaic_tpu | Recurrent (chunked-scan) GDN kernel for prefill; two `pallas_call`s (metadata + inner scan body). | [gdn/v1/fused_gdn_recurrent_kernel.py#L550](../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L550) | stable | **Novel** |
| GDN v2 decode | mosaic_tpu | GDN decode v2 with `compute_schedule_v2` metadata reduction. | [gdn/v2/gdn_decode_kernel.py#L564](../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/gdn_decode_kernel.py#L564) | stable | **Novel** |
| GDN v2 recurrent scan | mosaic_tpu | Recurrent-scan v2 body. | [gdn/v2/recurrent_scan_v2.py#L674](../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_v2.py#L674) | stable | **Novel** |
| GDN v3 (`compute_gdn` + `compute_conv1d`) | mosaic_tpu | v3 facade `pallas_call` orchestrating Conv1D + GDN for a tile; strict VMEM/DMA separation via `vmem_ldst`. | [gdn/v3/wrapper.py#L467](../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/wrapper.py#L467) | stable | **Novel** |
| GDN `triangle_solver` | mosaic_tpu | Triangular-matrix inverse via Newton-Schulz iteration (two `pallas_call`s) — the delta-rule chunk solve. | [gdn/triangle_solver.py#L78](../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L78) | stable | **Novel** |
| `causal_conv1d` (standalone) | mosaic_tpu | Causal depthwise conv1d with stateful conv-cache update; `strided_ldst` helper. | [causal_conv1d/causal_conv1d.py#L524](../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L524) | stable | **Novel** |

### Structured / sparse
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `structured_sparse_matmul` v1 (SPMM) | mosaic_tpu | Software-emulated N:M structured-sparse matmul (M ≤ 16), f32/bf16/int8, LHS- or RHS-sparse. "performance benefits limited to memory-bound workloads." | [structured_sparse_matmul/v1/spmm.py#L626](../../../raw/code/tpu-inference/tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py#L626) | experimental (TODOs: int4, subelement masking) | **Novel** |
| SparseCore `ragged_gather` / `_v2` | mosaic_tpu (SparseCore) | Offloads ragged gather to the TPU **SparseCore** (`pltpu.get_tpu_info().sparse_core`, `VectorSubcoreMesh`). | [sparse_core/ragged_gather.py](../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather.py), [ragged_gather_v2.py](../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_v2.py) | experimental — falls back to regular gather when SC absent | **Novel**, HW-gated (v5p / v7x only) |
| SparseCore `ragged_scatter` | mosaic_tpu (SparseCore) | SparseCore ragged scatter; sorts source rows by valid-write count. | [sparse_core/ragged_scatter.py#L253](../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_scatter.py#L253) | experimental | **Novel**, HW-gated |
| SparseCore gather-reduce family | mosaic_tpu (SparseCore) | `gather_reduce`, `ragged_gather_reduce` (+`_v2`), `dense_gather_reduce` — gather rows by index and sum (embedding-lookup / MoE-routing offload). | [sparse_core/gather_reduce.py#L1700](../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/gather_reduce.py#L1700), [ragged_gather_reduce_v2.py](../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py), [dense_gather_reduce.py#L133](../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/dense_gather_reduce.py#L133) | experimental | **Novel**, HW-gated |

### Imported / reference (not Pallas kernels defined here)
- **RPA v2 tuning** — the v2 kernel imports `get_tuned_block_sizes` from *upstream* `jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes`; the local `v2/tuned_block_sizes.py` (1,341 entries) has **no in-repo importer** and reads as a data snapshot / reference, not a wired table.
- **GDN reference** (`gdn/reference/ragged_gated_delta_rule_{ref,chunked,wrapper}.py`) — pure-JAX correctness reference for the GDN kernels; **no `pallas_call`**, not a Pallas kernel.

## Notes

**Autotune surface (verified against source at `8eb31ee`).** These `tuned_block_sizes.py` dictionaries are the closest thing to autotune-result artifacts in the ecosystem and are directly reusable as priors:

| Table | File | Hardware keys | Entries |
|-------|------|---------------|---------|
| RPA v2 | [`ragged_paged_attention/v2/tuned_block_sizes.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py) | v5, v6 | **1,341** |
| RPA v3 | [`ragged_paged_attention/v3/tuned_block_sizes.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py) | v7, v6e, v5e | **~3,254** leaf entries |
| RPA v3 hd64 | [`ragged_paged_attention/v3/tuned_block_sizes_hd64.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes_hd64.py) | v7, v6e, v5e | **~312** |
| quantized_matmul | [`quantized_matmul/tuned_block_sizes.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py) | v6, v7 | **491** |
| megablox gmm/v2 | [`megablox/tuned_block_sizes.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py) | dtype-keyed | ~50 |
| fused_moe v1 | [`fused_moe/v1/tuned_block_sizes.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py) | **v7 only** | ~30 |
| MLA v2 | [`mla/v2/tuned_params.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py) | dataclass configs | ~21 |
| all_gather_matmul | [`collectives/all_gather_matmul_tuned_block_sizes.py`](../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py) | v5e/v6e/v7x | microbench-driven (~6) |

**Total ≈ 5,480 tuned entries** — reconciling the catalog. Subpage 03 (2026-04-23) reported "tpu-inference has ~1,200+ tuned block-size entries," which corresponded to the RPA v2 table specifically. That table is now **1,341**, and the overall surface has grown ~4.5× since — the growth is almost entirely **RPA v3 filling in complete v5e/v6e/v7 tables** (3,254 leaf entries, from "hundreds" in the catalog), plus the new MLA v2 config set and the DeepSeek-V4 experimental kernels. RPA v3's key schema is deeper: `device → page_size → q_{dtype}_kv_{dtype} → q_head-N_kv_head-M_head-D → max_model_len-L-sw-W → (num_kv_pages_per_block, num_queries_per_block)`. `fused_moe` v1 remains **v7-only** with an in-source comment that v5/v6 formulas are still needed.

**Backend.** 100% `mosaic_tpu`. Every kernel imports `from jax.experimental.pallas import tpu as pltpu`; SparseCore kernels additionally use `pltpu.get_tpu_info().sparse_core` + `VectorSubcoreMesh`. No Triton, no `mosaic_gpu` / `plgpu`.

**VMEM budgets** worth recording as concept-level facts: MLA v1 default `100 MiB`; quantized_matmul uses per-generation budgets (v6 vs v7 differ). These are baked into the kernels and tables and are reusable priors.

**Vendoring lineage.** Only two families are vendored: `flash_attention` (`jax-ml/jax` pallas flash attention — identical API, Dao et al. 2022, no attribution header) and the `megablox` GMMs (JAX megablox — matching name/signature, no header, but with a repo-local tuned table). `ragged_kv_cache_update` carries a vLLM SPDX header (first-party). Everything else — RPA v2/v3 (+hd64), MLA v1/v2, fused_moe v1, quantized_matmul (per-tensor + blockwise), all_gather_matmul, hierarchical_reduce_scatter, GDN v1/v2/v3 + triangle_solver, causal_conv1d, SparseCore gather/scatter/reduce, structured_sparse_matmul v1, batched_rpa, and the DeepSeek-V4 set — is novel to this repo. This is the upstream that `sglang-jax` vendors from ("Adapted from …tpu-inference…" headers on its side) and that `aphrodite` reaches via `torch.ops.xla.ragged_paged_attention`.

**Stability / reuse gotchas.** `experimental/batched_rpa` (gated on `USE_BATCHED_RPA_KERNEL=1`, "not fully tested") and `experimental/deepseek_v4/*` are research-grade. SparseCore kernels are **hardware-gated** — present on v5p/v7x, absent on v5e/v6e (explicit fallback to regular gather). `structured_sparse_matmul` is software-emulated N:M with open TODOs (int4, subelement masking) and only wins on memory-bound shapes. The local RPA v2 tuned table is likely stale relative to the upstream table the kernel actually reads.

## See also
- [tpu-inference — overview](overview.md) — the repo's grounded wiki front door.
- [tpu_inference/kernels/mla/v2/kernel — grounded concept](concepts/tpu_inference-kernels-mla-v2-kernel.md) — per-symbol page for the MLA v2 kernel.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — cross-repo Pallas catalog; this repo is the authoritative author, detailed in [subpage 03 — inference engines](../../analyses/pallas-kernel-directory/03-inference-engines.md).

## Sources
- `raw/code/tpu-inference/tpu_inference/kernels/**` at commit `8eb31ee` (read: RPA v2/v3/hd64, MLA v1/v2 + transpose, flash_attention, batched_rpa, deepseek_v4/*, fused_moe v1, megablox gmm/gmm_v2, quantized_matmul kernel/blockwise, all_gather_matmul, hierarchical_reduce_scatter, gdn v1/v2/v3 + triangle_solver + reference, causal_conv1d, sparse_core/*, structured_sparse_matmul v1, ragged_kv_cache_update, and all `tuned_block_sizes.py` / `tuned_params.py` tables).
