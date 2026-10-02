---
title: "sglang-jax — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, sglang-jax]
created: 2026-07-04
updated: 2026-07-04
---

sglang-jax defines ~20 Pallas kernels under `python/sgl_jax/srt/kernels/**` (plus one multimodal image-hash kernel), almost all `mosaic_tpu` — the sole exception is a GPU-path image hash. The mix is **vendored-majority with a genuinely novel core**: attention/MoE/quant/KV-cache kernels are adapted from [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (v0.11.1), dense flash attention from [jax-ml/jax](https://github.com/jax-ml/jax), and the linear-attention (GLA/KDA) kernels from a now-private `primatrix/pallas-kernel` repo; the genuinely novel-to-sglang-jax kernels are the **EAGLE speculative-decoding tree kernels**, the **fused_moe v2 (Strix-style) kernel**, the **fused SwiGLU MLP**, the **simple_gla fused-decode kernel**, and the **sglang-specific RPA v3 extensions** (custom_mask / attention_sink / xai_temperature). The crown jewel is the **ragged_paged_attention v3 autotune table (`tuned_block_sizes_v3.py`, ~2,080 entries)** — see Notes; this is a new v3-specific split that did not exist at the 2026-04-23 catalog snapshot.

## Kernels

### Attention — paged / ragged / MLA

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `ragged_paged_attention` v3 | mosaic_tpu | Mixed prefill/decode ragged paged attention; 3 split pallas_calls (DECODE/PREFILL/MIXED), per-bq l/m/acc reinit, precise sliding-window skipping, GQA/MQA. sglang extras: `custom_mask` (spec decode), `attention_sink` (streaming), `xai_temperature_len` (Grok), `soft_cap`, `q/k/v_scale`, `cu_kv_lens`-based page offsets | [ragged_paged_attention_v3.py#L329](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L329) (pallas_call [#L1923](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1923)) | stable | vendored-from-tpu-inference (v0.11.1) **+ novel sglang extensions** |
| `ragged_paged_attention` v2 | mosaic_tpu | Prior-gen ragged paged attention (single kernel, `(bq, bkv_p)` blocking); still the fallback path | [ragged_paged_attention.py#L302](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L302) | stable | vendored-from-tpu-inference (v0.11.1) |
| `mla_ragged_paged_attention` v2 | mosaic_tpu | Multi-Head Latent Attention (DeepSeek-class) over ragged pages; adapts tpu-inference MLA v2 to sglang-jax's flat concatenated page-indices layout (no uniform `pages_per_seq` padding) | [mla/v2/kernel.py#L253](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L253) | stable | vendored-from-tpu-inference (MLA v2) |
| `paged_attention` | mosaic_tpu | Non-ragged paged attention (uniform lengths); SPMD via `shard_map`, `xai_temperature_len`, `sm_scale`. Two-pass online-softmax (partial out/exp_sums/max_logits) | [paged_attention.py#L41](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L41) (pallas_call [#L245](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L245)) | stable | vendored-from-jax + Yanko (@Yanko-7) SPMD/xai mods |

### Attention — flash (dense)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| Multimodal `flash_attention` | mosaic_tpu | Dense (non-paged) flash attention for vision/multimodal encoders; SegmentIds masking; ships its own tuned block-spec table | [multimodal/kernels/flash_attention.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/kernels/flash_attention.py) | stable | vendored-from-jax (`ops/tpu/flash_attention.py`) |

### KV-cache update

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `kv_cache_update_kernel` | mosaic_tpu | Packs ragged new-KV slices into the paged KV cache via slot-mapping + semaphore-synced async DMAs; two variants (with/without in-kernel new-KV staging). VMEM budget ~64 MB | [update_kv_cache.py#L69](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L69) (pallas_call [#L176](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py#L176)) | stable | vendored-from-tpu-inference (v0.11.1) |

### MoE / grouped matmul

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `fused_moe` v2 | mosaic_tpu | Strix-style fused EP MoE FFN: tile along intermediate dim, keep tokens persistent in VMEM, double-buffer W1/W3/W2 with deferred W2 wait, in-kernel all-reduce; fp8 dequant-in-VMEM path | [fused_moe/v2/kernel.py#L260](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L260) | experimental | **novel** |
| `fused_moe` v1 | mosaic_tpu | Fused expert routing + two grouped GMMs for MoE (Qwen3-MoE, Mixtral, Bailing-MoE); HBM ring-buffer A2A scratch scaled to device HBM | [fused_moe/v1/kernel.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py) | stable | vendored-from-tpu-inference |
| `megablox_gmm` | mosaic_tpu | Grouped matmul (ragged group boundaries) for MoE experts, `(tm, tk, tn)` tiling | [gmm/megablox_gmm_kernel/gmm.py#L416](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L416) (pallas_call [#L572](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L572)) | stable | vendored-from-tpu-inference / jax megablox |
| `megablox_gmm_v2` | mosaic_tpu | Grouped-matmul v2 (quantized-friendly); separate `tuned_block_sizes.py` | [gmm/megablox_gmm_kernel/gmm_v2.py#L182](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L182) (pallas_call [#L1041](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L1041)) | stable | vendored-from-tpu-inference / jax megablox — see [gmm_v2 catalog](catalog/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.md) |

### Quantized matmul

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `quantized_matmul_kernel` | mosaic_tpu | Weight-quantized matmul (W8A8 / W8A16 / FP8), per-tensor / per-channel scales; VMEM-limit-aware tiling | [quantized_matmul_kernels/kernel.py#L119](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py#L119) (pallas_call [#L228](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py#L228)) | stable | vendored-from-tpu-inference |
| `quantized_matmul_kernel` (blockwise) | mosaic_tpu | Subchannel/blockwise-quantized matmul (`MXU_SIZE=256`), block-scale layout conversion | [quantized_matmul_kernels/blockwise_kernel.py#L25](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L25) (inner [#L139](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L139)) | stable | vendored-from-tpu-inference |

### Norm / GLU / MLP

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `fused_mlp` (gated SwiGLU) | mosaic_tpu | Fuses gate/up projections + SiLU + down projection into one pipelined kernel, eliminating intermediate HBM traffic; sharded (`shard_map`) + padding variant | [fused_mlp.py#L19](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L19) (pallas_call [#L162](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_mlp.py#L162)) | experimental | **novel** |

### SSM / linear attention (GLA / KDA)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `simple_gla` | mosaic_tpu | Gated Linear Attention: fused-recurrent + chunked (`chunk_fwd_h` varlen, `chunk_fwd_o`) forward for SSM-style models; varlen (no per-seq chunk padding) | [simple_gla/simple_gla.py#L321](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L321) (pallas_call [#L532](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L532), [#L717](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L717)) | stable | vendored-from-primatrix/pallas-kernel (now-private) |
| `simple_gla_fused` (decode) | mosaic_tpu | Fused DECODE GLA: replaces JAX `fused_recurrent_simple_gla` scan + gather/scatter shard_maps with a single kernel doing in-kernel async-DMA gather/scatter of the recurrent state, token-level double-buffer, per-head 2D compute | [simple_gla/simple_gla_fused.py#L57](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L57) (pallas_call [#L310](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py#L310)) | experimental | **novel** |
| `kda` (chunked forward) | mosaic_tpu | Kimi Delta Attention chunked forward for varlen sequences: gate cumsum, intra-chunk lower-triangular solve, gated-delta-rule `chunk_h`, `chunk_o`; merges 8 upstream modules into one file (4 pallas_calls) | [kda/kda.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py) | stable | vendored-from-primatrix/pallas-kernel (now-private, rev 3c691ad3) |

### Speculative decoding (EAGLE)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `tree_speculative_sampling_target_only` | mosaic_tpu | EAGLE-style tree speculative-sampling acceptance on-device (traverses retrieve-index / next-token / next-sibling trees over target vs draft probs) | [speculative/tree_speculative_sampling_target_only_kernel.py#L12](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py#L12) | experimental | **novel** |
| `build_eagle_tree_structure` | mosaic_tpu | Builds the EAGLE draft-tree structure + tree attention mask on-device (uses `sync_copy`) | [speculative/build_eagle_tree_structure_kernel.py#L11](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L11) (pallas_call [#L381](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py#L381)) | experimental | **novel** |
| `verify_tree_greedy` | mosaic_tpu | Greedy tree verification of draft tokens against target predictions; emits accept indices + counts | [speculative/verify_tree_greedy_kernel.py#L12](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L12) (pallas_call [#L237](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py#L237)) | experimental | **novel** |

### Other

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `hash_tiles32` / `add_tree_reduce_u64` (`gpu_tensor_hash`) | GPU (triton/mgpu) | MurmurHash-style tensor hashing for multimodal image dedup/caching; **GPU-path only** (falls back to non-Pallas when no GPU) — low TPU-perf relevance | [multimodal/layers/image_hash.py#L28](../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/layers/image_hash.py#L28) (pallas_call [#L168](../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/layers/image_hash.py#L168)) | experimental | **novel** |

**Imported (not defined here):** none of consequence — sglang-jax vendors a *local copy* of every kernel it uses (that is its design). `speculative/kernel.py::filter_finished_cache_loc_kernel` is a plain JAX helper (no `pallas_call`), not a Pallas kernel.

## Notes

**Autotune surface — the crown jewel.** [Subpage 03](../../analyses/pallas-kernel-directory/03-inference-engines.md) (2026-04-23) flagged sglang-jax's `tuned_block_sizes.py` (~2,000+ entries) as the **largest tuning table in any ingested repo**. The source has since **diverged**: the RPA autotune tables are now *split by kernel version*:

| Table | File | Keying | Entries (approx) |
|-------|------|--------|------------------|
| RPA **v3** block sizes | [tuned_block_sizes_v3.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py) | `(stage∈{d,p,m}, sliding_window, q_dtype, kv_dtype, q_heads, kv_heads, head_dim, page_size, max_num_tokens)` → `(bq_sz, bkv_sz, bq_csz, bkv_csz)` | **~2,080** (v5 empty, v6e populated) |
| RPA **v2** block sizes | [tuned_block_sizes.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py) | `(device, q_dtype, kv_dtype, q_head, kv_head, head_dim, page_size, max_num_tokens)` → `(num_kv_pages_per_block, num_queries_per_block)` | **~1,400** (v4+; v4 small-memory special case) |
| quantized_matmul | [quantized_matmul_kernels/tuned_block_sizes.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/tuned_block_sizes.py) | `(tpu_ver, m, n, k, dtypes)` + VMEM-limit budget | **~610** |
| megablox gmm/v2 | [gmm/megablox_gmm_kernel/tuned_block_sizes.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py) | dtype/shape-keyed | ~100 |
| fused_moe v1 | [fused_moe/v1/tuned_block_configs.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py) | 8-tuple workload signature | ~170 |
| fused_moe v2 | [fused_moe/v2/tuned_block_configs.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py) | workload signature | ~20 |
| MLA v2 | [mla/v2/tuned_block_sizes.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/tuned_block_sizes.py) | workload signature | tens |
| multimodal flash | [multimodal/kernels/tuned_block_sizes.py](../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/kernels/tuned_block_sizes.py) | workload signature | separate table |

So the "crown jewel" now lives in **`tuned_block_sizes_v3.py` (~2,080 entries)** for the v3 kernel — plus the older `tuned_block_sizes.py` (~1,400) that drives v2 — a combined ~3,500 RPA autotune entries, the richest autotune corpus among ingested inference repos. Both are generated by `benchmark/kernels/flash_attention/get_block_spec_config_v3.py` / the v2 equivalent. The v3 table is currently **v6e-only** (the `"TPU v5"` sub-dict is empty). SWA layers get their own `sliding_window`-keyed buckets (bkv ≈ sliding_window is typically optimal, so SWA must not share a bucket with full-attention layers). Lookup misses fall back to a heuristic and warn once per unique `(stage, key)`.

**Vendoring lineage.**
- **tpu-inference (v0.11.1)** → RPA v2, RPA v3 (core), MLA v2, `update_kv_cache`, `fused_moe` v1, megablox `gmm`/`gmm_v2`, `quantized_matmul` (both kernels). Each file carries an "Adapted from https://github.com/vllm-project/tpu-inference" header.
- **jax-ml/jax** → multimodal `flash_attention` (direct, not via tpu-inference) and the megablox lineage of `gmm`.
- **JAX + Yanko (@Yanko-7)** → `paged_attention` (JAX Authors 2023 base + sgl-jax SPMD/`xai_temperature_len`/`sm_scale` mods).
- **primatrix/pallas-kernel (now private)** → `simple_gla` (rev 41431b1, +`feat/gla_varlen` patch) and `kda` (rev 3c691ad3). **Provenance risk:** upstream is inaccessible; both files vendor by merging ~8 upstream modules into one, documented in-header.

**Genuinely novel-to-sglang-jax** (deep-follow-up candidates for TPU-perf work):
1. **RPA v3 extensions** — `custom_mask` (speculative decode), `attention_sink` (streaming, prepends sink logits before softmax), `xai_temperature_len` (Grok-style length temperature), `cu_kv_lens`-based page-index offsets. These are grafted onto the vendored v3 kernel body ([ragged_paged_attention_v3.py#L102-L195](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L102) reference, [#L508](../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L508) in-kernel).
2. **fused_moe v2** — Strix-style weight-streaming EP MoE with deferred-W2 double-buffering and in-kernel all-reduce.
3. **fused_mlp** — SwiGLU fully fused, zero intermediate HBM.
4. **simple_gla_fused** — single-kernel GLA decode with in-kernel DMA gather/scatter of recurrent state.
5. **EAGLE speculative-decoding tree kernels** (×3).

**Gotchas for reuse.** RPA v3 default `DEFAULT_VMEM_LIMIT_BYTES = 120 MB`; `DEFAULT_MASK_VALUE = -0.7 × f32.max`. `disable_semaphore_checks` is passed only when the installed `pltpu.CompilerParams` supports it (version-guarded via `inspect`). `simple_gla_fused` documents a Mosaic-TPU limitation: `infer-vector-layout` rejects 1D→3D vector reshapes, forcing per-head 2D tiles rather than an H-vectorised body — relevant if adapting the kernel. `update_kv_cache` requires `head_dim` divisible by 128 (async-DMA slot mapping).

## See also
- [sglang-jax — overview](overview.md) — the repo's grounded wiki front door.
- [fused_moe v2 kernel (catalog)](catalog/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.md) · [fused_moe v1 (catalog)](catalog/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.md) · [megablox gmm_v2 (catalog)](catalog/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.md)
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 03 — inference engines](../../analyses/pallas-kernel-directory/03-inference-engines.md) §3.2 (that snapshot predates the v3 tuning-table split, MLA v2, fused_moe v2, KDA, fused_mlp, and simple_gla_fused documented here).
