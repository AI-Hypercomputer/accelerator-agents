---
title: "ejkernel — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, ejkernel]
created: 2026-07-04
updated: 2026-07-04
---

ejkernel carries the broadest community-authored TPU Pallas surface: **~25 kernel families** under `ejkernel/kernels/_pallas/tpu/**`, every one compiled through `jax.experimental.pallas.tpu` (`pl.pallas_call` → **`mosaic_tpu`** backend). All headers read `Copyright 2025/2026 The EasyDeL/ejKernel Author @erfanzar`, but the source has diverged materially from the 2026-04-23 [directory catalog](../../analyses/2026-04-23-pallas-kernel-directory.md): a large share of the serving/MoE kernels are now **explicitly adapted or vendored** from upstream JAX, JetStream, and Google `tpu_inference` (the catalog's "all Erfan-authored" framing no longer holds — see Notes). Novel-to-ejkernel work concentrates in the training attention family (flash/MLA/DSA), the fused-collective matmul pair (all_gather ∘ matmul, reduce_scatter ∘ matmul), the fused loss kernels, and the gated-delta-rule (GDN linear-attention) suite. Commit `f1b5eb128f`.

## Kernels

### Attention — dense / flash (training)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `flash_attention` | mosaic_tpu | O(N)-memory tiled MHA/GQA; fwd + separate dk/dv and dq/ds bwd passes; `BlockSizes` (major/minor) is the tuning dial | [_pallas_impl_fwd.py#L631](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_fwd.py#L631), [_pallas_impl_bwd.py#L434](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_pallas_impl_bwd.py#L434) | experimental | novel |
| `flash_mla` | mosaic_tpu | Flash attention specialized for DeepSeek Multi-head Latent Attention (compressed KV latent); fwd + bwd | [_pallas_impl_fwd.py#L465](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_fwd.py#L465), [_pallas_impl_bwd.py#L436](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L436) | experimental | novel |
| `deepseek_attn` | mosaic_tpu | DeepSeek Sparse Attention (DSA, V3.2): MLA + Lightning-Indexer top-k token selection; delegates the core matmul to `_flash_mla_pallas_call` and adds the dynamic index path | [_pallas_impl_fwd.py#L77](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L77) | research | novel (wraps `flash_mla`) |
| `blocksparse_attention` | mosaic_tpu | Splash-style block-sparse attention; a mask-algebra pass classifies blocks (empty/partial/full) into prefetch tables the kernel uses to skip masked work; fwd + dq + full-grad bwd | [_kernel.py#L1260](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/blocksparse_attention/_kernel.py#L1260) | experimental | based-on Google Splash |
| `ring_attention` | mosaic_tpu | Sequence-parallel long context: Splash attention + K/V ring rotation (`ppermute`) merged with online log-sum-exp; orchestration layer, not a from-scratch kernel | [_pallas_impl_fwd.py#L1](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.py), [_pallas_impl_bwd.py](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ring_attention/_pallas_impl_bwd.py) | experimental | wraps Google Splash |

### Attention — paged / ragged / MLA (serving)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `page_attention` | mosaic_tpu | vLLM-style paged-KV decode attention (fwd only) | [_interface.py#L312](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_interface.py#L312) | experimental | vendored-from-jax (`pallas/ops/tpu/paged_attention`) |
| `prefill_page_attention` | mosaic_tpu | Chunked prefill over paged KV (fwd only) | [_interface.py#L184](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py#L184) | experimental | vendored-from-JetStream (`chunked_prefill_attention`) |
| `ragged_page_attention_v2` | mosaic_tpu | Mixed variable-length ragged paged attention with online softmax; decode continuous-batching (fwd only) | [_interface.py#L221](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_interface.py#L221) | experimental | based-on Google ragged attn |
| `ragged_page_attention_v3` | mosaic_tpu | Mixed decode/prefill in one launch over paged KV; auto-selects head-dim-64 vs 128 kernel; tuned-block table | [_pallas_impl_fwd.py#L1724](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1724) | experimental | based-on Google/vLLM |
| `ragged_page_attention_v3` (h64) | mosaic_tpu | Head-dim-64 specialization of v3 (partitions sequences by type; `arbitrary` dim-semantics + `vmem_limit_bytes`) | [_pallas_impl_fwd_h64.py#L1575](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L1575) | experimental | based-on Google/vLLM |
| `ragged_decode_attention` | mosaic_tpu | Single-token ragged decode attention (fwd only); `("parallel","arbitrary")` grid | [_pallas_impl_fwd.py#L221](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L221) | experimental | novel |
| `multi_latent_ragged_page_attention` (v1) | mosaic_tpu | MLA paged decode (DeepSeek latent KV) — first iteration (fwd only) | [_pallas_impl_fwd.py#L1288](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L1288) | experimental | novel |
| `multi_latent_ragged_page_attention_v2` | mosaic_tpu | MLA paged decode v2 with explicit async pipelining; up to three `pallas_call`s (decode/prefill/mixed) | [_pallas_impl_fwd.py#L1360](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2/_pallas_impl_fwd.py#L1360) | experimental | adapted-from tpu_inference MLA v2 |
| `fused_conv_decode` | mosaic_tpu | Fused short-convolution decode step (causal-conv1d state update + output) for GDN/Mamba-style layers; `("parallel",)` grid | [_pallas_impl_fwd.py#L123](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py#L123) | experimental | novel |

### Linear attention / SSM — gated delta rule (GDN)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `gated_delta_rule` | mosaic_tpu | Gated-DeltaNet linear-attention training kernel; four dispatch paths on `seg_ids`; fwd + bwd, `("parallel","parallel")` grid | [_pallas_impl_fwd.py#L461](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py#L461), [_pallas_impl_bwd.py#L308](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py#L308) | experimental | novel |
| `ragged_gated_delta_rule` | mosaic_tpu | Ragged GDR decode; per-token GDR updates fused in-kernel — README claims ~3.6× over XLA (fwd only) | [_pallas_impl_fwd.py#L110](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py#L110) | experimental | novel |
| `ragged_gated_delta_rule_v2` | mosaic_tpu | v2 GDN decode: fused-GDN-decode + recurrent-scan-v2 + policy scheduler; chunked mixed-prefill vs decode branch on `request_distribution`; `vmem_limit_bytes` / `disable_bounds_checks` tuning | [_pallas_impl_fwd.py#L1086](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_pallas_impl_fwd.py#L1086), [_fused_gdn_decode.py#L474](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L474) | experimental | adapted-from tpu-inference branch |
| `gated_delta_rule_grouped_decode` | mosaic_tpu | Grouped (multi-request) GDN decode step; `("parallel",)` grid | [_pallas_impl_fwd.py#L163](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py#L163) | experimental | novel |

### MoE / grouped matmul

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `grouped_matmul` (v1) | mosaic_tpu | Per-expert (group) GEMM in one launch; group-boundary row masking; 128×128 tiles | [_pallas_impl.py#L677](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L677) | experimental | novel (megablox-style) |
| `grouped_matmulv2` | mosaic_tpu | 2nd-gen grouped GEMM; `("parallel","arbitrary","arbitrary")` grid | [_pallas_impl.py#L422](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L422) | experimental | novel |
| `grouped_matmulv3` | mosaic_tpu | 3rd-gen MoE FFN kernel: per-expert matmul + fused activation/dequant, causal group row-mask, dtype-driven `tile_m` selection | [_pallas_impl.py#L1382](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1382) | experimental | adapted-from tpu_inference `gmm_v2` |

### Quantized matmul

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `quantized_matmul` | mosaic_tpu | Quantized GEMM with packed / predecode paths and fp32 accumulation; fwd + bwd + shared core; GemLite-style kernel-family selection | [_pallas_impl_core.py#L1005](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L1005), [_pallas_impl_fwd.py#L233](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L233), [_pallas_impl_bwd.py#L252](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L252) | experimental | novel |

### Collectives-fused matmul (tensor parallelism)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `all_gather_matmul` | mosaic_tpu | Fused all-gather ∘ matmul over a device ring using async-copy descriptors (`AsyncCopyDescriptor`) to overlap collective with compute | [_pallas_impl.py#L707](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L707) | experimental | novel |
| `reduce_scatter_matmul` | mosaic_tpu | Fused matmul ∘ reduce-scatter (complement of the above); `collective_id` + `("arbitrary",)` semantics | [_pallas_impl.py#L668](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L668) | experimental | novel |

### Fused loss / elementwise

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `fused_cross_entropy` | mosaic_tpu | Chunked-logit fused cross-entropy (online max/sum-exp, never materializes full logits); fwd + bwd, `("parallel",...)` grid | [_pallas_impl_fwd.py#L248](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_fwd.py#L248), [_pallas_impl_bwd.py#L309](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_cross_entropy/_pallas_impl_bwd.py#L309) | experimental | novel |
| `fused_kl_divergence` | mosaic_tpu | Chunked fused KL divergence over teacher/student logits (two online LSE passes); fwd + bwd | [_pallas_impl_fwd.py#L361](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_fwd.py#L361), [_pallas_impl_bwd.py#L345](../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_kl_divergence/_pallas_impl_bwd.py#L345) | experimental | novel |

## Notes

**Vendoring lineage (divergence from the 2026-04-23 catalog).** The catalog concluded "all TPU kernels are Erfan-authored." The current source contradicts that for the serving/MoE tier — in-file docstrings name upstream origins explicitly:
- `page_attention` — header URL points to `jax-ml/jax .../pallas/ops/tpu/paged_attention/paged_attention_kernel.py`.
- `prefill_page_attention` — header URL points to AI-Hypercomputer **JetStream** `chunked_prefill_attention.py`.
- `ragged_page_attention_v2` — "Based on Google's ragged attention implementation with online softmax."
- `ragged_page_attention_v3` — "Based on Google/vLLM implementation with extensions."
- `grouped_matmulv3` — "adapted from upstream TPU inference `gmm_v2`."
- `multi_latent_ragged_page_attention_v2` — "adapted from the upstream TPU inference MLA v2 kernel."
- `ragged_gated_delta_rule_v2/_fused_gdn_decode` — "adapted from the TPU-inference branch."
- `blocksparse_attention` / `ring_attention` — built on Google's **Splash Attention**.

  Genuinely novel-to-ejkernel: the training attention family (`flash_attention`, `flash_mla`, `deepseek_attn` DSA), the fused-collective matmul pair (`all_gather_matmul` / `reduce_scatter_matmul`), the GDN kernels (`gated_delta_rule` + ragged/grouped decode variants + `fused_conv_decode`), the fused loss kernels (`fused_cross_entropy`, `fused_kl_divergence`), and `quantized_matmul`.

**Backend.** Uniform — every kernel calls `pl.pallas_call` on `jax.experimental.pallas.tpu` (`pltpu`), i.e. the **`mosaic_tpu`** backend; there is no `mosaic_gpu`/`triton` code in this subtree (the GPU surface lives under sibling `_pallas/gpu/`, `_cuda/`, `_cute/` — out of scope here).

**Autotune surface.** Tiling is the universal dial: flash `BlockSizes`, grouped/quantized-matmul `tile_m/tile_k/tile_n` (dtype-driven, 128×128 base), paged-attention KV-pages-per-block + queries-per-block heuristics, and `pltpu.CompilerParams` knobs seen in-source — `dimension_semantics` (`parallel`/`arbitrary`), `vmem_limit_bytes` (GDN v2 caps VMEM at 64 MiB), `collective_id` (fused collectives), `disable_bounds_checks`. These feed ejkernel's `ConfigSelectorChain`/autotuner (see [overview](overview.md)).

**Perf claims quoted from source.** `ragged_gated_delta_rule` README/docstring: "~3.6× speedup over the XLA implementation" for ragged GDR decode. `flash_attention`: "Memory-efficient O(N) computation." Others carry only qualitative "TPU-optimized" language; `quantized_matmul` ships XLA-comparison benchmarks under `benchmarks/`.

**Scope note.** The task brief mentioned `scatter_update_pages`; **no such kernel is defined in this repo** (it exists upstream in `tpu_inference`/JAX, not vendored here). The `_h64` head-dim-64 variant belongs to `ragged_page_attention_v3` (listed as its own row above). Reference/ground-truth pure-JAX implementations accompany most kernels (`*reference*` functions) for correctness testing — not counted as kernels.

## See also
- [ejkernel — overview](overview.md) — the repo's grounded wiki front door.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; ejkernel appears in [§6 Community & research repos](../../analyses/pallas-kernel-directory/06-community-research.md) (row 6.1).
- [flash_attention BlockSizes](concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md) — the flash tiling dial + cost model.
- [blocksparse mask algebra](concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_masks.md), [sparse-info pass](concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_info.md), [Splash kernel](concepts/ejkernel-kernels-_pallas-tpu-blocksparse_attention-_kernel.md).
- [ragged paged attention v3 kernel](concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md), [v3 tuned-block table](concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md).
- [MLA ragged paged attention v2 kernel](concepts/ejkernel-kernels-_pallas-tpu-multi_latent_ragged_page_attention_v2-_pallas_impl_fwd.md).
- [grouped matmul v3 kernel](concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md), [quantized matmul core](concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md).
- [ring attention backward](concepts/ejkernel-kernels-_pallas-tpu-ring_attention-_pallas_impl_bwd.md).

## Sources
- raw/code/ejkernel (commit `f1b5eb128f`) — `ejkernel/kernels/_pallas/tpu/**`
