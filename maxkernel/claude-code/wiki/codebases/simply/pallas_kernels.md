---
title: "simply — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, simply]
created: 2026-07-04
updated: 2026-07-04
---

simply (google-deepmind's minimal JAX LLM research codebase) defines **2 locally-authored Pallas
kernels**, both `mosaic_tpu`: (1) a full hand-written **ragged paged-attention** kernel vendored from
vLLM's tpu-inference v3 and modified in-repo, and (2) a **novel ragged-all-to-all (ra2a)** RDMA
collective kernel for expert-parallel MoE dispatch, using raw async DMAs + semaphores. Its MoE
grouped-matmul (`gmm`) is *not* a local kernel — it dispatches to upstream `megablox.gmm` or
`jax.lax.ragged_dot` (§ Imported below). **Note the divergence from the 2026-04-23 directory**, which
recorded simply as a wrapper-only repo with no kernels of its own — the current checkout
(commit `b1064e6`) has since brought both kernels in-tree.

## Kernels

### Attention — paged / ragged
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `ragged_paged_attention` (`_ragged_paged_attention_kernel` → `_ragged_paged_attention_kernel_loop`) | mosaic_tpu | Ragged (variable-length, concatenated-batch), paged-KV-cache attention with **mixed prefill + decode in one launch**. Per-sequence dispatch into DECODE / PREFILL / MIXED cases, each with independent tunable block sizes (`d/p/m_block_sizes = (bq, bkv, bq_c, bkv_c)`); **double-buffered async-copy pipeline** prefetches next KV/Q blocks from HBM while computing online-softmax flash-attention over the current ones, async-writes outputs back. Supports GQA, causal + sliding-window + soft-cap masking, quantized KV (`k_scale`/`v_scale`), in-kernel KV-cache update, and context sharding. | [kernels/ragged_paged_attention.py](../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1931) · [concept](concepts/simply-kernels-ragged_paged_attention.md) | stable (production serving) | vendored-from-vllm-tpu-inference (v3) + local mods |

### MoE — collective / dispatch
| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `make_ra2a_3d` ragged-all-to-all (`_ra2a_3d_kernel_async`, `start`/`wait` phases) | mosaic_tpu | Hand-written **async ragged-all-to-all collective** for expert-parallel MoE: moves each token to the device shard holding its assigned expert. Two `pl.pallas_call`s — a non-blocking `start` (issues RDMA DMAs into aliased `dst_ref` under `pltpu.SemaphoreType.DMA`, `memory_space=ANY`, `collective_id`) and a `wait` that blocks on the DMA semaphores — so the transfer overlaps the *previous* chunk's expert compute in the software-pipelined MoE loop. Ragged per-shard send/recv sizes via SMEM offset arrays. Alternative to XLA's `jax.lax.ragged_all_to_all`. | [utils/moe_lib.py](../../../raw/code/simply/simply/utils/moe_lib.py#L202) · [concept](concepts/simply-utils-moe_lib.md) | stable (training) | novel |

**Imported (not defined here):**
- **`gmm` (grouped/expert matmul)** — `model_lib.gmm` dispatches on `gmm_impl`: `'megablox'` →
  upstream `megablox.gmm` (jax Pallas), or `'ragged_dot'` → `jax.lax.ragged_dot` (XLA). No local
  grouped-matmul Pallas kernel; simply supplies only the *routing/communication* around it (the ra2a
  kernel above).
- **`ref_ragged_paged_attention`** — pure-JAX per-sequence reference in the same file; the correctness
  oracle + CPU fallback for the Pallas kernel, not itself a kernel.

## Notes
- **Vendoring lineage (attention):** `ragged_paged_attention.py` header cites
  `github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`
  and enumerates the local deltas verbatim: *"Added `update_kv_cache` option… Added `save_residuals`
  option to return logsumexp… Skipped sequences with empty queries or empty KVs… Caller feed page
  indices in 2d… Round-robin style per sequence context sharding… Added support for megacore."* So the
  numerics/pipeline are upstream v3; the deltas are integration + a context-sharding scheme.
- **Novelty (MoE):** the ra2a kernel is simply's own — a raw-DMA/semaphore Mosaic collective that XLA's
  `ragged_all_to_all` can't overlap as tightly; it exists specifically to hide cross-device token
  movement behind the previous chunk's FLOPs in `run_moe_pipelined_shard_map`'s `_overlap_fn`.
- **Autotune surface (attention):** block sizes are `static_argnames` on the top-level
  `ragged_paged_attention` (`d_block_sizes`/`p_block_sizes`/`m_block_sizes`, `chunk_prefill_size`,
  `vmem_limit_bytes`); the file carries VMEM/SMEM estimators (`get_vmem_estimate_bytes`,
  `get_smem_estimate_bytes`) and a `has_bank_conflicts` helper. `donate_argnames` donates
  queries/keys/values/kv_cache for in-place update.
- **Reconciliation with the cross-repo directory (2026-04-23):** subpage 04-research-labs records simply
  as *"serving wrapper, not a kernel"* that merely `import`s upstream RPA, with the contribution being an
  `autotune_block_sizes` DMA-overhead heuristic (~0.5 MiB virtual bytes). **This is now stale** — the
  current checkout (`b1064e6`, 2026-06-13) vendors the full v3 kernel in-tree under `simply/kernels/`
  *and* adds the novel ra2a MoE kernel, neither of which the 2026-04-23 survey saw. The directory's
  claim that "both [simply, graphcast] wrap upstream JAX Pallas kernels rather than defining their own"
  no longer holds for simply.

## See also
- [simply — overview](overview.md) — the repo's grounded wiki front door.
- [simply.kernels.ragged_paged_attention](concepts/simply-kernels-ragged_paged_attention.md) — deep-dive on the double-buffered paged-attention kernel.
- [simply.utils.moe_lib](concepts/simply-utils-moe_lib.md) — deep-dive on expert-parallel MoE + the ra2a collective kernel.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — cross-repo catalog; simply appears in [subpage 04 — research labs](../../analyses/pallas-kernel-directory/04-research-labs.md) (as a wrapper-only entry that this page supersedes).
