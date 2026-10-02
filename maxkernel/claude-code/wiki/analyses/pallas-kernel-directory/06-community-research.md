---
title: "Pallas kernel directory — §6 Community & research-companion repos"
type: analysis
tags: [directory, pallas, kernels, ejkernel, easydel, ringattention, community]
created: 2026-04-23
updated: 2026-04-23
---

Catalog of community and research-companion Pallas repositories — smaller repos, often paper companions or individual-author kernel libraries. Several advertise "Pallas" but target Triton/Mosaic-GPU, not TPU — flagged explicitly. Part of [2026-04-23 Pallas kernel directory](../2026-04-23-pallas-kernel-directory.md).

## Repo metadata (activity, license, stars)

| Repo | Stars | Last push | License | Notes |
|---|---|---|---|---|
| [erfanzar/ejkernel](https://github.com/erfanzar/ejkernel) | 22 | 2026-04-11 | Apache-2.0 | Active, sole maintainer Erfan Zare Chavoshi |
| [erfanzar/EasyDeL](https://github.com/erfanzar/EasyDeL) | 355 | 2026-04-22 | Apache-2.0 | Very active; consumes ejkernel |
| [erfanzar/jax-flash-attn2](https://github.com/erfanzar/jax-flash-attn2) | 34 | 2025-03-04 | Apache-2.0 | Stale ~13 months; superseded by ejkernel/EasyDeL |
| [haoliuhl/ringattention](https://github.com/haoliuhl/ringattention) | 770 | 2025-10-13 | Apache-2.0 | Original paper repo, low-frequency maintenance |
| [lengstrom/flashback](https://github.com/lengstrom/flashback) | 11 | 2025-03-28 | none | Research companion, stale; **GPU-only despite name** |
| [zhixuan-lin/gla-jax](https://github.com/zhixuan-lin/gla-jax) | 8 | 2025-01-26 | MIT | Research companion, stale; **GPU-only** |
| [sqtian/PALLAS_TPU_KERNEL_MATMUL](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL) | 87 | 2025-07-28 | MIT | Pedagogical, stale |
| [Essential-AI/maxtext-external](https://github.com/Essential-AI/maxtext-external) | 0 | 2025-01-29 | Apache-2.0 | Snapshot of maxtext, stale |
| [labyrinth-ssr/tpu-research](https://github.com/labyrinth-ssr/tpu-research) | 0 | 2026-02-06 | Apache-2.0 | **Vendors Google `tpu_inference` kernels** |
| [ashioyajotham/recompute_dont_restore](https://github.com/ashioyajotham/recompute_dont_restore) | 1 | 2026-04-21 | none | Pedagogical, freshly active |
| [AlexG1105/mamba2-jax-pallas](https://github.com/AlexG1105/mamba2-jax-pallas) | 0 | 2026-03-05 | Apache-2.0 | **GPU (Mosaic GPU), not TPU** |
| [rdyro/gpu_ragged_dot](https://github.com/rdyro/gpu_ragged_dot) | 3 | 2026-01-31 | Apache-2.0 | **GPU-targeted** |
| [rdyro/moe_in_jax](https://github.com/rdyro/moe_in_jax) | 3 | 2026-04-19 | none | **TPU + SparseCore** |
| [jondeaton/ring-attention-jax-pallas](https://github.com/jondeaton/ring-attention-jax-pallas) | — | — | — | **404 confirmed — repo does not exist** |

## 6.1 erfanzar/ejkernel

Single-author community kernel library positioning as "production-grade." Claims no AI-generated code (only AI-generated docs). Broadest community TPU Pallas surface: attention family, paged/ragged inference, linear-attention (GDR), collectives-fused matmul, quantized/grouped matmul. Kernels under `ejkernel/kernels/_pallas/tpu/<kernel>/{_interface.py, _pallas_impl_fwd.py, _pallas_impl_bwd.py}`. Every TPU kernel header: `Copyright 2025 The EasyDeL/ejKernel Author @erfanzar`. **Ring-attention is explicitly a Splash wrapper, NOT a from-scratch ring kernel.**

| Kernel | Source | Backend | Stability | Perf claim | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| flash_attention | [flash_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/flash_attention) | `mosaic_tpu` | experimental | "O(N) memory complexity" | training/inference MHA | EasyDeL `operations/kernels/flash_attention.py` | fwd+bwd split |
| flash_mla | [flash_mla/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/flash_mla) | `mosaic_tpu` | experimental | none | DeepSeek-style MLA training | EasyDeL MLA path | fwd+bwd |
| blocksparse_attention | [blocksparse_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/blocksparse_attention) | `mosaic_tpu` | experimental | none | sparse-mask training | EasyDeL `blocksparse_attention.py` | explicit mask module |
| deepseek_attn | [deepseek_attn/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/deepseek_attn) | `mosaic_tpu` | experimental | none | DeepSeek attention | EasyDeL | fwd+bwd |
| page_attention | [page_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/page_attention) | `mosaic_tpu` | experimental | none | decode inference | EasyDeL inference | fwd-only |
| prefill_page_attention | [prefill_page_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/prefill_page_attention) | `mosaic_tpu` | experimental | none | paged prefill | EasyDeL | fwd-only |
| ragged_page_attention_v2 | [ragged_page_attention_v2/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2) | `mosaic_tpu` | experimental | none | continuous batching decode | EasyDeL | parallels Google's RPA v2 |
| ragged_page_attention_v3 | [ragged_page_attention_v3/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3) | `mosaic_tpu` | experimental | none | continuous batching decode | EasyDeL | includes dedicated `_h64` variant |
| ragged_decode_attention | [ragged_decode_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/ragged_decode_attention) | `mosaic_tpu` | experimental | none | decode-only | EasyDeL | fwd-only |
| multi_latent_ragged_page_attention | [multi_latent_ragged_page_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention) | `mosaic_tpu` | experimental | none | MLA paged decode | EasyDeL | fwd-only |
| multi_latent_ragged_page_attention_v2 | [multi_latent_ragged_page_attention_v2/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention_v2) | `mosaic_tpu` | experimental | none | MLA paged decode v2 | EasyDeL | fwd-only |
| gated_delta_rule | [gated_delta_rule/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/gated_delta_rule) | `mosaic_tpu` | experimental | README: "3.6x speedup" for ragged GDR decode | GDN-style linear attention training | EasyDeL GDR op | fwd+bwd |
| ragged_gated_delta_rule | [ragged_gated_delta_rule/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule) | `mosaic_tpu` | experimental | see above | GDR inference | EasyDeL | fwd-only |
| grouped_matmul (v1/v2/v3) | [grouped_matmul/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/grouped_matmul), [grouped_matmulv2/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/grouped_matmulv2), [grouped_matmulv3/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/grouped_matmulv3) | `mosaic_tpu` | experimental | none | MoE / grouped gemm | EasyDeL MoE | three staged iterations in-tree |
| quantized_matmul | [quantized_matmul/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/quantized_matmul) | `mosaic_tpu` | experimental | benchmarks vs XLA in `benchmarks/` | quantized gemm | EasyDeL | fwd+bwd+core |
| all_gather_matmul | [all_gather_matmul/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/all_gather_matmul) | `mosaic_tpu` | experimental | none | TP-fused collective | EasyDeL | fused all-gather ∘ matmul |
| reduce_scatter_matmul | [reduce_scatter_matmul/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul) | `mosaic_tpu` | experimental | none | TP-fused collective | EasyDeL | complements all_gather_matmul |
| ring_attention | [ring_attention/](https://github.com/erfanzar/ejkernel/tree/main/ejkernel/kernels/_pallas/tpu/ring_attention) | `mosaic_tpu` | experimental | none | sequence-parallel long ctx | EasyDeL RingAttn | **Docstring: "wraps Splash Attention with ring communication topology" — orchestration, not novel kernel** |

Parallel GPU surface under `ejkernel/kernels/_pallas/gpu/` (ragged_decode_attention, scaled_dot_product_attention) + `_cuda/`, `_cute/` (FlashAttention, quantized_matmul, unified_attention, chunked_prefill_paged_decode, ragged_page_attention_v3, blocksparse_attention) — out of scope for TPU catalog but noted.

## 6.2 erfanzar/EasyDeL

Training/serving framework wrapping ejkernel into operation registry. TPU kernels under `easydel/operations/kernels/` are thin adapters re-exporting ejkernel implementations. Kernel folder lists `BlockSparseAttn` (described: "Splash Attention for TPU using Pallas kernels"), `FlashAttn`, `RingAttn`, `UnifiedAttn` ("vLLM-style" continuous-batching prefill+decode), `RaggedPageAttnV2/V3`, `AutoRegressiveDecodeAttn`, `SSM1Op`, `SSM2Op`, `GatedDeltaRuleOp`, `KernelDeltaAttnOp` (Kimi Linear).

| File | Source | Role |
|---|---|---|
| `operations/kernels/flash_attention.py` | [flash_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/flash_attention.py) | Registry wrapper for ejkernel flash_attention |
| `operations/kernels/ring_attention.py` | [ring_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/ring_attention.py) | Wraps ejkernel ring_attention (Splash-based) |
| `operations/kernels/blocksparse_attention.py` | [blocksparse_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/blocksparse_attention.py) | Splash/BlockSparse wrapper |
| `operations/kernels/ragged_page_attention.py` | [ragged_page_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/ragged_page_attention.py) | Paged attention wrapper |
| `operations/kernels/multi_latent_ragged_page_attention.py` | [multi_latent_ragged_page_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/multi_latent_ragged_page_attention.py) | MLA paged wrapper |
| `operations/kernels/{gated_delta_rule,inference_gdn,kda}.py` | [gated_delta_rule.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/gated_delta_rule.py), [inference_gdn.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/inference_gdn.py), [kda.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/kda.py) | GDN/KDA ops |
| `operations/kernels/{ssm1,ssm2}.py` | [ssm1.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/ssm1.py), [ssm2.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/ssm2.py) | Mamba-1 / Mamba-2 ops |
| `operations/kernels/glm_moe_dsa_indexer.py` | [glm_moe_dsa_indexer.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/glm_moe_dsa_indexer.py) | GLM MoE DSA index op — **unique to EasyDeL** |
| `operations/kernels/unified_attention.py` | [unified_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/unified_attention.py) | Unified prefill+decode |
| `operations/kernels/paged_flash_attention.py` | [paged_flash_attention.py](https://github.com/erfanzar/EasyDeL/blob/main/easydel/operations/kernels/paged_flash_attention.py) | Paged flash variant |

Stability: experimental — kernel set churns weekly. Apache-2.0.

## 6.3 erfanzar/jax-flash-attn2

Predecessor of ejkernel — multi-backend FA2 wrapper. Current tree (`jax_flash_attn2/flash_attention_triton/`, `flash_attention_jax/`) shows only Triton + reference-JAX backends; advertised Pallas path is not present and appears migrated to ejkernel. **Stale ~13 months; treat as historical.**

| Module | Source | Backend | Stability | Notes |
|---|---|---|---|---|
| flash_attention (Triton) | [flash_attention_triton/](https://github.com/erfanzar/jax-flash-attn2/tree/main/jax_flash_attn2/flash_attention_triton) | `triton` | experimental | GPU only |
| flash_attention (JAX ref) | [flash_attention_jax/](https://github.com/erfanzar/jax-flash-attn2/tree/main/jax_flash_attn2/flash_attention_jax) | `xla` | research | Reference, not a Pallas kernel |

## 6.4 haoliuhl/ringattention

**Canonical public Pallas Ring Attention**, Liu/Zaharia/Abbeel ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889)). Single file: `ringattention/ringattention_pallas_tpu.py`. Confirmed via source read:

- Uses `jax.experimental.pallas.tpu` (pltpu). **Does NOT use `pltpu.emit_pipeline`**; uses `pltpu.PrefetchScalarGridSpec` and a manual grid.
- Accepts `axis_name`, rotates K/V with `lax.ppermute(..., perm=[(i, (i+1) % axis_size) ...])` — unidirectional ring.
- Supports causal masking via `below_or_on_diag` helper (imported from sibling `ringattention_jax.py`). Straight causal mask check against current (q_block, k_block) position in the ring iteration.
- **NOT a zig-zag / load-balanced causal variant** (Brandon et al. 2023). The load imbalance that zig-zag fixes is not addressed here.
- No TPU-version check in-source; targets TPU generically via pltpu. README lists no v4/v5e/v5p/v6e matrix.
- Only top-level API: `ring_flash_attention_tpu` (fwd/bwd private helpers). **No inference-specialized Pallas variant** — `ringattention_jax_inference.py` is pure JAX.

| Kernel | Source | Backend | Stability | Perf claim | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| `ring_flash_attention_tpu` | [ringattention/ringattention_pallas_tpu.py](https://github.com/haoliuhl/ringattention/blob/main/ringattention/ringattention_pallas_tpu.py) | `mosaic_tpu` | research (paper companion) | README: "train with tens of millions of tokens in context size without adding any communication or computation overhead" — **no measured numbers** | sequence-parallel long-context training | Used by `LargeWorldModel` family (same authors); vendored into many downstream forks | Causal via straight ppermute ring; **no zig-zag variant present**; 770⭐, Apache-2.0 |

## 6.5 lengstrom/flashback

Backward-over-backward FlashAttention — rare Pallas higher-order-gradient example, motivated by differentiating through training ([arXiv:2503.13751](https://arxiv.org/abs/2503.13751)). **Confirmed GPU-only** via headers (`from jax.experimental.pallas import triton as plgpu`, `compiler_params=dict(triton=dict(num_stages=..., num_warps=...))`). **No TPU Pallas here.** Included because task called it out explicitly. No LICENSE file.

| Kernel | Source | Backend | Stability | Perf claim | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| flash_softmax fwd/bwd/bwd² | [flashback/attentions/flash_softmax_kernels.py](https://github.com/lengstrom/flashback/blob/main/flashback/attentions/flash_softmax_kernels.py) + `flash_softmax_bob_kernel.py` | `triton` (Pallas) | research | README: "double backwards roughly as fast as naive attention, with significant memory savings" | meta-learning, hyperparam optim, differentiating through training | authors' own research | "barely modified from original jax source code" per header; bob = backward-over-backward |
| flash_sigmoid fwd/bwd/bwd² | [flashback/attentions/flash_sigmoid_kernels.py](https://github.com/lengstrom/flashback/blob/main/flashback/attentions/flash_sigmoid_kernels.py) + `flash_sigmoid_op.py` | `triton` (Pallas) | research | README: "very fast double backwards, strong walltime improvements over naive" | sigmoid-attention higher-order grads | authors' own research | Novel compared to stock FA |

## 6.6 zhixuan-lin/gla-jax

Gated Linear Attention from Yang et al. ([arXiv:2312.06635](https://arxiv.org/abs/2312.06635)). `ops.py` defines **Triton-backed** Pallas kernels (`plgpu` / Triton compiler_params, `num_warps=4`). **No TPU Pallas despite README claim.**

| Kernel | Source | Backend | Stability | Perf claim | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| cumsum_kernel | [ops.py](https://github.com/zhixuan-lin/gla-jax/blob/main/ops.py) | `triton` | research | none | bidirectional cumsum in GLA | `gla.py` benchmark | `max_items=16384`, num_warps=4 |
| rnn_kernel | [ops.py](https://github.com/zhixuan-lin/gla-jax/blob/main/ops.py) (same file as `cumsum_kernel`) | `triton` | research | none | parallel-scan linear RNN with log-space gates | `gla.py` | `max_items=8192`, key-value outer loop |

8⭐, MIT, last push Jan 2025 — essentially abandoned.

## 6.7 sqtian/PALLAS_TPU_KERNEL_MATMUL

Pedagogical progression of TPU matmul kernels with plotted benchmarks. MIT, 87⭐, stale since 2025-07-28. README gives sizes each version handles + qualitative verdicts.

| Kernel | Source | Backend | Stability | Perf claim | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| matmul_v1 (naive) | [src/kernels/matmul_v1.py](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL/blob/main/src/kernels/matmul_v1.py) | `mosaic_tpu` | pedagogical | README: "limited to M=K=N ≤ 1024" | teaching single-VMEM-tile matmul | `visualize_performance.py` | loads whole matrices into VMEM |
| matmul_v2 (parallel rows/cols) | [.../matmul_v2_parallel.py](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL/blob/main/src/kernels/matmul_v2_parallel.py) | `mosaic_tpu` | pedagogical | "M=K=N ≤ 2048, slower than jnp.matmul" | teach 2D grid partition | bench | parameter N controls split |
| matmul_v3 (3D block grid) | [.../matmul_v3_block.py](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL/blob/main/src/kernels/matmul_v3_block.py) | `mosaic_tpu` | pedagogical | "enables all sizes, poor performance vs baseline" | teach K-accumulation | bench | blocks `[bm, bk, bn]` |
| matmul_v4 (optimal block) | [.../matmul_v4_optimal_block_size.py](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL/blob/main/src/kernels/matmul_v4_optimal_block_size.py) | `mosaic_tpu` | pedagogical | "`(bm,bk,bn) = (512,512,512)` tuned for TPU MXU" | teach MXU-aware blocking | bench | — |
| matmul_v5 (bf16/int8 + fp32 accum) | [.../matmul_v5_quant_prec.py](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL/blob/main/src/kernels/matmul_v5_quant_prec.py) | `mosaic_tpu` | pedagogical | "`(bm,bk,bn)=(512,1024,1024)` nearly matches XLA" | teach precision choice | bench | bf16/int8 inputs, fp32 accum |
| matmul_batch_matmul | [.../matmul_batch_matmul.py](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL/blob/main/src/kernels/matmul_batch_matmul.py) | `mosaic_tpu` | pedagogical | none | teach batched variant | bench | batch-axis extension |

## 6.8 Essential-AI/maxtext-external

Snapshot of upstream MaxText. `MaxText/kernels/` contains `megablox/` (common.py, gmm.py, ops.py) and `ragged_attention.py`. File headers attribute "Copyright 2024 Google LLC" — **verbatim from upstream MaxText, not Essential-AI originals.** Last push 2025-01-29, 0⭐. Essential-AI appears to have used this as a pinned vendor, not a fork with novel kernels. **No original Pallas kernels beyond upstream.**

| Kernel | Source | Backend | Stability | Notes |
|---|---|---|---|---|
| megablox/gmm | [MaxText/kernels/megablox/gmm.py](https://github.com/Essential-AI/maxtext-external/blob/main/MaxText/kernels/megablox/gmm.py) | `mosaic_tpu` | stable (upstream) | Vendored from google-deepmind/maxtext; header "Copyright 2024 Google LLC" |
| ragged_attention (ragged_mqa, ragged_mha, ragged_gqa) | [MaxText/kernels/ragged_attention.py](https://github.com/Essential-AI/maxtext-external/blob/main/MaxText/kernels/ragged_attention.py) | `mosaic_tpu` | stable (upstream) | Vendored |

## 6.9 labyrinth-ssr/tpu-research

Experimental research repo. `tpu_inference_kernel/` subtree is **a wholesale vendor of Google's `tpu_inference` library** — every kernel file carries `SPDX-License-Identifier: Apache-2.0`, `Copyright 2025 Google LLC` headers; files import from `tpu_inference.kernels.*`. Effectively a mirror. Also contains `tokamax` submodule + two original sub-projects (`delta_attention_comparison/` for Kimi delta-attention bench, `matmul-shape/` for padding studies). **No novel Pallas kernels authored by labyrinth-ssr.**

| Kernel | Source | Backend | Stability | Notes |
|---|---|---|---|---|
| flash_attention | [tpu_inference_kernel/flash_attention/kernel.py](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/flash_attention/kernel.py) | `mosaic_tpu` | stable (vendored Google) | Copyright 2025 Google LLC |
| fused_moe v1 | [tpu_inference_kernel/fused_moe/v1/kernel.py](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/fused_moe/v1/kernel.py) | `mosaic_tpu` | stable (vendored) | Uses SMEM/VMEM/semaphores, double-buffered weights, a2a-sharded experts, optional per-group quantized weights |
| megablox/gmm | [`tpu_inference_kernel/megablox/gmm.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/megablox/gmm.py) | `mosaic_tpu` | stable (vendored) | — |
| mla/v1 | [`tpu_inference_kernel/mla/v1/kernel.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/mla/v1/kernel.py) | `mosaic_tpu` | stable (vendored) | "TPU-Friendly and Data-Movement-Friendly MLA Ragged Paged Attention" |
| ragged_paged_attention/v2 | [`tpu_inference_kernel/ragged_paged_attention/v2/kernel.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/ragged_paged_attention/v2/kernel.py) | `mosaic_tpu` | stable (vendored) | + [`ragged_kv_cache_update.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/ragged_paged_attention/v2/ragged_kv_cache_update.py), [`tuned_block_sizes.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/ragged_paged_attention/v2/tuned_block_sizes.py) |
| ragged_paged_attention/v3 | [`tpu_inference_kernel/ragged_paged_attention/v3/kernel.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/ragged_paged_attention/v3/kernel.py) + [`kernel_hd64.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/ragged_paged_attention/v3/kernel_hd64.py) | `mosaic_tpu` | stable (vendored) | head-dim-64 specialization |
| quantized_matmul | [`tpu_inference_kernel/quantized_matmul/kernel.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/quantized_matmul/kernel.py) | `mosaic_tpu` | stable (vendored) | + tuned_block_sizes |
| collectives/all_gather_matmul | [`tpu_inference_kernel/collectives/all_gather_matmul.py`](https://github.com/labyrinth-ssr/tpu-research/blob/main/tpu_inference_kernel/collectives/all_gather_matmul.py) | `mosaic_tpu` | stable (vendored) | imports from `tpu_inference.kernels.collectives` |

## 6.10 ashioyajotham/recompute_dont_restore

Pedagogical FlashAttention fwd/bwd on TPU Pallas, built from first principles. Freshly active (last push 2026-04-21), 1⭐, no LICENSE. Useful learning resource; not something to depend on. Benchmarks sweep v4/v5e/v5p. `flash_fwd.py` module docstring spells out Pallas grid convention (`kv_tiles` as `"arbitrary"` because of online-softmax state dependency) — good reference writeup.

| Kernel | Source | Backend | Stability | Perf claim | Use case | Callers | Notes |
|---|---|---|---|---|---|---|---|
| flash_fwd | [03_pallas_kernels/flash_fwd.py](https://github.com/ashioyajotham/recompute_dont_restore/blob/main/03_pallas_kernels/flash_fwd.py) | `mosaic_tpu` | pedagogical | benchmarks in `05_benchmarks/` | teach FA fwd | tests, notebooks | grid `(B, H, q_tiles, kv_tiles)` |
| flash_bwd | [.../flash_bwd.py](https://github.com/ashioyajotham/recompute_dont_restore/blob/main/03_pallas_kernels/flash_bwd.py) | `mosaic_tpu` | pedagogical | — | teach "recompute, don't restore" bwd | tests | recomputes attention matrix in-kernel |
| gqa extension | [04_gqa_extension/grouped_query_attention.py](https://github.com/ashioyajotham/recompute_dont_restore/blob/main/04_gqa_extension/grouped_query_attention.py) | `mosaic_tpu` | pedagogical | — | adapt to GQA | tests | educational extension |

## Additional small repos

- **[AlexG1105/mamba2-jax-pallas](https://github.com/AlexG1105/mamba2-jax-pallas)** — Mamba-2 SSD kernels (`bmm_chunk_fwd`, `chunk_cumsum_fwd`, `chunk_scan_fwd`, `chunk_state_fwd`, `ssd_combined`, `state_passing_fwd`). File headers: "Mosaic GPU (H100/H200) Pallas implementation." **GPU-only, not TPU.** Apache-2.0, 0⭐, 2026-03-05. Also has `chunk_cumsum_fwd_pallas` Triton variant advertised "~1.3× of Triton." Useful for GPU Mamba-2.
- **[rdyro/gpu_ragged_dot](https://github.com/rdyro/gpu_ragged_dot)** — Name says GPU; header "Copyright 2025 The JAX Authors". JAX-derived ragged-dot Pallas example. Apache-2.0, 3⭐.
- **[rdyro/moe_in_jax](https://github.com/rdyro/moe_in_jax)** — TPU-targeted MoE. `moe/megablox/kernels.py` vendors DeepMind's megablox ("Copyright 2025 DeepMind Technologies Limited"). `moe/sc_kernels.py` is **TPU SparseCore** (imports `jax.experimental.pallas.tpu_sc as plsc`, sets `--xla_tpu_use_tc_device_shape_on_sc=true`) — rare community SparseCore Pallas example. No LICENSE, 3⭐, active.
- **[jondeaton/ring-attention-jax-pallas](https://github.com/jondeaton/ring-attention-jax-pallas)** — **404 confirmed.** User has no such repo. Previous survey was chasing a dead link; drop it.

## Cross-repo observations

### Legitimately novel vs vendored

- **Legitimately novel community-authored TPU Pallas**: ejkernel (all TPU kernels are Erfan-authored, including the interesting all_gather_matmul / reduce_scatter_matmul fused-collective pair and the ring_attention Splash wrapper), haoliuhl/ringattention (canonical paper impl), recompute_dont_restore (pedagogical originals), sqtian matmul series (pedagogical originals). moe_in_jax `sc_kernels.py` looks author-original — rare SparseCore Pallas example.
- **Vendored from Google**: **labyrinth-ssr/tpu-research** (full vendor of `tpu_inference`), **Essential-AI/maxtext-external** (upstream MaxText), rdyro/moe_in_jax megablox (vendored DeepMind megablox).
- **Vendored / derived from JAX**: rdyro/gpu_ragged_dot ("Copyright The JAX Authors"), flashback ("barely modified from the original jax source code").
- **GPU-only (mislabeled as Pallas)**: gla-jax, flashback, mamba2-jax-pallas all advertise Pallas but are Triton/Mosaic-GPU only. **Only haoliuhl, ejkernel, sqtian, recompute_dont_restore, labyrinth-ssr (vendored), Essential-AI (vendored), and rdyro/moe_in_jax are actually TPU Pallas.**

### Activity / maintenance judgment

- **Active** (≤ 3 months since last push): EasyDeL, ejkernel, recompute_dont_restore, moe_in_jax, labyrinth-ssr, mamba2-jax-pallas, gpu_ragged_dot.
- **Stale-but-canonical**: haoliuhl/ringattention (paper companion — maintenance-mode acceptable).
- **Effectively abandoned** (> 9 months, low usage): jax-flash-attn2 (superseded), gla-jax, sqtian matmul (pedagogical — acceptable), flashback, Essential-AI/maxtext-external.

### Worth a proper wiki ingest

1. **erfanzar/ejkernel** — highest priority. Broadest community TPU Pallas surface, actively maintained, every attention variant a modern TPU LLM serving stack needs + fused-collective matmul. Ingest full `ejkernel/kernels/_pallas/tpu/` tree.
2. **erfanzar/EasyDeL** — codebase page with subpage for kernels-operation-registry; de-facto "application" of ejkernel.
3. **haoliuhl/ringattention** — canonical Ring Attention. Essential for long-context experiments and baseline before introducing zig-zag.
4. **sqtian/PALLAS_TPU_KERNEL_MATMUL** — lightweight ingest; staged matmul optimization reference. Per-version benchmark plots instructive for block-size reasoning.
5. **ashioyajotham/recompute_dont_restore** — pedagogical cross-reference when writing FlashAttention concept pages; `05_benchmarks/xprof_guide.md` may be reusable.
6. **labyrinth-ssr/tpu-research** — not worth ingesting as original work; worth a stub noting it mirrors Google `tpu_inference`. Ingest `tpu_inference` directly.

Lower priority: jax-flash-attn2 (superseded), gla-jax (GPU-only, abandoned), flashback (GPU-only; ingest only if pursuing higher-order-grad on TPU), mamba2-jax-pallas (GPU-only), Essential-AI/maxtext-external (snapshot), rdyro/gpu_ragged_dot (GPU-only). rdyro/moe_in_jax borderline — SparseCore Pallas example rare enough for a one-page stub.

### Zig-Zag / load-balanced ring attention variants

**NONE LOCATED IN THIS GROUP.** Confirmed absent:
- haoliuhl/ringattention: single unidirectional ppermute ring, standard `below_or_on_diag` causal check; no striped/zig-zag load-balancing.
- ejkernel ring_attention: docstring says it's a Splash-wrapped ring — no zig-zag logic.
- EasyDeL RingAttn: thin wrapper over ejkernel, inherits the limitation.

**If a zig-zag ring attention Pallas TPU implementation exists publicly, it is not in any of the ten target repos or the four meta-repos.** The canonical Brandon et al. striped variant remains unimplemented in community Pallas code as of 2026-04-23.

## Sources

- Web-research agent, 2026-04-23.

## See also

- [Directory main page](../2026-04-23-pallas-kernel-directory.md)
- §1 [Upstream JAX + tokamax](01-upstream-jax-tokamax.md) — upstream that these libs vendor from
- §3 [Inference engines](03-inference-engines.md) — labyrinth-ssr vendors from here
