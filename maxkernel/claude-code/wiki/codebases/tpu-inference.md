---
title: "tpu-inference (vllm-project/tpu-inference)"
type: codebase
tags: [vllm, tpu-inference, pallas, mosaic-tpu, ragged-paged-attention, mla, megablox, fused-moe, quantized-matmul, all-gather-matmul, gdn, sparsecore, crown-jewel-tuning-tables]
repo: vllm-project/tpu-inference
commit: a657060d
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [tpu-inference overview](tpu-inference/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

vLLM's TPU inference backend and the **most comprehensive Pallas kernel collection** surveyed in this wiki — broadest novel-content surface on TPU. Authoritative author for: Ragged-Paged-Attention v2/v3 (+ `_hd64`), Multi-Latent Attention v1/v2, Fused-MoE v1, blockwise Quantized Matmul (600+ tuned entries), Fused AllGather+Matmul (collective overlap), GDN/triangle-solver (Qwen-Next), SparseCore gather/scatter (v5p/v7x), structured sparse matmul v1. **Crown-jewel tuning artifacts**: ~1,200 RPA v2 entries, 600+ quantized_matmul entries (v6 96 MiB VMEM / v7 48 MiB VMEM), hundreds of RPA v3 entries. Per-kernel detail in [§3.1 of the Pallas kernel directory](../analyses/pallas-kernel-directory/03-inference-engines.md#31-vllm-projecttpu-inference).

## Overview

All kernels `mosaic_tpu`, Apache-2.0. README recommends **v7x (Ironwood), v5e, v6e**; v3/v4/v5p experimental. Per-hardware support is tracked in `support_matrices/{release,nightly}/{v6e,v7x}/{default,flax_nnx,vllm}/kernel_support_matrix.csv`.

Three novel-content axes worth ingesting:

1. **Inference-attention frontier** — RPA v2/v3 + MLA v1/v2 + GDN. Two trends: mixed-prefill-decode in a single dispatch (RPA), and multi-latent compressed-cache (MLA for DeepSeek-V2/V3).
2. **Quantization frontier** — `quantized_matmul/blockwise_kernel.py` with 600+ tuned entries keyed by `(tpu_ver, m, n, k, lhs_dtype, rhs_dtype)`. W8A8 / W4A16 / W8A16 / FP8 covered. Zero-point / subchannel quant explicitly `NotImplementedError`.
3. **Collective overlap** — `collectives/all_gather_matmul.py` pipelines remote-copy, HBM→VMEM DMA, and MXU compute with semaphores. Fuses TP all-gather with the matmul it feeds. Constraints: `k`,`n` div by 128; `m` div by `tp_size*2*8`.

## Architecture

Kernels at `tpu_inference/kernels/`:

```
tpu_inference/kernels/
  ragged_paged_attention/
    v2/{kernel,tuned_block_sizes,ragged_kv_cache_update}.py  ← prior-gen RPA (1,200+ tuned entries)
    v3/{kernel,kernel_hd64,tuned_block_sizes,tuned_block_sizes_hd64}.py  ← current, v6e/v7
  mla/{v1,v2}/kernel.py                                       ← Multi-Latent Attention
  flash_attention/kernel.py                                   ← vendored from jax-ml/jax (no attrib hdr)
  megablox/{gmm,gmm_v2,tuned_block_sizes}.py                  ← MoE grouped matmul (47 entries, bf16×fp8)
  fused_moe/v1/{kernel,tuned_block_sizes}.py                  ← full fused routing+GMM×2 (28 v7-only entries)
  quantized_matmul/{kernel,blockwise_kernel,tuned_block_sizes}.py  ← 600+ entries, v6 96MiB / v7 48MiB VMEM
  collectives/{all_gather_matmul,all_gather_matmul_tuned_block_sizes}.py
  gdn/{fused_gdn_decode_kernel,fused_gdn_recurrent_kernel,triangle_solver}.py  ← Qwen-Next GDN
  sparse_core/{ragged_gather,ragged_scatter,gather_reduce}.py  ← v5p/v7x SparseCore
  structured_sparse_matmul/v1/spmm.py                         ← N:M sparse, emu-software
  experimental/batched_rpa/{kernel,schedule,wrapper,flash_attention}.py  ← USE_BATCHED_RPA_KERNEL=1
support_matrices/{release,nightly}/{v6e,v7x}/{default,flax_nnx,vllm}/
  kernel_support_matrix.csv
.buildkite/kernel_microbenchmarks/
  ragged_paged_attention_v3_head_dim_64/                       ← microbench yaml
  mla/
  ...
```

## Key abstractions

Full per-kernel row schema in [the directory §3.1 table](../analyses/pallas-kernel-directory/03-inference-engines.md#31-vllm-projecttpu-inference). Highlights:

| Kernel | Novel? | Tuning entries | Why it matters |
|---|---|---|---|
| `ragged_paged_attention` v3 | **novel** | hundreds (v6e, v7) | Three `pallas_call` split (DECODE/PREFILL/MIXED); precise sliding-window skipping |
| `ragged_paged_attention` v3 `_hd64` | **novel** spec. | separate hd64 table | Specialization for Gemma-class small head dims |
| `ragged_paged_attention` v2 | **novel** | ~1,200 (v5, v6) | `(q_dtype, kv_dtype, q_head, kv_head, head_dim, page_size, max_context, max_seq)` → `(kv_pages_per_block, queries_per_block)` |
| `mla_ragged_paged_attention` v1/v2 | **novel** | heuristic (v1), improved pipelining (v2) | DeepSeek-V2/V3 MLA; mixed prefill/decode |
| `fused_moe` v1 | **novel** | **28 v7-only** entries | Fused routing + two GMMs; Qwen3-MoE / Mixtral / Llama4-MoE |
| `quantized_matmul` blockwise | **novel** | **600+** entries | v6 96 MiB VMEM, v7 48 MiB VMEM; int8×int8, fp8×fp8 |
| `all_gather_matmul` | **novel** | microbench table | Pipelines remote-copy + DMA + MXU with semaphores |
| `fused_gdn_decode` + `fused_gdn_recurrent` + `triangle_solver` | **novel** | — | Qwen-Next Gated Delta Net |
| `sparse_core` gather/scatter/gather_reduce | **novel** (hw-gated) | — | `VectorSubcoreMesh` on v5p/v7x; fallback to regular gather elsewhere |
| `structured_sparse_matmul` v1 | **novel** | — | N:M ≤16 software emulation; memory-bound benefit only |
| `megablox_gmm` / `gmm_v2` | vendored (no attrib hdr) | 47 entries | Vendored from `jax-ml/jax` megablox; all bf16×fp8_e4m3fn |
| `flash_attention` | vendored | — | Vendored from `jax-ml/jax` flash_attention (no attrib hdr) |
| `batched_rpa` | **novel** | — | Experimental; gated on `USE_BATCHED_RPA_KERNEL=1` |

## Entry points

```bash
cd raw/code/tpu-inference

# install in a TPU environment
pip install -e .

# microbenchmarks (these exist; numbers live in CI artifacts, not README)
ls .buildkite/kernel_microbenchmarks/

# support matrix
cat support_matrices/release/v6e/default/kernel_support_matrix.csv
cat support_matrices/release/v7x/default/kernel_support_matrix.csv
```

Python import paths:
```python
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import ragged_paged_attention
from tpu_inference.kernels.quantized_matmul.blockwise_kernel import ...
from tpu_inference.kernels.collectives.all_gather_matmul import all_gather_matmul
```

Downstream dispatch (torch-xla path):
```python
torch.ops.xla.ragged_paged_attention   # → this repo via torch_xla.experimental.custom_kernel
```

## Dependencies

- `jax` + `jaxlib` + `libtpu` (TPU runtime).
- `torch_xla` (optional — consumed by aphrodite / torch-based callers via `torch.ops.xla.*`).
- No GPU deps — this repo is TPU-only.

## Performance-relevant surfaces

### 1. Crown-jewel tuning tables
- **~1,200 entries** in `ragged_paged_attention/v2/tuned_block_sizes.py` — the de-facto reference for RPA v2 on v5/v6.
- **600+ entries** in `quantized_matmul/tuned_block_sizes.py` — keyed `(tpu_ver, m, n, k, lhs_dtype, rhs_dtype)`. **v6 VMEM budget = 96 MiB, v7 = 48 MiB** (baked into the config struct, not just data). These numbers are wiki-concept-facts.
- **Hundreds** in `ragged_paged_attention/v3/tuned_block_sizes.py` + tens in `tuned_block_sizes_hd64.py`.
- **47** in `megablox/tuned_block_sizes.py` (all bf16×fp8_e4m3fn).
- **28 v7-only** in `fused_moe/v1/tuned_block_sizes.py` — explicit code comment: *"formulas only applied to tpu-v7; need more for other generations"*. MoE is under-tuned on v5/v6.
- `collectives/all_gather_matmul_tuned_block_sizes.py` — microbench-driven across `w16a16`…`w4a4`.

These tables are **directly usable priors** for any matching hypothesis — don't rediscover from scratch.

### 2. VMEM budget discipline
`quantized_matmul/blockwise_kernel.py` hard-codes **96 MiB on v6** and **48 MiB on v7**. Any Pallas kernel this wiki writes should respect these budgets. Gemma4 program should record these as concept-level facts.

### 3. Collective overlap via `all_gather_matmul`
`collectives/all_gather_matmul.py` fuses TP AG with the matmul it feeds. Pipelines: remote-copy → HBM→VMEM DMA → MXU compute, semaphore-synchronized. Constraints `k`, `n` div by 128; `m` div by `tp_size*2*8`. Candidate for gemma4 TP hypotheses.

### 4. Hardware-gated SparseCore kernels
`sparse_core/` kernels require SC hardware — present on v5p/v7x, absent on v5e/v6e. Explicit fallback path: *"Sparse core is not available. Fallback to regular gather"*. Relevant if/when this wiki's program moves to v5p/v7x.

### 5. MLA as architectural knob
`mla/v1` + `mla/v2` are the only public Pallas MLA for TPU. DeepSeek-class compressed-cache attention. Directly applicable if a DeepSeek-class model enters the program.

### 6. Experimental `batched_rpa` gate
`experimental/batched_rpa/__init__.py`: *"all of the code in this directory is experimental and not fully tested"*; gated on `USE_BATCHED_RPA_KERNEL=1`. Ships a repo-local `flash_attention.py` variant alongside. Don't cite without a microbench.

### 7. Support-matrix CSVs
`support_matrices/{release,nightly}/{v6e,v7x}/...` — per-hardware kernel availability. Read before citing a kernel is "stable on v6e".

### 8. CI microbench YAMLs as perf-claim source
`.buildkite/kernel_microbenchmarks/*.yml` — where the actual numeric perf claims live (README is qualitative). Parse these rather than re-measuring when possible.

## Connections

- **[Pallas kernel directory §3.1](../analyses/pallas-kernel-directory/03-inference-engines.md#31-vllm-projecttpu-inference)** — full kernel detail.
- [sglang-jax](sglang-jax.md) — **vendors most of its kernels from this repo** (RPA v2/v3, fused_moe v1, quantized_matmul blockwise, update_kv_cache, megablox gmm/v2) with SGLang-specific extras.
- [pytorch/xla](https://github.com/pytorch/xla) — dispatches `torch.ops.xla.ragged_paged_attention` to this repo.
- [jax](jax.md) — `flash_attention/kernel.py` + `megablox` are vendored from here.
- [maxtext](maxtext.md) — `config.attention == "vllm_rpa"` routes here via `tpu_inference.layers.common.attention_interface.sharded_ragged_paged_attention`.
- [aphrodite-engine](https://github.com/aphrodite-engine/aphrodite-engine) (not ingested — AGPL-3.0) — consumer.

## See also

- [pallas-kernel](../concepts/pallas-kernel.md)
- [kv-cache](../concepts/kv-cache.md)
- [static-cache](../concepts/static-cache.md)
- [continuous-batching](../concepts/continuous-batching.md)
- [attention-block-sizes](../concepts/attention-block-sizes.md)
- [sparsecore](../concepts/sparsecore.md)
- [vmem](../concepts/vmem.md)
- [int8-quantization](../concepts/int8-quantization.md)
- [all-gather](../concepts/all-gather.md)
- [ragged-dot](../concepts/ragged-dot.md)
- [autotuning](../concepts/autotuning.md)

## Sources

- `raw/code/tpu-inference/tpu_inference/kernels/`
- `raw/code/tpu-inference/support_matrices/`
- `raw/code/tpu-inference/.buildkite/kernel_microbenchmarks/`
- Upstream: <https://github.com/vllm-project/tpu-inference>
