---
title: "How to Scale Your Model — Ch 7: All About Transformer Inference"
type: source
tags: [docs, book, scaling-book, inference, prefill, decode, kv-cache, latency, throughput, pareto]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 7
upstream: https://jax-ml.github.io/scaling-book/inference
created: 2026-04-23
updated: 2026-04-23
---

Chapter 7 of the scaling-book. **Prefill / generate split, KV cache, and the latency-throughput Pareto frontier** — the theoretical foundation for every LLM serving decision. Generation is **always memory-bandwidth-bound at small-to-moderate batch**; prefill is **compute-bound for sequences > ~240 tokens**.

## Key claims

1. **Naive sampling** (reprocess full prefix per new token): O(n²) FFW, O(n³) attention — never used.
2. **KV caching** reduces generation to O(n) FFW + O(n²) attention; first-class technique.
3. **Prefill arithmetic intensity** ≈ `T/2` FLOPs/byte → compute-bound for `T > ~480` tokens on v5e.
4. **Generation arithmetic intensity** ≈ 1 FLOP/byte (T=1, weights + KV must load per step) → **always memory-bandwidth-bound**, independent of batch size.
5. **Critical batch size for compute-bound generate**: `B > 120` (int8 weights + bf16 FLOPs) or `B > 240` (bf16 weights).
6. **Latency lower bound**: `step_time ≥ (param_size + B × KV_size) / total_BW`.
7. **Throughput lower bound**: `tokens/s ≤ B × BW / (B × KV_size + param_size)`.
8. **Latency-throughput Pareto**: small B (1–32) → low latency, poor utilization; large B (120+) → good throughput, high per-request latency. No free lunch.
9. **KV cache dominance**: at long contexts (S > 2k), KV loading time >> param loading time — reducing KV via GQA, int4 KV, paged caches is critical.

## Key data points

### LLaMA-2-13B generation on v5e 4×4 (16 chips), int8 params + bf16 FLOPs, 8k ctx

| Batch | KV memory | Step | Throughput (tok/s) | Latency/tok |
|---:|---:|---:|---:|---:|
| 1 | 6.7 GB | ~5 ms | 200 | 5 ms |
| 16 | 107 GB | ~20 ms | 787 | 20.3 ms |
| 32 | 214 GB | ~37 ms | 873 | 36.6 ms |
| 240 | 1608 GB | ~249 ms | 963 | 249 ms |

At B=32: 26 GB params + 214 GB KV → **90% of memory is cache**.

### Regime summary

- Prefill: compute-bound, MFU-limited, matches training kernel economics.
- Generate: bandwidth-bound, KV-cache-dominated, scales with batch up to critical B.

## Techniques referenced

- Flash / Splash attention (no `[B, N, T, T]` materialization).
- **Paged KV caches** (vLLM-style page blocks) — see [tpu-inference](../codebases/tpu-inference.md) / [sglang-jax](../codebases/sglang-jax.md) RPA.
- **Continuous batching** (vLLM scheduler) — fills idle compute by batching across requests.
- **Grouped-Query Attention (GQA)** — fewer KV heads than Q heads shrinks cache.
- **int4 KV quantization** — halves cache again.
- **Disaggregated serving** — separate prefill-only and generate-only servers (typical ratio 3:1).

## Gaps & caveats

- Assumes dense attention; sparse / MoE inference has different communication.
- Doesn't cover scheduler optimizations (continuous batching, priority queues, preemption).
- Multi-device generation (TP during decode) only lightly touched.
- Book is Feb 2025 — pre-dates ecosystem improvements like MLA (DeepSeek-V2/V3) which change KV-cache economics; see [tpu-inference MLA v1/v2](../codebases/tpu-inference.md).
- Long-context (> 32k tokens) quadratic memory effects not deeply addressed.
- bf16 / int8 / int4 comparison is v5e-specific; v6e/v7 numbers shift.

## Connections

- [concepts/kv-cache](../concepts/kv-cache.md) / [static-cache](../concepts/static-cache.md) / [continuous-batching](../concepts/continuous-batching.md).
- [concepts/decode-profile-signature](../concepts/decode-profile-signature.md).
- [concepts/serving-warmup](../concepts/serving-warmup.md).
- [codebases/tpu-inference](../codebases/tpu-inference.md) — MLA + RPA + paged-KV production impls.
- [codebases/sglang-jax](../codebases/sglang-jax.md) — EAGLE spec decoding.

## See also

- [Ch 4 — Transformer Math](2025-scaling-book-ch4-transformers.md)
- [Ch 8 — Applied Inference (LLaMA-3-70B)](2025-scaling-book-ch8-applied-inference.md)

## Sources

- `raw/code/scaling-book/inference.md`
- Upstream: <https://jax-ml.github.io/scaling-book/inference>
