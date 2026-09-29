---
title: "How to Scale Your Model — Ch 8: Serving LLaMA 3-70B on TPUs"
type: source
tags: [docs, book, scaling-book, llama3, serving, inference, applied, v5e, quantization, int8, int4]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 8
upstream: https://jax-ml.github.io/scaling-book/applied-inference
created: 2026-04-23
updated: 2026-04-23
---

Chapter 8 of the scaling-book. **Applied serving analysis for LLaMA-3-70B on v5e** — minimum-topology / latency / throughput numbers by dtype, with disaggregated serving ratios. Companion to the Ch 7 theory.

## Key claims

1. **v5e is the cost winner** for inference: 5.8×10¹⁷ FLOPs/$ vs. 3.3×10¹⁷ for H100 and 3.9×10¹⁷ for v5p (as of Feb 2025 GCP pricing).
2. **LLaMA-3-70B KV/token**: `2 × 8 × 128 × 80 = 160 kB`; at 32 k ctx → **5.3 GB/sequence**.
3. **Memory fit by dtype**: bf16 140 GB → 4×4 (16 chips); int8 70 GB → 4×2 (8); int4 35 GB → 2×2 (4).
4. **Step latency** (bandwidth-bound) ≈ `(params + KV) / total_BW`; on 4×2 int8 → **~17 ms/step**.
5. **Generation compute-bound threshold**: `B > 120` (int8) — hard to hit because requests are sequential.
6. **Serving sharding**: pure model parallelism (16-way) is compute-bound without becoming ICI-bound; larger TP (4×8) needs `B > 26` to avoid ICI dominance.
7. **Prefill on 16 chips at 40% MFU**: 8 k prompt → ~910 ms — significant relative to generation.
8. **Quantization wins**: int8 halves critical batch size → cheaper topology; int4 quarters it.
9. **Disaggregated serving**: prefill-servers : generate-servers ≈ **3:1** to keep both pipelines saturated (8 k prefill + 512 generate).

## Key data points

### Minimum topology by dtype (LLaMA-3-70B, 8 k ctx, 43 concurrent seqs)

| Dtype | Min topology | Chips | Memory/chip | Step time |
|---|---|---:|---:|---:|
| bf16 | 4×4 | 16 | 11.4 GB | 9.5 ms |
| int8 | 4×2 | 8 | 14 GB | 17 ms |
| int4 | 2×2 | 4 | 14.2 GB | 19 ms |

### Throughput per chip (512-token median, B=43)

| Dtype | Topology | QPS/chip |
|---|---|---:|
| bf16 | 4×4 | 0.27 |
| int8 | 4×2 | 0.55 |
| int4 | 2×2 | 1.11 |

Prefill: 8 k tokens → ~910 ms on 16 chips (40% MFU). Generate: 512 tokens → 10–20 ms/tok latency.

## Techniques referenced

- Model parallelism (AllGather + ReduceScatter).
- Batch sharding.
- **Quantization** (int8 weights → halve critical batch; int4 → halve again).
- Disaggregated prefill/generate services.
- Latency-vs-throughput Pareto plots (interactive in book).

## Gaps & caveats

- Assumes synchronous batch processing; **continuous batching (vLLM)** not modeled.
- Working-memory-during-prefill not quantified.
- Uniform request patterns assumed (8 k prefill + 512 generate); real workloads are bursty with SLOs.
- Multi-slice serving (DCN) is impractical — not analyzed.
- No adaptive-batch / priority-queue coverage.
- **Book is Feb 2025**; MLA-class models (DeepSeek-V2/V3) change economics — see [tpu-inference MLA kernels](../codebases/tpu-inference.md).
- v6e / v7 inference numbers will differ — primary consumer guidance is still v5e here.

## Connections

- [Ch 7 — Inference theory](2025-scaling-book-ch7-inference.md) — this is the applied companion.
- [codebases/tpu-inference](../codebases/tpu-inference.md) — production serving-kernel impls.
- [codebases/sglang-jax](../codebases/sglang-jax.md).
- [codebases/maxtext](../codebases/maxtext.md) — inference configs `llama3_70b_v5e-16.yml`, `llama3_405b_v5e-64.yml`.
- [concepts/int8-quantization](../concepts/int8-quantization.md).

## See also

- [Ch 6 — Applied Training](2025-scaling-book-ch6-applied-training.md)

## Sources

- `raw/code/scaling-book/applied-inference.md`
- Upstream: <https://jax-ml.github.io/scaling-book/applied-inference>
