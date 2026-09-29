---
title: "How to Scale Your Model — Ch 4: All the Transformer Math You Need to Know"
type: source
tags: [docs, book, scaling-book, transformers, flops, parameters, kv-cache, mlp, attention, grouped-query-attention, gradient-checkpointing]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 4
upstream: https://jax-ml.github.io/scaling-book/transformers
created: 2026-04-23
updated: 2026-04-23
---

Chapter 4 of the scaling-book. **Closed-form formulas for transformer FLOPs, parameters, memory, and arithmetic intensities.** The reference for any back-of-the-envelope estimate in this wiki.

## Key claims

1. **FLOPs for a contraction** over shared/batching dims: `2 × ∏(non-contracting+batching dims) × ∏(contracting dims)`.
2. **Training FLOPs ≈ 6 × num_params × num_tokens** (2 forward + 4 backward per param).
3. **Inference FLOPs ≈ 2 × num_params × num_tokens** (forward only).
4. **MLP FLOPs/layer**: `18 B T D F` (with gating: W_in1, W_in2, W_out — SwiGLU-like).
5. **Attention QKVO FLOPs/layer**: `12 B T D (N+K) H` (N = num heads, K = KV heads, H = head dim).
6. **Attention QK·V FLOPs/layer**: `12 B T² N H` (quadratic in seq length; causal mask halves).
7. **MLP dominates FLOPs unless `T > 8 D`** (e.g., D=4k → attention dominates only past T=32k).
8. **KV cache size**: `2 × S × L × K × H` bytes/sequence (S=context, L=layers, K=kv heads, H=head dim); ~8 GB at (S=8k, L=64, K=8, H=128, int8).
9. **Attention arithmetic intensity** differs sharply by regime: **prefill** ≈ O(T) (compute-bound for T>~240); **generation** ≈ 1 FLOP/byte (**always bandwidth-bound**).
10. **Gradient checkpointing**: "block" policy ~20 activations/layer to save; "big-matmuls-only" policy ~7.

## Key data points

### Per-layer FLOPs/params example (D=4k, F=16k, N=64 heads, K=8 kv heads, H=128, B=1024)

| Component | Params | FLOPs/token/layer |
|---|---:|---:|
| MLP (SwiGLU) | 3 D F = 192 M | 18 B T D F = 3.1×10¹² |
| Attn QKVO | 4 D (N+K) H = 33 M | 12 B T D N H = 6.3×10¹¹ |
| Attn QK·V | — | 12 B T² N H (quadratic) |
| Embedding | 2 D V | per-layer varies |

### KV cache sizing worked example (LLaMA-3-70B)

- `L = 80`, `D = 8192`, `N = 64`, `K = 8`, `H = 128`, bf16.
- KV/token = `2 × L × K × H × 2` bytes = 160 kB.
- Context 32k → 5.3 GB/sequence — drives the serving chapter's memory analysis.

### Training total

For LLaMA-3-70B on 15 T tokens: `6.3e24` total FLOPs. See Ch 6 for the chip-count and time estimates.

## Techniques referenced

- Flash / Splash attention (online softmax — avoids materializing the `[B, N, T, T]` tensor).
- Gating einsums (SwiGLU / GeGLU, see [gated-linear-unit](../concepts/gated-linear-unit.md)).
- Grouped-query attention (GQA) — `K < N` reduces KV cache.
- Multi-query attention (MQA) — `K = 1`.
- Gradient checkpointing (`jax.checkpoint` via [`ad_checkpoint`](../codebases/jax.md)).
- Ring attention (Ch 5/7 context parallelism).
- Structured sparsity (brief, not quantified).
- Mixture-of-Experts (brief, with AllToAll routing — see [concepts/expert-parallelism](../concepts/expert-parallelism.md)).

## Gaps & caveats

- Layer norm and positional encoding FLOPs ignored (small but non-zero; matters for fusion analysis — see Gemma4 exp 33).
- Causal mask halves attention FLOPs but kernel still typically processes full QKV.
- MoE **sparse activation** not fully quantified (E/k ratio matters).
- Positional-encoding variants (RoPE, ALiBi) perf impact not addressed.
- Assumes Flash-family attention; naive attention materializes `[B, N, T, T]` — catastrophic memory.
- Book is **2025-02-04** — predates some ecosystem changes; formulas are generation-independent but calibration constants move.

## Connections

- [concepts/mfu](../concepts/mfu.md) — the measurement this chapter enables.
- [concepts/kv-cache](../concepts/kv-cache.md) / [concepts/static-cache](../concepts/static-cache.md).
- [concepts/rematerialization](../concepts/rematerialization.md).
- [concepts/flash-attention](../concepts/flash-attention.md) / [splash-attention](../concepts/splash-attention.md).
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — worked Llama-2 forward.

## See also

- [Ch 5 — Parallelize for Training](2025-scaling-book-ch5-training.md)
- [Ch 6 — Applied Training (LLaMA-3)](2025-scaling-book-ch6-applied-training.md)
- [Ch 7 — Inference](2025-scaling-book-ch7-inference.md)

## Sources

- `raw/code/scaling-book/transformers.md`
- Upstream: <https://jax-ml.github.io/scaling-book/transformers>
