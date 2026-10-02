---
title: "Source: transformers.md"
type: source-page
doc: transformers.md
updated: 2026-07-03
---
# transformers.md — Transformer Math (101)

Part 4. Derives per-layer FLOPs and parameter counts for the MLP block ($\approx 18BTDF$ FLOPs,
$3DF$ params) and attention block ($12BTD(N+K)H$ FLOPs for QKVO projections plus $12BT^2NH$ for the
dot-product itself)
([global-flops-and-params-calculation](src:transformers.md#global-flops-and-params-calculation)).
Derives the famous rule of thumb that total training FLOPs $\approx 6 \cdot
\text{num\_params} \cdot \text{num\_tokens}$ for dense Transformers when dot-product attention FLOPs
are neglected, and shows attention FLOPs only dominate once sequence length $T > 8D$
([general-rule-of-thumb-for-transformer-flops](src:transformers.md#general-rule-of-thumb-for-transformer-flops),
[fractional-cost-of-attention-with-context-length](src:transformers.md#fractional-cost-of-attention-with-context-length)).
Covers MoE as "E MLP blocks per layer, k active per token" plus the two AllToAlls this adds
([sparsity-and-mixture-of-experts](src:transformers.md#sparsity-and-mixture-of-experts)), gradient
checkpointing strategies (block remat vs. big-matmuls-only) trading recomputation FLOPs for
activation memory
([gradient-checkpointing](src:transformers.md#gradient-checkpointing)), and KV cache sizing
($2SLKH$ per cache)
([key-value-kv-caching](src:transformers.md#key-value-kv-caching)).

## Feeds
- [Transformer FLOPs and memory accounting](../topics/transformer-flops-and-memory-accounting.md) —
  the core topic this chapter defines.
- [Inference serving, latency and throughput](../topics/inference-serving-latency-throughput.md) —
  consumes the KV-cache sizing formula.
