---
title: "Source: inference.md"
type: source-page
doc: inference.md
updated: 2026-07-03
---
# inference.md — Serving LLMs

Part 7. Splits inference into prefill (compute-bound, one-shot, latency dominated by TTFT) and
generation (memory-bandwidth-bound, serial token-by-token, requiring batching across requests to
reach the ~240-token critical batch size)
([the-basics-of-transformer-inference](src:inference.md#the-basics-of-transformer-inference)).
Shows attention is compute-bound during prefill for any reasonable sequence length but always
memory-bandwidth-bound during generation, since its arithmetic intensity is low and roughly constant
([what-about-attention](src:inference.md#what-about-attention)). Covers KV-cache-size-reduction
techniques (GMQA, MLA) that can cut cache size by an order of magnitude
([theoretical-estimates-for-llm-latency-and-throughput](src:inference.md#theoretical-estimates-for-llm-latency-and-throughput)).
Derives sharding strategy differences between prefill (model-parallel up to the ICI bound, then
sequence-parallel) and generation (model-parallel moving activations, not KV caches/params, up to
the FLOPs-ICI bound $F/\alpha$)
([distributing-inference-over-multiple-accelerators](src:inference.md#distributing-inference-over-multiple-accelerators)).
Introduces continuous batching, prefix caching, and disaggregated prefill/generation serving via the
JetStream architecture
([designing-an-effective-inference-engine](src:inference.md#designing-an-effective-inference-engine)).

## Feeds
- [Inference serving, latency and throughput](../topics/inference-serving-latency-throughput.md) —
  the core topic this chapter defines.
- [Transformer FLOPs and memory accounting](../topics/transformer-flops-and-memory-accounting.md) —
  consumes the KV-cache sizing formula this chapter builds on.
