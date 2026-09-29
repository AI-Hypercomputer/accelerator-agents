---
title: "Source: applied-inference.md"
type: source-page
doc: applied-inference.md
updated: 2026-07-03
---
# applied-inference.md — Serving LLaMA 3-70B on TPUs

Part 8. Applies Part 7's inference model to LLaMA 3-70B, showing decode latency can be lower-bounded
by the time to load all parameters (plus KV cache at larger batch sizes) from HBM into the MXU
([whats-the-llama-serving-story](src:applied-inference.md#whats-the-llama-serving-story)). Shows
useful model-parallelism degree is capped around 8-32 depending on model size and sharding axis
count, with further sharding trading throughput for latency
([thinking-about-throughput](src:applied-inference.md#thinking-about-throughput)). Demonstrates
LLaMA 3-70B is strongly KV-cache-memory-bandwidth-bound across nearly all realistic serving
configurations, visualizing the latency/throughput Pareto tradeoff directly
([visualizing-the-latency-throughput-tradeoff](src:applied-inference.md#visualizing-the-latency-throughput-tradeoff)).

## Feeds
- [Inference serving, latency and throughput](../topics/inference-serving-latency-throughput.md) —
  the worked example this chapter contributes.
