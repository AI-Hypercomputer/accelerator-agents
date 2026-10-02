---
title: "Source: microbenchmarks/requirements.txt"
type: source-page
doc: microbenchmarks/requirements.txt
updated: 2026-07-03
---
# microbenchmarks/requirements.txt

The single-chip microbenchmark suite's dependency pin: `jax[tpu]==0.5.2`
([microbenchmarks/requirements.txt](src:microbenchmarks/requirements.txt)) — a fixed JAX version,
distinct from the per-recipe `jax-stable-stack`/`jax-ai-image` Docker base images the training and
inference recipes pin independently.

## Feeds
- [TPU microbenchmarks](../topics/tpu-microbenchmarks.md) — the JAX version the single-chip
  matmul/HBM benchmarks in that topic are run against.
