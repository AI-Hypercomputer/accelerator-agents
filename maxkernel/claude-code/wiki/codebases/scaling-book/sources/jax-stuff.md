---
title: "Source: jax-stuff.md"
type: source-page
doc: jax-stuff.md
updated: 2026-07-03
---
# jax-stuff.md — How to Parallelize a Transformer in JAX

Part 10. Describes JAX's three multi-device programming modes — Auto (`jax.jit`, XLA/Shardy infers
sharding and communication), Explicit ("sharding in types", propagation happens in the JAX type
system so ambiguous ops raise a trace-time error), and Manual (`jax.shard_map`, the programmer
writes every collective explicitly)
([how-does-parallelism-work-in-jax](src:jax-stuff.md#how-does-parallelism-work-in-jax)). Shows how
`jax.jit` with sharded inputs and `out_shardings` causes XLA to insert an AllReduce automatically,
visible directly in the compiled HLO, and how `jax.lax.with_sharding_constraint` can correct the
compiler when it inserts unwanted communication (the "compiler tickling" workflow). Demonstrates
Explicit mode's `jax.typeof` sharding introspection and its trace-time ambiguity errors when a
contracting dimension is sharded on both operands.

## Feeds
- [JAX parallelism programming model](../topics/jax-parallelism-programming-model.md) — the core
  topic this chapter defines.
- [Profiling methodology and reading XLA/HLO](../topics/profiling-methodology-and-hlo.md) — the HLO
  inspection technique this chapter introduces and the profiling chapter builds on.
