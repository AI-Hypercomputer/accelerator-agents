---
id: STRAT-MLA-08
title: On SparseCore, pl.when may wrap emit_pipeline and its grid length may be traced
applies_to: sparsecore
lever: pipelining
generality: unit-wide
phase: [implement]
verify: "manual: compile pl.when around pltpu.emit_pipeline with a traced grid length on a VectorSubcoreMesh and compare against the unskipped kernel"
---

# STRAT-MLA-08 — Data-dependent skipping may lower on SparseCore; verify before relying

**Rule.** Toolchain-dependent; verify before relying. Two constructs widely assumed unsupported on
SparseCore have been seen to lower: `pl.when` wrapping `pltpu.emit_pipeline` on a
`VectorSubcoreMesh`, and a **traced** (runtime-valued) grid length for `emit_pipeline`. Together they
are the enabling primitives for any data-dependent skip on the unit, so assuming they are unavailable
is expensive — it rules out a class of design before it is costed. Where both spellings lower, they
are expected to perform alike, so pick the more readable one (hypothesis; time both once).

**Mechanism.** None to state: this is a lowering capability of a particular build, not a behaviour
derived from hardware. It is recorded because a *negative* assumption about an API is
self-confirming — nobody measures what they believe will not compile — and because the probe that
settles it takes minutes. Probe `sc_data_dependent_control_flow` covers the neighbouring question of a
data-dependent `while` trip count.

**Scope.** SparseCore vector-subcore kernels. A lowering capability expires with the toolchain, so
re-probe after any jax/jaxlib/libtpu change. For traced DMA *extents* (not grid lengths) see
[STRAT-MLA-12](STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md).

**Does not apply as a performance claim.** That a predicate *lowers* does not mean the skipped work
costs nothing. Whatever fixed cost the call and its traversal carry survives the skip; localize it with
[STRAT-MLA-11](STRAT-MLA-11-localize-fixed-cost-before-building-a-grid-lever.md) before expecting a
saving. Lowering buys the design; it does not buy the saving.

**How to check on your kernel.** Write a two-line lowering probe of both constructs on the target
toolchain; check byte-exactness against the unmodified kernel; then time skipped versus unskipped at
constant live work.
