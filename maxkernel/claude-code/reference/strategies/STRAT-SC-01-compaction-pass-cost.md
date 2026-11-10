---
id: STRAT-SC-01
title: A pass with cross-lane serializing ops in its body is priced by those ops, not by its trip count
applies_to: sparsecore
lever: algorithm
generality: general
phase: [plan]
verify: "manual: time one chunk of the serializing-op body against a plain masked pass over the same width on the target unit"
---

# STRAT-SC-01 — Price a pass by its serializing ops, not its trip count

**Rule.** "This deletes two passes" is a trip-count argument, and trip count does not price a pass
whose body contains **cross-lane serializing ops**. Cost the body first. A restructuring that trades
several cheap full-width passes for one pass with serializing ops in it can be a net loss even when
the trip-count arithmetic is correct.

**Mechanism.** A cross-lane op forces the lanes to agree before the chunk can retire, so its cost is
paid per chunk and does not amortize over the pass. A body with a couple of such ops can cost a
multiple of a masked pass over the same width: the trip-count model prices the *iteration* while
the hardware prices the *body*. Hypothesis: a compaction pass built from `plsc.store_compressed` and
`plsc.all_reduce_population_count` can lose to the several masked passes it replaces; verify per
target.

**Scope.** General wherever a lane-cooperative primitive exists. On SparseCore the usual suspects
are `store_compressed` and `all_reduce_population_count` (check they lower at all with probe
`sc_primitive_lowering`). The per-chunk cost ratio is unit- and toolchain-specific and must be
measured, never carried over.

**Does not apply when** the pass being deleted is itself serializing, or when the compaction shrinks
later passes enough to pay for itself — which is a measurement, not an assumption.

**How to check on your kernel.** Before committing to a compaction or other lane-cooperative
restructuring, microbenchmark one chunk of the proposed body against one chunk of the masked pass it
replaces, at the real width. Multiply the per-chunk ratio by the trip counts on both sides; adopt
the restructuring only if that product, not the pass count, favours it.
