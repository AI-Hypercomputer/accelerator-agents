---
id: STRAT-MLA-15
title: Removing register spills does not move a kernel whose VALU issue binds; spills ride in free slots
applies_to: tensorcore
lever: method
generality: general
phase: [plan, profile]
verify: "manual: compare static bundle count in the LLO dump before and after; if it does not fall, the removed ops were co-issued"
---

# STRAT-MLA-15 — Price a spill by the slot it occupies, not by its count

**Rule.** A large spill count is not a lever unless spill traffic sits on the binding slot. In a VLIW
bundle, spill stores and reloads use the vector store / load slots; if VALU issue already fills the
bundles, the spills are co-issued for free and deleting them leaves the bundle count — and the clock —
unchanged. Before restructuring a kernel to remove spills, check which slot's op count sets the bundle
floor.

**Mechanism.** Bundle count ≈ max over slots of (ops in that slot / slots of that kind per bundle),
plus dependency stalls. Removing ops from a non-binding slot changes neither term. A restructuring
that keeps an accumulator in registers can cut spill stores sharply while VALU op count, and hence
bundle count, stays put.

**Scope.** Any VLIW-scheduled kernel. Read per-slot op counts and bundle count from the profile
summary / LLO dump (see [tpu_profiling_and_diagnostics.md](../tpu_profiling_and_diagnostics.md) and
[tpu_mxu_and_register_optimization.md](../tpu_mxu_and_register_optimization.md)). Even a change that
does cut bundles may move wall time little if the remaining bundles are VALU-bound.

**Does not apply when** vector load/store is the binding slot — then spill removal is a first-order
lever.

**How to check on your kernel.** From the LLO dump, count ops per slot (VALU, vector load, vector
store, MXU) and the static bundle count. If VALU ops / VALU slots is the max term, spills are free;
leave them. After any spill-removal change, confirm the bundle count fell before timing it.
