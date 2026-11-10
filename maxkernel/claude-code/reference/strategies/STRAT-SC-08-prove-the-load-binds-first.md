---
id: STRAT-SC-08
title: Every transfer lever measures the same on a compute-bound kernel
applies_to: sparsecore
lever: pipelining
generality: general
phase: [plan, implement]
verify: probe sc_vs_tc_streaming_bandwidth
---

# STRAT-SC-08 — Prove the load is the bottleneck before restructuring it

**Rule.** Every transfer lever — `pl.emit_pipeline`, a hand-rolled double buffer, a traced DMA
extent — changes **when** bytes arrive. On a compute-bound kernel they therefore change nothing and
measure the same as each other. Do not spend a run ranking them; establish that the roofline binds
on transfer first, and if it does not, pick whichever is shortest to read.

**Mechanism.** These levers hide latency under compute. Where there is no exposed latency to hide,
their delta is the noise floor. The ceiling they must beat is an **aggregate** bandwidth figure, not
a per-worker one — dividing a chip's streaming bandwidth by its worker count invents headroom that
does not exist, and that invented headroom is usually what made the lever look worth trying.

**Scope.** General as a gating rule. On SparseCore, streaming bandwidth is shared by all subcores —
there is no per-subcore escape hatch — and achieved bandwidth depends on row width (see
[STRAT-MLA-06](STRAT-MLA-06-sparsecore-bandwidth-is-width-dependent.md)). An achieved bandwidth is
not a ceiling. Compare it only with a figure for the **same scope** (per core vs all cores), never
an aggregate against one core; the unit's own ceiling is an unverified vendor figure, not something
the probes measure (see [tpu_hardware_facts.md](../tpu_hardware_facts.md) §2).
Hypothesis: on a compute-bound SparseCore kernel `emit_pipeline` and a hand-rolled double buffer are
indistinguishable; verify on the target (probes `emit_pipeline_vs_blockspec_grid` and
`manual_dma_vs_managed` ask the TensorCore version of this question). When several transfer arms all move together, suspect a shared confound
([STRAT-SC-10](STRAT-SC-10-shared-arm-confound.md)).

**Does not apply when** the kernel is bandwidth-bound — on such a pipeline buffering depth is the
whole point, and these levers separate immediately.

**How to check on your kernel.** Measure the unit's aggregate streaming bandwidth with probe
`sc_vs_tc_streaming_bandwidth` (and `sc_gather_row_width_scaling` for indirect rows at your width).
Compute the kernel's bytes moved over its measured time and place it on the roofline in
`reference/tpu_memory_overlapping.md` §1. Only if it sits on the bandwidth side, rank transfer
levers.
