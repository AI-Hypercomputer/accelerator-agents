---
id: STRAT-SC-09
title: Unit choice follows what binds the algorithm, not which unit is wider
applies_to: sparsecore
lever: algorithm
generality: unit-wide
phase: [plan]
verify: "manual: name the binding parameter at your sizes, then time both units on the real shape (e.g. probe sc_topk_vs_lax_topk or sc_gather_vs_xla_gather)"
---

# STRAT-SC-09 — Unit choice follows what binds, not pipe width

**Rule.** Pick the execution unit from **what binds the algorithm**, not from which unit has more
throughput on paper. An algorithm bound by a parameter other than bandwidth will run the wide unit
far below its rated number, and the narrow unit that matches its access pattern can win by a margin
that looks impossible until you notice the wide unit was never bandwidth-bound to begin with.

**Mechanism.** A selection algorithm's work scales with the number of items selected (`k`), not with
the bytes scanned. A wide SIMD unit prices that as poor effective bandwidth — it moves the bytes at
full rate and then serializes on the selection — while a unit with gather/scatter and lane-private
state prices it as ordinary work. Likewise an indirect gather is bound by **descriptor issue rate**:
rewriting it as a TensorCore `pallas_call` to avoid a SparseCore hop pays per descriptor, and no
blocking or layout change removes that.

**Scope.** SparseCore vs TensorCore. What transfers is the question, *what binds?*, not any ratio.
The right unit with the wrong cooperation granularity is still the wrong choice: a SparseCore
algorithm that forces many-way cross-subcore cooperation pays a `subcore_barrier` per step and can
lose to a less cooperative layout.

**Does not apply when** `k` is small enough that the scan dominates. Then the algorithm really is
bandwidth-bound and the wide unit is correct.

**How to check on your kernel.** State the binding parameter at your sizes (`k`, descriptor count,
bytes). Compute the wide unit's *effective* bandwidth on the real shape (bytes / measured time) and
compare it to the rated figure; a large gap means bandwidth does not bind. Then time both units:
probes `sc_topk_vs_lax_topk`, `sc_gather_vs_xla_gather`, `sc_moe_route_vs_xla` cover the common
cases. Count `subcore_barrier`s per call in any cooperative design.
