---
id: STRAT-MLA-13
title: Bounding a kernel's loops by the live length does not bound its DMA — and index-assigned work leaves the live units at full cost
applies_to: any
lever: algorithm
generality: general
phase: [plan]
verify: "manual: time the kernel at several live lengths with the static extent fixed; a flat time means the transfer, not the body, binds"
---

# STRAT-MLA-13 — A compute bound is not a DMA bound

**Rule.** A kernel whose body skips work past a runtime length can still pay for all of it: an
auto-pipelined `BlockSpec` (or any static-extent copy) moves the **whole block** whatever the body
reads. When the kernel is transfer-bound, its cost follows the static extent, not the live length.
Fixing the DMA alone is not enough if work is assigned by **index**: the live slices land on a subset
of the units, which keep their full cost while the rest go idle. Make the partition width itself
dynamic (e.g. sub-row width `ceil(max_len / num_units)`) so the live range spreads over every unit,
then bound each unit's DMA extent to its live slice.

**Mechanism.** The pipeline emitter sizes transfers from the BlockSpec, which is static; the length
only reaches the loops inside the body. On a unit whose throughput is set by streaming (SparseCore),
the skipped compute was never the cost. Index assignment (`row · P + j` → unit `j mod W`) is the
lockstep problem of [STRAT-SC-05](STRAT-SC-05-wave-longest-row.md) in another form: the critical path
is the busiest unit. After rebalancing, the saving should track the dead fraction of the static extent
and vanish when there is no dead extent — use that case as the null control.

**Scope.** Any kernel with a static-extent input pipeline and a runtime live length, on either unit.
Traced DMA extents on SparseCore have their own lowering constraints; see
[STRAT-MLA-12](STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md).

**Does not apply when** the kernel is compute-bound on the live work — then the extra transfer is
hidden (see [STRAT-SC-08](STRAT-SC-08-prove-the-load-binds-first.md)).

**How to check on your kernel.** Check whether the input BlockSpec / DMA extent is static while the
body loops are length-bounded. Time the kernel at several live lengths: if time is flat, the static
transfer binds. Then check how live work maps onto units (cores, subcores, grid shards).
