---
id: STRAT-MLA-14
title: An XLA scatter into a sub-32-bit tiled array can relayout the entire operand — cost scales with the array, not the update
applies_to: tensorcore
lever: layout
generality: general
phase: [plan]
verify: "manual: dump post-optimization HLO of the scatter at the production update count and look for reshape/copy ops whose shape is the whole operand"
---

# STRAT-MLA-14 — Look for a whole-operand relayout around a small scatter

**Rule.** Toolchain-dependent; verify before relying. `cache.at[i, j, k].set(rows)` on a large
sub-32-bit array can compile to "relayout the whole array, scatter, relayout back" — two full copies
of the operand around a small update. Whether it does depends on the **update count**, so a
microbenchmark at a small update count will not show it. Inspect the post-optimization HLO at the
production update count. When a cache write costs in proportion to the cache rather than to the
update, replace it with an aliased Pallas kernel (`input_output_aliases`) that DMAs only the touched
tiles.

**Mechanism.** A sub-32-bit array's native tiled layout packs several elements per sublane. At large
update counts the XLA scatter emitter wants a different layout, so layout assignment inserts
relayouts of the whole operand; at small update counts the same scatter runs in place. If the update
granularity aligns with whole native tiles, a DMA per tile group writes in place with no relayout.

**Scope.** XLA scatter / dynamic-update-slice into sub-32-bit tiled arrays on TPU, e.g. quantized KV
caches. See [tpu_hardware_facts.md](../tpu_hardware_facts.md) for sublane tiling by dtype.

**Does not apply when** the post-optimization HLO shows the scatter in place. At small update counts
the XLA path can beat the Pallas kernel, so dispatch per static update count rather than replacing
the scatter everywhere.

**How to check on your kernel.** Dump post-optimization HLO at each production update count (prefill
and decode). Look for reshape, copy or transpose ops whose shape is the whole operand adjacent to the
scatter; in the profile, check whether the write's time scales with array size. A/B against an aliased
Pallas DMA kernel at each update count and keep whichever wins per count.
