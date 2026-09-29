---
title: "Kernel class: grouped / ragged / indirection (GMM, MoE, ragged dot, paged KV)"
type: concept
tags: [kernel-class, grouped, ragged, moe, paged, pallas, briefs-tier]
created: 2026-07-22
---

**Route here when** K1 shows per-group serialization, index-gather chains, offset/length metadata, or page tables. Paged/ragged attention takes this as SECONDARY to [attention](attention.md).

- **Strategy**: one program per group/tile; recover the group→row map from offset metadata (`cumsum(sizes)`); block so no tile straddles a group boundary; scalar-prefetch the index arrays (catalog class 2 machinery).
- **In-kernel manual DMA gather beats outer gather loops**: an outer per-item gather before `pallas_call` compiles to a serialized dynamic-slice/DUS chain (64 sequential steps staging ~67 MB observed) — move it inside via `pltpu.make_async_copy`.
- **Streaming granularity is a sweepable lever with a rig-measurable optimum**: pages-per-block coarsening AND re-refinement both won on the same kernel at different points; the optimum shifts with the rest of the kernel's VMEM footprint.
- **DMA-semaphore flag wall**: per-page/per-item semaphores exhaust near 256 flags — share one semaphore per (array, slot) (BRIEFS §6).
- **Dispatch floor is the class's standing refute risk**: tiny per-program work (per-element gather, single-token decode) loses to XLA's fused path regardless of inner-loop quality — check granularity before authoring; the rewrite is often the right tool.
- **Uniform-group ceiling**: when groups are uniform, the dense-equivalent einsum can run ~68% MXU and nothing ragged beats it (even library ragged primitives run severalfold slower) — the honest verdict is refute on the ceiling.
- Portability: loop-carried `pl.ds` offsets must be tiling-aligned; negative indices lower to dynamic_slice and die on fleet libtpu (BRIEFS §5).
