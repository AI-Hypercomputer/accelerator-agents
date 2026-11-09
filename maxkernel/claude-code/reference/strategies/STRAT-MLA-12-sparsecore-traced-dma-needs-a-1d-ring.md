---
id: STRAT-MLA-12
title: A SparseCore DMA with a traced offset and extent lowers — into a 1-D ring slot with multiple_of hints, not a 2-D ring row
applies_to: sparsecore
lever: pipelining
generality: unit-wide
phase: [implement]
verify: "manual: compile a 2-D (nbuf, len) VMEM ring sliced as ring.at[slot, pl.ds(0, ext)] with traced ext; if it lowers, the 1-D form is not required"
---

# STRAT-MLA-12 — Traced-extent SparseCore DMAs want a flat ring

**Rule.** Toolchain-dependent; verify before relying. On SparseCore, `make_async_copy` from HBM at a
traced offset with a traced extent has been seen to work only into a destination slice the verifier
can prove tile-aligned: a **1-D** VMEM ring of `nbuf × slot_len` words, addressed at
`pl.multiple_of(slot * slot_len, m)`, with `pl.multiple_of(ext, m)` and `pl.multiple_of(off, m)` for
the alignment multiple `m` the verifier demands (the compile error names the tile). Slicing a **2-D**
`(nbuf, slot_len)` ring as `ring.at[slot, pl.ds(0, ext)]` was rejected at compile time.

**Mechanism.** The 2-D ring gets a tiled layout, so a dynamic-size slice along the tiled row dimension
cannot be proven a multiple of the tile; Mosaic refuses rather than guessing ("Slice sizes along tiled
dimensions must be aligned to tiles"). A flat ring with explicit `multiple_of` hints gives the
verifier the proof it needs. Skip zero-extent copies under `pl.when(ext > 0)` on both `.start()` and
`.wait()`: every started DMA must be waited (see [tpu_hardware_facts.md](../tpu_hardware_facts.md)).

**Scope.** SparseCore vector-subcore kernels moving 32-bit data. No upstream test covers a contiguous
traced-size DMA on SparseCore; the nearest is the traced-size destination slice in
`tests/pallas/tpu_sparsecore_pallas_test.py` (find it with `tools/search_api.py`). The TensorCore
analogue (traced extent into a fixed buffer) is covered by probes `traced_dma_length` and
`traced_dma_offset_by_dtype`. Sharpens
[STRAT-MLA-08](STRAT-MLA-08-sparsecore-data-dependent-skip-lowers.md), which covers traced *grid
lengths* and `pl.when` around `emit_pipeline`, not traced DMA extents.

**Does not apply when** the extent is static — then an ordinary 2-D ring or BlockSpec works — or when
a newer toolchain lowers the 2-D slice.

**How to check on your kernel.** Compile the 2-D ring slice with a traced extent on the target
toolchain. If it is rejected, use the flat ring with `multiple_of` hints and check bit-exactness
against a static-BlockSpec version across extents including zero.
