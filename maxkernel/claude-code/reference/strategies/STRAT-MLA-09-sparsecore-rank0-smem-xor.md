---
id: STRAT-MLA-09
title: On SparseCore a rank-0 load is legal only from SMEM, and an SMEM load only at rank 0
applies_to: sparsecore
lever: layout
generality: unit-wide
phase: [implement]
verify: "manual: lower a SparseCore kernel that reads a scalar from a VMEM in_spec and one that reads a vector from SMEM; both should be rejected"
---

# STRAT-MLA-09 — The SparseCore rank-0 / SMEM exclusive-or

**Rule.** Toolchain-dependent; verify before relying. On SparseCore the two properties have been
seen locked together: a **rank-0 (scalar) load is legal only from SMEM**, and an **SMEM load is legal
only at rank 0**. You then cannot read a scalar out of a vector-memory `in_spec`, nor a vector out of
SMEM. This rules out the obvious shape of a per-window scalar predicate.

**Mechanism.** Enforced in the SparseCore lowering pass at its rank/memory-space check
(`jax/_src/pallas/mosaic/sc_lowering.py`; locate the check with `tools/search_api.py`). The design it
kills is the natural one: one scalar predicate per window in an SMEM `in_spec`, branched on. The shape
that works is to **vector-load a row subchunk into VMEM and lane-extract element 0** — the predicate
arrives as a vector and is narrowed afterwards rather than fetched as a scalar.

**Scope.** SparseCore kernels. A lowering restriction is a fact about a build; re-read the check after
a toolchain change. The workaround is cheap and is the thing to carry: if a predicate must cross into
SparseCore, plan for it to arrive as a vector. Related lowering facts: probes `sc_primitive_lowering`
and `sc_pl_ops_inside_kernel`.

**Does not apply on** TensorCore, where scalar reads from an SMEM / scalar-prefetch operand are
routine — this asymmetry is what makes the restriction surprising when porting a working TensorCore
Pallas kernel onto the unit.

**How to check on your kernel.** Re-read the rank/memory-space check in the SparseCore lowering pass,
or compile the SMEM-predicate design once; if it lowers on your toolchain, the vector-load workaround
is no longer required.
