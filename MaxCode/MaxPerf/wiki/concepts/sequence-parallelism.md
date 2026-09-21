<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "Sequence Parallelism"
type: concept
tags: [stub, parallelism]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Companion to tensor parallelism: shards the non-matmul ops (LayerNorm, Dropout, residual) along the sequence axis so their activation memory scales with `sp_size`. Replaces the TP block's internal `all_reduce` with a `reduce_scatter` + `all_gather` pair of the same aggregate volume — no extra communication, reduced activation memory. In JAX this falls out of a `PartitionSpec` over the sequence axis.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Tensor Parallelism](tensor-parallelism.md)
- [Context Parallelism](context-parallelism.md)
- [Sharding (GSPMD)](sharding.md)
- [Async Collectives](async-collectives.md)

## Sources

- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Section 4 (sequence parallelism subsection)
