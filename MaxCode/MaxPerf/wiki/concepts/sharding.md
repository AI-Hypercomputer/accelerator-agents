---
title: "Sharding (GSPMD)"
type: concept
tags: [stub, parallelism]
created: 2026-04-22
updated: 2026-04-22
sources: 3
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->


Partitioning tensors across a device mesh; mesh design allocates dims to ICI/DCN based on expected collective bandwidth.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [FSDP (Fully Sharded Data Parallelism)](fsdp.md)
- [Tensor Parallelism](tensor-parallelism.md)
- [ICI (Inter-Chip Interconnect)](ici.md)
- [DCN (Data Center Network)](dcn.md)
- [Collective Communication](collective-communication.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof Megascale Stats](../sources/2026-xprof-megascale-stats.md) — `raw/code/xprof/docs/megascale_stats.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — 5D parallelism composition (DP × TP × PP × CP × EP); the five axes partition over a device mesh, axis assignment is the core design choice
- [JAX HuggingFace Part 2](../sources/2026-jax-huggingface-part-2.md) — `jax.make_mesh` + `NamedSharding` + `PartitionSpec` primitives in practice; per-weight sharding map for Llama-2-7B; inputs must be replicated (not sharded) for TP
