---
title: "ICI (Inter-Chip Interconnect)"
type: concept
tags: [stub, hardware, interconnect]
created: 2026-04-22
updated: 2026-04-22
sources: 6
---

Intra-slice high-bandwidth interconnect between TPU chips within an island; used for tensor-parallel and FSDP collectives. Distinct from DCN.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [DCN (Data Center Network)](dcn.md)
- [Megascale](megascale.md)
- [ICI Roofline](ici-roofline.md)
- [Tensor Parallelism](tensor-parallelism.md)
- [FSDP (Fully Sharded Data Parallelism)](fsdp.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof Utilization Viewer](../sources/2026-xprof-utilization-viewer.md) — `raw/code/xprof/docs/utilization_viewer.md`
- [xprof Megascale Stats](../sources/2026-xprof-megascale-stats.md) — `raw/code/xprof/docs/megascale_stats.md`
- [xprof Roofline Model](../sources/2026-xprof-roofline-model.md) — `raw/code/xprof/docs/roofline_model.md`
- [JAX HuggingFace Part 2](../sources/2026-jax-huggingface-part-2.md) — 8-chip TP on TPU v6e host; sub-linear scaling (3.8× on 8 chips) attributed to collectives riding ICI
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — TP axis placement and `tp=8` GPU NVLink cliff vs topology-dependent ICI-ring limits on TPU; interconnect bandwidth table (NVLink ~900 GB/s H100 vs ICI varies by generation)
