---
title: "MXU (Matrix Unit)"
type: concept
tags: [stub, hardware, compute]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Systolic array on each TensorCore that executes matrix multiply; 128×128 on v5e and earlier, 256×256 on v6e. Issues one instruction per 8 cycles.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [VPU (Vector Programmable Unit)](vpu.md)
- [TensorCore (Tensor Node)](tensor-node.md)
- [MXU Utilization](mxu-utilization.md)
- [Dimension Alignment](dimension-alignment.md)
- [Peak FLOPs](peak-flops.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof Utilization Viewer](../sources/2026-xprof-utilization-viewer.md) — `raw/code/xprof/docs/utilization_viewer.md`
