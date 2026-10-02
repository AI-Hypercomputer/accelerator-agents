---
title: "Dtype Strategy"
type: concept
tags: [stub, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Choice and mixing of bf16, fp32, fp8, int8 across params/activations/compute; bf16 is native on MXU and fp32 weights force per-matmul cast (~17% cost).

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Int8 Quantization](int8-quantization.md)
- [MXU (Matrix Unit)](mxu.md)
- [Peak FLOPs](peak-flops.md)
- [MXU Utilization](mxu-utilization.md)
- [Dimension Alignment](dimension-alignment.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — BF16 vs FP16 vs FP8 format comparison; DeepSeek-V3 FP8 tile scheme (1×128 activations, 128×128 weights+scales); no loss-scaler path on TPU (BF16 native)
