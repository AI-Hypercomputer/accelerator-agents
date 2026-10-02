---
title: "Tensor Parallelism"
type: concept
tags: [stub, parallelism]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Splits individual ops (matmul) across devices; cheap within an ICI island (≤8 on most generations), expensive across nodes.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [FSDP (Fully Sharded Data Parallelism)](fsdp.md)
- [Sharding (GSPMD)](sharding.md)
- [ICI (Inter-Chip Interconnect)](ici.md)
- [ICI Roofline](ici-roofline.md)
- [All-Reduce](all-reduce.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — column+row GEMM pair inside transformer block (Megatron); GPU caps at tp=8 at NVLink edge, on TPU the cap is topology-dependent (ICI-ring size), not a fixed number
- [JAX HuggingFace Part 2](../sources/2026-jax-huggingface-part-2.md) — 8-way TP of Llama-2-7B on TPU v6e via gSPMD + NamedSharding; column/row recipe (Q/K/V/gate/up = `P('axis', None)`, O/down = `P(None, 'axis')`); ~3.8× speedup vs single-chip
