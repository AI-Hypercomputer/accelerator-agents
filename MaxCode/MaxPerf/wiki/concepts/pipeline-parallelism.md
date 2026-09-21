<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "Pipeline Parallelism"
type: concept
tags: [stub, parallelism, scheduling]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Splits a model's **layers** across devices so each device holds a stage. Schedules determine how micro-batches flow through stages and how much bubble (idle time) the pipeline incurs. Family: **AFAB** (GPipe, bubble `(p−1)/m`), **1F1B** (same bubble but lower activation memory), **interleaved 1F1B** (v virtual chunks, bubble `(p−1)/(m·v)`), **Zero-Bubble** (near-zero given balanced chunks), **DualPipe** (DeepSeek-V3: splits backward into `B_input` and `B_weight`). On TPU, PP is rarely needed within a single pod because ICI bandwidth lets TP + FSDP scale further than NVLink on GPU; PP becomes relevant across DCN boundaries (multi-pod).

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Tensor Parallelism](tensor-parallelism.md)
- [FSDP (Fully Sharded Data Parallelism)](fsdp.md)
- [DCN](dcn.md)
- [ICI (Inter-Chip Interconnect)](ici.md)
- [Async Collectives](async-collectives.md)

## Sources

- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Section 6 (Pipeline Parallelism)
