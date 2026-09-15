<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# DMA Idle Time

## Prerequisites

- [TPU v6e Architecture](01-tpu-v6e-architecture.md)

## Leads to

- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md)
- [HLO Barrier Removal](24-hlo-barrier-removal.md)

## Used by agents

- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — detects DMA stalls in xprof traces
- [AutoRefactor](../../agents/auto_refactor.md) — removes barriers that prevent DMA overlap

## What it is

DMA (Direct Memory Access) engines on TPU v6e asynchronously transfer data between HBM and VMEM while compute units execute instructions. When the DMA pipeline is idle — no outstanding transfers are in flight — compute units eventually stall waiting for data. DMA idle time appears in xprof traces as gaps between transfer completions and the next transfer initiation.

Three primary causes of DMA idle time: (1) unnecessary barriers in the HLO graph that serialize DMA and compute, (2) insufficient double-buffering where the compiler fails to prefetch the next tile, and (3) fusion boundaries that force materialization to HBM between operations that could share VMEM.

Eliminating DMA idle time is about ensuring the memory system is always working — that by the time compute finishes with one tile, the next tile is already in VMEM.

## Why it matters for MaxPerf

DMA idle time is the most common cause of low MXU utilization in xprof profiles. When TPUDiagnoseAgent identifies DMA gaps, method M4 (Execution Graph Unblocking) targets the root cause — typically unnecessary control dependencies or overly conservative barriers inserted by XLA. AutoRefactor can then remove these barriers or restructure the schedule to overlap transfers with computation.

## Worked example

An xprof trace shows a matmul consuming 0.5 ms of compute, but the total op time is 0.9 ms. The trace reveals 0.4 ms where the DMA engine has no outstanding requests — compute finishes tile N but tile N+1 hasn't arrived from HBM. The fix: AutoRefactor enables double-buffering by removing a spurious control dependency between the matmul and an unrelated operation, allowing XLA to schedule prefetch of tile N+1 during computation of tile N. Post-fix time: 0.55 ms.

## See also

- [TPU v6e Architecture](01-tpu-v6e-architecture.md) — hardware context for DMA engines
- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md) — the memory levels DMA connects
- [HLO Barrier Removal](24-hlo-barrier-removal.md) — technique for eliminating unnecessary barriers
- [XLA Compilation Passes](08-xla-compilation-passes.md) — where DMA scheduling decisions are made
