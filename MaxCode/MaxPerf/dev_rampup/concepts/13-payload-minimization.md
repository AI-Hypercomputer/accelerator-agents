<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Payload Minimization

## Prerequisites

- [Collective Operations](11-collective-operations.md)
- [Commutativity & Graph Commutation](12-commutativity-and-graph-commutation.md)

## Leads to

- [Pipeline Stage Balancing](17-pipeline-stage-balancing.md)

## Used by agents

- [MaxShard](../../agents/max_shard.md) — implements payload reduction rewrites
- [DeepResearch](../../agents/deep_research.md) — derives minimal-communication graph structures

## What it is

Payload minimization is the practice of reducing the number of bytes that flow through collective operations. Every byte transmitted over ICI costs time — minimizing payload directly reduces communication latency and frees ICI bandwidth for overlapping with other traffic.

Strategies for payload minimization:
1. **Commutation** — push collectives past dimension-reducing operations (see [concept 12](12-commutativity-and-graph-commutation.md))
2. **Compression** — use lower precision for communication (e.g., all-reduce in bf16 instead of fp32, or quantize to int8 before gather)
3. **Selective communication** — only transmit what's needed (e.g., top-k sparsification for gradient all-reduce)
4. **Collective fusion** — merge multiple small collectives into one larger one to amortize fixed overhead
5. **Topology-aware placement** — arrange data so that most communication is within high-bandwidth neighborhoods

The challenge is maintaining numeric equivalence (for compression) or convergence properties (for sparsification) while reducing bytes.

## Why it matters for MaxPerf

Method M7 (Payload Minimization via Algebraic Commutativity) focuses primarily on strategies 1 and 4 — restructuring the graph to reduce communication without losing any information. This is lossless optimization: the results are bitwise identical, just computed with less traffic. When TPUDiagnoseAgent shows that ICI is saturated or that collectives dominate step time, payload minimization is the primary lever.

## Worked example

A transformer model does gradient all-reduce on 4 weight matrices per layer:
- Q: [4096, 4096] = 32 MB
- K: [4096, 4096] = 32 MB
- V: [4096, 4096] = 32 MB
- O: [4096, 4096] = 32 MB

Total: 4 all-reduces × 32 MB = 128 MB payload per layer.

Optimization 1 (collective fusion): merge into one all-reduce of concatenated gradients = 128 MB but with 1x fixed overhead instead of 4x. Saves ~0.1 ms per layer from reduced synchronization.

Optimization 2 (commutation): if output projection O = concat(heads) × W_o, and gradient flows through this linearly, commute all-reduce before the concat-projection. Gradient before projection: [4096, 1024] per head × 4 heads. All-reduce the smaller pre-projection gradients: 4 × 16 MB = 64 MB. Savings: 50% payload reduction.

## See also

- [Collective Operations](11-collective-operations.md) — the operations being optimized
- [Commutativity & Graph Commutation](12-commutativity-and-graph-commutation.md) — algebraic technique for payload reduction
- [Pipeline Stage Balancing](17-pipeline-stage-balancing.md) — hiding remaining communication latency
- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md) — identifying when communication is the bottleneck
