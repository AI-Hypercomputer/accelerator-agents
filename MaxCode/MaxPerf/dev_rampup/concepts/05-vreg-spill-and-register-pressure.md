<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# VREG Spill & Register Pressure

## Prerequisites

- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md)
- [ISA Latency Tables](04-isa-latency-tables.md)

## Leads to

- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md)
- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md)

## Used by agents

- [MaxKernel](../../agents/max_kernel.md) — designs kernels to stay within register budget
- [DeepResearch](../../agents/deep_research.md) — models spill cost in algebraic analysis

## What it is

The TPU v6e has a finite vector register file (VREG). When a kernel requires more live values simultaneously than registers available, the compiler "spills" excess values to VMEM (or worse, HBM), inserting store/load pairs that consume DMA bandwidth and add latency. This is register pressure — the demand for simultaneous register slots exceeds supply.

Spills are especially costly on TPU because unlike CPUs where L1 cache access is 1-4 cycles, a VMEM round-trip costs 4-8 cycles and an HBM round-trip costs 50-200+ cycles. A kernel that spills even a few registers to HBM can see 2-3x slowdown compared to one that fits entirely in VREG.

Register pressure is a function of tile size, loop unroll factor, and the number of simultaneously live accumulator values. Larger tiles improve MXU utilization but increase register demand. This creates a tension that MaxPerf must navigate.

## Why it matters for MaxPerf

When MaxKernel writes Pallas kernels, it must choose tile dimensions that balance MXU efficiency against register pressure. Method M3 (Algebraic ISA Modeling) includes spill cost as a term in the cost function — a tile size that exceeds the register budget gets penalized by the predicted spill overhead. This prevents the system from recommending configurations that look good on paper but suffer in practice due to hidden spill costs.

## Worked example

A Pallas matmul kernel uses tile size 256x256 with 4 accumulator registers (each 128x128 in bf16 = 32 KB). Total register demand: 4 accumulators × 32 KB + 2 input tiles × 32 KB + workspace = 224 KB. If VREG capacity is 192 KB, the kernel spills 32 KB to VMEM each iteration. At 6 cycles per spill pair and 4 spills per iteration: 24 extra cycles. Reducing to tile size 128x256 cuts accumulators to 2 × 32 KB = 64 KB input, totaling 160 KB — no spills. Net effect: 15% fewer FLOPs per cycle from smaller tiles, but 20% time saved from eliminated spills. The smaller tile wins.

## See also

- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md) — where spilled data goes
- [ISA Latency Tables](04-isa-latency-tables.md) — cycle cost of spill operations
- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md) — incorporating spill cost into models
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md) — practical kernel design under pressure
