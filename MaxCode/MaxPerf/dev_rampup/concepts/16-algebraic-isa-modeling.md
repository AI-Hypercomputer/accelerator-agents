<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Algebraic ISA Modeling

## Prerequisites

- [ISA Latency Tables](04-isa-latency-tables.md)
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md)
- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md)

## Leads to

- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md)

## Used by agents

- [DeepResearch](../../agents/deep_research.md) — builds symbolic cost expressions
- [MaxTile](../../agents/max_tile.md) — evaluates configurations without running them

## What it is

Algebraic ISA modeling replaces empirical grid search with symbolic cost expressions built from ISA latency tables. For a given kernel configuration (tile size, unroll factor, pipeline depth), the model computes total cycle count as a closed-form expression:

```
T(m, n, k) = ⌈m/128⌉ · ⌈n/128⌉ · (MXU_latency(k) + pipeline_drain)
           + DMA_overhead(m, n, k)
           + spill_cost(live_regs(m, n) - REG_BUDGET)⁺
```

Each term is derived from the ISA table: MXU latency per tile, DMA initiation cost, transfer time per byte, and spill penalty per register eviction. The model predicts execution time as a function of configuration parameters, enabling optimization without measurement.

The key advantage over grid search: a grid search of 100 configurations requires 100 compilations and runs. The algebraic model evaluates all 100 in milliseconds on paper, then runs only the top 3 to validate. This is method M3's core contribution.

## Why it matters for MaxPerf

Method M3 (Algebraic ISA Modeling, not Grid Search) uses this technique to bypass the combinatorial explosion of autotuning. DeepResearch derives the cost expression symbolically, MaxTile evaluates it across the configuration space, and the system only runs experiments on the predicted-optimal configurations. This turns hours of autotuning into minutes of algebraic reasoning + a handful of validation runs.

## Worked example

For a VMEM-resident matmul kernel with tile size (m, n) and inner dim k=8192:

```
T(m=256, n=256, k=8192) =
  MXU:   (256/128) × (256/128) × (8192/128 × 8 cycles) = 2 × 2 × 512 = 2048 cycles
  DMA:   2 loads × (256 × 8192 × 2 bytes) / (800 GB/s / clock) = 1640 cycles
  Spill: live = 4 accum × 128×128×4B = 256KB > 192KB budget → 2 spills × 12 cycles = 24 cycles
  Total: 2048 + 1640 + 24 = 3712 cycles

T(m=128, n=256, k=8192) =
  MXU:   1 × 2 × 512 = 1024 cycles
  DMA:   2 loads × (128 × 8192 × 2B) / rate = 820 cycles
  Spill: live = 2 accum × 128×128×4B = 128KB < 192KB → 0
  Total: 1024 + 820 + 0 = 1844 cycles (but half the output, so 3688/output-equivalent)
```

Smaller tile wins due to zero spill cost, even with slightly less MXU efficiency.

## See also

- [ISA Latency Tables](04-isa-latency-tables.md) — the data feeding this model
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md) — spill term derivation
- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md) — using the model to prune search space
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md) — implementing the optimal configuration
