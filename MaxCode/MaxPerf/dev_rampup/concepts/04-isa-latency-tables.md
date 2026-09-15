<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# ISA Latency Tables

## Prerequisites

- [TPU v6e Architecture](01-tpu-v6e-architecture.md)

## Leads to

- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md)
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md)
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md)

## Used by agents

- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — validates expected vs actual op latencies
- [DeepResearch](../../agents/deep_research.md) — models instruction cost algebraically
- [MaxTile](../../agents/max_tile.md) — predicts tile-size cost without running experiments
- [MaxKernel](../../agents/max_kernel.md) — selects instruction sequences for Pallas kernels

## What it is

The ISA (Instruction Set Architecture) latency table catalogs the cycle cost of each instruction on TPU v6e. Key entries include MXU matmul (varies by tile size, typically 8-16 cycles per 128x128 output tile depending on accumulation depth), vector ALU operations (1-4 cycles), scalar operations (1 cycle), and DMA initiation (fixed overhead + transfer time proportional to payload size).

Latencies are not always fixed — some instructions have data-dependent timing (e.g., division) and some depend on operand location (VREG vs VMEM). The table also includes pipeline depths: the MXU has a deep pipeline (~22 stages), meaning it takes 22 cycles before the first result emerges, but can accept a new input every cycle once full.

These tables are not published in full by Google but are inferred empirically through microbenchmarks. MaxPerf uses these inferred values to predict performance without exhaustive grid search.

## Why it matters for MaxPerf

Method M3 (Algebraic ISA Modeling) builds symbolic cost expressions from these latency values. Instead of running hundreds of tile-size experiments, DeepResearch and MaxTile compose instruction costs algebraically to predict optimal configurations. This turns a combinatorial search into a closed-form calculation, which is the core insight that makes MaxPerf faster than brute-force autotuning.

## Worked example

Consider choosing between two Pallas kernel implementations for a reduction:
- Option A: 4 vector adds (4 cycles each) + 1 reduce (8 cycles) = 24 cycles/iteration
- Option B: 1 tree-reduce pattern = 3 levels × 8 cycles = 24 cycles/iteration, but with half the register pressure

The ISA table shows both have equal cycle cost, but Option B's lower register pressure (from [VREG Spill](05-vreg-spill-and-register-pressure.md)) avoids spills that would add 6 extra DMA cycles. Total: A = 30 cycles, B = 24 cycles. Choose B.

## See also

- [TPU v6e Architecture](01-tpu-v6e-architecture.md) — the hardware executing these instructions
- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md) — building cost models from these tables
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md) — hidden costs not in the basic table
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md) — writing kernels that exploit these latencies
