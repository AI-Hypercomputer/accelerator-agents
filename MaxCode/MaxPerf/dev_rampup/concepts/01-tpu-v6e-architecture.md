<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# TPU v6e Architecture

## Prerequisites

None — this is a starting concept.

## Leads to

- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md)
- [DMA Idle Time](03-dma-idle-time.md)
- [ISA Latency Tables](04-isa-latency-tables.md)

## Used by agents

- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — profiles hardware utilization
- [Orchestrator](../../agents/maxperf_orchestrator.md) — selects optimization strategies based on hardware limits

## What it is

The TPU v6e (Trillium) is Google's sixth-generation Tensor Processing Unit. Each chip contains a matrix unit (MXU) capable of bf16/fp32 matrix multiplications, a vector unit for element-wise operations, a scalar unit for control flow, and high-bandwidth memory (HBM) for data storage. The chip is connected to other chips via inter-chip interconnect (ICI) for distributed computation.

The MXU operates on 128x128 tiles and delivers peak throughput when both operands stream continuously from registers. The vector unit handles activations, reductions, and non-linear functions. Both units share a common register file (VREG) that sits between compute and memory.

Key specifications: the v6e provides approximately 920 TFLOPS bf16 peak, ~820 GB/s HBM bandwidth, and 4.5 TB/s ICI bisection bandwidth per pod slice. These numbers define the theoretical ceilings that MaxPerf optimization targets.

## Why it matters for MaxPerf

Every optimization method in MaxPerf ultimately targets one of the hardware ceilings: MXU utilization, memory bandwidth, or ICI throughput. Understanding the architecture means knowing which ceiling a workload hits and which agent should respond. When TPUDiagnoseAgent reports that MXU is at 40% utilization, you need the architectural context to know whether memory stalls, pipeline bubbles, or insufficient parallelism is the cause.

## Worked example

A transformer layer performs a matmul of shape [2048, 8192] x [8192, 8192] in bf16. The MXU processes 128x128 tiles, so this decomposes into (2048/128) x (8192/128) x (8192/128) = 16 x 64 x 64 = 65,536 tile operations. At 920 TFLOPS peak, the compute time for 2 x 2048 x 8192 x 8192 = 274.9 GFLOP is ~0.30 ms. If the measured time is 0.75 ms, the MXU utilization is 0.30/0.75 = 40%, indicating a non-compute bottleneck that needs further diagnosis.

## See also

- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md) — the memory system feeding the MXU
- [ISA Latency Tables](04-isa-latency-tables.md) — per-instruction cost on this architecture
- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md) — formal framework for ceiling analysis
