<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Checkpoint: Section 1 — Hardware

Test your understanding of concepts 1-5 with these scenario-based questions.

---

## Question 1: Roofline Classification

An xprof trace shows a fused operation taking 1.8 ms. From the HLO, the operation performs a batched matmul [64, 2048, 512] × [64, 512, 2048] followed by element-wise GeLU. Total FLOPs: ~270 GFLOP. Total bytes loaded from HBM: 256 MB.

**A)** What is the arithmetic intensity of this operation?

**B)** Is this operation compute-bound or memory-bound on TPU v6e (ridge point ≈ 1122 FLOPs/byte)?

**C)** What is the achieved FLOP/s, and what percentage of peak (920 TFLOPS) does it represent?

---

## Question 2: Memory Hierarchy Decision

A model has a pointwise activation function (SiLU) applied to a [8192, 16384] bf16 tensor between two matmuls. XLA has placed this as a separate unfused op, causing a full HBM round-trip.

**A)** How many bytes does this unnecessary materialization cost (write + read)?

**B)** At 820 GB/s HBM bandwidth, how much time does this waste per occurrence?

**C)** Which agent would you route this to, and why: MaxKernel (Pallas kernel) or AutoRefactor (fusion flags)?

---

## Question 3: DMA Idle Diagnosis

An xprof timeline shows repeating pattern: 0.4 ms compute → 0.25 ms DMA idle → 0.4 ms compute → 0.25 ms idle. The kernel is a loop body that processes tiles from HBM.

**A)** What is the effective utilization of the compute units (fraction of time doing useful work)?

**B)** What is the most likely root cause: insufficient double-buffering, spurious barriers, or HBM bandwidth saturation?

**C)** If the DMA transfer takes 0.35 ms and compute takes 0.40 ms, what would the timeline look like with correct double-buffering?

---

## Question 4: Register Pressure Tradeoff

MaxKernel proposes a Pallas kernel with tile size 256×256. Each tile requires:
- 4 accumulator registers (128×128, fp32, 64 KB each = 256 KB)
- 2 input tiles (128×128, bf16, 32 KB each = 64 KB)
- 16 KB workspace

Total: 336 KB. VREG budget: 256 KB.

**A)** How much excess register demand exists (spill volume per iteration)?

**B)** If each VMEM spill pair costs 8 cycles, and there are 5 spills per tile, what is the overhead per tile?

**C)** Propose a tile size reduction that eliminates spills. What is the tradeoff?

---

## Question 5: ISA Cost Reasoning

Two kernel designs for a vector reduction:
- Design A: serial accumulation — 128 vector adds at 4 cycles each = 512 cycles
- Design B: tree reduction — 7 levels × 64 parallel adds at 4 cycles each = 28 cycles, but requires 64 temporary registers

**A)** Which design has lower cycle count?

**B)** If the VREG budget only has room for 32 temporary registers, what happens to Design B?

**C)** At what register budget does Design A become preferable to Design B (including spill costs of 8 cycles per evicted register)?

---

## Answers guidance

After working through these, verify your reasoning against the concept pages:
- [01 - TPU v6e Architecture](../concepts/01-tpu-v6e-architecture.md)
- [02 - HBM & Memory Hierarchy](../concepts/02-hbm-and-memory-hierarchy.md)
- [03 - DMA Idle Time](../concepts/03-dma-idle-time.md)
- [04 - ISA Latency Tables](../concepts/04-isa-latency-tables.md)
- [05 - VREG Spill & Register Pressure](../concepts/05-vreg-spill-and-register-pressure.md)
