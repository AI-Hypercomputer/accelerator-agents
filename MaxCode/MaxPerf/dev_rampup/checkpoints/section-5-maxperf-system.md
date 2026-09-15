<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Checkpoint: Section 5 — MaxPerf System

Test your understanding of concepts 20-22 with these scenario-based questions.

---

## Question 1: Pallas Kernel Design Decision

Fusion boundary accounting shows a boundary between LayerNorm and a subsequent matmul costing 0.08 ms per layer × 128 layers = 10.2 ms per step. The matmul is currently handled by a custom-call to an optimized GEMM library.

**A)** What are the two options for eliminating this boundary?

**B)** If you write a Pallas kernel that fuses LayerNorm + matmul, what risk do you face regarding the matmul's performance compared to the library GEMM?

**C)** How would you calculate whether the fusion benefit (10.2 ms saved) outweighs potential matmul performance loss? What measurement would you need?

---

## Question 2: Sublane Layout Impact

A Pallas kernel loads a [256, 256] bf16 tile from HBM. The data is stored in row-major layout in HBM, but the kernel's first operation is a column-wise reduction (reduce along dim=0).

**A)** Why does the mismatch between storage layout and access pattern matter on TPU?

**B)** What are two approaches to fix this: one at kernel design time, one at graph level?

**C)** If the transpose costs 32 cycles per 128×128 sub-tile and there are 4 sub-tiles, what is the overhead? Compare to the DMA transfer time for 256×256×2 bytes at 800 GB/s effective.

---

## Question 3: Hypothesis Classification

TPUDiagnoseAgent produces a diagnostic vector showing:
```
mxu_utilization: 0.52
hbm_bw_utilization: 0.71
bottleneck_class: memory
top_ops: [("unfused_layernorm", 180ms, memory), ("dot.qkv", 320ms, compute)]
dma_idle_fraction: 0.15
fusion_boundary_cost_ms: 85
```

**A)** Generate two hypotheses from this vector, specifying class and origination source for each.

**B)** Estimate the predicted impact of each hypothesis.

**C)** Which hypothesis would you pursue first and why?

---

## Question 4: Diagnostic Vector Interpretation

Two models have these diagnostic vectors:

Model A:
```
step_time_ms: 900, mxu_utilization: 0.72, hbm_bw_utilization: 0.35
bottleneck_class: compute, collective_time_ms: 45, bubble_fraction: 0.03
```

Model B:
```
step_time_ms: 900, mxu_utilization: 0.31, hbm_bw_utilization: 0.82
bottleneck_class: memory, collective_time_ms: 180, bubble_fraction: 0.18
```

Both have the same step time. Which is closer to optimal and why?

**A)** For Model A, what does high MXU + low HBM BW indicate about the workload?

**B)** For Model B, what are the top 3 optimization opportunities (in priority order)?

**C)** Which model has more total optimization potential, and which agents would you assign?

---

## Question 5: Hypothesis Origination Comparison

Three hypotheses for the same model:

| # | Hypothesis | Origin |
|---|-----------|--------|
| H1 | Fuse 3 unfused ops identified in xprof | Profile-driven |
| H2 | Apply attention optimization that worked on a similar model | Analogy-driven |
| H3 | Commute all-reduce past linear layer (algebraically valid) | Pattern-driven |

**A)** Assign confidence levels (high/medium/low) to each and justify.

**B)** If H2's predicted impact is 3× larger than H1 or H3, does it necessarily go first? What other factor matters?

**C)** Describe a scenario where the orchestrator would pursue H3 before H1 despite H1's higher confidence.

---

## Answers guidance

Verify your reasoning against:
- [20 - Pallas Kernels & Sublane Layout](../concepts/20-pallas-kernels-and-sublane-layout.md)
- [21 - Hypothesis Classes & Origination](../concepts/21-hypothesis-classes-and-origination.md)
- [22 - Diagnostic Vector](../concepts/22-diagnostic-vector.md)
