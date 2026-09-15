<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Checkpoint: Section 4 — Optimization Methods

Test your understanding of concepts 14-19 with these scenario-based questions.

---

## Question 1: Roofline Interpretation

An xprof trace shows an operation with:
- 62% HBM bandwidth utilization
- Arithmetic intensity: 3 FLOPs/byte
- MXU utilization: 18%

**A)** Is this operation compute-bound or memory-bound? (Ridge point ≈ 1122 FLOPs/byte)

**B)** Why is MXU utilization only 18% despite the op being in one regime? What's the likely cause?

**C)** Which agent would investigate this, and which method would they use?

---

## Question 2: Constraint Narrowing

You need to optimize tile size for a custom kernel. The problem dimensions are [2048, 4096] × [4096, 4096]. Constraints:
- Tiles must divide dimensions evenly
- Tiles must be multiples of 128 (MXU alignment)
- VREG budget: 256 KB
- Each tile config needs 2 input tiles (bf16) + 1 accumulator (fp32) in registers

**A)** List all valid tile sizes for the M dimension (multiples of 128 that divide 2048).

**B)** For tile M=256, N=256: compute register demand (2 inputs [M×4096 slice, 4096×N slice] are loaded in inner-loop chunks of 128; accumulator is M×N in fp32). Does this fit in 256 KB?

**C)** What additional constraint from profiling could further narrow the space?

---

## Question 3: Algebraic vs Empirical

DeepResearch's algebraic model predicts tile config A costs 1800 cycles and config B costs 2100 cycles. Empirical measurement shows A = 2200 cycles, B = 2150 cycles.

**A)** Why might the algebraic model be wrong (give two possible reasons)?

**B)** Does this invalidate the algebraic approach? How should MaxPerf respond?

**C)** How would you update the cost model to account for the discrepancy?

---

## Question 4: Pipeline Stage Decision

A 4-stage pipeline has measured stage times: [8.2 ms, 8.5 ms, 12.1 ms, 7.8 ms]. Microbatch count: 8.

**A)** What is the throughput-limiting stage?

**B)** Calculate the bubble fraction with these unbalanced stages.

**C)** If moving one layer from stage 3 to stage 4 changes times to [8.2, 8.5, 10.8, 9.1] ms, is this a net improvement? By how much?

---

## Question 5: Specialization Ladder

AutoRefactor has tested 4 rungs of the specialization ladder:

| Rung | Step time | Numeric valid? |
|------|-----------|---------------|
| 1 (baseline) | 1850 ms | yes |
| 2 (standard opt) | 1690 ms | yes |
| 3 (TPU-specific) | 1580 ms | yes |
| 4 (experimental) | 1520 ms | **no** — max diff 0.3 in fp32 |

**A)** Which rung should be accepted as the current best?

**B)** What should AutoRefactor do next with rung 4's flags?

**C)** If rung 4 contains 5 new flags, describe the bisection process to find the problematic one.

---

## Question 6: Fusion Boundary Accounting

Post-fusion HLO shows these boundaries in the critical path:

| Boundary | Tensor shape | Dtype | Occurrences/step |
|----------|-------------|-------|-----------------|
| A → B | [2048, 16384] | bf16 | 96 (once per layer) |
| C → D | [2048, 4096] | fp32 | 96 |
| E → F | [2048, 2048] | bf16 | 48 |

**A)** Calculate total materialization cost per step at 820 GB/s HBM.

**B)** Rank these boundaries by cost — which should MaxKernel target first?

**C)** If writing a Pallas kernel to eliminate boundary A→B takes significant effort but boundary C→D can be eliminated by changing a fusion flag, which do you pursue first?

---

## Answers guidance

Verify your reasoning against:
- [14 - Roofline Model & Bottleneck Analysis](../concepts/14-roofline-model-and-bottleneck-analysis.md)
- [15 - Iterative Constraint Narrowing](../concepts/15-iterative-constraint-narrowing.md)
- [16 - Algebraic ISA Modeling](../concepts/16-algebraic-isa-modeling.md)
- [17 - Pipeline Stage Balancing](../concepts/17-pipeline-stage-balancing.md)
- [18 - Compile-Time Specialization Ladder](../concepts/18-compile-time-specialization-ladder.md)
- [19 - Fusion Boundary Cost Accounting](../concepts/19-fusion-boundary-cost-accounting.md)
