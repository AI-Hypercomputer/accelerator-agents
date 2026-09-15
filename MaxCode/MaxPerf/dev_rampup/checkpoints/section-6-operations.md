<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Checkpoint: Section 6 — Operations

Test your understanding of concepts 23-25 with these scenario-based questions.

---

## Question 1: Graph Rewrite Identification

You see this pattern in HLO:
```
%gradient = bf16[4096, 16384] ...
%all-reduce.1 = bf16[4096, 16384] all-reduce(%gradient), replica_groups={{0,1,2,3,4,5,6,7}}
%dot.proj = bf16[4096, 2048] dot(%all-reduce.1, %weight[16384, 2048])
```

**A)** What graph rewrite pattern applies here?

**B)** Write the rewritten HLO (pseudocode is fine).

**C)** Calculate the payload reduction: original all-reduce size vs new all-reduce size.

**D)** What precondition must be verified before this rewrite is safe?

---

## Question 2: Barrier Removal Analysis

An xprof timeline shows the following schedule for two independent operations:
```
Time:   0----1----2----3----4----5----6----7----8 ms
MXU:    [==matmul_A (3ms)==]..idle..[==matmul_B (3ms)==]
DMA:    ..idle..[=prefetch_B=]..idle..
```

A control dependency forces `prefetch_B` to wait until `matmul_A` completes.

**A)** What is the current total time for both operations?

**B)** If the barrier is removed and DMA can start at time 0, sketch the new timeline.

**C)** What is the speedup?

**D)** What would you check to confirm the barrier is safe to remove?

---

## Question 3: Experiment Protocol Application

You've been optimizing a model for 8 experiments. Results:

| Exp | Optimization | Delta | Cumulative step time |
|-----|-------------|-------|---------------------|
| 1 | Fusion flag change | -45 ms | 1355 ms |
| 2 | Collective commutation | -28 ms | 1327 ms |
| 3 | Barrier removal | -62 ms | 1265 ms |
| 4 | Pallas attention kernel | -95 ms | 1170 ms |
| 5 | Pipeline rebalance | -12 ms | 1158 ms |
| 6 | Secondary fusion | -4 ms | 1154 ms |
| 7 | Tile size optimization | -3 ms | 1151 ms |
| 8 | Scheduling tweak | -2 ms | 1149 ms |

Target: 1100 ms. Diminishing returns threshold: 3 consecutive experiments < 0.5% improvement.

**A)** Has the diminishing returns halt rule triggered? Show the calculation.

**B)** Should the orchestrator continue, halt, or change strategy? Why?

**C)** What would you do differently given that the target is still 49 ms away?

---

## Question 4: Halt Rule Scenarios

Consider these three situations:

**Scenario A**: Experiment 5 produces a 35 ms improvement but the numeric check shows max_diff = 0.02 in fp32 (tolerance: 1e-5).

**Scenario B**: The next 3 prioritized hypotheses are all "analogy-driven" with medium confidence (55-65%).

**Scenario C**: After experiment 4, a re-profiling shows the bottleneck_class has shifted from "memory" to "communication".

**A)** For Scenario A: what is the correct decision and next action?

**B)** For Scenario B: should the orchestrator halt, or is there a better response?

**C)** For Scenario C: how should the orchestrator adapt? Which agents get activated/deactivated?

---

## Question 5: End-to-End Operational Decision

A new model arrives for optimization. The diagnostic vector shows:
```
step_time_ms: 2100
mxu_utilization: 0.38
hbm_bw_utilization: 0.55
ici_bw_utilization: 0.44
bottleneck_class: memory
fusion_boundary_cost_ms: 210
collective_time_ms: 380
dma_idle_fraction: 0.22
bubble_fraction: 0.05
```

Target: 1600 ms (24% improvement needed).

**A)** What is the maximum theoretical improvement from eliminating all fusion boundaries? Is that enough alone?

**B)** What is your top-3 prioritized action plan? For each, name the method, agent, and estimated impact.

**C)** After actions complete, what measurement would trigger a strategy pivot to collective optimization?

---

## Answers guidance

Verify your reasoning against:
- [23 - Graph Rewrite Patterns](../concepts/23-graph-rewrite-patterns.md)
- [24 - HLO Barrier Removal](../concepts/24-hlo-barrier-removal.md)
- [25 - Experiment Protocol & Halt Rules](../concepts/25-experiment-protocol-and-halt-rules.md)
