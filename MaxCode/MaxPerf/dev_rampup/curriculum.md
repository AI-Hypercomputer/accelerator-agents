<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# dev_rampup Curriculum

A structured learning path for understanding and operating the MaxPerf system.

---

## Dependency Graph

```
                    [01 TPU v6e Architecture]
                     /          |          \
        [02 HBM & Memory]  [04 ISA Latency]  [03 DMA Idle Time]
              |         \       |
   [05 VREG Spill]    [14 Roofline Model]
                            |
                   [15 Iterative Constraint Narrowing]
                            |
                   [16 Algebraic ISA Modeling]

        [06 JAX Tracing]
              |
        [07 HLO IR]
           /      \
  [08 XLA Passes]  [10 Numeric Equivalence]
        |
  [09 XLA Fusion & Custom-Call]
     /        \
[19 Fusion Boundary]  [18 Specialization Ladder]
                            |
                      [24 HLO Barrier Removal]

  [11 Collective Ops]
     /            \
[12 Commutativity]  [13 Payload Minimization]
        |
[23 Graph Rewrite Patterns]

  [14 Roofline] + [09 Fusion] + [04 ISA]
        |
  [20 Pallas Kernels & Sublane Layout]

  [15 Constraint Narrowing] + [10 Numeric Equiv]
              |
  [21 Hypothesis Classes & Origination]
              |
  [22 Diagnostic Vector]

  [17 Pipeline Stage Balancing] ← [14 Roofline] + [11 Collective Ops]

  [25 Experiment Protocol] ← [22 Diagnostic Vector] + [10 Numeric Equiv]
```

---

## Reading Order

### Section 1 — Hardware (Concepts 1-5)

1. [TPU v6e Architecture](concepts/01-tpu-v6e-architecture.md)
2. [HBM & Memory Hierarchy](concepts/02-hbm-and-memory-hierarchy.md)
3. [DMA Idle Time](concepts/03-dma-idle-time.md)
4. [ISA Latency Tables](concepts/04-isa-latency-tables.md)
5. [VREG Spill & Register Pressure](concepts/05-vreg-spill-and-register-pressure.md)

**Checkpoint:** [Section 1 — Hardware](checkpoints/section-1-hardware.md)

---

### Section 2 — Compiler Pipeline (Concepts 6-10)

6. [JAX Tracing Model](concepts/06-jax-tracing-model.md)
7. [HLO Intermediate Representation](concepts/07-hlo-intermediate-representation.md)
8. [XLA Compilation Passes](concepts/08-xla-compilation-passes.md)
9. [XLA Fusion & Custom-Call Boundaries](concepts/09-xla-fusion-and-custom-call-boundaries.md)
10. [Numeric Equivalence](concepts/10-numeric-equivalence.md)

**Checkpoint:** [Section 2 — Compiler Pipeline](checkpoints/section-2-compiler.md)

---

### Section 3 — Distributed Systems (Concepts 11-13)

11. [Collective Operations](concepts/11-collective-operations.md)
12. [Commutativity & Graph Commutation](concepts/12-commutativity-and-graph-commutation.md)
13. [Payload Minimization](concepts/13-payload-minimization.md)

**Checkpoint:** [Section 3 — Distributed Systems](checkpoints/section-3-distributed.md)

---

### Section 4 — Optimization Methods (Concepts 14-19)

14. [Roofline Model & Bottleneck Analysis](concepts/14-roofline-model-and-bottleneck-analysis.md)
15. [Iterative Constraint Narrowing](concepts/15-iterative-constraint-narrowing.md)
16. [Algebraic ISA Modeling](concepts/16-algebraic-isa-modeling.md)
17. [Pipeline Stage Balancing](concepts/17-pipeline-stage-balancing.md)
18. [Compile-Time Specialization Ladder](concepts/18-compile-time-specialization-ladder.md)
19. [Fusion Boundary Cost Accounting](concepts/19-fusion-boundary-cost-accounting.md)

**Checkpoint:** [Section 4 — Optimization Methods](checkpoints/section-4-methods.md)

---

### Section 5 — MaxPerf System (Concepts 20-22)

20. [Pallas Kernels & Sublane Layout](concepts/20-pallas-kernels-and-sublane-layout.md)
21. [Hypothesis Classes & Origination](concepts/21-hypothesis-classes-and-origination.md)
22. [Diagnostic Vector](concepts/22-diagnostic-vector.md)

**Checkpoint:** [Section 5 — MaxPerf System](checkpoints/section-5-maxperf-system.md)

---

### Section 6 — Operations (Concepts 23-25)

23. [Graph Rewrite Patterns](concepts/23-graph-rewrite-patterns.md)
24. [HLO Barrier Removal](concepts/24-hlo-barrier-removal.md)
25. [Experiment Protocol & Halt Rules](concepts/25-experiment-protocol-and-halt-rules.md)

**Checkpoint:** [Section 6 — Operations](checkpoints/section-6-operations.md)

---

## Concept-to-Agent Mapping

| Agent | Primary Concepts |
|-------|-----------------|
| [TPUDiagnoseAgent](../agents/tpu_diagnose.md) | [01](concepts/01-tpu-v6e-architecture.md), [02](concepts/02-hbm-and-memory-hierarchy.md), [03](concepts/03-dma-idle-time.md), [04](concepts/04-isa-latency-tables.md), [14](concepts/14-roofline-model-and-bottleneck-analysis.md), [22](concepts/22-diagnostic-vector.md) |
| [DeepResearch](../agents/deep_research.md) | [14](concepts/14-roofline-model-and-bottleneck-analysis.md), [15](concepts/15-iterative-constraint-narrowing.md), [16](concepts/16-algebraic-isa-modeling.md), [17](concepts/17-pipeline-stage-balancing.md), [12](concepts/12-commutativity-and-graph-commutation.md), [21](concepts/21-hypothesis-classes-and-origination.md) |
| [MaxKernel](../agents/max_kernel.md) | [04](concepts/04-isa-latency-tables.md), [05](concepts/05-vreg-spill-and-register-pressure.md), [09](concepts/09-xla-fusion-and-custom-call-boundaries.md), [15](concepts/15-iterative-constraint-narrowing.md), [19](concepts/19-fusion-boundary-cost-accounting.md), [20](concepts/20-pallas-kernels-and-sublane-layout.md) |
| [MaxShard](../agents/max_shard.md) | [11](concepts/11-collective-operations.md), [12](concepts/12-commutativity-and-graph-commutation.md), [13](concepts/13-payload-minimization.md), [15](concepts/15-iterative-constraint-narrowing.md), [23](concepts/23-graph-rewrite-patterns.md) |
| [AutoRefactor](../agents/auto_refactor.md) | [08](concepts/08-xla-compilation-passes.md), [09](concepts/09-xla-fusion-and-custom-call-boundaries.md), [18](concepts/18-compile-time-specialization-ladder.md), [19](concepts/19-fusion-boundary-cost-accounting.md), [24](concepts/24-hlo-barrier-removal.md) |
| [MaxTile](../agents/max_tile.md) | [04](concepts/04-isa-latency-tables.md), [16](concepts/16-algebraic-isa-modeling.md), [17](concepts/17-pipeline-stage-balancing.md), [20](concepts/20-pallas-kernels-and-sublane-layout.md) |
| [Orchestrator](../agents/maxperf_orchestrator.md) | [14](concepts/14-roofline-model-and-bottleneck-analysis.md), [21](concepts/21-hypothesis-classes-and-origination.md), [22](concepts/22-diagnostic-vector.md), [25](concepts/25-experiment-protocol-and-halt-rules.md) |

---

## Concept-to-Method Mapping (M1-M9)

| Method | Name | Concepts |
|--------|------|----------|
| M1 | Bottleneck Analysis & Symbolic Graph Refactoring | [14](concepts/14-roofline-model-and-bottleneck-analysis.md), [07](concepts/07-hlo-intermediate-representation.md), [23](concepts/23-graph-rewrite-patterns.md) |
| M2 | Compiler Overhead Detection & Algorithmic Exploration | [08](concepts/08-xla-compilation-passes.md), [03](concepts/03-dma-idle-time.md), [22](concepts/22-diagnostic-vector.md) |
| M3 | Algebraic ISA Modeling (not Grid Search) | [16](concepts/16-algebraic-isa-modeling.md), [04](concepts/04-isa-latency-tables.md), [05](concepts/05-vreg-spill-and-register-pressure.md) |
| M4 | Execution Graph Unblocking | [24](concepts/24-hlo-barrier-removal.md), [08](concepts/08-xla-compilation-passes.md), [03](concepts/03-dma-idle-time.md) |
| M5 | Iterative Constraint Narrowing | [15](concepts/15-iterative-constraint-narrowing.md), [10](concepts/10-numeric-equivalence.md), [21](concepts/21-hypothesis-classes-and-origination.md) |
| M6 | Pipeline Stage Balancing | [17](concepts/17-pipeline-stage-balancing.md), [11](concepts/11-collective-operations.md), [14](concepts/14-roofline-model-and-bottleneck-analysis.md) |
| M7 | Payload Minimization via Algebraic Commutativity | [13](concepts/13-payload-minimization.md), [12](concepts/12-commutativity-and-graph-commutation.md), [11](concepts/11-collective-operations.md) |
| M8 | Compile-Time Specialization Ladder | [18](concepts/18-compile-time-specialization-ladder.md), [08](concepts/08-xla-compilation-passes.md), [09](concepts/09-xla-fusion-and-custom-call-boundaries.md) |
| M9 | Fusion Boundary Cost Accounting | [19](concepts/19-fusion-boundary-cost-accounting.md), [09](concepts/09-xla-fusion-and-custom-call-boundaries.md), [20](concepts/20-pallas-kernels-and-sublane-layout.md) |
