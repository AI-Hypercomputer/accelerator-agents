---
name: strategies-index
description: >-
  One row per strategy rule: what it says, which unit it applies to, and the
  MaxKernel phase it informs. Read this table, then open only the rules whose
  row matches the kernel in front of you.
---

# Strategy index

Each strategy is one rule learned from optimizing real Pallas TPU kernels, stated without the
measurements it came from. A rule tells you what to check and why; it never tells you a number.
Any number you need comes from a probe run on your target (`tools/run_probe.py`, see
[`../probe_catalog.md`](../probe_catalog.md)). Each file's `verify:` field names the probe or the
manual check that tests the rule on your kernel.

Read [`../tpu_hardware_facts.md`](../tpu_hardware_facts.md) whole before using any rule here.

**Columns.** `applies to` is the unit the rule is about. `carry` is how far it generalizes:
**unit-wide** = any kernel on that unit; **general** = the mechanism is not specific to the unit.
A rule whose text says *toolchain-dependent* must be re-verified after any jax or libtpu upgrade.
The `SC`/`MLA` prefixes are stable ids, not scope: many `SC` rules are not about SparseCore.

| id | applies to | phase | lever | carry | rule |
| --- | --- | --- | --- | --- | --- |
| [STRAT-MLA-01](STRAT-MLA-01-bucket-padding-is-not-free-upstream.md) | sparsecore | plan | algorithm | general | Bucket padding is free only downstream of the stage that multiplies it out |
| [STRAT-MLA-02](STRAT-MLA-02-truncation-granularity-has-an-interior-optimum.md) | any | plan, implement, autotune | pipelining | general | A knob that sets grid-truncation granularity has an interior optimum, not a direction |
| [STRAT-MLA-03](STRAT-MLA-03-shortened-axis-biases-the-sweep.md) | any | plan, profile, autotune | method | general | A knob swept on a shortened axis is biased when that axis is the one that penalizes it |
| [STRAT-MLA-04](STRAT-MLA-04-carry-must-reach-each-sub-kernel.md) | any | implement, profile | method | general | In a multi-kernel op the loop carry must reach each kernel's own inputs, not just the op's |
| [STRAT-MLA-05](STRAT-MLA-05-never-pad-gather-indices-with-a-constant.md) | sparsecore | plan, implement | layout | general | Pad unused gather-index slots with distinct addresses; a repeated index is the degenerate bank conflict |
| [STRAT-MLA-06](STRAT-MLA-06-sparsecore-bandwidth-is-width-dependent.md) | sparsecore | plan | layout | unit-wide | The SparseCore streaming figure is access-width dependent; it is not a single ceiling for wide-row indirect reads |
| [STRAT-MLA-07](STRAT-MLA-07-widening-roofline-ratio-means-fixed-cost-bound.md) | any | plan, profile | method | general | A roofline ratio that widens as you optimize proves the floor models the wrong resource |
| [STRAT-MLA-08](STRAT-MLA-08-sparsecore-data-dependent-skip-lowers.md) | sparsecore | implement | pipelining | unit-wide | On SparseCore, pl.when may wrap emit_pipeline and its grid length may be traced |
| [STRAT-MLA-09](STRAT-MLA-09-sparsecore-rank0-smem-xor.md) | sparsecore | implement | layout | unit-wide | On SparseCore a rank-0 load is legal only from SMEM, and an SMEM load only at rank 0 |
| [STRAT-MLA-10](STRAT-MLA-10-a-global-constant-must-pay-on-shapes-you-did-not-measure.md) | any | plan, profile, autotune | method | general | A module-level constant must pay for itself on the shapes you did not measure |
| [STRAT-MLA-11](STRAT-MLA-11-localize-fixed-cost-before-building-a-grid-lever.md) | sparsecore | plan, profile | method | general | Before building a lever that shortens a grid, measure how much of the cost scales with that grid |
| [STRAT-MLA-12](STRAT-MLA-12-sparsecore-traced-dma-needs-a-1d-ring.md) | sparsecore | implement | pipelining | unit-wide | A SparseCore DMA with a traced offset and extent lowers — into a 1-D ring slot with multiple_of hints, not a 2-D ring row |
| [STRAT-MLA-13](STRAT-MLA-13-a-compute-bound-is-not-a-dma-bound.md) | any | plan | algorithm | general | Bounding a kernel's loops by the live length does not bound its DMA — and index-assigned work leaves the live units at full cost |
| [STRAT-MLA-14](STRAT-MLA-14-xla-scatter-into-subword-array-can-relayout-the-whole-operand.md) | tensorcore | plan | layout | general | An XLA scatter into a sub-32-bit tiled array can relayout the entire operand — cost scales with the array, not the update |
| [STRAT-MLA-15](STRAT-MLA-15-spill-removal-cannot-move-a-valu-bound-kernel.md) | tensorcore | plan, profile | method | general | Removing register spills does not move a kernel whose VALU issue binds; spills ride in free slots |
| [STRAT-SC-01](STRAT-SC-01-compaction-pass-cost.md) | sparsecore | plan | algorithm | general | A pass with cross-lane serializing ops in its body is priced by those ops, not by its trip count |
| [STRAT-SC-02](STRAT-SC-02-absolute-savings-compose.md) | any | plan, profile | method | general | Predict a stacked optimization from absolute time saved, never from multiplied percentages |
| [STRAT-SC-03](STRAT-SC-03-barrier-the-loop-carry.md) | any | implement, profile, test | method | general | A chained-invocation clock must barrier the loop carry, or LICM hoists the kernel out of its own loop |
| [STRAT-SC-04](STRAT-SC-04-emit-pipeline-scratches-noop.md) | any | implement | pipelining | general | An ignored keyword argument is a silent no-op, and emit_pipeline(scratches=) is one |
| [STRAT-SC-05](STRAT-SC-05-wave-longest-row.md) | sparsecore | plan | algorithm | general | A lockstep group costs its longest member, so mean extent does not model ragged work |
| [STRAT-SC-06](STRAT-SC-06-pad-scatter-stride-off-bank-count.md) | sparsecore | plan, implement | layout | general | A per-lane stride sharing a factor with the bank count collapses every lane onto one bank |
| [STRAT-SC-07](STRAT-SC-07-distribution-sensitivity-is-a-conflict.md) | any | profile | method | general | Distribution sensitivity at constant work is a memory conflict, not a workload |
| [STRAT-SC-08](STRAT-SC-08-prove-the-load-binds-first.md) | sparsecore | plan, implement | pipelining | general | Every transfer lever measures the same on a compute-bound kernel |
| [STRAT-SC-09](STRAT-SC-09-unit-choice-follows-what-binds.md) | sparsecore | plan | algorithm | unit-wide | Unit choice follows what binds the algorithm, not which unit is wider |
| [STRAT-SC-10](STRAT-SC-10-shared-arm-confound.md) | any | profile | method | general | Arms that all move by the same amount are measuring what they share |
| [STRAT-SC-11](STRAT-SC-11-oracle-needs-a-positive-control.md) | any | implement, profile, test | method | general | A correctness oracle that has never failed is not known to be able to fail |
| [STRAT-SC-12](STRAT-SC-12-measure-at-the-scope-the-work-lives.md) | any | plan, profile | method | general | A lever that acts between components reads as zero on either component alone |
| [STRAT-SC-13](STRAT-SC-13-gate-the-estimator-not-the-samples.md) | any | profile | method | general | A stability gate must bound the estimator you report, not the spread of the samples |

## By phase

Open these when the MaxKernel loop is in that phase.

| phase | rules |
| --- | --- |
| plan — design, floors, unit choice, layout | STRAT-MLA-01 · STRAT-MLA-02 · STRAT-MLA-03 · STRAT-MLA-05 · STRAT-MLA-06 · STRAT-MLA-07 · STRAT-MLA-10 · STRAT-MLA-11 · STRAT-MLA-13 · STRAT-MLA-14 · STRAT-MLA-15 · STRAT-SC-01 · STRAT-SC-02 · STRAT-SC-05 · STRAT-SC-06 · STRAT-SC-08 · STRAT-SC-09 · STRAT-SC-12 |
| implement — lowering and kernel code | STRAT-MLA-02 · STRAT-MLA-04 · STRAT-MLA-05 · STRAT-MLA-08 · STRAT-MLA-09 · STRAT-MLA-12 · STRAT-SC-03 · STRAT-SC-04 · STRAT-SC-06 · STRAT-SC-08 · STRAT-SC-11 |
| test — correctness checks | STRAT-SC-03 · STRAT-SC-11 |
| autotune — sweeping knobs | STRAT-MLA-02 · STRAT-MLA-03 · STRAT-MLA-10 |
| profile — measurement and attribution | STRAT-MLA-03 · STRAT-MLA-04 · STRAT-MLA-07 · STRAT-MLA-10 · STRAT-MLA-11 · STRAT-MLA-15 · STRAT-SC-02 · STRAT-SC-03 · STRAT-SC-07 · STRAT-SC-10 · STRAT-SC-11 · STRAT-SC-12 · STRAT-SC-13 |
