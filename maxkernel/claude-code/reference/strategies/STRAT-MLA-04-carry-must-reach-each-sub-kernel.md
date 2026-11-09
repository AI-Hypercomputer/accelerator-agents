---
id: STRAT-MLA-04
title: In a multi-kernel op the loop carry must reach each kernel's own inputs, not just the op's
applies_to: any
lever: method
generality: general
phase: [implement, profile]
verify: "manual: count tpu_custom_call per chained iteration and require it to equal the op's static kernel count"
---

# STRAT-MLA-04 — A per-op carry is not a per-kernel carry

**Rule.** [STRAT-SC-03](STRAT-SC-03-barrier-the-loop-carry.md) requires a chained-invocation clock to
route its dependency through the loop carry. That is necessary but **not sufficient** when the op
lowers to more than one kernel. Each kernel is eliminated on its own operand set, so the carry must
reach the inputs that *the stage you are timing* depends on. For a gather, perturb the **indices**
(e.g. rotate them by the carry, changing which rows are fetched but not how many); perturbing the
values it is later combined with leaves the gather invariant.

**Mechanism.** Common-subexpression elimination works per call, on operands. A gather depends on
`(table, indices, page_table)`; its consumer depends on the gather's result and on other values. A
carry folded into the consumer's values makes the consumer vary — so the loop looks healthy — while
the gather subgraph stays bit-identical and `K` gathers collapse to one. Because CSE preserves
correctness, **the answer is right and the question was wrong**: no oracle, bit-exactness check or NaN
guard fires. The tell is a physically impossible *relation* rather than an impossible absolute, e.g.
"the consumer is cheaper than the gather it contains".

**Scope.** Any op lowering to more than one kernel, on any compiler that CSEs calls. The self-check
generalizes unchanged: count calls per chained iteration, compare against the static kernel count,
and require equality rather than "greater than one".

**Does not apply when** the op is a single kernel; then
[STRAT-SC-03](STRAT-SC-03-barrier-the-loop-carry.md) alone covers it.

**How to check on your kernel.** Count the op's static custom calls in the compiled HLO, then count
`tpu_custom_call` executions per chained iteration in the profile; they must be equal.
