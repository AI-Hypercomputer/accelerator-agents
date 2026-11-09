---
id: STRAT-SC-07
title: Distribution sensitivity at constant work is a memory conflict, not a workload
applies_to: any
lever: method
generality: general
phase: [profile]
verify: "manual: sweep input distributions at a proven-constant op count and compare the latency spread before and after the fix"
---

# STRAT-SC-07 — Distribution sensitivity at constant work is a memory conflict

**Rule.** If latency moves with the input *distribution* while the operation count is provably
unchanged, stop modelling work and go look for a bank or port conflict. Use the **collapse of the
cross-distribution spread** as the confirming signature that you fixed the right thing, rather than
the headline delta on any one distribution.

**Mechanism.** Conflict severity tracks address *concentration*, which is exactly what changing a
distribution changes while leaving update counts fixed. No work-based account can produce a latency
that depends on values rather than counts, so the observation itself narrows the search to the
memory system. A typical fix is the coprime stride padding of
[STRAT-SC-06](STRAT-SC-06-pad-scatter-stride-off-bank-count.md).

**Scope.** General, on any unit. It is also a **diagnostic-order** rule: a pre-LLO Mosaic dump
exposes no bank counter, so the instrument is a controlled sweep across distributions at fixed work,
not the profiler. Reaching for the profile first costs a run and answers nothing.

**Does not apply when** the operation count is not actually constant across your distributions —
prove that first, and expect to be wrong about it for any kernel with a data-dependent early exit.
If counts move, you have a workload difference and this rule says nothing.

**How to check on your kernel.** Pick several input distributions (uniform, concentrated/skewed,
and the production one). Prove the op count is identical across them (count updates, or confirm
there is no data-dependent exit). Time each; a spread that survives is a conflict. After the fix,
re-time all of them and require the spread to collapse, not just one number to improve.
