---
id: STRAT-MLA-02
title: A knob that sets grid-truncation granularity has an interior optimum, not a direction
applies_to: any
lever: pipelining
generality: general
phase: [plan, implement, autotune]
verify: probe dispatch_floor
---

# STRAT-MLA-02 — Read what a "batch size" actually does before you sweep it

**Rule.** A knob named like a tile size may instead set the **granularity at which a data-dependent
grid stops early**. Such a knob does not trade against occupancy or VMEM; it trades *padding
truncated* against *grid steps walked*. Two opposing terms mean an interior optimum: find it by
sweeping at both ends of the workload, and never assume the knob has a direction.

**Mechanism.** For a chunk of extent `C` walked in steps of `B`, a grid that stops at the last step
holding live work runs `ceil(live / B)` steps out of `C / B`. Cost is roughly
`steps × per_step + truncation_waste × per_row`. Shrinking `B` cuts the waste linearly but raises the
step count as `C / B`. Waste dominates while padding is most of the chunk (decode); step count
dominates once the chunk is full (prefill). Each end is optimal for a different regime, so a single
value is a compromise that must be priced at both ends.

**The two-term model is incomplete.** A smaller `B` can also win on a chunk with no truncation waste
at all, so a third term favouring smaller steps (plausibly overlap across narrower steps) exists
independently of truncation. hypothesis: it is visible only where the step count is small; verify by
sweeping the full-chunk case too.

**Scope.** Any grid whose trip count is truncated against a runtime extent — ragged attention,
bucketed serving, variable-length reductions. The mechanism is arithmetic; only the per-step cost,
which sets where the turning point lands, is hardware-specific. When two levers ship together, only an
in-program profile says which one paid; see
[STRAT-MLA-01](STRAT-MLA-01-bucket-padding-is-not-free-upstream.md). Sweep at the production chunk
extent; see [STRAT-MLA-03](STRAT-MLA-03-shortened-axis-biases-the-sweep.md).

**Does not apply when** the grid always runs to completion — then the knob is a tile size and the
usual occupancy and working-set reasoning governs. Do not infer from a zero truncation term that the
knob is neutral; sweep it there too.

**How to check on your kernel.** Identify the chunk extent your grid walks and measure the per-grid-step
cost on the target unit (probe `dispatch_floor` reports the per-step cost). Then sweep `B` at both a
mostly-empty and a full chunk, at the production extent.
