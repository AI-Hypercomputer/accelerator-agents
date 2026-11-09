---
id: STRAT-MLA-05
title: Pad unused gather-index slots with distinct addresses; a repeated index is the degenerate bank conflict
applies_to: sparsecore
lever: layout
generality: general
phase: [plan, implement]
verify: "manual: time the gather with constant-filled pad slots against pad slot i filled with i % N"
---

# STRAT-MLA-05 — Scatter the pad indices of an indirect gather

**Rule.** Unused slots in an indirect-gather index array must be filled with **distinct** addresses,
not a repeated one. The two idioms everyone reaches for — pad with zero, or pad with the last valid
index — concentrate every lane's read on one address. On banked memory that is the worst case, not a
neutral one. Padding with something that varies per slot costs nothing and avoids it.

**Mechanism.** This is [STRAT-SC-06](STRAT-SC-06-pad-scatter-stride-off-bank-count.md)'s mechanism
reached without a stride. Bank index is the address modulo the bank count, so identical addresses are
identical banks by definition — no unlucky stride needed, and no stride padding can fix it. It is worth
its own entry because the **trigger** is invisible: "choose a fill value for an index array" does not
read as a layout decision.

**Scope.** Any indirect gather or scatter into banked memory. The fix is free and local: fill pad slot
`i` with something that varies in `i` (`i % N` suffices) rather than a constant. The penalty is a
ratio within one instrument, so it does not depend on absolute byte accounting.

**Does not apply when** the unit coalesces or broadcasts identical addresses — some do, and then a
repeated index is the cheap case. hypothesis: SparseCore does not coalesce, so treat a constant fill
as a performance bug there until measured otherwise.

**How to check on your kernel.** Find whether the target unit coalesces identical addresses, and its
bank count (see [STRAT-SC-06](STRAT-SC-06-pad-scatter-stride-off-bank-count.md)); then A/B a constant
fill against a per-slot-varying fill at your gather shape.
