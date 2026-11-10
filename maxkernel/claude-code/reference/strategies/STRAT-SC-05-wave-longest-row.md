---
id: STRAT-SC-05
title: A lockstep group costs its longest member, so mean extent does not model ragged work
applies_to: sparsecore
lever: algorithm
generality: general
phase: [plan]
verify: probe sc_module_surface
---

# STRAT-SC-05 — A lockstep group costs its longest member

**Rule.** Where `W` units execute in lockstep, a group costs the **maximum extent over its `W`
members**, not the mean. Model ragged work as `E[max over W]`. A mean-based or bytes-based cost
model is wrong by the full raggedness factor, always optimistically, so it will manufacture a lever
that cannot exist — check this analytically before building a "traced extent" lever.

**Mechanism.** Lockstep means every member waits for the slowest, so raggedness that would average
out across independent workers is instead rounded up per group to its worst member. For
`rl ~ U(0, n)`, `E[max over W] = n·W/(W+1)`, which is already close to `n` for a modest `W`.
**Halving the mean extent moves the clock by almost nothing.**

**Scope.** General to any lockstep execution group. On SparseCore, work assigned by index forms a
wave across all vector subcores of all cores — row `w` of a wave is `w * (cores * subcores) + flat`
— so `W = num_cores × num_subcores`. Read those from `SparseCoreInfo` at run time (probe
`sc_module_surface` reports the geometry); never hard-code `W`. `E[max over W]` is the part that
transfers, and it moves with `W`.

**Does not apply when** work is assigned to groups by *extent* rather than by index. Sorting or
bucketing members by length is the fix this rule points at, and it shifts the governing statistic
from `E[max]` back toward the mean.

**How to check on your kernel.** Determine how rows map to lockstep members (by index or by
extent) and read `W` from `SparseCoreInfo`. Compute `E[max over W]` of the real extent distribution
and use it, not the mean, in the cost model. If a lever's predicted gain disappears under `E[max]`,
drop it or add length bucketing first.
