---
id: STRAT-MLA-11
title: Before building a lever that shortens a grid, measure how much of the cost scales with that grid
applies_to: sparsecore
lever: method
generality: general
phase: [plan, profile]
verify: "manual: sweep the grid extent at constant live work and fit t = a + b * steps"
---

# STRAT-MLA-11 — Localize fixed cost before building a lever against it

**Rule.** "Fixed cost binds" is a diagnosis, not a target. A kernel's time is
`fixed + per_step × steps + per_unit × work`, and a skip, truncation or compaction lever attacks
**only the middle term**. Before building one, sweep the grid extent at **constant live work** and
fit `t = a + b × steps`: the slope is the lever's entire ceiling, and the intercept is what it can
never reach. A few-arm sweep, run before the design, not after it.

**Mechanism.** A residual identified by subtraction — measured time minus modelled work — is
attributed, not measured, and it lands in the intercept and the slope indistinguishably. The two imply
opposite levers: a per-step cost is removed by shortening a grid, a per-call cost only by issuing
fewer calls, and no grid work touches the second. Holding live work constant while moving the grid
extent separates them, because only the per-step term responds. The subtraction is usually done
against a variable-cost model (bytes, rows, FLOPs), so its residual is silently assumed to be what a
grid lever reaches. A confirmed linearity is also not a confirmed lever: the traversal term can scale
exactly as predicted while total time barely moves, because the intercept dominates.

**Scope.** General to any kernel with a grid and a candidate skip/truncation/compaction lever, on any
unit. What transfers is the sweep and the reading of its two coefficients. Sharpens
[STRAT-MLA-07](STRAT-MLA-07-widening-roofline-ratio-means-fixed-cost-bound.md): a widening roofline
ratio diagnoses fixed cost; this sweep says which kind.

**Does not apply when** live work cannot be held constant while the grid extent moves — then the
slope confounds the terms and the fit is not a ceiling. It does not argue against fixed-cost levers;
it says a *grid* lever cannot be one. Where the intercept is the target, the lever is call-count
reduction, and [STRAT-SC-09](STRAT-SC-09-unit-choice-follows-what-binds.md) constrains which
direction a cross-unit fusion may go.

**How to check on your kernel.** Re-derive which term of your cost model the lever attacks. Probe
`dispatch_floor` gives the generic per-call and per-grid-step cost; for your kernel, sweep its grid
extent at fixed live work and fit intercept and slope.
