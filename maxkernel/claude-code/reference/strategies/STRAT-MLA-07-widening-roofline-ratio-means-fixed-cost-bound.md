---
id: STRAT-MLA-07
title: A roofline ratio that widens as you optimize proves the floor models the wrong resource
applies_to: any
lever: method
generality: general
phase: [plan, profile]
verify: "manual: recompute measured time / analytic floor at every step of the optimization path"
---

# STRAT-MLA-07 — Track the roofline ratio across the optimization, not at the start of it

**Rule.** Recompute the ratio of measured time to the analytic floor at **every step** of an
optimization. If removing work moves the kernel *further* from its floor, the remaining cost is not the
resource the floor models: the next lever must attack fixed cost, and further work on the modelled
resource is capped. **Then localize the fixed cost before acting on it** — per-call and per-grid-step
fixed costs need opposite levers, and an attribution by subtraction cannot tell them apart; see
[STRAT-MLA-11](STRAT-MLA-11-localize-fixed-cost-before-building-a-grid-lever.md).

**Mechanism.** A roofline divides measured time by a modelled resource. Let time be
`fixed + bytes / BW`. Cutting `bytes` shrinks the floor proportionally while measured time falls only by
the variable part, so the ratio **rises**. A monotonically rising ratio along an optimization path is
direct evidence of a fixed-cost component, and its size bounds that component from below. This inverts
the usual reading: a gap is normally headroom to close, but a *widening* gap is the floor disqualifying
itself. It says nothing about *which* fixed cost — per call, per grid step or per window — binds.

**Scope.** General to any roofline, on any unit. The diagnostic needs at least two points on an
optimization path. Use the roofline in
[tpu_memory_overlapping.md](../tpu_memory_overlapping.md) §1 for the floor.

**Does not apply when** the ratio is already near 1, where its movement is inside noise, or when the
work removed also removed fixed cost (fewer invocations, not smaller ones) — then both terms fall and
the ratio is uninformative. Complements
[STRAT-SC-08](STRAT-SC-08-prove-the-load-binds-first.md), which says prove the load binds *before*
restructuring it; this says how to notice afterwards that it stopped binding.

**How to check on your kernel.** Recompute your own floor at each step of the path, not once. When the
ratio widens, run the constant-live-work grid sweep of
[STRAT-MLA-11](STRAT-MLA-11-localize-fixed-cost-before-building-a-grid-lever.md) before choosing a lever.
