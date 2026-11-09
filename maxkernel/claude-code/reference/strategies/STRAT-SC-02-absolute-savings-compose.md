---
id: STRAT-SC-02
title: Predict a stacked optimization from absolute time saved, never from multiplied percentages
applies_to: any
lever: method
generality: general
phase: [plan, profile]
verify: "manual: record each lever's absolute saving alone, confirm the levers touch disjoint work, and compare the stacked result to the summed savings"
---

# STRAT-SC-02 — Absolute savings compose; percentages do not

**Rule.** When stacking two optimizations, forecast the composite from the **absolute time each
saves**, not from its percentage against a baseline that the other lever is moving. The intuition
"this lever is worth more once the baseline is smaller" is an artifact of the denominator, not a
property of the silicon.

**Mechanism.** Deleted work costs what it costs regardless of what else was deleted. A percentage is
a ratio to a moving denominator, so stacking levers changes the percentage without changing the
physics. Levers touching disjoint work therefore compose *near*-multiplicatively in ratio terms —
`(1+d1)(1+d2)−1` — but the quantity that is actually stable is the numerator. Hypothesis: on
disjoint levers the stacked result lands close to, and slightly better than, the multiplicative
forecast; verify per kernel.

**Scope.** General. Strongest where the levers touch disjoint passes; the residual interaction is
the part worth measuring rather than modelling.

**Does not apply when** two levers contend for the same resource — then the second saves less than
it did alone, and neither absolute nor multiplicative composition holds.

**How to check on your kernel.** For each lever, record the absolute time saved when applied alone
to the same base. Confirm from the code that the levers touch disjoint passes or resources. Forecast
the stack as base minus the summed savings; if the measured stack falls well short, the levers
share a resource and must be re-planned together.
