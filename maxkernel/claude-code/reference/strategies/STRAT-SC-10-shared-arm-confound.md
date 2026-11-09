---
id: STRAT-SC-10
title: Arms that all move by the same amount are measuring what they share
applies_to: any
lever: method
generality: general
phase: [profile]
verify: "none: experiment-design rule"
---

# STRAT-SC-10 — Arms that agree are probably measuring what they share

**Rule.** When every arm of a probe moves by about the same amount, the default reading is **not**
"the lever works." It is that the arms share something the control does not, and that shared thing
is what moved. Before crediting a lever, name what its arms have in common and add an arm that
isolates it.

**Mechanism.** Implementing a lever usually requires a restructuring that is not the lever — a fused
pass, a changed guard, a different buffer. If that restructuring is present in every treatment arm
and absent from the control, it is perfectly confounded with the lever, and the probe cannot tell
them apart no matter how many arms it has or how tight the error bars are. Agreement between arms
is the signature, and it is easy to misread as robustness: structurally unrelated mechanisms do not
land on nearly identical deltas by chance.

**Scope.** General experiment design, any hardware. Cheap to apply — it costs one extra arm holding
the shared change with the lever removed.

**Does not apply when** the arms are genuinely minimal edits off a common base and you can point at
the diff proving it. That proof is the work; arm agreement is not a substitute for it.

**How to check on your kernel.** Diff each treatment arm against the control and list every change
that is not the lever itself. If any such change is common to all arms, add an arm with that change
alone and the lever removed. Credit the lever only with the difference between the full arm and
that isolating arm.
