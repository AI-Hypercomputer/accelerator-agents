---
id: STRAT-SC-12
title: A lever that acts between components reads as zero on either component alone
applies_to: any
lever: method
generality: general
phase: [plan, profile]
verify: "manual: time the lever on the whole call as well as on each component, and compare against the call-size-ratio forecast"
---

# STRAT-SC-12 — Measure at the scope where the work actually lives

**Rule.** Measure a lever on the **whole call**, and derive its dilution from *where its work sits*,
not from the ratio of call sizes. A lever that deletes work **between** two components is invisible
when either component is timed alone — a stage-scoped number can even have the wrong sign and reject
a real win. A lever that is **global** to the call does not dilute the way a component-local one
does. Scaling a component-scoped measurement by a size ratio gets both cases wrong, in opposite
directions.

**Mechanism.** The size-ratio heuristic assumes a lever's savings are proportional to the fraction
of the call its component occupies. That holds only for work confined to that component. Inter-
component work belongs to neither, so it is excluded from both measurements and appears only in the
total; call-wide state is present in every component, so it saves everywhere and dilutes less than
the ratio predicts.

**Scope.** General to any multi-stage or multi-kernel op. The direction of the error is predictable
from where the work sits, so this is usable as a forecast, not only as a postmortem.

**Does not apply when** the lever's work is genuinely confined to one component and nothing crosses
the boundary — which is a claim about the dataflow that costs nothing to check and is wrong more
often than it looks.

**How to check on your kernel.** Classify the lever: component-local, inter-component (deletes a
hand-off, gather or materialization between stages), or call-global (a layout or stride used
everywhere). For the last two, time the whole call, never a single stage. If you forecast from a
component measurement, state the classification and the direction of expected error.
