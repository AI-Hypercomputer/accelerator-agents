---
id: STRAT-MLA-10
title: A module-level constant must pay for itself on the shapes you did not measure
applies_to: any
lever: method
generality: general
phase: [plan, profile, autotune]
verify: "manual: grep the constant's callers and enumerate the shape classes compiled against it"
---

# STRAT-MLA-10 — Tune through an argument; a global owes evidence on every shape it reaches

**Rule.** The evidence a tuning owes is set by its **blast radius**, not by the size of the diff.
Changing a call-site argument obliges you to measure that call site. Changing a module-level constant
obliges you to measure every shape class compiled against it, including those with no test — and if
you cannot, thread the value as an argument and leave the default alone.

**Mechanism.** The two changes look identical in review — one number, one line — which makes the
asymmetry easy to miss. A global is read by every caller at every shape, so a win on the shape under
optimization is one point in the space the change actually covers. Globals that tune a *granularity*
tend to have the two-sided cost of
[STRAT-MLA-02](STRAT-MLA-02-truncation-granularity-has-an-interior-optimum.md): the setting that helps
the sparse shape adds overhead to the dense one, so the untested shapes are systematically on the
losing side. Example: a finer gather window makes a padding skip more selective on mostly-empty
buckets and is pure overhead on full ones, where the skip is a no-op.

**Scope.** General; this is about change blast radius, not accelerators. It is the cheapest method
rule to apply — a grep for callers, run before the A/B rather than after it. Threading each tuning as
a keyword argument from the dispatcher keeps kernel defaults untouched and each change revertible on
its own line.

**Does not apply when** the constant provably has one caller and one compiled shape. Verify that by
grep rather than assumption — and prefer the argument anyway: it costs nothing, documents the tuning
at the site that chose it, and keeps the revert local.

**How to check on your kernel.** Grep the constant's callers, enumerate the shape classes compiled
against it, and A/B at each (at least the sparse and the dense extreme) before changing the default.
