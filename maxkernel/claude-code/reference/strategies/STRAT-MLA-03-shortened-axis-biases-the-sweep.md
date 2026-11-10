---
id: STRAT-MLA-03
title: A knob swept on a shortened axis is biased when that axis is the one that penalizes it
applies_to: any
lever: method
generality: general
phase: [plan, profile, autotune]
verify: "none: experiment-design rule"
---

# STRAT-MLA-03 — Before shrinking a problem to sweep it, ask which term you shrank

**Rule.** Sweeping a knob at a reduced problem size is standard and usually fine. It stops being fine
when the axis you shortened carries the knob's **cost** rather than its benefit: the sweep then
understates the penalty by the shrink factor, and its trend does not extrapolate — not in magnitude,
and not in sign.

**Mechanism.** A knob's downside is often proportional to `extent / knob` — steps, launches, tiles,
passes. Its upside is usually proportional to something else, often a fixed fraction of the work.
Shrinking `extent` scales the penalty down while leaving the benefit intact, so the measurement is
biased toward the knob by exactly the shrink factor. The result is a clean, monotone, convincing trend
pointing the wrong way; monotonicity is no defence, it is what the bias produces. A grid-truncation
knob swept on a shortened chunk is the canonical case; see
[STRAT-MLA-02](STRAT-MLA-02-truncation-granularity-has-an-interior-optimum.md).

**Scope.** General. This is the *extent* sibling of
[STRAT-SC-12](STRAT-SC-12-measure-at-the-scope-the-work-lives.md), which is about measuring at the
wrong **scope** (which component). Both produce sign errors; this one is easier to miss, because
shrinking a problem to sweep it faster is normal practice and leaves no trace in the numbers.

**Does not apply when** the knob's cost provably does not scale with the shortened axis. That is a
claim to check, not assume: an axis is usually worth shortening because it is expensive, and it is
usually expensive because work is proportional to it.

**How to check on your kernel.** Nothing to measure on the mechanism. Before any reduced-size sweep,
write down which axis carries the knob's cost term; if it is the axis you shortened, confirm the
winning setting at the production extent before adopting it.
