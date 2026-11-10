---
id: STRAT-SC-13
title: A stability gate must bound the estimator you report, not the spread of the samples
applies_to: any
lever: method
generality: general
phase: [profile]
verify: "none: experiment-design rule"
---

# STRAT-SC-13 — Gate the uncertainty of the estimator, not the dispersion of the samples

**Rule.** A benchmark's quarantine or stability criterion must bound the uncertainty of **the
statistic you report**. Gate the dispersion of the individual rounds instead and the rule is
backwards: **collecting more rounds makes it fire more often**, while the estimate it is guarding
gets better. Read the harness's criterion before trusting the cells it did not quarantine.

**Mechanism.** Sample dispersion is a property of the distribution and does not shrink with `n`; the
standard error of a median does. A conjunct written on a p10–p90 spread therefore tightens nothing
as evidence accumulates — it just samples the tail more times. The failure is silent in the useful
direction too: the gate looks strict while bounding a quantity nobody reports.

**Scope.** General to any measurement harness. The replacement is a bootstrap CI on the reported
statistic, or agreement between independent replicates; a ratio-based conjunct on the same rounds is
sound and worth keeping.

**Does not apply when** the reported statistic *is* a dispersion — a tail-latency target, for
instance. Then gating spread is gating the estimator, and this rule is satisfied already.

**How to check on your kernel.** Identify which statistic the harness reports (median, mean, min)
and which quantity its quarantine rule bounds. If they differ, replace the dispersion gate with a
bootstrap confidence interval on the reported statistic, or run two independent replicates and use
their agreement as the error bar.
