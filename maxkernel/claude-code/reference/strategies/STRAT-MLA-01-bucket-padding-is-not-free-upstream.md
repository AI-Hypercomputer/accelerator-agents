---
id: STRAT-MLA-01
title: Bucket padding is free only downstream of the stage that multiplies it out
applies_to: sparsecore
lever: algorithm
generality: general
phase: [plan]
verify: "manual: time the upstream gather at a full and a mostly-empty bucket, and read the consumer's wait on it in the profile"
---

# STRAT-MLA-01 — Bucket padding is free only downstream of the mask

**Rule.** Padding a batch up to a compiled bucket buys static shapes, and the stage that masks the
padding out pays nothing for it. Every stage **upstream** of that mask pays full price for rows whose
contribution is already known to be zero. When an indirect gather or prefetch sits upstream of the
mask, make it skip the padding at whatever granularity its traversal allows, and bound the consumer's
grid to the live rows too: the gather-side bound pays fully only once the consumer stops reading the
dead rows.

**Mechanism.** The masking stage knows which rows are live; the stage feeding it does not, because
liveness arrives as a runtime value while the gather's cost is set by its static extent. A gather's
work is `rows × fanout` whatever survives, so its wasted fraction is the bucket's padding fraction
`(P − S) / P`: negligible at prefill, nearly total at decode with mostly-empty buckets. The saving is
structurally zero when the bucket is full, so the change is safe to enable unconditionally. Under data
parallelism the bucket is set by the busiest rank, so a rank with almost no live work can still run a
large-bucket program and pay the full gather.

**Scope.** Any bucketed serving kernel with a gather, prefetch or scatter stage ahead of its mask. The
skip granularity is the traversal window, not the row, so the realizable win is the padding fraction
rounded down to whole windows. Measure at the bucket the workload actually lands in (usually the
smallest one on the ladder), not at a bucket the serving configuration never reaches.

**A device-time saving is not a step saving.** If the gather runs on a parallel engine (SparseCore)
while the TensorCore runs the consumer, the saving can be entirely hidden at decode and exposed only
where the consumer actually waits — which can be the opposite regime from where the padding fraction
predicts the win. Measure the consumer's wait, not the producer's duration; see
[STRAT-SC-12](STRAT-SC-12-measure-at-the-scope-the-work-lives.md).

**Does not apply when** the upstream output feeds a cross-row reduction that needs padding rows
materialized to a neutral element, or when buckets are typically full. It is not a licence to shrink
the window: a finer window is more selective but costs real work on full buckets; see
[STRAT-MLA-10](STRAT-MLA-10-a-global-constant-must-pay-on-shapes-you-did-not-measure.md).

**How to check on your kernel.** First, whether the upstream stage is on the critical path at all
(the consumer's wait in an in-program profile). Then the smallest bucket on your padding ladder, the
live-row fraction that lands in it, and the traversal granularity the skip can reach.
