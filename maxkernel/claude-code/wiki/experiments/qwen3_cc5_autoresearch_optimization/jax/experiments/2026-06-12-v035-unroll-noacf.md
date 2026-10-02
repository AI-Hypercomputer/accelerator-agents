---
title: "Qwen3-8B jax v035 — scan unroll=2 without async-collective-fusion flags (XLA-bug workaround) + flag-price control"
type: experiment
hypothesis: qwen3-jax-manual-weight-prefetch
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork .repo/2026-06-12-v034-wprefetch; image digest bc8d7950 (qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v034-unroll)
status: complete
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, scan, unroll, fsdp-collectives, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v035 — unroll=2 minus async-collective-fusion flags

[v034](2026-06-12-v034-scan-unroll.md) hit an XLA INTERNAL bug in the
triple (unroll ≥ 2 × offload_attn × async-collective-fusion flags). This
run drops the suspected trigger — the 3-flag async-collective-fusion family
(`enable_async_collective_fusion`, `…_fuse_all_gather`,
`…_multiple_steps`; `…_fuse_all_reduce=false` also removed as part of the
family) — and re-tries unroll, with a control arm pricing what the dropped
flags were worth (never isolated within v026's 13-flag bundle).

## Hypothesis under test

[qwen3-jax-manual-weight-prefetch](../../../../hypotheses/qwen3-jax-manual-weight-prefetch.md)
round 1 continued: if the AG overlap unlocked by unroll=2 exceeds the value
of the async-collective-fusion flags, arm B beats the frontier.

**Arms (bs4/seq8192, scan + offload_attn + splash + chunked CE; flag stack
= v026's 25 minus the 4 ACF flags = 21 flags):**
- **A — control: unroll=1, 21-flag stack** — prices the removed flags
  against v027 (6,040).
- **B — unroll=2, 21-flag stack (profiled)** — the workaround attempt;
  PASS > 6,161; bucket test async-done < 624 ms.

**Falsification criterion**: B still crashes (bug deeper than the ACF
flags) or B ≤ max(A, 6,040) + 1% → the unroll route closes; escalate to
carry-prefetch or close the hypothesis on this libtpu generation.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: digest-pinned `qwen3-8b-jax@sha256:bc8d7950…` (= v034-unroll tag).
- Workload: `alekseyv-qwen3-cc5-jax-v035-noacf`.

## Results

**Arm A (unroll=1, 21-flag stack)** — 20 steps clean, loss series EXACT
match to v027:

| Metric | v027 (25 flags) | Arm A (21 flags) | delta |
|--------|------------------|-------------------|-------|
| Step (median 2–19) | 5,425.0 ms | 5,425.4 ms | 0 |
| tok/s/chip | 6,040 | **6,039** | **−0.02% (noise)** |

**The 4 async-collective-fusion flags are worth ~ZERO** on this config —
v026's +10.3% lives entirely in the remaining 21 flags (windowed-einsum
disable + layout + scheduler). Recipe simplification available but not a
win.

**Arm B (unroll=2, 21-flag stack)** — crashed at step 0 with the IDENTICAL
v034 INTERNAL error (`async-start … T(8,128)(2,1)` vs `…S(5)`), proving the
bug is **NOT the ACF flags**: it is the unroll≥2 × offload_attn pair — the
post-optimization pass fails to propagate the host-space annotation of the
unrolled stacked offload buffer into the async-start shape.

## Profile

Arm A unprofiled (control); arm B crashed pre-profile. Throughput + exact
loss parity carry arm A's finding; the error text carries arm B's.

## HLO Dump

`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v035-noacf/hlo_b/`
(trivial early modules only — the crash precedes the train_step dump).

## Verdict

**refuted** (as the workaround it tested): removing the ACF flags neither
costs (arm A flat) nor unblocks (arm B same crash). Net knowledge:
1. ACF flags ≈ 0 value — 21-flag stack is equivalent to the 25-flag lane
   standard.
2. The XLA bug is structural to **scan-unroll ≥ 2 + pinned-host offload**
   on this libtpu — upstream-fileable with the exact error + before-opt HLO.
3. The carry-prefetch alternative is memory-infeasible (gathered weights in
   the scan carry are saved per-iteration by scan AD: +16 GB at bs4).

The mechanism itself (AG overlap via unroll) remains UNPROVEN rather than
disproven → final mechanism probe = [v037](2026-06-12-v037-unroll-saveattn-bs2.md):
unroll **without offload** (bs2 + save_attn fits in HBM, no S(5) buffers) —
quantifies the prize and provides the evidence package for the XLA bug
report.

## Next hypotheses

- [qwen3-jax-manual-weight-prefetch](../../../../hypotheses/qwen3-jax-manual-weight-prefetch.md) — final mechanism probe v037 (unroll=2 on the no-offload substrate); then the hypothesis resolves either "mechanism proven, blocked by XLA bug at the frontier op-point" or "mechanism refuted".
- None else.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v035-noacf/`
- Prior: [v034 (the XLA bug)](2026-06-12-v034-scan-unroll.md), [v027 (frontier)](2026-06-12-v027-bs4-mtfl.md), [v026 (the flag bundle)](2026-06-12-v026-mt-flags-bs3.md)
