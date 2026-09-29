---
title: "Qwen3 jax — async collective fusion (overlap grad reduce-scatter)"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "7-13% MFU (analyzer estimate)"
confidence: medium
effort: S
origin: 2026-06-02-v018-xla-flag-stack
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — async collective fusion (overlap grad reduce-scatter)

The v018 frontier profile shows the FSDP backward gradient **reduce-scatter runs synchronously** —
12.9% of step (~149 ms), the largest non-matmul cost, not pipelined with the backward matmuls
(see [observation](../observations/qwen3-jax-fsdp-rs-not-overlapped.md)). XLA's
**async-collective-fusion** flags schedule collectives to overlap with compute. This is a
*compiler/scheduler* lever — the category that demonstrably transfers to qwen3-jax (XLA scheduler
flags gave +3.4 pp at v018), unlike collective *offload* (SparseCore, refuted: offload ≠ overlap).

## Statement

Adding the XLA async-collective-fusion flags (fuse + multi-step, covering reduce-scatter) to the v018
frontier overlaps the grad reduce-scatter with the backward matmuls and raises MFU above 35.8%
(analyzer estimate +7–13 pp; count any ≥ +1 pp beyond noise), with loss parity and exit 0.

## Rationale

- **Profile-driven**: the analyzer pinpointed the synchronous reduce-scatter as the #1 recoverable
  chunk; MXU util is only 48.3%, so overlapped collectives can fill idle MXU cycles.
- **Category match**: scheduler/overlap levers transfer to qwen3 (v018); offload levers do not (v003/v017).
  Async-fusion is overlap, not offload → high prior.
- **Flag set** (MaxText `xla_flags_library`, LIBTPU_INIT_ARGS — drop any the build rejects):
  `xla_tpu_enable_async_collective_fusion=true`,
  `xla_tpu_enable_async_collective_fusion_fuse_all_gather=true`,
  `xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true`,
  `xla_tpu_enable_async_collective_fusion_multiple_steps=true`.

## Proposed experiment

v023: v018 frontier launch + the 4 async-collective-fusion flags appended to `LIBTPU_INIT_ARGS`
(keep the 7 scheduler flags + scoped_vmem). Flag-only, reuses the splash image. Measure MFU/tok-s-chip
vs 35.8%; re-profile to confirm the reduce-scatter bucket shrinks / moves under compute. Verify loss
parity.

## Risks

- Some async-fusion flags may be invalid on this libtpu build (report rejections; LIBTPU unknown flags
  tend to warn not crash — unlike the XLA_FLAGS jf_spmd crash in v019). Keep them in LIBTPU_INIT_ARGS.
- Async fusion can increase peak HBM (in-flight buffers); v018 is at 65.4% HBM so there's headroom.
- Falsified if MFU ≤ 35.8% beyond noise.

## Dependencies

None — flag-only on the v018 frontier image.
