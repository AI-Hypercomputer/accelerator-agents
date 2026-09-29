---
title: "Qwen3-8B jax v023 — async collective fusion (overlap grad reduce-scatter)"
type: experiment
hypothesis: qwen3-jax-async-collective-fusion
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-splash-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, xla-flags, collective, reduce-scatter, remat, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v023 — async collective fusion (overlap grad reduce-scatter)

## Hypothesis under test

**Hypothesis**: The v018 profile showed the FSDP backward gradient **reduce-scatter is synchronous**
(12.9% of step, ~149 ms, the #1 non-matmul cost). XLA async-collective-fusion flags overlap collectives
with compute; adding them to the v018 frontier should shrink/hide that bucket and raise MFU above 35.8%
(analyzer estimate +7–13 pp). This is an *overlap* (scheduler) lever — the category that transfers to
qwen3 (v018 +3.4 pp) — not an *offload* lever (SparseCore, refuted v003/v017).

**Mechanism**: v018 frontier (`--use_remat --use_splash --batch_size=4 --seqlen=2048` + 7 LIBTPU
scheduler flags + scoped_vmem) **+ 4 async-collective-fusion flags** in `LIBTPU_INIT_ARGS`:
`xla_tpu_enable_async_collective_fusion=true`,
`xla_tpu_enable_async_collective_fusion_fuse_all_gather=true`,
`xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true`,
`xla_tpu_enable_async_collective_fusion_multiple_steps=true`. Flag-only, no rebuild. No jf_spmd, no tokamax.

**Predicted signal**: MFU > 35.8% (target +5–13 pp); re-profile shows the all-reduce-scatter bucket
shrinks or moves under compute; MXU util rises above 48.3%; loss parity; exit 0.

**Falsification criterion**: MFU ≤ 35.8% beyond noise (the build doesn't overlap RS, or async overhead
cancels the gain), or any flag crash.

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v006-splash`. Flag-only (re-dispatched once after a transient API-529 that never launched).
- **Launch**: v018 frontier (`--use_remat --use_splash --batch_size=4 --seqlen=2048` + 7 LIBTPU scheduler flags + scoped_vmem) **+ 4 async-collective-fusion flags** (`enable_async_collective_fusion`, `..._fuse_all_gather`, `..._fuse_reduce_scatter`, `..._multiple_steps`). All 4 **accepted** by libtpu (no rejections).

## Results

| Metric | v018 frontier | v023 (+async-collective-fusion) | Δ |
|--------|---------------|----------------------------------|---|
| MFU | **35.8%** | 32.3% | **−3.5 pp** |
| tok/s/chip | 6,964 | 6,275 | **−9.9%** |
| steady step time (ms) | ~1,154 | ~1,285 | +11.4% |
| loss | 12.10→12.07 | 12.0954→12.0690 stable | parity |
| exit | 0 | 0 | — |

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-jax-v023-async-collective-fusion
- **Run name**: `2026-06-02-qwen3-jax-v023-async-collective-fusion`
- **On-disk pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v023-async-collective-fusion/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v023-async-collective-fusion/)
- **GCS**: `.../2026-06-02-qwen3-jax-v023-async-collective-fusion/plugins/profile/2026_06_02_08_10_38/` (2 hosts; steps 12–14).
- Contents: xprof trace + HLO dump.

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v023-async-collective-fusion/hlo/` — 1,060 files (text + proto).

## Verdict

**Refuted.** Falsification criterion ("MFU ≤ 35.8% beyond noise") met: 32.3% < 35.8%, a −3.5 pp / −9.9% regression with step time +11.4%. All 4 async-collective-fusion flags were accepted and the run was clean (no OOM/NaN, loss parity), but the lever is **net-negative** here. Interpretation: the v018 frontier already runs the MaxText latency-hiding scheduler bundle (`enable_all_experimental_scheduler_features` + `latency_hiding_scheduler_rerun=2` + `ag_backward_pipelining` + `host_transfer_overlap`), which *already* overlaps collectives aggressively; stacking async-collective-fusion **on top over-subscribes the scheduler** (extra in-flight buffers / conflicting placement) and costs more than the reduce-scatter overlap it adds. **Nuance to the "overlap levers transfer" pattern**: a *second* overlap mechanism layered on an already-overlap-tuned schedule is not additive — it regresses. The synchronous reduce-scatter ([observation](../../../../observations/qwen3-jax-fsdp-rs-not-overlapped.md)) remains the #1 cost but is NOT cheaply addressable by this flag. Frontier stays v018 (35.8%).

## Next hypotheses

- **Reassess** — this is the 5th consecutive post-v018 non-win (v019 inconclusive; v020/v021/v023 refuted; v022 neutral); the profile's #1 lever (RS overlap) backfired. Per the loop branch logic, a fresh [retrospective](../../../../analyses/) is warranted to synthesize the v018-win + 5-refutation arc and decide whether 35.8% is near the practical ceiling for *transferable* levers (gap to llama3-jax 43.3% may require non-transferable deep kernel work).
- None high-prior remaining among cheap flag levers — block-tuning, tokamax-splash, async-fusion, scheduler-flags@seq8192 all exhausted/refuted; norms-HBM bucket is already XLA-fused (RMSNorm Pallas confirmed-counterproductive on TPU); scan refuted as dominant lever.
