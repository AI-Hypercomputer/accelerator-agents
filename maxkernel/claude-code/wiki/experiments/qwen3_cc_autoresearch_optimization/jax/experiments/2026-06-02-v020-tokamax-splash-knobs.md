---
title: "Qwen3-8B jax v020 — tokamax-splash perf knobs (frontier)"
type: experiment
hypothesis: qwen3-jax-tokamax-splash-knobs
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-splash-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, splash, tokamax, xla-flags, remat, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v020 — tokamax-splash perf knobs (frontier)

## Hypothesis under test

**Hypothesis**: Swapping upstream JAX splash for the **tokamax-shipped** splash kernel with its perf
knobs (`use_base2_exp`, `fuse_reciprocal`) — llama3-jax's +4.4% lever, currently OFF on this lane
(`USE_TOKAMAX_SPLASH` defaults to 0) — lifts the v018 seq2048 frontier above 35.8% MFU.

**Mechanism**: v018 frontier config (`--use_remat --use_splash --batch_size=4 --seqlen=2048`, the 7
LIBTPU scheduler flags + scoped_vmem) + env `USE_TOKAMAX_SPLASH=1`. The trainer's `splash_attn.py`
already routes to `tokamax._src.ops.experimental.tpu.splash_attention` when this env is set, with
`use_base2_exp`/`fuse_reciprocal` defaulting on. Image `qwen3-8b-jax:v006-splash` (tokamax importable).
Flag-only, no rebuild. NOTE: the invalid `xla_jf_spmd_threshold` flag is excluded (crashes this build).

**Predicted signal**: MFU > 35.8% (target +1–4 pp); splash custom-call time-share drops; loss parity
(~12.07, no NaN; base2/fuse_reciprocal are exact reformulations); exit 0.

**Falsification criterion**: MFU ≤ 35.8% beyond noise (tokamax knobs don't help qwen3's head config),
or a tokamax-splash VMEM OOM / numerics issue.

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v006-splash`. Flag-only (env toggle).
- **Launch**: v018 frontier (`--use_remat --use_splash --batch_size=4 --seqlen=2048` + 7 LIBTPU scheduler flags + scoped_vmem) **+ env `USE_TOKAMAX_SPLASH=1`**. No jf_spmd flag.
- **tokamax path confirmed**: all 28 layers logged `[tokamax-splash]` with `bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 base2_exp=True fuse_recip=True exp_sched=False`.

## Results

| Metric | v018 frontier (upstream splash) | v020 (tokamax-splash + knobs) | Δ |
|--------|---------------------------------|-------------------------------|---|
| MFU | **35.8%** | 32.8% | **−3.0 pp** |
| tok/s/chip | 6,964 | 6,375 | **−8.5%** |
| throughput tok/s | 56,782 | 51,004 | −10.2% |
| steady step time (ms) | 1,154 | 1,285 | +11.4% |
| loss | 12.10→12.07 | 12.095→12.069 stable | parity |
| exit | 0 | 0 | — |

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-jax-v020-tokamax-splash-knobs
- **Run name**: `2026-06-02-qwen3-jax-v020-tokamax-splash-knobs`
- **On-disk pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v020-tokamax-splash-knobs/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v020-tokamax-splash-knobs/)
- **GCS**: `.../2026-06-02-qwen3-jax-v020-tokamax-splash-knobs/plugins/profile/2026_06_02_07_23_22/` (2 host dirs).
- **Steps captured**: 12–14.
- Contents: xprof trace + HLO dump.

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v020-tokamax-splash-knobs/hlo/` — 54 modules (text + proto).

## Verdict

**Refuted.** Falsification criterion ("MFU ≤ 35.8% beyond noise") met: 32.8% < 35.8%, a clear −3.0 pp / −8.5% regression with step time +11.4%. The tokamax-splash path is confirmed active (`base2_exp`/`fuse_reciprocal` both True) yet it is **slower than upstream JAX splash** on qwen3 — the **opposite** of the llama3-jax sibling (where tokamax-splash + these knobs was +4.4%). Loss parity confirms the knobs are numerically sound; this is purely a throughput regression in the tokamax kernel routing for qwen3's head config (32q/8kv hd128) at this shape. **tokamax-splash retired for this lane; upstream JAX splash stays the frontier kernel.** This is the second llama3-transferable kernel lever that fails to transfer to qwen3 (after [SparseCore-offload](2026-06-02-v017-sc-bs4.md)) — while the *scheduler-flag* lever did transfer (v018). Pattern: qwen3-jax's kernel/collective economics differ from llama3-jax; compiler/scheduler levers transfer, kernel-swap levers don't.

## Next hypotheses

- [Splash block-size sweep](../../../../hypotheses/qwen3-jax-splash-block-tuning.md) — on **upstream** splash (the frontier kernel): `splash_attn.py` notes symmetric `bq==bkv==1024` as a torchax "universal winner", but the lane currently runs `bq=2048 bkv=1024`. Try `SPLASH_BQ=1024`. Flag-only. **Dispatched as v021.**
- [seq16384](../../../../hypotheses/qwen3-jax-seq16384.md) — does upstream splash scale to 2× the target seq? Flag-only.
- None further on tokamax-splash — refuted (regresses on qwen3's head config).
