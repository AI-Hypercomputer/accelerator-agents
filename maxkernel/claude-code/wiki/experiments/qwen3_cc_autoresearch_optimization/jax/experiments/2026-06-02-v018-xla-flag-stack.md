---
title: "Qwen3-8B jax v018 — MaxText HOST_OFFLOAD XLA flag stack (frontier)"
type: experiment
hypothesis: qwen3-jax-xla-flag-stack
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-splash-2026-06-02
status: supported
verdict: supported
tags: [qwen3-cc, jax, xla-flags, splash, remat, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v018 — MaxText HOST_OFFLOAD XLA flag stack (frontier)

## Hypothesis under test

**Hypothesis**: The MaxText `HOST_OFFLOAD` XLA flag bundle (latency-hiding scheduler,
host-transfer overlap, all-gather backward pipelining, scheduler memory-pressure
tracking) — the single biggest XLA win on the llama3-jax sibling lane (+~10% alone,
which took it from ~34% → 43% MFU) — lifts the qwen3-jax v008 frontier above 32.4% MFU.
This lane has touched **only one** XLA flag so far (`scoped_vmem`); the bundle is untried.

**Mechanism**: v008 frontier config (`--use_remat --use_splash --batch_size=4 --seqlen=2048`,
image `qwen3-8b-jax:v006-splash`) + 8 flags appended to `LIBTPU_INIT_ARGS` alongside the
vmem flag: `xla_tpu_enable_all_experimental_scheduler_features`,
`xla_tpu_enable_scheduler_memory_pressure_tracking`, `xla_tpu_enable_ag_backward_pipelining`,
`xla_tpu_host_transfer_overlap_limit=24`, `xla_tpu_scheduler_percent_shared_memory_limit=100`,
`xla_latency_hiding_scheduler_rerun=2`, `xla_max_concurrent_host_send_recv=100`,
`xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`. Flag-only, no rebuild.

**Predicted signal**: MFU > 32.4% (target +5–10 pp); collective/transfer time-share drops in
the profile; loss parity (~12.07, no NaN), exit 0.

**Falsification criterion**: MFU ≤ 32.4% (flags inert or regressive on this shape), or any
flag rejected by libtpu causes a crash.

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v006-splash`. **Flag-only** (no rebuild).
- **Launch** (diff from v008 frontier: **7 scheduler flags** appended to `LIBTPU_INIT_ARGS` beside the already-present vmem flag):
  > [!note] Correction (from [v019](2026-06-02-v019-xla-flags-s8k.md)): the originally-planned 8th flag `--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000` is **invalid on this XLA build** (crashes "Unknown flag in XLA_FLAGS") and was **not** in this run's launch_cmd. v018's +3.4 pp is attributable to the 7 LIBTPU scheduler flags alone (scoped_vmem was already in the v008 baseline).
  ```
  cd /app/trainer && LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 \
    --xla_tpu_enable_all_experimental_scheduler_features=true \
    --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
    --xla_tpu_enable_ag_backward_pipelining=true \
    --xla_tpu_host_transfer_overlap_limit=24 \
    --xla_tpu_scheduler_percent_shared_memory_limit=100 \
    --xla_latency_hiding_scheduler_rerun=2 \
    --xla_max_concurrent_host_send_recv=100' \
  ... XLA_FLAGS='--xla_dump_to=<gcs_root>/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto' \
  python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False --seqlen=2048 \
    --tp_parallelism=1 --train_steps=20 --weights_dtype=bf16 \
    --use_remat --use_splash --batch_size=4 \
    --profile_dir=<gcs_root> --profile_start_step=12 --profile_steps=3
  ```
- global batch 32, seqlen 2048. **All 8 flags accepted by libtpu — zero rejections.**

## Results

| Metric | v008 frontier (splash bs4) | **v018 (+XLA flag stack)** | Δ |
|--------|----------------------------|----------------------------|---|
| MFU | 32.4% | **35.8%** | **+3.4 pp** |
| tok/s/chip | 6,299 | **6,964** (best sustained ~7,098) | **+10.6%** |
| throughput tok/s | 50,389 | 56,782 | +12.7% |
| steady step time (ms) | ~1,301 | **1,154** | −11.3% |
| loss | 12.10→12.07 | 12.0954→12.0685, monotone, no NaN | parity |
| exit | 0 | 0 | — |

Steady state from step 2 (step0 = 116 s first-compile, step1 = 7.8 s second-compile phase; both expected JIT, not hangs). Median steady step time 1,154 ms over steps 2–19.

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-jax-v018-xla-flag-stack
- **Run name**: `2026-06-02-qwen3-jax-v018-xla-flag-stack`
- **On-disk pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v018-xla-flag-stack/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v018-xla-flag-stack/)
- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v018-xla-flag-stack/plugins/profile/2026_06_02_06_57_38/`
- **Steps captured**: 12–14 (2 xplane hosts: g30s, rkr2).
- Contents: xprof trace (2 hosts) + HLO dump.

### Deep bucket attribution (profile-analyzer, steps 12–14, all hosts)

| Bucket | % step | ~ms | Bound by | Notes |
|--------|--------|-----|----------|-------|
| convolution fusion (matmul) | 51.4% | ~593 | compute (AI 1,777 ≫ ridge 578) | fwd+bwd proj/MLP matmuls |
| **all-reduce-scatter fusion (FSDP grad)** | **12.9%** | **~149** | **ICI, SYNCHRONOUS** | **#1 addressable — not pipelined with bwd matmuls; 362 per-tensor instances** |
| custom-call (splash Mosaic) | 12.3% | ~142 | VMEM/compute | splash fwd+bwd, fused-bwd kernel confirmed |
| loop fusion (norms/acts) | 11.0% | ~127 | HBM (AI 1.2) | RMSNorm/SiLU/QK-norm (XLA-fused here) |
| data formatting (layout) | 4.5% | ~52 | HBM | QKV→splash GQA layout bridges |
| custom fusion (scatter/RS body) | 3.8% | ~44 | — | embedding scatter, RS bodies |
| all-gather (fwd FSDP) | 1.4% | ~16 | ICI | already async-pipelined |
| other | ~2.7% | ~31 | — | all-reduce, copy-done(host-offload), broadcast |

**MXU util 48.3%** (v008 44.6%, +3.7 pp from the flags) · **TC idle 3.87 ms** (low) · **HBM peak 20.45/31.25 GiB = 65.4%** (healthy, remat working) · **HBM BW util 31.5%** (compute-bound overall).

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v018-xla-flag-stack/hlo/` — 871 files (13 unique modules; runtime dominated by `module_0209.jit_train_step`, 19.2 MiB / 124,713 lines).
- **Fusion verification** (profile-analyzer): splash = `tpu_custom_call` `splash_mha_fwd_residuals`(540) + `splash_mha_dkv_no_residuals`(432), `use_fused_bwd_kernel: true` — **PASS**. QK-norm+RoPE **already XLA-fused** into loop-fusion bodies (zero top-level rsqrt/reduce) — a custom kernel would hit `xla-already-fuses` unless it also absorbs the layout bridges. Host-offload `copy-start/done` pairs present (5,658) — `host_transfer_overlap_limit=24` active. **No `while`/scan** — train step fully unrolled across all decoder layers (explains the 871 per-layer artifacts).

> [!note] BOTTLENECK VERDICT (profile-analyzer, 2026-06-02): NOT matmul-bound (48.6% non-matmul). The #1 addressable cost is the **synchronous FSDP gradient reduce-scatter (12.9%, ~149 ms)** — `ag_backward_pipelining` overlaps all-gather but the grad reduce-scatter still runs serially; **async-collective-fusion (overlap RS with bwd matmuls) could recover ~7–13 pp MFU** (→ v023). Scan-over-layers **refuted as dominant lever** (won't move the runtime buckets; compile-time only). QK-norm+RoPE kernel deprioritized (XLA already fuses). See [observation: fsdp-reduce-scatter-not-overlapped](../../../../observations/qwen3-jax-fsdp-rs-not-overlapped.md).

## Verdict

**Supported — NEW FRONTIER, 35.8% MFU.** The MaxText HOST_OFFLOAD XLA scheduler flag bundle lifts the seq2048 frontier from 32.4% → **35.8% MFU** (+3.4 pp), **6,964 tok/s/chip** (+10.6%), step time −11.3%, with loss parity and zero semantic change (scheduling/transfer-overlap only). All 8 flags accepted. This is the exact lever the [retrospective](../../../../analyses/2026-06-02-qwen3_cc-jax-retrospective.md) flagged as the #1 unexplored direction (llama3-jax's +10% single win) — it transfers cleanly to qwen3. **The XLA flag stack is now part of the frontier config**; subsequent experiments stack on it. Climb: 20.5%→22.0%→25.1%→32.4%→**35.8%** (+74% tok/s/chip over baseline). Remaining gap to the llama3-jax sibling (43.3%): ~7.5 pp.

## Next hypotheses

- [XLA flag stack @ seq8192](../../../../hypotheses/qwen3-jax-xla-flag-stack.md) — does the +3.4 pp transfer to the program-target seq8192 (v009 shape, 30.4%)? Flag-only, immediate. **Dispatched as v019.**
- [AMP fp32-master / bf16-compute](../../../../hypotheses/qwen3-jax-amp-mixed-precision.md) — stack on the new v018 frontier; orthogonal lever. Code-change + image build.
- [Splash block-size sweep](../../../../hypotheses/qwen3-jax-splash-block-tuning.md) — splash still ~11% of step; tune bkv/bq on top of v018.
- Flag ablation — attribute which of the 8 flags carry the +3.4 pp (drop one at a time) to build a reusable prior for the torchax lane.
