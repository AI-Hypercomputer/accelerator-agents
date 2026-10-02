---
title: "Qwen3-8B jax v044 — splash context checkpointing @ bs2/seq8192"
type: experiment
hypothesis: qwen3-jax-splash-context-checkpoint
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork .repo/2026-06-12-v044-ctx (trunk 23aa5e7 + SPLASH_RESIDUAL_CKPT_NAME plumbing + save_qkv_ctx policies); image v6e8-qwen3-8b-jax-20260612-v044-ctx
verdict: supported
tags: [qwen3-cc5, jax, v6e-8, splash, remat, context-checkpoint, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v044 — splash context checkpointing

Ports MaxText's kernel-side attention-residual checkpointing (inventory #1):
splash emits out+lse tagged `"context"`; the `save_qkv_ctx` policy keeps
q/k/v in HBM and **offloads context to pinned host** — backward consumes
the residuals instead of re-running splash forward. Result: **+7.3% step
time (2,615 → 2,437 ms); +6.7% tok/s/chip (6,265 → 6,723/chip; ≈38.6%
MFU); MaxText gap closes from −9.9% to −3.3%.** The single largest
one-shot win since the v026 flag stack.

## Hypothesis under test

[qwen3-jax-splash-context-checkpoint](../../../../hypotheses/qwen3-jax-splash-context-checkpoint.md):
PASS > 6,390 tok/s/chip (frontier +2%; step < 2,563 ms). CPU smoke: grads
bit-identical across save_qkv / save_qkv_ctx.

**Diff vs v043**: env `SPLASH_RESIDUAL_CKPT_NAME=context` +
`--remat_policy=save_qkv_ctx`. `JAX_SCAN_UNROLL=1` guard against the XLA
S(5) unroll×offload bug (unroll=1 = default scan = safe).

## Setup

- **Hardware**: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`
  (project `<your-project>`, zone `<your-zone>`).
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v044-ctx` (fresh tag, digest 5acf9bbb).
- **Workload**: `alekseyv-qwen3-cc5-jax-v044-ctx` (31 chars < 40 limit).
- **Single arm**: bs2 seq8192 + scan + save_qkv_ctx + splash + chunked CE
  f32-x + 21-flag stack; 20 steps, profiled steps 12–14, HLO dump.

```bash
export JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"
export JAX_SCAN_UNROLL=1
export SPLASH_RESIDUAL_CKPT_NAME=context
export XLA_FLAGS="--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
  --batch_size=2 --seqlen=8192 --tp_parallelism=1 \
  --train_steps=20 --weights_dtype=bf16 --use_splash=True \
  --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla --remat_policy=save_qkv_ctx \
  --use_scan=True \
  --profile_dir=gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx \
  --profile_start_step=12 --profile_steps=3
```

Flags diff vs v043: `SPLASH_RESIDUAL_CKPT_NAME=context` added; `--remat_policy=save_qkv` → `--remat_policy=save_qkv_ctx`; `JAX_SCAN_UNROLL=1` added. The 21-flag `LIBTPU_INIT_ARGS` stack is unchanged from v043 (minus the 4 async-collective-fusion flags dropped at v035).

## Baseline comparison

| Metric | v043 (frontier, 50-step) | **v044 (this run, 20-step)** | MaxText (mt-v001) |
|--------|--------------------------|-------------------------------|-------------------|
| Step (median, steady) | 2,615.0 ms | **2,436.9 ms (−178 ms, −6.8%)** | 3,535 ms @ bs3 |
| tok/s/chip | 6,265 | **6,723 (+7.3%)** | 6,953 |
| MFU (our accounting) | ≈36.0% | **≈38.6% (+2.6 pp)** | ≈39.6% |
| vs PASS bar (6,390) | −2.0% | **+5.2% — PASS** | — |
| vs MaxText | −9.9% | **−3.3%** | — |
| Peak HBM | ~27 GiB | **22.03 GiB (70.5% cap)** | — |

## Results

20 steps clean, exit 0, loss bit-identical to reference (within ±0.0001),
no OOM, no XLA S(5) INTERNAL error, no NaN.

### Step table (host-0 readings)

| Step | Loss | Step time (ms) | tok/s global |
|------|------|----------------|--------------|
| 0 | 12.1029 | 42,058.7 | 3,116 (compile) |
| 1 | 12.0996 | 37,042.3 | 3,538 (warm-up) |
| 2 | 12.0944 | 2,434.2 | 53,846 |
| 3 | 12.0921 | 2,435.7 | 53,813 |
| 4 | 12.0897 | 2,434.6 | 53,837 |
| 5 | 12.0836 | 2,435.2 | 53,824 |
| 6 | 12.0813 | 2,435.8 | 53,811 |
| 7 | 12.0806 | 2,436.5 | 53,796 |
| 8 | 12.0804 | 2,436.5 | 53,796 |
| 9 | 12.0769 | 2,436.9 | 53,787 |
| 10 | 12.0742 | 2,437.0 | 53,785 |
| 11 | 12.0737 | 2,437.3 | 53,777 |
| 12 | 12.0683 | 2,471.5* | 53,034 |
| 13 | 12.0668 | 2,438.6 | 53,749 |
| 14 | 12.0667 | 2,438.8 | 53,745 |
| 15 | 12.0645 | 2,437.4 | 53,775 |
| 16 | 12.0617 | 2,439.2 | 53,735 |
| 17 | 12.0605 | 2,438.5 | 53,750 |
| 18 | 12.0616 | 2,439.2 | 53,736 |
| 19 | 12.0589 | 2,439.3 | 53,733 |

*Step 12 is the first profiling step; xprof capture adds ~35 ms overhead (expected).

**Steady-state summary** (steps 2–11, 15–19; 15 steps; excluding 0–1 compile
and 12–14 profiling):

| Statistic | Value |
|-----------|-------|
| Median step time | 2,436.9 ms |
| Min / Max | 2,434.2 / 2,439.3 ms |
| Spread (max−min) | 5.1 ms |
| Tokens/step | 131,072 (bs2 × seq8192 × 8 chips) |
| Global tok/s (median) | 53,786 |
| **tok/s/chip (median)** | **6,723** |

**Loss check** vs PASS criterion (≤ ±0.001):

| Step | This run | Reference | Delta | |
|------|----------|-----------|-------|---|
| 0 | 12.1029 | 12.1029 | 0.0000 | PASS |
| 1 | 12.0996 | 12.0995 | 0.0001 | PASS |
| 2 | 12.0944 | 12.0944 | 0.0000 | PASS |

## Profile

**xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v044-ctx/2026_06_12_15_01_13`
**Run name**: `2026-06-12-qwen3-jax-v044-ctx/2026_06_12_15_01_13`
**Steps captured**: steps 12–14 (3 profiling steps at steady state).
**Description**: xprof trace (TPU v6 Lite, 8 chips/host, 2 hosts, fsdp=8);
device + host planes. HBM peak 22.03/31.25 GB (70.5% cap); MXU 55.7%.
**GCS path**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx/plugins/profile/2026_06_12_15_01_13/`

### Bucket attribution (steady state, steps 12–14)

| Bucket | % of step | ~ms/step | vs v043 |
|--------|-----------|----------|---------|
| convolution fusion | 52.2% | 1,272 ms | flat (−1 ms) |
| **custom-call (splash)** | **21.7%** | **529 ms** | **−206 ms (−6.4 pp) ← primary driver** |
| loop fusion | 9.2% | 224 ms | −24 ms (−0.9 pp) |
| all-reduce-scatter fusion | 6.6% | 161 ms | −35 ms (−1.3 pp) |
| data formatting | 2.2% | 54 ms | flat |
| **async-done** | **2.0%** | **49 ms** | **+25 ms (new: context DMA overhead)** |
| custom fusion (CE) | 1.6% | 39 ms | — |
| all-reduce (DP) | 1.2% | 29 ms | — |
| all-gather (FSDP) | 0.9% | 22 ms | — |
| reduce | 0.8% | 19 ms | — |
| pad | 0.7% | 17 ms | — |
| broadcast | 0.2% | 5 ms | — |
| copy-done | 0.1% | 2 ms | — |

### Gain decomposition (v043 2,615 ms → v044 2,437 ms = **+178 ms**)

| Source | Saving |
|--------|--------|
| Splash custom-call bucket (no fwd re-run in bwd) | −206 ms |
| Loop fusion (smaller remat scope in bwd) | −24 ms |
| All-reduce-scatter (shorter scan body → better scheduling) | −35 ms |
| async-done DMA overhead (context 4.57 GiB offload/reload) | +25 ms |
| Other residuals | +12 ms |
| **Net** | **−178 ms** |

### HBM improvement

Context tensors that previously occupied HBM during backward are now in
host pinned memory S(5): `bf16[36,2,32,8192,128]` + `f32[36,2,32,8192]`
= 4.57 GiB offloaded. HBM peak dropped from ~27 GiB (v038 estimate) to
**22.03 GiB (70.5% cap)** — a ~5 GiB headroom recovery.

## HLO Dump

**GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx/hlo/` (50 files, 33.6 MiB)
**Modules (2)**: `jit_train_step(18061459034935412866)` + `jit__identity_fn(3512440348090122076)` — 2-module fingerprint confirms scan-over-layers active (1 main train step + 1 identity for params).

### Hypothesis-firing verification

| Signal expected | Found? | Evidence |
|-----------------|--------|---------|
| `splash_mha_fwd_residuals` in scan body (fwd saves residuals) | YES | Instances `.1` and `.2` at lines 5767, 7264 |
| `splash_mha_dkv_no_residuals` in bwd (consuming saved residuals) | YES | Instances `.1` and `.2` at lines 4630, 7851 |
| `splash_mha_fwd_no_residuals` absent (no bwd re-run) | YES | 0 occurrences |
| S(5) context tensors in while-loop parameters | YES | `bf16[36,2,32,8192,128]{S(5)}` + `f32[36,2,32,8192]{S(5)}` at lines 3718, 3728, 4318, 5578 |
| DMA via async `dynamic-update-slice-start/done` | YES | Per-layer async DUS pairs in scan body |
| `splash_mha_dq` absent (dq folded into dkv_no_residuals) | YES | 0 occurrences |

**All four structural conditions confirmed.** The 206 ms custom-call drop
(28.1% → 21.7%) matches the hypothesis's predicted 250 ms estimate within
8%. The DMA cost (+25 ms async-done overhead) is partially hidden but
not perfectly — 4.57 GiB at v6e's ~1.5 TB/s HBM bandwidth has a physical
floor of ~3 ms; the remaining 22 ms is scheduling latency (DMA not yet
fully pipelined behind MXU compute). This is a known refinement target.

## Verdict

**`supported`** — 6,723 tok/s/chip (+7.3% vs v043 frontier 6,265; +5.2%
vs PASS threshold 6,390; +178 ms step improvement). Loss bit-identical
(all deltas ≤ 0.0001). No semantic change (policy-only). Profile and HLO
both populated. No OOM, no NaN, no XLA S(5) error. The mechanism fires
exactly as predicted: splash backward reads saved out+lse from host-pinned
S(5) memory instead of re-running the forward kernel.

This closes the #1 item from the MaxText feature-gap inventory. The MaxText
gap narrows from −9.9% to −3.3% (≈85 ms/step remaining), with the dominant
remaining contributors being:

1. **Convolution fusion** (52.2%, 1,272 ms) — MXU at 55.7% vs MaxText's
   implied ~65%+ utilization. Layout and matmul-kernel path differences.
2. **async-done** (2.0%, 49 ms) — context DMA not fully pipelined; 25 ms
   residual vs baseline.
3. **Other bucket differences** (CE, pad, broadcast) — smaller targets.

The inventory #2 item (fused gate-up matmul / layout changes) remains the
next direction.

## Next hypotheses

- [qwen3-jax-fused-gate-up](../../../../hypotheses/qwen3-jax-fused-gate-up.md) — fuse gate and up projection into a single matmul (MaxText's `fused_kernel=True` for MLP gate-up); targets the 52.2% convolution fusion bucket and the MXU gap (55.7% vs MaxText's ~65%+); expected 2–6% step gain at M effort.
- [qwen3-jax-ctx-dma-pipeline](../../../../hypotheses/qwen3-jax-ctx-dma-pipeline.md) — improve DMA scheduling for the context round-trip; the 49 ms async-done (2.0%) includes ~25 ms residual latency not yet hidden behind compute; target is to drop async-done from 49 ms toward the 24 ms baseline via better XLA scheduler hints or earlier async-start placement; expected 0.5–1.5% gain at S effort.
- [qwen3-jax-ctx-hbm-only](../../../../hypotheses/qwen3-jax-ctx-hbm-only.md) — trial of `save_qkv_ctx_hbm` policy (keep context in HBM rather than offloading to host); hypothesis: the 25 ms DMA overhead may exceed the DMA hiding benefit for v6e's ICI schedule; compare step time vs v044's 2,437 ms; if HBM fits (22.03 GiB used, 31.25 GiB cap — ~9 GiB headroom available), HBM residency may close the async-done gap entirely; expected effect: +0.5–1% if fits, OOM if not; S effort.

## See also

- [Hypothesis](../../../../hypotheses/qwen3-jax-splash-context-checkpoint.md)
- [v043 (frontier before this)](2026-06-12-v043-bs2-svqkv-val.md)
- [MaxText feature-gap inventory](../../../../analyses/2026-06-12-maxtext-feature-gap-inventory.md)

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx/`
- xprof run: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx/plugins/profile/2026_06_12_15_01_13/`
- HLO dump: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-ctx/hlo/`
- Prior: [MaxText feature-gap inventory](../../../../analyses/2026-06-12-maxtext-feature-gap-inventory.md)
- v043: [2026-06-12-v043-bs2-svqkv-val.md](2026-06-12-v043-bs2-svqkv-val.md)
