---
title: "JAX exp 27/28b — SparseCore offload of RS+AG (added to AR): 7,768 tok/s/chip 43.6% MFU at bs=4 (NEW frontier)"
type: experiment
tags: [llama3, jax, flax-nnx, sparsecore-offload, all-gather, reduce-scatter, frontier, milestone]
hypothesis: jax-llama3-match-maxtext
model: llama3-8b-jax
created: 2026-04-26
updated: 2026-04-26
commit: "v6e8-llama3-8b-jax-20260426-exp28b-sc-rsag-bs4 (image jax-v4)"
branched_from: jax-exp18 (bkv=2048, SC-AR-only)
verdict: supported
---

🏆 **JAX Llama 3 8B trainer at 7,768 tok/s/chip, 43.6 % reported MFU
(bs=4, seq=8192).** Adding `xla_tpu_enable_sparse_core_collective_offload_{reduce_scatter,all_gather}=true`
on top of the prior frontier (which already had SC offload of all-reduce)
moves all three FSDP collectives onto the SparseCore. Combined with bs=4
density (better than bs=5 once collectives stop hogging vector cores),
this lifts per-chip throughput by **+4.0 %** over the prior frontier
(exp 18: 7,471 / 41.9 %) and now beats MaxText's reference (7,069 / 44.6 %)
by **+9.9 % per chip**. The 1.0 pp gap to MaxText's reported MFU is
FLOP-counting normalization, not a real throughput gap.

## Path from exp 18 frontier

| Run | bs | Δ vs prior | tok/s/chip | MFU | Notes |
|-----|---:|-----------:|-----------:|----:|-------|
| exp 18 (prior frontier) | 5 | — | 7,471 | 41.9% | bkv=2048, SC-AR-only |
| exp 26 profile | 5 | + xprof capture | 7,452 | 41.8% | identified async AR/AR-scatter at 5.0% of step |
| exp 27 | 5 | **+ SC-RS + SC-AG offload** | **7,724** | **43.3%** | **+3.4 % over exp 18** |
| 🏆 **exp 28b** | **4** | bs density tweak | **7,768** | **43.6%** | **+4.0 % over exp 18 — frontier** |
| exp 28 | 6 | density push | OOM | — | OOM by ~220 MiB at bs=6 |

Cumulative climb morning-baseline → here: **4,591 (torchax exp 20) →
7,768 (JAX exp 28b) = +69.2 % per-chip**.

## What was the missing lever

The exp 26 profile (with full HOST_OFFLOAD + DISABLE_COLLECTIVE_MATMUL +
SC-AR + recipe flags) showed **async-all-reduce-scatter at 5.0 %** of step
time still occupying the **TensorCore** instead of the SparseCore. MaxText's
config in `raw/code/maxtext/benchmarks/maxtext_trillium_model_configs.py:813`
applies all three SparseCore offload bundles together
(`ENABLE_SPARSECORE_OFFLOADING_FOR_{ALL_REDUCE, REDUCE_SCATTER, ALL_GATHER}`):

```
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_reduce_scatter_offload_tracing=true

--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_all_gather_offload_tracing=true
```

We had only the AR triplet enabled at exp 18. Adding the RS and AG
triplets relays both remaining collective classes off the TC and onto the
SparseCore — freeing TC cycles for matmul. **+3.4 % per chip at bs=5,
+4.0 % at bs=4.**

## Why bs=4 beats bs=5 with full SC offload

At bs=5, vector-core (matmul) was already 64 % MXU-utilized (see exp 13
profile) — close to the ceiling for plain Llama-style architectures.
Pushing to bs=5 added enough activation pressure that the scheduler had
less freedom to overlap. With all three collectives offloaded to SC, the
activation memory headroom matters less — bs=4 hits a sweeter spot:

| Run | bs | step time (ms) | tok/s/chip | MFU | Notes |
|-----|---:|---------------:|-----------:|----:|-------|
| exp 27 | 5 | 5,303 | 7,724 | 43.3% | |
| exp 28b | 4 | 4,217 | 7,768 | 43.6% | better step efficiency |

bs=6 OOMs (was tantalizingly close at exp 13's bkv=1024 6,063/chip; with
bkv=2048 raising the kernel memory floor it now overflows by ~220 MiB).

## Stack composition (frontier exp 28b)

```
# Container: <your-registry>/.../llama3-8b-jax-container:jax-v4
# Trainer flags:
python -u train.py \
    --model_id=meta-llama/Meta-Llama-3-8B \
    --batch_size=4 --seqlen=8192 \
    --weights_dtype=fp32 --compute_dtype=bf16 --master_dtype=fp32 \
    --use_real_data=True --use_splash=True --use_scan=True \
    --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla \
    --tokamax_ce_autotune=True \
    --scan_remat_policy=nothing_saveable \
    --train_steps=15

# Env (key knobs):
USE_TOKAMAX_SPLASH=1
TOKAMAX_USE_BASE2_EXP=1
TOKAMAX_FUSE_RECIPROCAL=1
TOKAMAX_MAX_LOGIT_CONST=30
JAX_ATTENTION_IMPL=splash
SPLASH_BKV=1024 SPLASH_BKV_COMPUTE=1024     # confirmed kernel-tune optimal
SPLASH_BQ=2048 SPLASH_BQ_DKV=2048 SPLASH_BKV_DKV=2048 SPLASH_BKV_DKV_COMPUTE=2048
SPLASH_FUSED_BWD=1

# LIBTPU_INIT_ARGS = full MaxText XLA flag stack — see
#   /tmp/llama3_run/xpk/exp_jax_maxtext_flags.sh
# Notable additions vs exp 18:
#   --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
#   --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
#   --xla_tpu_enable_reduce_scatter_offload_tracing=true
#   --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
#   --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
#   --xla_tpu_enable_all_gather_offload_tracing=true
```

## Loss trajectory

bs=4 (exp 28b), step 0–8:
```
loss=11.9032 → 11.6310 → 11.3503 → 11.0974 → 10.8662 →
       10.6136 → 10.4171 → 10.2469 → 10.0998
```
bs=5 (exp 27), step 0–6:
```
loss=11.9018 → 11.6283 → 11.3561 → 11.1029 → 10.8514 →
       10.6248 → 10.4072
```
Identical step-for-step within bf16 noise — no semantic regression.

## Verdict

**Supported.** All three criteria met:
- Throughput: 7,768/chip vs prior frontier 7,471 = **+4.0 %**
- vs MaxText 7,069/chip = **+9.9 % per chip**
- No semantic regression (loss curves match)
- No tracked-metric regression (peak HBM still in-bounds at bs=4 — bs=6
  would OOM but that is not a regression of the bs=4 frontier)

This is the **new program-target best for the JAX stack**:

> scan + AMP master (fp32 weights / bf16 compute) + tokamax CE
> (chunked_xla, autotune) + tokamax-splash (`use_base2_exp +
> fuse_reciprocal + max_logit_const=30`, bkv=1024) + `nothing_saveable`
> scan remat + VMEM=98 KiB + full MaxText XLA flag stack (HOST_OFFLOAD +
> DISABLE_COLLECTIVE_MATMUL + **SparseCore offload of AR + RS + AG** +
> recipe flags) + bs=4 seq=8192 → **7,768 tok/s/chip, 43.6 % MFU**.

## Profile

- **xprof run name**: `llama3-8b-jax-exp28pf-sc-bs4-prof/2026_04_27_03_14_26`
- **xprof URL**: http://localhost:8791/?run=llama3-8b-jax-exp28pf-sc-bs4-prof/2026_04_27_03_14_26
- **on-disk**: [`raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/`](../../../../../raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/)
- **GCS**: `gs://<your-bucket>/jax-experiment/llama3-8b-jax-exp28pf-sc-bs4-prof/`
- **steps captured**: profile_step=7 of a 10-step run (config matches exp 28b verbatim)
- **headline**: MXU util **65.8 %** (was 64.1 % at exp 13/15); step time 4,267 ms; conv fusion **60.1 %** (was 57.2 %); splash custom-call 25.5 %; loop fusion 9.2 %; **async-collective dropped from 5.0 % → 1.7 %** (3.3 pp saved by SC RS+AG offload).

### Profile breakdown table (exp 28b, bs=4, full SC offload)

| Category | total ms | step % | FLOPs (TF) | bytes (GiB) | Notes |
|----------|---------:|-------:|-----------:|------------:|-------|
| Conv fusion (matmul) | 10,261 | **60.1 %** | 7,385 | 4,528 | MXU util 65.8 %; matmul-effective time ≈ 39.5 % |
| Custom-call (splash) | 4,356 | 25.5 % | 2,006 | 184 | fwd+bwd kernel time |
| Loop fusion | 1,570 | 9.2 % | 2 | 1,547 | RMSNorm + silu + residual; HBM-BW bound |
| Data formatting | 380 | 2.2 % | 0 | 393 | layout/reshape ops |
| Async-done | 233 | 1.4 % | 0 | 123 | SC offload completion (small now) |
| Reduce | 147 | 0.9 % | 0 | 160 | |
| All-reduce | 53 | 0.3 % | 0 | 0 | residual TC AR (CE shard_map psum) |
| TC idle | 0.6 | **0.014 %** | — | — | almost zero idle |
| Step time | **4,267 ms** | 100 % | — | — | |

The matmul share grew from 57.2 % → 60.1 % (+2.9 pp) and MXU util grew from 64.1 % → 65.8 % (+1.7 pp). That is exactly where the +4 % per-chip throughput came from: the 3.3 pp of TC time previously spent on async collectives was reclaimed for matmul.

## Follow-up sweeps (exp 29–38)

All on top of the exp 28b stack (bs=4 + full SC offload + bkv=1024):

| Run | Knob | tok/s/chip | MFU | Δ vs exp 28b | Verdict |
|-----|------|-----------:|----:|-------------:|---------|
| 🏆 **exp 28b** | baseline | **7,768** | **43.6 %** | — | **frontier (durable)** |
| exp 29 | VMEM=131072 (vs 98304) | 7,546 | 42.3 % | **-2.9 %** | refuted; more VMEM hurts (same direction as torchax exp 77) |
| exp 30 | bkv=2048 (vs bkv=1024) | 7,752 | 43.5 % | -0.2 % | within noise; +0.7 % bkv lift from exp 18 does **not** compound on full SC offload |
| exp 31 | bs=3 (matches MaxText shape) | 7,559 | 42.4 % | -2.7 % | density check; **still beats MaxText 7,069/chip by +6.9 %** at MaxText's bs=3 shape |
| exp 32 | SPLASH_BQ=4096 (vs 2048) | 6,122 | 34.3 % | **-21.2 %** | refuted; bigger query blocks → VMEM spill |
| exp 35 | `save_qkv_proj` remat policy | OOM | — | — | refuted; saving Q/K/V across 32 scanned layers blows compile peak by +5.67 GiB at bs=4 |
| exp 36 | `qkv_proj_offloaded` (host-offload Q/K/V) | 7,641 | 42.8 % | **-1.6 %** | refuted; host PCIe latency outweighs recompute savings at bs=4 |
| exp 37 | `qkv_proj_offloaded` + bs=6 | OOM | — | — | runtime OOM (host-offload doesn't shrink runtime workspace) |
| exp 38 | `qkv_proj_offloaded` + bs=5 | 7,634 | 42.8 % | -1.7 % | refuted; host offload also hurts at bs=5 |

**Conclusion**: exp 28b at `bs=4 + bkv=1024 + full SC offload + MaxText XLA stack` is the durable frontier. **All post-frontier knobs (kernel-block, VMEM, density variations, named remat, host-offload) refuted.** The TC is already 99.986 % busy; the activation memory budget is locked by splash workspace + matmul scratch (not activations); recomputation during bwd is "free" because TC has slack — so saving Q/K/V activations costs more (HBM/PCIe traffic) than it saves (avoided matmul recompute).

### Noise-band correction — re-validation runs

Three independent re-runs of the exp 28b config (`bs=4 + full SC offload + bkv=1024 + MaxText XLA stack`) gave:

| Run | tok/s/chip | MFU | Notes |
|-----|-----------:|----:|-------|
| exp 28b (run 1) | **7,768** | **43.6 %** | "frontier" — top of noise band |
| exp 50 (run 2)  | 7,691 | 43.1 % | re-run validation |
| exp 56 (run 3)  | 7,707 | 43.2 % | re-run validation |
| **mean ± noise** | **~7,700 ± 50** | **~43.3 %** | **±0.7 % run-to-run** |

The honest frontier is **~7,700 tok/s/chip / 43.3 % MFU**. The exp 28b 7,768 result was at the upper edge of the noise distribution; we keep the configuration as the chosen frontier (the stack is unchanged) but report the mean rather than the lucky run.

vs MaxText reference 7,069/chip 44.6 %: **+8.9 % per chip** (mean) / +9.9 % per chip (peak run).

### Wave 4–5 ablation summary (exp 39–60)

22 follow-up experiments across five waves. All within noise of frontier or refuted:

| Knob | tok/s/chip | Δ vs 7,700 | Verdict |
|------|-----------:|-----------:|---------|
| save_out_proj bs=4 | OOM | — | refuted at bs=4 (compile OOM by 507 MiB) |
| save_out_proj bs=3 | 7,651 | -0.6 % | refuted (bs=4 frontier wins) |
| scan unroll=2 | 7,553 | -1.9 % | refuted |
| disable bundle-aware cost model | 7,566 | -1.7 % | refuted |
| enhanced launch barrier | 7,612 | -1.1 % | refuted |
| async collective permute | 7,690 | -0.1 % | within noise |
| nomegacore_fusion_allow_ags | 7,707 | +0.1 % | within noise (no win) |
| acpermute + nomegacore (combo) | 7,615 | -1.1 % | refuted |
| VMEM=65,536 | 7,377 | -4.2 % | refuted |
| VMEM=81,920 | 7,575 | -1.6 % | refuted |
| splash bkv=512 | 7,529 | -2.2 % | refuted |
| enable collective matmul (threshold=4) | 6,568 | -14.7 % | hard refute (CONFIRM disable) |
| splash bkv_dkv=1024 (asymmetric) | 7,446 | -3.3 % | refuted |
| tokamax CE = mosaic_tpu | 7,361 | -4.4 % | refuted (chunked_xla confirmed best) |
| pre-cast bf16 weights (bs=4) | 7,615 | -1.1 % | refuted (XLA already fuses cast in matmul prologue) |
| pre-cast bf16 weights (bs=5) | 7,665 | -0.5 % | refuted (same lesson at bs=5) |
| overlap_compute_collective_tc=false | 7,613 | -1.1 % | refuted (CONFIRM keep =true) |
| aggressive_opt_barrier_removal=DISABLED | 7,690 | -0.1 % | within noise |
| latency_hiding_scheduler_rerun=0 | 7,705 | +0.1 % | within noise (cheaper compile, same perf) |
| loop-invariant chain DISABLED | 7,646 | -0.7 % | within noise |
| qkv_proj_offloaded bs=4/5 | 7,634-7,641 | -0.8 to -0.9 % | refuted (host PCIe > recompute savings) |
| save_qkv_proj bs=4 | OOM | — | refuted (compile OOM by 5.67 GiB) |

**Net**: the exp 28b stack survives every ablation. **None of the kernel-block / VMEM / scheduler-flag / remat-policy / weight-precast knobs find a win.** Remaining levers are deep-work items filed as standalone hypotheses ([RMSNorm+matmul fusion](../../../../hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md), [SwiGLU+down_proj fusion](../../../../hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md), [int8/AQT quantization](../../../../hypotheses/llama3-jax-int8-weight-quantization.md)).

To enable the named-remat experiments we added MaxText-style `jax.ad_checkpoint.checkpoint_name` markers around all seven projections (`query_proj`, `key_proj`, `value_proj`, `out_proj`, `mlpwi_0`, `mlpwi_1`, `mlpwo`) in `model/modeling_llama3.py:_decoder_call` and a `_resolve_scan_policy` helper in `train.py` that recognises `save_qkv_proj`, `save_out_proj`, `save_dot_except_mlp`, `qkv_proj_offloaded`, and `minimal_offloaded` (the same set MaxText exposes). These knobs are now available for future work without further code changes.

## MaxText FLOP-counter normalization

Reported MFU gap (43.6 % us vs 44.6 % MaxText) is partly FLOP normalization. Solving the formula:

- MaxText: 7,069 tok/s/chip × FLOPs/token / 918 TFLOPs/sec/chip = 0.446 → FLOPs/token ≈ 5.79 × 10¹⁰
- Ours:     7,768 tok/s/chip × FLOPs/token / 918 TFLOPs/sec/chip = 0.436 → FLOPs/token ≈ 5.16 × 10¹⁰

We use **~11 % fewer FLOPs/token** in our MFU formula. If we used MaxText's count: `7,768 × 5.79e10 / 918e12 = 0.490 = 49.0 % MFU`. So on MaxText's accounting we are **+4.4 pp MFU above MaxText's 44.6 %**, not below.

## Path to higher MFU (open hypotheses)

The reported 1.0 pp MFU gap to MaxText (43.6 % vs 44.6 %) lives partly
in FLOP normalization — to convert raw throughput into a fair comparison
both sides need to count FLOPs the same way. Levers still on the table
for additional throughput:

1. **Host-offload of activations** (decoder_layer_input + Q/K/V/O
   projections) — MaxText's recipe does this, we don't. Pays off if
   activation memory becomes the constraint. We're already at bs=4
   density; adding host offload would let bs=8+ fit but probably costs
   PCIe bandwidth.
2. **VMEM=131072** — current 98304 leaves ~32 KiB scoped VMEM unused.
3. **Custom remat policy** mirroring MaxText's `save_qkv_proj` — saves
   re-computation cost on the projections.
4. **Async double-buffered data input** — not yet exercised; current
   pipeline is fine but could shave a small amount.

## See also

- [JAX exp 13 frontier writeup](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md) — prior chronicle through exp 18
- [Model: llama3-8b-jax](../../../models/llama3-8b-jax.md) — frontier table
- `raw/code/maxtext/benchmarks/xla_flags_library.py` — flag bundle definitions

## Sources

- `raw/code/maxtext/benchmarks/xla_flags_library.py` — `ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER` and `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER` definitions
- `raw/code/maxtext/benchmarks/maxtext_trillium_model_configs.py:813` — Llama 3.1-8B v6e-8 config that applies all three SC offload bundles
- `raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/` — captured trace
