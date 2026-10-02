---
title: "Llama 3 8B torchax v6e-8 — 2026-04-26 day summary: morning baseline → end-of-day frontier"
type: analysis
tags: [llama3, torchax, summary, mfu, optimization-loop, milestone]
created: 2026-04-26
updated: 2026-04-26
---

Single-day summary of the optimization loop on `meta-llama/Meta-Llama-3-8B`,
torchax + JAX on v6e-8 (single slice, 8 chips, 2 hosts × 4 local devices),
program target = bf16 compute + fp32 master weights + seq=8192, bs sweep
across {2, 3, 4}.

## Day climb

| Frontier waypoint | Stack delta | bs=3 seq=8192 | Δ vs morning |
|-------------------|-------------|---------------|--------------|
| Morning baseline (exp 20: AMP fp32-mu/nu master, bs=1, no scan, no tokamax) | per-layer remat + jax-splash | 4,591/chip 31.6 % MFU | — |
| exp 55b — fp32 master weights + chunked_xla CE bring-up | + scan + tokamax CE (mosaic_tpu impl) + AMP master | 6,178/chip 34.6 % | +34.6 % |
| exp 56 — autotune | + tokamax CE autotune wrap | 6,202/chip 34.8 % | +35.1 % |
| exp 62b — chunked_xla CE | swap mosaic_tpu → chunked_xla impl | 6,303/chip 35.3 % | +37.3 % |
| exp 65 — chunked_xla + autotune + fp32 cast | reinstated fp32 cast at boundary (lse precision) | 6,313/chip 35.4 % | +37.5 % |
| exp 72a — tokamax-shipped splash | swap jax-splash → tokamax-splash + `use_base2_exp + fuse_reciprocal` | 6,392/chip 35.8 % | +39.2 % |
| 🏆 **exp 74b — final frontier** | **+ `max_logit_const=30`** | **6,559/chip 36.8 %** | **+42.9 %** |

Gap to MaxText reference (44.6 % MFU at the same shape): **7.8 pp**.

## Final stack (exp 74b)

```
--weights_dtype=fp32 --compute_dtype=bf16 --master_dtype=fp32
--use_real_data=True --use_splash=True --use_scan=True
--use_tokamax_ce=True --tokamax_ce_impl=chunked_xla --tokamax_ce_autotune=True
--scan_remat_policy=nothing_saveable
--batch_size=3 --seqlen=8192

# env
USE_TOKAMAX_SPLASH=1
TOKAMAX_USE_BASE2_EXP=1
TOKAMAX_FUSE_RECIPROCAL=1
TOKAMAX_MAX_LOGIT_CONST=30
LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304
```

## What landed (validated wins)

1. **Scan-over-layers** (exp 40 architecture, used everywhere ≥ exp 55b):
   collapses 32-layer unrolled HBM peak; required to fit anything beyond
   bs=2 at seq=8192.
2. **tokamax CE — chunked_xla impl** (exp 62b): replaces upstream
   `mosaic_tpu` Pallas CE bwd kernel (which recomputes blocks at 21 % MXU
   efficiency) with XLA matmul-based chunked impl. **+1.6 %** at bs=3.
3. **fp32 cast at CE boundary** (mandatory for chunked_xla — see exp 66
   invalidation): chunked_xla's `lse` and `loss_sum` accumulators inherit
   `x.dtype`, so bf16 input collapses precision. fp32 cast on inputs is
   required for correctness.
4. **tokamax-shipped splash** (exp 72a): swap upstream JAX splash for
   tokamax's variant; exposes extra perf knobs. **+1.3 %** at bs=3.
5. **`use_base2_exp=True`** (exp 73): TPU's `exp2` is faster than `exp`;
   the `/ ln 2` rescale fuses with upstream multiply. ~+0.7 % at bs=3
   (independent contribution).
6. **`fuse_reciprocal=True`** (exp 73): fuses the `output / lse` division
   into the kernel — saves one HBM round-trip on the post-attention
   output buffer per layer. ~+0.8 % at bs=3 (independent contribution).
   With (5) and (6) both on: super-additive +1.3 % combined.
7. **`max_logit_const=30`** (exp 74b): lets the kernel skip a softmax
   stabilization pass since logits never exceed 30 in practice. Loss
   identical step-for-step to the no-mlc run (mathematical equivalence).
   **+2.6 %** at bs=3.

## Refuted and invalidated

| Optimization | Result | Reason |
|--------------|--------|--------|
| ~~bf16-native chunked_xla (skip fp32 cast)~~ | INVALID | kernel accumulates lse in input dtype → loss precision lost (loss=11.0000 plateau). Per "no model-quality optimizations" rule. |
| ~~q_seq_shards=2 / 4~~ | INVALID | NaN loss from step 1 — kernel needs context-parallel mesh axis we don't have. Throughput "win" is artifact. |
| `dots_saveable` / `dots_with_no_batch_dims_saveable` remat | OOM by 19-42 GiB | matmul-output saves don't fit at this shape. |
| Recipe XLA flags (async-collective-fusion bundle) | neutral / -0.3 % | comms already well-overlapped at this density. |
| TP=2 (FSDP=4) | -14 % per chip | v6e ICI cannot absorb the extra TP traffic at this scale. |
| VMEM=131,072 / 65,536 KiB | -2.7 % / -3.1 % | VMEM=98 KiB optimum; bigger steals HBM, smaller costs scheduler workspace. |
| Splash block sweeps (bq=4096; 4096/4096 sym; 2048/2048 sym; bkv=512; bq=1024; MaxText-default 512-all) | -0.1 to -5.8 % | exp 8/9/10's autotuned 2048/1024/2048/2048-fused config remains optimal for our shape. |
| Unfused-bwd splash | -3.9 % | tokamax splash only supports fused-bwd anyway. |
| `dq_reduction_steps=3` | -0.6 % | conservative reduction-conflict avoidance hurts at this shape. |
| `use_experimental_scheduler=True` | -0.2 % | softmax overlap heuristic doesn't pay at this shape. |
| `cost_estimate_flops_fwd/bwd` (XLA hint) | within noise | XLA already overlaps the splash kernel with FSDP collectives. |
| `JAX_DEFAULT_MATMUL_PRECISION=bfloat16` / `tensorfloat32` | within noise / -0.5 % | `default` already does bf16 matmul efficiently on TPU. |
| bs=4 with full optimal stack | 6,415/chip 36.0 % | bs=3 still wins by +2.2 % per chip; bs=4 has memory pressure that cuts MXU efficiency. |
| bs=5 | OOM by 2.91 GiB | density ceiling is bs=4 at seq=8192 even with chunked_xla CE. |

## Bottleneck on the frontier

From the [exp 79 profile](../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md#updated-breakdown--2026-04-26-latest-frontier-exp-79-profile-on-the-exp-74b-stack)
of the exp 74b stack:

| Category | ms/step (per chip) | %  | HBM-BW util |
|----------|-------------------:|----|------------:|
| Convolution fusion (matmul) | 2,040 | 54.4 % | 0.54 |
| Splash MHA (fwd + dkv bwd) | 816 | 21.8 % | — |
| Loop fusion (RMSNorm + silu + residual + mul) | 570 | 15.2 % | 0.75 |
| Misc / data formatting / collectives | ~322 | ~8.6 % | — |
| Tokamax CE | ~0 | ~0 % | — |
| **Total** | **3,748** | **100 %** | |

MXU **utilization 55.0 %**; **matmul achieves 73.9 % MXU efficiency when
running**. The MFU shortfall is "matmul time-share" — too much non-matmul
cost — not "matmul-tile efficiency."

## What it would take to close the 7.8 pp gap

Theoretical analysis: to climb from 36.8 % to MaxText's 44.6 %, we need
to cut step time from 3.748 s → 3.09 s — save ~660 ms / step (-18 %).

The savings have to come from non-matmul time (matmul itself is at 74 %
MXU efficiency, hard ceiling without a different kernel). Three deep-work
levers, each requiring kernel/framework rewrites:

1. **Faster splash MHA bwd kernel** (~-400 ms achievable target).
   The current `splash_mha_dkv_no_residuals` runs at ~24 % HBM-BW util;
   a Pallas-mosaic kernel that better overlaps the dQ/dK/dV emitters
   could halve its time. Custom Pallas work, weeks of effort.
2. **Custom matmul prologue** (~-200 ms achievable target).
   A Pallas matmul that fuses the `weights.astype(bfloat16)` cast into
   the matmul prologue would skip the materialized bf16 weight buffer
   in HBM (one round-trip per matmul × 32 layers × 7 matmuls per layer
   × 3 fwd+bwd = ~672 round-trips per step). Custom Pallas work.
3. **MaxText-style layer fusion** (~-150 ms achievable target).
   Compress the loop-fusion 570 ms (RMSNorm + silu + residual + mul +
   embeddings) by emitting a single fused HLO region per
   `LlamaDecoderLayer` instead of per-op. Requires moving from HF
   transformers + torchax into a Flax/JAX-native module structure.

None of these are reachable by sweeping flags or env vars on the existing
HF-Llama-via-torchax call sites. The current stack has been pushed to the
ceiling of the knob-tunable surface.

## Day-end deliverable

- [exp 55b page (chunked_xla CE bring-up)](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md)
- [exp 62b page (chunked_xla iteration)](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md)
- [exp 72a page (tokamax-splash + final frontier)](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md)
- [bottleneck breakdown observation](../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md)

## See also

- [Llama 3 8B program README](../experiments/llama3_8B_autoresearch_optimization/README.md)
- MaxText reference: 44.6 % MFU at bs=2 seq=8192 (Llama 3.1-8B; same
  shape we benchmark against)
- The full day's experiments span exp 48 → exp 82 (35 numbered
  experiments + their lettered variants).

## Sources

- All exp_* xpk shell scripts under `/tmp/llama3_run/xpk/`
- Trainer: `/mnt/disks/persist/docker-build-llama3/trainer/{train.py,splash_attn.py,model/scan.py,model/sharding.py}`
- Profile artifact: `gs://<your-bucket>/jax-experiment/llama3-8b-exp79-tk-prof/`
