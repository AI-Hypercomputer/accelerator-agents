---
title: "Qwen3 JAX v015 - splash TP2 tokamax remat fused rope bs8"
type: experiment
hypothesis: "Adding fused RoPE on top of the v014 Splash+tokamax+remat TP2 batch-8 configuration will preserve the CE memory win and reduce elementwise/loop-fusion overhead."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
verdict: supported
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, supported, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v015 - splash TP2 tokamax remat fused rope bs8

## Hypothesis under test

**Hypothesis**: Adding fused RoPE on top of the v014 Splash+tokamax+remat TP2 batch-8 configuration will preserve the CE memory win and reduce elementwise/loop-fusion overhead.

**Mechanism**: The workload image includes the `v015-splash-tp2-tokamax-remat-fused-rope-bs8` code path. The runtime shape matches v014 (`fsdp=4,tp=2`, per-chip batch 8, global batch 32, seq_len 2048) and keeps Splash enabled, so the intended delta is the fused RoPE implementation rather than a sharding or batch change.

**Predicted signal**: The run should complete with `EXIT_CODE=0`, keep peak HBM under 31.25 GiB, retain or improve the v014 throughput/MFU observation (~47,558 tok/s / 30.6% MFU), and show lower loop-fusion/elementwise share in XProf if fused RoPE fires.

**Falsification criterion**: Compile/runtime failure, peak HBM regression back to OOM, or clean completion with lower throughput/MFU than the v014 observation and no profile evidence of reduced loop-fusion/elementwise time.

## Setup

GKE workload `alekseyv-qwen3-v015`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=8 --seqlen=2048 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Both pods completed cleanly; `EXIT_CODE=0`.

| Metric | Worker 0 | Worker 1 | Notes |
|--------|----------|----------|-------|
| Steady step time | ~1.370 s | ~1.370 s | step 0 cold compile excluded |
| Throughput | 47,846 tok/s | 47,781 tok/s | 5,981 and 5,973 tok/s/chip |
| MFU | 30.8% | 30.7% | trainer estimate, v6e bf16 peak |
| Global batch | 32 | 32 | seq_len=2048 |
| Exit code | 0 | 0 | 20 train steps |

Formal frontier comparison:

| Run | Mesh | Global batch | Throughput | MFU | Delta vs v012 |
|-----|------|--------------|------------|-----|---------------|
| v012 Splash TP4 bs8 | fsdp=2,tp=4 | 16 | 39,438 tok/s | 25.4% | reference |
| v015 Splash TP2 tokamax remat fused RoPE bs8 | fsdp=4,tp=2 | 32 | 47,846 tok/s | 30.8% | +21.7% throughput, +5.4 pp MFU |

Observation-only comparison against backfilled v014: v015 improves trainer throughput from 47,558 to 47,846 tok/s and MFU from 30.6% to 30.8%. This is a small measurement-level improvement, not a large structural shift.

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8/2026_06_03_00_51_58`
- **XProf URL**: `http://localhost:8791/?run=2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **MXU utilization**: 35.5% (xprof overview)
- **Op profile**: `jit_train_step` accounts for 99.0% of device time; idle is 1.0%.

Top HLO op families by time:

| Op family | Time share | Notes |
|-----------|------------|-------|
| convolution fusion | 35.8% | matmul-heavy dense path |
| loop fusion | 12.8% | memory-bound elementwise/RMSNorm/RoPE area |
| all-gather | 11.5% | TP communication |
| all-reduce-scatter fusion | 11.0% | TP/FSDP collective path |
| custom-call | 10.4% | Splash/Pallas attention signal |
| all-to-all | 5.4% | TP communication |
| data formatting | 5.0% | layout/copy pressure |
| custom fusion | 3.2% | memory-heavy fused elementwise |

Compared with the v014 observation, loop-fusion share is unchanged at 12.8%, but total loop-fusion time drops slightly from 4,253 ms to 4,243 ms and non-fusion elementwise drops from 149.3 ms to 147.4 ms. The fused-RoPE signal is therefore only partial; the main win is preserving v014's tokamax/remat throughput while nudging it upward.

Memory profile:

```text
capacity: 31.25 GiB
peak:    31.23 GiB
stack:   25.2969 GiB
heap:     5.9352 GiB
free:     0.014 GiB
util:    99.94%
```

## HLO Dump

HLO dump path:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8/hlo/
```

Artifact count: 24 GCS objects, 270.07 MiB total.

Dominant optimized train-step dump:

```text
module_0089.jit_train_step.cl_854318611.after_optimizations.txt
size: 19.77 MiB
```

## Verdict

`supported`.

v015 is a clean frontier shift over the formal v012 baseline/frontier: throughput rises from 39,438 tok/s to 47,846 tok/s, and MFU rises from 25.4% to 30.8%. The specific fused-RoPE sub-hypothesis is only weakly supported: the profile does not show a major loop-fusion share reduction versus v014, only small total-time and elementwise reductions. Peak HBM is almost full at 31.23 GiB, leaving almost no memory margin, so the next iteration should not simply increase sequence length or batch without first freeing memory or changing the sharding/CE path.
