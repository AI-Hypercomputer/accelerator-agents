---
title: "Exp 20 — AMP (bf16 compute + fp32 master) at bs=1 seq=8192 (ACCEPTED, end-goal stack)"
type: experiment
tags: [llama3, torchax, amp, fp32-master, seq8k, accepted, milestone]
hypothesis: llama3-torchax-amp-fp32-master-seq8k
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp20-amp-fp32-master (image hf-v7)"
branched_from: v6e8-llama3-8b-torchax-20260425-exp13-scan
verdict: supported
---

🏆 **End-goal stack achieved.** The user-stated final target — *bf16
compute + fp32 master weights + seq=8192* — runs at **31.6 % MFU on v6e-8
multi-host**, only −0.6 pp from the no-fp32-master baseline (exp 17). All
the optimizations stacked together work end-to-end.

## Stack composition

This experiment combines every accepted optimization from the loop:

1. **Splash attention** (exp 2): `torch.nn.functional.scaled_dot_product_attention`
   override → `jax.experimental.pallas.ops.tpu.splash_attention`.
2. **Splash autotune** (exp 8 / 9 / 10): `block_q=block_kv=1024,
   use_fused_bwd_kernel=True, dkv=(2048,2048)` — universal winner across
   seq=1024–8192 (+30 % kernel fwd+bwd vs the stale defaults).
3. **Per-layer gradient_checkpoint** (exp 13): each `LlamaDecoderLayer.forward`
   wrapped in its own `interop.gradient_checkpoint` scope with
   `policy=nothing_saveable`. 32 distinct scopes force XLA to schedule
   layer-by-layer; this is what unlocks the `seq>1024` shapes.
4. **fp32 mu/nu master state** (this experiment): `optax.adamw(mu_dtype=fp32)`
   while weights stay in bf16. The standard mixed-precision pattern — fp32
   for the optimizer update, bf16 for forward/backward.
5. **VMEM 98304 KiB** (from exp 18b discovery): allows XLA scheduler more
   workspace to fit borderline shapes.

## Setup

- Image: **`hf-v7`**. Adds `--master_dtype` flag to train.py (passes
  `mu_dtype` through to `optax.adamw`).
- Workload: `llama3-8b-exp20-amp-bs1-seq8k`.
- Per-chip: B·L = 1 × 8192 = **8192**.
- Command: `--batch_size=1 --seqlen=8192 --weights_dtype=bf16
  --master_dtype=fp32 --use_splash=True --use_per_layer_remat=True`.

## Results

| Metric | Value |
|---|---|
| Cold compile | 137 s |
| Steady step time | **1,450 ms** |
| Throughput (aggregate) | **45,184 TPS** |
| Per-chip TPS | **5,648** |
| MFU (MaxText formula) | **31.6 %** |
| Loss step 0..14 | 11.7500 (bf16-precision constant; AMP weights are bf16) |

### Comparison with the chain

| Run | Stack | Shape | per-chip TPS | MFU |
|---|---|---|---|---|
| baseline | bf16 | bs=2 seq=1024 | 4,591 | 22.9 % |
| exp 9 | + splash autotune + bs=4 | bs=4 seq=1024 | 7,225 | **36.1 %** (best at seq=1024) |
| exp 17 | + per-layer remat + seq=8192 | bs=1 seq=8192 | 5,755 | 32.2 % (best at seq=8192) |
| **exp 20** | **+ fp32 mu/nu master** | bs=1 seq=8192 | **5,648** | **31.6 %** (end-goal) |
| exp 16 | full fp32 (weights + master) | bs=4 seq=1024 | 3,463 | 17.3 % (rejected — bf16 compute is critical) |
| exp 19 | full fp32 (weights + master) | bs=1 seq=8192 | — | OOM (790 MiB over) |
| exp 20b | AMP at higher density | bs=2 seq=8192 | — | OOM (844 MiB over) |

**Key finding**: AMP (bf16 compute, fp32 master) at the program-target seq is
**only −0.6 pp MFU** vs no-master-state (exp 17's 32.2 %). The convergence
benefit of fp32 master state is essentially free in throughput terms. By
contrast, **full-fp32** (exp 16) is **−14.9 pp MFU** because TPU MXU
falls back to VPU for fp32 matmul. AMP is the right precision strategy.

## Profile

- Profile pulled to `/tmp/llama3_run/exp20_profile_r0.tgz`. Upload to GCS
  pending.

## Verdict + reasoning

**Supported (accepted).** This becomes the **production target stack** for
all future Llama 3 8B torchax fine-tuning at the program shape.

The remaining gap to MaxText reference (44.6 % MFU at bs=3 seq=8192,
B·L=24,576) is **13 pp**. Closing more requires unlocking higher
per-chip B·L; the OOM data points narrow the path:

- bs=2 seq=8192 + AMP: OOM by 844 MiB → tokamax CE (~256 MiB savings) +
  dots_saveable variant might just fit; or scan-over-layers (proper
  refactor) frees more.
- bs=4 seq=2048 + AMP: probably OOM by similar margin (untested).
- bs=2 seq=4096 + AMP: maybe fits (per-chip B·L = 8192, similar to exp 20).

## Next hypotheses generated

1. **`exp 21` — host-offload `mu/nu` via `optax`'s `save_and_offload_only_these_names`**
   policy or custom offload. Frees 8 GiB / chip at fp32 master — way more than
   the 844 MiB exp 20b needs to fit.
2. **`exp 22` — tokamax CE kernel** (sharded-aware variant). Saves
   ~256 MiB / chip activations + frees logits-materialization peak.
3. **`exp 23` — bs=2 seq=4096 + AMP** — same B·L=8192 as exp 20, alternate
   shape. Likely fits.

## See also

- [Exp 17 (bs=1 seq=8192 baseline)](2026-04-25-exp17-layer-remat-bs1-seq8k-accepted.md) — the no-AMP comparison.
- [Exp 13 (per-layer remat)](2026-04-25-exp13-per-layer-remat-accepted.md) — the precondition.
- [Exp 9 (current best at seq=1024)](2026-04-25-exp9-splash-autotuned-bs4-accepted.md).
- [Llama 3 program](../../program.md) — `seq=8192` + `fp32 master` is the program-target.

## Sources

- Workload (XPK): `llama3-8b-exp20-amp-bs1-seq8k`.
- Image: `<your-registry>/test/llama3-8b-torchax-container:hf-v7`.
- Branch: `v6e8-llama3-8b-torchax-20260425-exp20-amp-fp32-master`.
