---
title: "Qwen3 JAX v258 - MLP up projection before gate projection"
type: experiment
hypothesis: "Computing the MLP up projection before the gate projection may alter producer ordering around the hot MLP fused matmul/SwiGLU region without changing Qwen3 math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: supported-same-band
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v258 - MLP up projection before gate projection

## Hypothesis under test

**Hypothesis**: The current frontier's largest device-time bucket remains MLP
convolution fusion. The v256 explicit-SwiGLU rewrite changed HLO but regressed.
A smaller source-order probe may influence XLA's producer ordering around the
two MLP input projections and the SwiGLU epilogue while preserving exact math.

**Mechanism**: Derive a thin image from `qwen3-8b-jax:v169-no-mlp-sac` and
change only `Qwen3MLP.__call__`:

```python
u = cn(self.up_proj(x), "mlpwi")
g = cn(self.gate_proj(x), "mlpwi")
return self.down_proj(jax.nn.silu(g) * u)
```

The original expression computes `gate_proj` before `up_proj`. Both projections
keep the same params and the same `checkpoint_name(..., "mlpwi")` tag; the
final value is unchanged.

Keep the v230 RS2+AR2 runtime stack fixed: no-scan, remat, Tokamax Splash,
MaxText CE, activation sharding, scoped VMEM 100352 KiB, all-gather on async
collective fusion, SparseCore reduce-scatter/all-reduce offload plus
aggregator, RS latency multiplier 2, AR latency multiplier 2.

**Predicted signal**: Support requires clean finite loss and throughput above
the v230/v236 band of **59,749-59,750 tok/s / 42.9% MFU**, or a same-band
throughput with materially reduced convolution-fusion self-time.

**Falsification criterion**: Compile/runtime failure, invalid loss, same HLO
with no throughput gain, or clean completion below the frontier.

## Setup

GKE workload: `alekseyv-q3-v258-mlp-up-first`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Base digest**: `sha256:21a390217fc38d05a8638d2ae681f00ece3d255d57ed88a0651e4054304f6854`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v258-no-scan-sparsecore-rs-ar-rs2-ar2-mlp-up-proj-first`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

The edit only changes source evaluation order for two independent linear
projections whose results are multiplied after `jax.nn.silu(g)`. It preserves
Qwen3 math. Cluster loss and HLO/profile comparison gate validity.

## Results

- **Worker 0 summary**: 59,904 tok/s, 7,488 tok/s/chip, approx **43.0% MFU**.
- **Worker 1 summary**: 59,875 tok/s, 7,484 tok/s/chip, approx **42.9% MFU**.
- **Best post-profile step**: worker0 step 15 reached **59,991 tok/s**.
- **Loss**: normal fake-data trajectory, ending at **12.0458** on both workers.
- **Exit**: `EXIT_CODE=0`.

This is a clean same-band result against the v230/v236 frontier. It does not
materially raise the train-log MFU frontier, but it does improve the profiled
step-time signal versus the prior best profile band.

## Profile

- **XProf run**:
  `2026-06-05-qwen3-jax-v258-no-scan-sparsecore-rs-ar-rs2-ar2-mlp-up-proj-first/2026_06_05_13_25_35`
- **Average step time**: **4391.1 ms**
- **MXU utilization**: **67.6%**
- **Peak HBM**: **30.94 GiB** of 31.25 GiB
- **Idle**: 0.3% in op profile
- **Top device buckets**:
  - `convolution fusion`: 57,126.2 ms, 54.2%
  - `data formatting`: 2,958.4 ms, 2.8%

## HLO Dump

- **Local optimized train-step HLO**:
  `/tmp/qwen3-v258-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **SHA256**:
  `e413f9db56a30764ebd38a27dc3587ae3889261d16b56473026515e2c91f758a`
- **Size**: 19,544,295 bytes, 116,493 lines
- **Counts**:
  - `all-gather`: 6901
  - `all-reduce`: 165
  - `reduce-scatter`: 2774
  - `async-start`: 361
  - `async-done`: 361
  - `convolution`: 3087
  - `custom-call`: 3536
  - `fusion`: 30655
  - `copy`: 21320
  - `tokamax`: 2
  - `jvp_jit_silu__`: 576

The HLO is distinct from v256/v257 and contains the expected Tokamax markers,
so the source-order probe did change compilation and was launched with the
correct frontier environment.

## Verdict

**Supported same-band / carry as a candidate for confirmation only if the next
structural probe needs this ordering.**

The MLP `up_proj`-first ordering produced the best profiled step time seen in
this narrow frontier family (**4391.1 ms**) and held nominal **42.9-43.0% MFU**,
but the full train-log throughput is still same-band with v230/v236 rather than
a clear primary-metric breakthrough. Treat this as useful evidence that MLP
source ordering can move XLA scheduling, but do not stop the search here. The
next hypothesis should use the v258 HLO/profile diff to target a larger
structural gap rather than continue simple MLP expression rewrites.
