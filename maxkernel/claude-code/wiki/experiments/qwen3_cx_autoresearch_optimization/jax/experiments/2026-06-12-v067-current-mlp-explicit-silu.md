---
title: "Qwen3 JAX current v067 - explicit SwiGLU SiLU algebra"
type: experiment
hypothesis: "Replacing jax.nn.silu(gate) with gate * sigmoid(gate) inside the MLP SwiGLU may alter the compiler's fused MLP schedule without changing model math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, silu, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v067 - explicit SwiGLU SiLU algebra

## Hypothesis under test

**Hypothesis**: The current HLO still contains `jvp_jit_silu__=576` and is
dominated by MLP matmul/convolution-fusion work. Rewriting the MLP gate
activation from `jax.nn.silu(g)` to the algebraically equivalent
`g * jax.nn.sigmoid(g)` may give XLA a different fusion boundary for the
SwiGLU forward/backward path without changing the mathematical function.

**Mechanism**: Build a thin image from
`qwen3-8b-jax:v258-mlp-up-first` and change only `Qwen3MLP.__call__`:

```python
return self.down_proj(jax.nn.silu(g) * u)
```

to:

```python
return self.down_proj((g * jax.nn.sigmoid(g)) * u)
```

Keep the v064 runtime stack and np0 placement fixed with a unique compile
cache. This is a source-level MLP fusion probe; no sharding, remat, CE,
attention, data, optimizer, or runtime scheduler flags change.

**Predicted signal**: Support requires clean finite loss and either throughput
above v064's **59,326 tok/s / 4433.8 ms** or a same-band result with a smaller
or faster MLP/convolution-fusion bucket that motivates carrying the rewrite.
Failure to compile, significant numerical mismatch, larger HLO family, or
clean completion below v064 refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v067-silu`
- Base image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Base digest: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- Experiment image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Experiment digest: `sha256:d752d2d9720ddc54602c2476689a40c20d322386c47339f047dbf7efa60ede21`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v067-v037-explicit-silu`
- Placement: nodepool `<your-cluster>-np-0`, the current best
  v064/v037 placement.
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Equivalence

Local CPU MLP equivalence against the base image passed on a small config after
copying identical weights:

- `max_abs_diff=0.0`
- `allclose_1e-6=True`

## Result

The workload completed cleanly with finite loss on np0. Worker0 summary:

- Throughput: **59,347 tok/s**
- Per-chip throughput: **7,418 tok/s/chip**
- Old lane MFU: **42.6%**

XProf:

- Run: `2026-06-12-qwen3-jax-v067-v037-explicit-silu/2026_06_12_14_29_48`
- Average step time: **4432.5 ms**
- Step-time stddev: **1.1 ms**
- MXU utilization: **66.8%**
- Peak HBM: **29.03 GiB / 31.25 GiB**, **2.2202 GiB** free

The second collected profile directory,
`2026-06-12-qwen3-jax-v067-v037-explicit-silu/2026_06_12_14_29_49`, matched
the same **4432.5 ms** step-time band with **67.0%** MXU.

Op profile remains dominated by the same buckets:

- Convolution fusion: **28,399.104 ms / 53.3%**
- Splash custom-calls: **14,713.642 ms / 27.6%**
- Loop fusion: **5,354.498 ms / 10.1%**
- Idle: **0.3%**

Fresh train-step HLO was copied to
`/tmp/qwen3-v067-hlo/train_step.after_optimizations.txt`:

- SHA256:
  `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`
- Size: **19,895,815 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`, `exp=692`.

Compared with v064, the HLO keeps the same high-level op counts for
collectives, fusions, convolutions, Splash calls, and copies, but the explicit
algebra removes the `jvp_jit_silu__=576` name family and slightly reduces HLO
text size. Runtime improves from v064's **59,326 tok/s / 4433.8 ms** to
**59,347 tok/s / 4432.5 ms**, a small **+21 tok/s** and **-1.3 ms** movement.

## Verdict

**Provisional support.** The source rewrite is numerically equivalent in the
local MLP check, compiles cleanly, removes the explicit `jvp_jit_silu__` HLO
family, and produces the current best live result. The gain is small enough
that it must be confirmed by an exact fresh-cache rerun before treating v067 as
a durable frontier.
