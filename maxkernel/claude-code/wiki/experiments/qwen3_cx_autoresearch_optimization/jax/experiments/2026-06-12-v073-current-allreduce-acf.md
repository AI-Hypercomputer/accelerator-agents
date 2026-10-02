---
title: "Qwen3 JAX current v073 - explicit-SiLU stack with all-reduce ACF enabled"
type: experiment
hypothesis: "Enabling all-reduce async collective fusion while keeping SparseCore all-reduce offload may reduce residual all-reduce scheduling tail on the current explicit-SiLU RS3+AR2 HLO."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, all-reduce, async-collective-fusion, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v073 - explicit-SiLU stack with all-reduce ACF enabled

## Hypothesis under test

**Hypothesis**: The carried current stack keeps all-gather on async collective
fusion but explicitly sets `--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false`
while using SparseCore all-reduce offload. Older all-reduce isolates coupled
this ACF flag with removing SparseCore all-reduce offload, so the single-flag
interaction on the current explicit-SiLU RS3+AR2 HLO remains open. Enabling
all-reduce ACF may be ignored, or it may alter residual scheduling enough to
reduce profile step time without changing model math.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and change only:

```text
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather on async collective fusion, SparseCore reduce-scatter/all-reduce
offload with aggregator, RS3+AR2, scoped VMEM 100352, shared-memory limit 90,
scheduler rerun 3, normal offload tracing, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the current durable **59.33k tok/s / 4433.8 ms** band or a lower XProf
step time/profile breakdown with unchanged HBM. Compile failure, HBM pressure,
or same/lower throughput and same/slower profile refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v073-aracf`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v073-v067-allreduce-acf`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v073_allreduce_acf`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v072 refuted scheduler resource sharing as an
identical-HLO profile regression.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-9rkl`: **59,322 tok/s**, **7,415 tok/s/chip**,
  **42.5%** old MFU.
- `gke-tpu-0d0013c1-l5jf`: **59,294 tok/s**, **7,412 tok/s/chip**,
  **42.5%** old MFU.

XProf
`2026-06-12-qwen3-jax-v073-v067-allreduce-acf/2026_06_12_15_56_32`
reported **4432.1 ms** average step time, **0.1 ms** stddev, **66.8% MXU**,
and **0.3%** idle. Peak HBM stayed unchanged at **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free.

Top XProf buckets:

- `convolution fusion`: **56,932.672 ms / 53.5%**
- `custom-call`: **29,433.682 ms / 27.6%**
- `loop fusion`: **10,722.739 ms / 10.1%**

Fresh train-step HLO was copied to `/tmp/qwen3-v073-hlo` and is byte-identical
to v068/v071/v072:

- SHA256:
  `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`
- Size: **19,895,815 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`, `exp.=180`.

## Verdict

Refuted/tie. Enabling all-reduce async collective fusion is accepted but inert
for the compiled train-step HLO under SparseCore all-reduce offload. The XProf
window is slightly lower than v068/v071, but host throughput is lower and the
profile shape is unchanged, so this is not enough to carry the flag.
