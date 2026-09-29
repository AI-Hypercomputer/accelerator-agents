---
title: "Qwen3 JAX current v069 - async all-gather cap 1 on lean shmem90 HLO"
type: experiment
hypothesis: "Limiting concurrent async all-gathers to one may reduce residual all-gather scheduling pressure on the current shmem90 lean HLO family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, all-gather, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v069 - async all-gather cap 1 on lean shmem90 HLO

## Hypothesis under test

**Hypothesis**: The v064-v068 current stack keeps all-gather on async collective
fusion while offloading reduce-scatter/all-reduce to SparseCore. Historical
v232 showed `--xla_max_concurrent_async_all_gathers=1` was a tie on the older
RS2/AR2 frontier, but this cap has not been tested on the current shmem90 lean
HLO family with **2.22 GiB** free HBM and the explicit-SiLU HLO cleanup.
Restricting concurrent async all-gathers to one may reduce residual
async-done/all-gather pressure without reintroducing the large direct
all-gather bucket seen with cap 2.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and change only the XLA runtime flags by adding:

```text
--xla_max_concurrent_async_all_gathers=1
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
SparseCore RS/AR offload with aggregator, scoped VMEM 100352, shared-memory
limit 90, scheduler rerun 3, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v064/v068 **59.3k tok/s / 4433.8 ms** band or a same-band result with
lower XProf all-gather/custom-fusion time and unchanged HBM. A changed HLO that
exposes large direct all-gather time or any clean slowdown refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v069-ag1`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v069-v067-async-allgather-cap1`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v069_ag1`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v068 confirmed that explicit SiLU algebra is an HLO
cleanup but not a durable throughput gain by itself.

## Result

Workload `alekseyv-qwen3-cc5-jax-v069-ag1` completed on np0 with finite loss.
Worker summaries were in the same current-frontier band:

- `gke-tpu-0d0013c1-9rkl`: **59,332 tok/s**, **7,416 tok/s/chip**,
  **42.5% old MFU**.
- `gke-tpu-0d0013c1-l5jf`: **59,301 tok/s**, **7,413 tok/s/chip**,
  **42.5% old MFU**.

XProf
`2026-06-12-qwen3-jax-v069-v067-async-allgather-cap1/2026_06_12_14_59_35`
reported **4433.9 ms** average step time, **1.3 ms** stddev, **66.8% MXU**,
and **0.3%** idle. Peak HBM stayed unchanged at **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. The top op-profile buckets remained the same shape:
convolution fusion **56,939.055 ms / 53.5%**, Splash/custom-call
**29,434.380 ms / 27.6%**, and loop fusion **10,725.883 ms / 10.1%**.

The fresh train-step HLO was copied to `/tmp/qwen3-v069-hlo` and is
byte-identical to v067/v068: SHA256
`84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**. Counts stayed unchanged:
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
`logistic=0`, `exponential=183`, `exp.=180`.

## Verdict

**Refuted / tie.** `--xla_max_concurrent_async_all_gathers=1` did not change
HLO, memory, or the profile shape, and the measured XProf step time
**4433.9 ms** is effectively identical to v064/v068's **4433.8 ms** band and
slower than v067's high-side **4432.5 ms** sample. Do not carry the all-gather
concurrency cap on the current lean shmem90 stack.
