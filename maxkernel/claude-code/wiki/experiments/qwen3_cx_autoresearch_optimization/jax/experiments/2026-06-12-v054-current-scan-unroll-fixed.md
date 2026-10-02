---
title: "Qwen3 JAX current v054 - fixed scan-unroll probe"
type: experiment
hypothesis: "Partial scan unrolling may reduce scan-loop overhead while keeping the lower-HBM scan/offload-attention stack viable."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v054-scan-unroll-fixed
status: invalid-refuted
tags: [qwen3-cc, jax, flax-nnx, scan, scan-unroll, remat, offload-attn, splash, tokamax-ce, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v054 - fixed scan-unroll probe

## Hypothesis under test

**Hypothesis**: The current no-scan frontier is faster but carries a very large
unrolled HLO and leaves little room for structural CE/logit changes. The
scan/offload-attention stack has a smaller compiled footprint but has been
slower. Setting `JAX_SCAN_UNROLL` to a small fixed value may reduce scan-loop
overhead without crossing the HBM limit.

**Mechanism**: Rerun the previously broken scan-unroll probe with a fresh image
tag and explicit runtime entrypoint hygiene. The image is
`qwen3-8b-jax:v054-scan-unroll-fixed`, digest
`sha256:bc8d7950aa89ae8bebb689649fe397bd8764608a41a066324fcf8c6de771e836`.
The launch sets `imagePullPolicy: Always`, `cd /app/trainer`, and
`PYTHONPATH=/app/trainer:${PYTHONPATH:-}` before invoking
`/opt/venv/bin/python -u train.py`.

Two arms run sequentially in the same workload:

- Phase A: `JAX_SCAN_UNROLL=2`, profiled, HLO dumped under `hlo_a`
- Phase B: `JAX_SCAN_UNROLL=4`, unprofiled, HLO dumped under `hlo_b`

Both arms use scan-over-layers, `remat_policy=offload_attn`, Splash attention,
Tokamax CE `chunked_xla`, batch size 4, sequence length 8192, TP1/FSDP8, scoped
VMEM 98304, scheduler shared-memory limit 100, and the earlier unroll-probe
collective/scheduler flags.

**Predicted signal**: Support requires clean finite loss and either arm
beating the current scan-family throughput while staying within HBM. It is not
expected to beat the no-scan v036/v037 frontier unless the scan-unroll schedule
is unexpectedly strong.

**Falsification criterion**: Import/package failure, compile/runtime failure,
invalid loss, HBM OOM, or clean completion materially below the scan-family
frontier.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v054-unrollfix`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v054-scan-unroll-fixed`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v054-scan-unroll-fixed`
- Mesh: `fsdp=8,tp=1`; global batch 32; phase A profile window steps 12-14.

## Results

The packaging fix worked: phase A passed model setup, optimizer setup,
Tokamax CE setup, fake dataloader creation, and reached the first call to the
jitted train step. It then failed during XLA post-optimization before any step
completed:

```text
jax.errors.JaxRuntimeError: INTERNAL: during context [post-optimization]:
The async-start expects the shape of operand 0 to match the async shape at
index {0} (bf16[18,2,4,8192,32,128]{5,3,4,2,1,0:T(8,128)(2,1)} vs
bf16[18,2,4,8192,32,128]{5,3,4,2,1,0:T(8,128)(2,1)S(5)}).
```

The JobSet failed with `EXIT_CODE=1`, so no profile was produced and phase B
did not run. HLO dumping stopped before an `after_optimizations` file was
emitted:

- HLO dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v054-scan-unroll-fixed/hlo_a/`
- Available train-step artifacts include only `before_optimizations.txt`,
  debug/options/config files, and the pre-crash dump subtree.

## Verdict

Invalid/refuted. The stale-image/import issue from the broken v034 unroll
attempt is fixed, but the actual scan-unroll configuration is not valid under
the current async-collective/offload-attention flag stack. Do not carry
`JAX_SCAN_UNROLL=2` or pursue phase-B `JAX_SCAN_UNROLL=4` on this exact stack
without first changing the async collective configuration.
