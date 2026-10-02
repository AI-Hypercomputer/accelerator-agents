---
title: "Qwen3 JAX v170 - no MLP intermediate sharding constraint rerun"
type: experiment
hypothesis: "If v169's no-MLP-intermediate-sharding-constraint gain is real, an exact rerun should preserve the same HLO family and stay above the confirmed v153/v154 frontier band."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac
status: supported
tags: [qwen3-cc, jax, flax-nnx, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v170 - no MLP intermediate sharding constraint rerun

## Hypothesis under test

**Hypothesis**: v169 improved the best-worker average from the confirmed
57,754-57,757 tok/s frontier band to 57,947 tok/s by removing only the `_sac`
constraint around the MLP gate/up product. If this is a real layout win rather
than run noise, an exact rerun should preserve the same train-step HLO family
and remain above the confirmed frontier.

**Mechanism**: Exact rerun of v169:

```python
return self.down_proj(jax.nn.silu(g) * u)
```

All image, code, runtime flags, batch/sequence shape, profile window, CE,
attention, remat, and scan settings are held fixed.

**Predicted signal**: Support requires clean loss, same HLO family or persistent
cache replay of v169, and average throughput above 57,754-57,757 tok/s.

**Falsification criterion**: Compile/runtime failure, invalid loss, fallback to
the v153 HLO family, or clean completion within/below the old frontier band
without a profile win.

## Setup

GKE workload `alekseyv-qwen3-v170-mlpsacr`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image digest**: `sha256:21a390217fc38d05a8638d2ae681f00ece3d255d57ed88a0651e4054304f6854`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v170-no-mlp-intermediate-sac-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 57,913 tok/s average, 7,239 tok/s/chip, 41.5% MFU.
- Worker 1: 57,935 tok/s average, 7,242 tok/s/chip, 41.5% MFU.
- Best observed post-warmup step: 58,050 tok/s.
- Final loss: 12.0461 on both workers.

This confirms v169's gain over the prior v153/v154 frontier band
(57,754-57,757 tok/s average).

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v170-no-mlp-intermediate-sac-rerun/plugins/profile/2026_06_04_19_39_56/`

Artifacts:

- `gke-tpu-46dd3e54-g30s.xplane.pb`
- `gke-tpu-46dd3e54-rkr2.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

No fresh HLO dump appeared under the v170 prefix, consistent with persistent
cache replay of v169's train-step program:

- v169 train-step HLO: `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,769 bytes
- SHA256: `443b5f34d11a802519fca1f96187c541566bdb9d532c86177249d46b11e3d644`
- Compiled memory: 26.86 GiB

## Verdict

Supported. v170 confirms the no-MLP-intermediate-`_sac` layout change as the
new frontier:

- v169: 57,947 tok/s best-worker average, 41.5% MFU.
- v170: 57,935 tok/s best-worker average, 41.5% MFU.
- Prior confirmed frontier v153/v154: 57,754-57,757 tok/s, 41.4% MFU.

Carry the removal of the MLP intermediate sharding constraint for subsequent
experiments.
