---
title: "Qwen3 JAX current v062 - final hidden FSDP constraint retest"
type: experiment
hypothesis: "Retesting a final hidden-state FSDP activation constraint on the current no-scan SparseCore stack may improve the lm-head/CE schedule without changing math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v062-final-hidden-fsdp-current
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, lm-head, ce, layout, sharding-constraint, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v062 - final hidden FSDP constraint retest

## Hypothesis under test

**Hypothesis**: v164 refuted adding `_sac(hidden)` before `lm_head` on the older
scan frontier, but the current no-scan SparseCore/MaxText-CE stack has a
different lm-head/CE and collective schedule. Reapplying the final hidden-state
FSDP layout constraint may reduce relayout or all-gather pressure before the
full-vocabulary projection without changing math.

**Mechanism**: Build a thin image from `qwen3-8b-jax:v258-mlp-up-first` and
change only `Qwen3ForCausalLM.__call__`:

```python
hidden = _sac(hidden)
```

immediately after `hidden = self.model(input_ids, position_ids)` and before the
`return_hidden` check / `lm_head` call. All weights, CE math, remat policy,
Splash attention, sharding plan, and backend flags remain fixed.

**Predicted signal**: Support requires clean finite loss and throughput above
the current-cluster v036/v037 frontier (**59,330 tok/s / 4433.0 ms**), or a
same-band result with lower CE/lm-head collective or layout cost and no HBM
regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op,
or clean completion below v036/v037 without useful profile evidence.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v062-finalhidden`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v062-final-hidden-fsdp-current`
- Image digest: `sha256:c562e776daf0b45f75e427608fab78163fba3ac44d05eab195f4b3e71e4da4e6`
- Base image: `qwen3-8b-jax:v258-mlp-up-first`
- Base digest: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v062-v037-final-hidden-fsdp`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.
- Runtime stack: v037 frontier flags, including no-scan/remat, activation
  sharding, Tokamax Splash max-logit 30, MaxText CE entrypoint, scoped VMEM
  100352, selective resources, rerun3, SparseCore RS/AR offload, RS3+AR2, and
  shared-memory limit 90.
- Local gate: `py_compile` passed; source diff is exactly one inserted
  `hidden = _sac(hidden)` line.

## Result

**Verdict: refuted.**

The workload completed cleanly with finite loss but regressed below the current
v036/v037 frontier:

- worker summary: **58,944 tok/s**, **7,368 tok/s/chip**, **42.3% old MFU**;
- XProf:
  `2026-06-12-qwen3-jax-v062-v037-final-hidden-fsdp/2026_06_12_13_29_25`;
- XProf step time: **4463.5 ms** average, **1.4 ms** stddev;
- MXU utilization: **66.7%**;
- peak HBM: **29.03 GiB / 31.25 GiB**, with **2.2202 GiB** free.

The optimized train-step HLO was copied to `/tmp/qwen3-v062-hlo`:

- SHA256: `5a51ec39665a854cad511367b69bd400f6956a034cd295edfa9029489632349a`;
- size: **19,915,837 bytes / 119,352 lines**;
- counts: `all-gather=6997`, `all-reduce=165`, `reduce-scatter=2774`,
  `async-start=394`, `async-done=394`, `convolution=3081`,
  `custom-call=3849`, `copy=22117`, `fusion=30896`, `dot_general=7302`,
  `splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

This is the same larger HLO family seen in v061 and is slower than the v036/v037
current-cluster frontier. The final-hidden FSDP constraint remains closed; do
not carry it.
