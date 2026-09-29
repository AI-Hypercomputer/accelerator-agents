---
title: "Qwen3-8B jax (Flax NNX) v6e-8 baseline"
type: experiment
hypothesis: baseline
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: minimal-jax-trainer-2026-06-02
verdict: baseline
tags: [qwen3-cc, jax, flax-nnx, baseline, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax (Flax NNX) v6e-8 baseline

First measured baseline for the **native-JAX (Flax NNX)** Qwen3-8B trainer on
**TPU v6e-8** (2 hosts × 4 chips, fsdp=8). The model is the from-scratch Flax port
in [`../model/modeling_qwen3.py`](../model/modeling_qwen3.py) (equivalence-verified
vs HF — see the lane README), random-init bf16, plain softmax CE, AdamW, XLA SDPA
attention. No splash / tokamax-CE / scan. Direct sibling to the
[torchax baseline](../../torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md)
at the identical shape, so the two lanes compare directly.

## Hypothesis under test

None — reference baseline for the `8B/v6e-8` variant of the jax lane.

## Setup

- **Hardware**: TPU v6e-8 — 2 hosts × 4 chips, fsdp=8, tp=1. Cluster
  `<your-cluster>` (`<your-project>`, zone `<your-zone>`), 1 slice.
- **Dispatch**: GKE/XPK via the gke-cluster-runner agent. Workload
  `alekseyv-qwen3-jax-v000-base`.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:latest`
  — built `FROM` the torchax lane image (which already carries flax/optax/
  transformers) with only the jax trainer code swapped in. **No torch/torchax at
  run time** (pure JAX).
- **Env**: `JAX_COMPILATION_CACHE_DIR=gs://…/qwen3_cc/jax_lane_cache`,
  `XLA_FLAGS=--xla_dump_to=…/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto`.
- **Data**: synthetic random tokens (`use_real_data=False`); random-init weights
  → loss not a convergence signal (perf baseline).
- **Command**:
  ```bash
  python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
    --batch_size=1 --seqlen=2048 --tp_parallelism=1 \
    --train_steps=20 --weights_dtype=bf16 \
    --profile_dir=gs://…/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline \
    --profile_start_step=12 --profile_steps=3
  ```
  global_batch = 1 × 8 = 8; tokens/step = 8 × 2048 = 16,384.

## Baseline comparison

This is the jax-lane baseline. Cross-lane comparison against the torchax baseline
(identical Qwen3-8B / v6e-8 / fsdp=8 / bs1 / seq2048 / synthetic):

| Lane | Step time | Throughput | tok/s/chip | MFU |
|------|-----------|------------|-----------|-----|
| torchax | 519 ms | 29,795 tok/s | 3,724 | 19.2% |
| **jax (Flax NNX)** | **512 ms** | **31,955 tok/s** | **3,994** | **20.5%** |
| **Δ (jax − torchax)** | −1.4% | — | **+7.3%** | **+1.3 pp** |

The native-JAX lane edges torchax at the baseline (**+7.3% tok/s/chip**),
consistent with the llama3 program's torchax→jax finding (less framework dispatch
+ collective overhead in the pure-JAX path). The op profile corroborates: jax
spends **21.7%** on the FSDP collective vs torchax's **31.3%**.

## Results

| Metric | This run (jax baseline) | Notes |
|--------|------------------------|-------|
| Steady step time | **512.6 ms** | step 0 cold (114.6 s) excluded |
| Throughput | **31,955 tok/s** (3,994 / chip) | 18 steps measured |
| MFU | **20.5%** (trainer) / MXU util 19.9% (xprof) | v6e bf16 peak 918 TFLOPS/chip |
| Cold compile | ~2 min (step 0 = 114.6 s) | cache now warm at `…/qwen3_cc/jax_lane_cache` |
| Exit code | 0 | clean 20-step run |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-02-qwen3-jax-v6e8-baseline`
  (run `2026-06-02-qwen3-jax-v6e8-baseline/2026_06_02_05_07_29`).
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline/plugins/profile/2026_06_02_05_07_29/`
  (2 hosts: `gke-tpu-32cb1c36-gs53.xplane.pb`, `…-jj8f.xplane.pb`).
- **Local pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline/GCS_LOCATION.txt).
- **Profiled steps**: 12–14.
- **Op profile** (xprof-mcp): `convolution fusion` (matmul) **37.6%**,
  `collective-permute-done` (FSDP) **21.7%**, `loop fusion` 15.7% (HBM-bound,
  RMSNorm/elementwise), `custom fusion` 8.7%. MXU util 19.9%.

### HLO Dump

- **GCS**: `gs://…/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline/hlo/` — **967
  module files** (text + proto, dumped direct to GCS during compile).

## Observations

- **Cross-lane: jax > torchax at the baseline** (+7.3% tok/s/chip). The gap
  tracks the lower collective share in the jax profile (21.7% vs 31.3%).
- The jax lane reaches a comparable matmul share (~37%) — the two lanes run the
  same dense GQA compute; the difference is overhead, not FLOPs.
- Both lanes are far below compute-bound (MXU ~20%): the headroom levers
  (batch scaling, splash, CE) apply to both.

## Verdict + reasoning

**baseline** — reference established. `EXIT_CODE=0`; profile (both hosts) + 967
HLO modules + compile cache persisted to GCS; MFU/throughput captured and
analyzed via xprof-mcp. Sets the `8B/v6e-8` jax matrix row at
**512 ms / 31,955 TPS / 20.5% MFU @ (seq 2048, global batch 8)**.

## Next hypotheses

- [Per-chip batch scaling (jax)](../../../../hypotheses/qwen3-jax-batch-scaling.md) — raise batch to lift MXU occupancy; cheapest first move, no code change.
- [Splash attention (jax)](../../../../hypotheses/qwen3-jax-splash-attention.md) — port the GQA-native splash kernel into the Flax attention; cuts the N² SDPA and unlocks seq 8192.
- [tokamax streamed cross-entropy (jax)](../../../../hypotheses/qwen3-jax-tokamax-ce.md) — drop the `[B,L,V]` logit materialization at the lm_head; unlocks seq 8192.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline/`
- Local pointer: `raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline/GCS_LOCATION.txt`
- Trainer: `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/` (train.py, model/, sharding.py).
- Sibling: [torchax baseline](../../torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md).
