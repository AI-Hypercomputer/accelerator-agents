---
title: "Qwen3 JAX v021 - splash TP2 tokamax remat fused s4k bs4"
type: experiment
hypothesis: "The v015/v018 TP2 stack can move to seq4096 while preserving an even per-chip batch, using batch 4 to avoid the odd-batch layout OOM seen in v020."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, long-seq, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v021 - splash TP2 tokamax remat fused s4k bs4

## Hypothesis under test

**Hypothesis**: The v015/v018 TP2 stack can move to seq4096 while preserving an
even per-chip batch, using batch 4 to avoid the odd-batch layout OOM seen in
v020.

**Mechanism**: Reuse the v015 image, successful XLA flag stack, TP2 sharding,
Splash attention, tokamax CE, remat, and fused RoPE. Increase `--seqlen` from
2048 to 4096 and reduce per-chip batch to 4, giving global batch 16.

**Predicted signal**: The run should at least compile; if it runs, it establishes
sequence 4096 as the next shape ladder point after v018's memory-relief result.

**Falsification criterion**: Compile/runtime OOM, or a clean run with such low
MFU that the longer-sequence path needs a different memory mechanism first.

## Setup

Planned GKE workload `alekseyv-qwen3-v021`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v021-splash-tp2-tokamax-remat-fused-s4k-bs4`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=4 --seqlen=4096 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 16
- **Profile window**: steps 12-14

## Results

Invalid: both workers failed during `jit_train_step` compile with HBM OOM.

| Metric | Value |
|--------|-------|
| Exit code | 1 |
| Failure point | `jit_train_step` compile |
| Per-chip batch | 4 |
| Global batch | 16 |
| Sequence length | 4096 |
| HBM used | 33.99 GiB |
| HBM capacity | 31.25 GiB |
| Over capacity | 2.74 GiB |
| Program HBM | 33.99 GiB |
| HLO temp | 33.97 GiB |

The largest allocation is again tokamax backward (`fusion.4291.remat5`, 2.32
GiB), followed by a 593.50 MiB all-gather and many 192.00 MiB hidden-state /
async collective temps. The tokamax CE token shape is `labels=i32[65536]`,
matching the total-token shape of v015 batch 8 at seq2048, but the seq4096 graph
does not fit the same way.

## Profile

No runtime profile was produced because compile failed before step 0.

## HLO Dump

The OOM dump is under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v021-splash-tp2-tokamax-remat-fused-s4k-bs4/hlo/`

It contains 41 objects totaling 186.27 MiB, including
`module_0092.jit_train_step.cl_854318611.oom_intermediate_module.txt` and
`module_0092.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`.

## Verdict

Invalid.

Seq4096 with per-chip batch 4 is still too large for the v015/v018 TP2 stack.
The next long-sequence probe should cut the even per-chip batch to 2.
