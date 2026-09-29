---
title: "Qwen3 JAX v014 - splash TP2 tokamax remat bs8"
type: experiment
hypothesis: "Hypothesis not recovered - page filed by BACKFILL after the run completed without a stub."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v014-splash-tp2-tokamax-remat-bs8
verdict: inconclusive
status: inconclusive
backfilled: true
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, tp2, backfilled, inconclusive, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v014 - splash TP2 tokamax remat bs8

## Hypothesis under test

**Hypothesis not recovered** - page filed by BACKFILL after the run completed without a stub. The original dispatch context was not durable in the wiki; treat this experiment as observation-only.

## Setup

Backfilled from completed GKE workload `alekseyv-qwen3-v014`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v014-splash-tp2-tokamax-remat-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v014-splash-tp2-tokamax-remat-bs8`
- **XProf run**: `2026-06-03-qwen3-jax-v014-splash-tp2-tokamax-remat-bs8/2026_06_03_00_38_54`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=8 --seqlen=2048 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 32

## Results

Clean completion: `EXIT_CODE=0`.

| Metric | Observed |
|--------|----------|
| Steady step time | ~1,376 ms |
| Throughput | 47,558 tok/s |
| tok/s/chip | 5,945 |
| MFU | 30.6% |
| Profile window | steps 12-14 |

## Profile

Backfilled page. No profile-analyzer pass was dispatched before completion, so this page intentionally carries `backfilled: true` and is observation-only.

## HLO Dump

Backfilled page. HLO path from command:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v014-splash-tp2-tokamax-remat-bs8/hlo
```

## Verdict

`inconclusive`.

The run completed and the raw metrics exceed the v012 frontier, but the `/start-experiment` loop protocol does not assign supported/refuted verdicts to runs that completed without a durable pre-run stub and profile-analyzer audit. Use this as a strong observation when choosing the next properly stubbed iteration.
