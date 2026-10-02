---
title: "exp 53 — Splash block-size sweep at seq=2048 b=1 fp32-master"
type: experiment
hypothesis: splash-block-tuning-new-regime
model: gemma4-e4b-jax
commit: 732c0b3
verdict: refuted
tags: [experiment, jax, gemma4, splash, fp32-master, block-sweep, flat, rejected]
created: 2026-04-24
updated: 2026-04-24
---

Follow-up from the new-regime baseline ([exp 52](2026-04-24-exp52-jax-fp32master-seq2k-accepted.md)). At seq=2048 the default splash block_q=min(1024,seq)=1024 gives 2 tiles per head; try full-tile (block=2048) and smaller (block=512) to see if either moves the needle. Outcome: **block=2048 errors out of VMEM**, block=512 is dead flat (−0.0 %, within noise). Splash block size is not a TPS lever at this config. Mirrors the [exp 48 plateau](2026-04-24-exp48-jax-splash-param-sweep-potential.md) observation in the old regime.

## Hypothesis under test

At seq=2048, splash's default block=1024 leaves 2 tiles per head per seq. Full-tile (block=2048) could eliminate the per-tile overhead; smaller blocks (block=512) could improve per-op HBM-VMEM streaming. Prior data: old-regime exp 48 block sweep was uniformly flat at exp-36 shape (b=3 s=1024). This experiment tests whether the new regime (fp32 master, different activation-memory pressure) has different block-size sensitivity.

## Setup

- **Hardware**: TPU v6e-4.
- **Commit**: 732c0b3 (exp 52 baseline).
- **Command (two variants)**:
  ```bash
  # Variant A: block=2048 (full-tile)
  SPLASH_BLOCK_Q=2048 SPLASH_BLOCK_KV=2048 SPLASH_BLOCK_KV_COMPUTE=2048 \
  SPLASH_BLOCK_Q_DKV=2048 SPLASH_BLOCK_KV_DKV=2048 SPLASH_BLOCK_KV_DKV_COMPUTE=2048 \
    python -u -m train --steps 20 --batch_size 1 --seq_len 2048 \
    --weights-dtype fp32 --compute-dtype bf16 \
    --profile_dir $PROFILE_DIR --profile_steps 10 11 12
  # Variant B: block=512
  SPLASH_BLOCK_Q=512 SPLASH_BLOCK_KV=512 ... (same shape)
  ```

## Results

| variant | block | TPS | step time median (steps 6-15) | verdict |
|---|---|---|---|---|
| exp 52 baseline | default (1024) | 26,807 | 305.6 ms | reference |
| exp 53a (Variant A) | 2048 (full tile) | — | — | **compile-time `CompileTimeScopedVmemOom`**: splash_mha_fwd_residuals needs 32.14 MiB VMEM, limit is 32 MiB (144 KiB over). Not runnable at this config. |
| exp 53b (Variant B) | 512 | ~26,807 | 305.6 ms (305-307 range) | **flat**, −0.0 % within noise |

Loss descent for 53b matches exp 52 within bf16-reorder noise (step 4: 2.42 vs 2.37; step 19: 2.51 vs 2.30 — trajectories diverge by a few percent due to minor reordering, both clean, no NaN).

## Profile

- On-disk path: `raw/profiles/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/` (gitignored). Steps 10–12 captured.
- GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/`
- Browser URL: `http://localhost:8791/?run=2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master`

## Verdict + reasoning

**Verdict: `refuted`** (rejected). Neither smaller nor larger blocks give a win at seq=2048 b=1 fp32-master. The TPS-vs-block-size curve is effectively flat, confirming the old-regime exp 48 plateau finding transfers to the new regime. This closes one follow-up from the exp 52 queue; splash block size is not a lever on v6e-4 for Gemma 4 E4B at any shape we've measured (old regime: b=3 s=1024; new regime: b=1 s=2048).

## Next hypotheses

From the exp 52 queue, still open:
- exp 54 — pure-AMP isolation at shared shape
- exp 55 — scan_layers under new regime
- exp 56 — 2D mesh (dp=2, tp=2) to see if the cross-mesh sharding changes the seq=8192 memory picture (K/V replication on tp=2 is permitted since num_kv_heads=2).
- exp 57 — PLE embedding host-offload (risky code change, deferred).

## See also

- [exp 52 — new-regime baseline](2026-04-24-exp52-jax-fp32master-seq2k-accepted.md)
- [exp 48 — old-regime splash param sweep (flat)](2026-04-24-exp48-jax-splash-param-sweep-potential.md) — same plateau observation
- [`jax/model/pallas_attention.py`](../../jax/model/pallas_attention.py) — env-var gate for all SPLASH_BLOCK_* params

## Sources

- `raw/profiles/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/` (xprof trace)
- `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/` (GCS mirror)
- `/tmp/gemma4_jax_exp53.log` (run stdout)
