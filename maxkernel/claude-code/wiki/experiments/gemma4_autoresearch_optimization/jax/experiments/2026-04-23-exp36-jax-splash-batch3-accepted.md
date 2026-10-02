---
title: "Exp 36 — splash + batch=3 in JAX (ACCEPTED, +13.9 % TPS, new JAX-stack best, beats torchax session-best by +3.7 %)"
type: experiment
tags: [experiment, gemma4, jax, flax-nnx, splash-attention, batch-growth, tps-win, session-best]
hypothesis: jax-splash-batch3
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: e6fb4c6
verdict: supported
---

Stack [exp 35](2026-04-23-exp35-jax-splash-potential.md) (splash attention wired into the native-JAX port) with **batch=3**. The hypothesis was: splash's per-token-fixed launch overhead is mostly constant at batch=1 seq=1024, so the matmul savings it created only pay rent when activation work grows. Raise batch to 3 and the ratio flips. Result: **34,614 TPS at 355.0 ms/step** — **+13.9 % over exp 35's 30,386 TPS**, new JAX-stack best. The JAX stack now **beats the torchax session-best** ([exp 25: 33,372 TPS](../../torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md)) by **+3.7 %**, without bf16 cross-entropy yet (exp 37).

Peak HBM: **27.11 GiB / 31.25 GiB = 86.75 %** (stack/static 16.44 GiB + heap/activations 10.67 GiB + free 4.14 GiB). Fits comfortably — the JAX stack is leaner than torchax at the same config (torchax exp 15 at batch=3 without bf16 CE hit 98.78 % HBM / 30.87 GiB; the native-JAX port frees ~3.8 GiB of dispatch / intermediate state). Loss trajectory descends cleanly (3.81 → 1.84 over 20 steps) — consistent with b=1 per-token rate; no divergence.

## Hypothesis

Does increasing `--batch_size` from 1 to 3, with `JAX_ATTENTION_IMPL=splash` already on, produce the analogous **+8.0 % TPS** gain seen in torchax when going from b=2 → b=3 ([exp 18](../../torchax/experiments/2026-04-23-exp18-fused-bwd-batch3-accepted.md)) — or a larger gain because we start from b=1 and splash's per-call overhead is a larger fraction of step time at b=1?

Mechanism: splash's Pallas custom-call launch overhead (~0.4 ms / call × 42 layers × ~3 calls/step ≈ ~50 ms / step at b=1 per exp 35's analysis) is **batch-independent**. Matmul + attention FLOPs scale with batch. So at b=3 the overhead is amortized across 3× the tokens — step time grows sub-linearly in batch, TPS grows.

Prediction: **+5–10 % TPS**, peak HBM ~55–60 % (stack constant at 16.4 GiB + ~3 × per-batch activation delta of ~1.5 GiB).

## Setup

**No code changes.** Only flag/env delta vs [exp 35](2026-04-23-exp35-jax-splash-potential.md):

- `--batch_size 1` → `--batch_size 3`.
- `JAX_ATTENTION_IMPL=splash` (unchanged).

Command (run from `.../jax/`):

```bash
WIKI_ROOT=/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki
PROFILE_DIR=$WIKI_ROOT/raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3
JAX_ATTENTION_IMPL=splash python -m train \
  --batch_size 3 --seq_len 1024 --steps 20 \
  --profile_dir $PROFILE_DIR \
  --profile_steps 10 11 12
```

Global batch = per-chip 3 × fsdp=4 = **12**. Tokens / step = 12 × 1024 = 12,288.

## Correctness

**Loss trajectory (b=3)** vs exp 35 (b=1):

| step | exp 36 b=3 | exp 35 b=1 |
|---|---|---|
| 0 | 3.8125 | 3.9062 |
| 2 | 2.7344 | 2.9688 |
| 4 | 2.1875 | 2.0156 |
| 9 | 1.8203 | — |
| 19 | 1.8359 | 2.2969 |

Not a bit-match — and shouldn't be, because the data loader packs 3 sequences per micro-batch step for b=3 vs 1 for b=1, so the per-step gradient signal differs. But **loss descends cleanly**, stays in the same 1.8–2.7 regime, and the smoke-5 trajectory (3.81 → 2.19) is identical between the 5-step smoke run and the 20-step benchmark run (seeded data). No NaN, no divergence. Compared with the torchax-b=3 baseline (exp 15 terminal loss ~1.9–2.2), we're within the expected region.

## Results

| Metric | exp 35 (b=1, splash) | **exp 36 (b=3, splash)** | Δ |
|---|---|---|---|
| **TPS (median, steps 6–15)** | 30,386 | **34,614** | **+13.9 %** |
| TPS (mean, steps 2–19) | 30,408 | 34,583 | +13.7 % |
| Step time (median 6–15) | 134.8 ms | 355.0 ms | +163 % (2.63× for 3× tokens) |
| Step time (xprof profile avg) | 142.2 ms | **375.5 ms** | (captures steps 10–12; includes profile overhead) |
| Step 0 compile | 132 s | 167 s | +35 s |
| Step 1 recompile | 133 s | 168 s | +35 s |
| **Peak HBM** | 16.43 GiB (52.6 %) | **27.11 GiB (86.75 %)** | +10.68 GiB (+34.2 pt) |
|  — stack (static) | ~16.44 GiB | 16.44 GiB | unchanged (weights + opt state) |
|  — heap (activations) | small | 10.67 GiB | +10.67 GiB (scales with batch) |
| Loss step 0 | 3.9062 | 3.8125 | different-batch, healthy |
| Loss step 19 | 2.2969 | 1.8359 | different-batch, healthy |
| Tokens per step | 4,096 | 12,288 | 3× |

**Throughput math sanity check**: step-time grew 2.63× for 3× tokens → per-token cost dropped by **3 / 2.63 = 1.14×**, which matches the measured +13.9 % TPS improvement. Clean amortization.

**Against torchax session-best** (exp 25: splash + batch=3 + bf16 CE + fused_bwd + SEQ_MINOR + block=1024 + selective_remat, 33,372 TPS):

- **exp 36 = 34,614 TPS → +3.7 % over torchax-best**, and this is the JAX stack *without* bf16 CE yet. The native-JAX port's lower dispatch overhead is the source of the remaining gap — the Flax NNX layer wiring skips torchax's aten-to-JAX function-level dispatch on every call.
- Headroom for exp 37 (bf16 CE) on top of this: the ~1.5 GiB fp32 logits materialization is still present in the native-JAX port's `forward_loss` — swapping to tokamax or hand-rolled bf16 CE should free 1.5 GiB (→ peak ~82 %) and remove one extra pass over `[B, S, V]`.

### HLO-op diff (top ops, exp 35 vs exp 36, 3-step profile window)

| Op | exp 35 time_ms | exp 36 time_ms | Δ | Notes |
|---|---|---|---|---|
| convolution fusion (matmuls) | 548.97 | 1512.18 | **+963.2 ms (2.75×)** | scales with batch × tokens |
| loop fusion (elementwise + norms) | 332.22 | 1264.78 | **+932.6 ms (3.81×)** | RMSNorm / residual-add — scales ≥ batch |
| custom fusion (splash kernel) | 169.11 | 174.80 | **+5.7 ms (1.03×)** | **near-constant** — per-call launch dominates |
| collective-permute-done | ~11 (from exp 35's full list) | **549.68 (12.2 %)** | **+540 ms** | **new pattern at b=3** — SPMD data reshuffles |
| collective-permute-start | ~14 | 45.70 | +32 ms | paired with -done |
| all-gather | 138.13 | 46.02 | −92.1 ms | weight gathers — fewer, because ? (same shard plan; check) |
| all-reduce-scatter fusion | 83.41 | 53.87 | −29.5 ms | smaller relative footprint |
| data formatting | 24.10 | 157.06 | +133 ms | layout conversions scale |
| custom-call | 47.08 | 135.61 | +88.5 ms | per-call infra — scales with ops |

**What the diff says**:

1. **`convolution fusion` at 2.75× is sub-linear in batch (3×)** — MXU is being utilized better at larger shapes. This is the asymptotic win that flat exp 35 pointed at.
2. **`custom fusion` (splash kernel) at 1.03× is near-constant.** The splash kernel launch overhead is the same 42 × 3-step = 126 calls at both batch sizes; only the per-call compute inside grows slightly. This is exactly the mechanism hypothesized — the constant-per-call overhead amortized over 3× tokens drives most of the TPS win.
3. **`loop fusion` (RMSNorm + residual-add) at 3.81× grows super-linearly.** This is the next bottleneck surface: per-token memory traffic is slightly worse at larger batches (likely due to sharded-tensor copy patterns). Candidate for exp 40+ (Mosaic RMSNorm kernel, already catalogued in `program.md`'s Pallas build-targets).
4. **`collective-permute-done` 12.2 % is new.** At b=1 this op was ~0.6 % of step time; at b=3 it's the #3 hotspot. Root cause: SPMD re-sharding on the batch dimension during the jitted step body. Worth a look — might be reducible via `in_shardings` / `out_shardings` tightening. Follow-up idea, not in this experiment's scope.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp36-splash-batch3](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp36-splash-batch3) — opens the interactive trace viewer.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-jax-exp36-splash-batch3/2026_04_24_03_01_54`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/) (gitignored; 321 MB).
- **GCS mirror** (for xprof_mcp / xprof browser server): `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp36-splash-batch3/`
- **Steps captured**: 10, 11, 12 (xprof trace + xplane.pb).
- **What's inside**: trace of the native-JAX trainer at batch=3 seq=1024 fsdp=4 bf16 with `JAX_ATTENTION_IMPL=splash`. Step-time 375.5 ms (xprof avg over 3 steady-state steps, profile-capture inflates slightly vs the 355 ms median); peak HBM 27.11 GiB / 86.75 %; splash custom fusion at 3.9 % of step time.

## Verdict

**SUPPORTED / accepted.** Beats the `+0.5 %` accepted bar by a wide margin (+13.9 %). New JAX-stack best at 34,614 TPS, 355.0 ms/step, 86.75 % HBM. **Exceeds torchax session-best** (33,372 TPS, exp 25) by +3.7 %, without bf16 CE or fused_bwd-specific tuning.

**Why the prediction landed on the upper end of the expected range** (+13.9 % vs predicted +5–10 %): two compounding effects.

1. **Per-call overhead amortization** (the predicted mechanism): splash's custom-fusion time grew 1.03× while everything else grew 2.7–3.8×. At b=1 it was 9.9 % of step time; at b=3 it's 3.9 %. This alone gives ~6 % back.
2. **XLA MXU utilization improvement** (bonus): `convolution fusion` time grew 2.75× for 3× tokens — matmul throughput at the larger `[B=3, ...]` shapes is closer to peak than at `[B=1, ...]`. At b=1 the matmul tiles had sub-peak occupancy (bound by the small batch contribution to GEMM inner-dims), b=3 fills the MXU rows better. Gives another ~7 % back.

These two amortize in parallel: neither is possible without splash landing in exp 35 first.

**Trunk effect**: no code to merge. The `JAX_ATTENTION_IMPL=splash` env-var and `--batch_size 3` flag are now the recommended-run config for the JAX stack. The trunk README/best-state is updated. All subsequent JAX-stack experiments should base-line against this.

## Follow-ups (ranked)

1. **exp 37 — splash + b=3 + bf16 cross-entropy.** Direct analog of torchax exp 12 → exp 15 stacking. Removes the ~1.5 GiB fp32 logits materialization (frees peak HBM from 86.8 % → ~82 %) and removes one pass over the `[B=3, S=1024, V=262144]` logits tensor. Expected: +1–3 % TPS + ~1.5 GiB freed. Confidence medium. Effort S (hand-roll a bf16 CE first — simplest — tokamax version as a second variant). This is the next ratchet to pass the torchax best by a comfortable margin.
2. **exp 38 — splash + b=3 + investigate collective-permute-done.** At 12.2 % of step time this is the single largest remaining non-compute cost. If `in_shardings` / `out_shardings` on the jitted step can be tightened so the batch dim doesn't get re-sharded mid-step, 5–10 % reclaim is possible. Confidence medium. Effort S-M (pytree sharding audit in train.py).
3. **exp 39 — splash + b=3 + RMSNorm Pallas kernel.** `loop fusion` is now 28.1 % of step time at b=3 (was 19.5 % at b=1). The RMSNorm pattern is 210 calls/step × 42 layers × 5 norms. A Mosaic RMSNorm kernel (one HBM pass instead of 3) could reclaim 3–8 %. Confidence medium. Effort M.
4. **exp 40 — scan-over-layers (JAX-native).** Drops step-0 compile from ~167 s → ~5 s. Throughput neutral, latency-to-first-signal win. Confidence high. Effort M.
5. **exp 41 — try b=4 / b=5.** HBM at 86.8 % has room for one more batch raise (+~3.5 GiB per batch). If b=4 fits, estimate +3–5 % TPS; if b=5 (after bf16 CE frees 1.5 GiB), maybe +5–7 % on top. Gated on exp 37 landing.

## See also

- [exp 35 — splash in JAX at b=1 (potential, flat)](2026-04-23-exp35-jax-splash-potential.md) — the prerequisite.
- [exp 34 — JAX port baseline](2026-04-23-exp34-jax-baseline-accepted.md) — anchor.
- [Torchax exp 15 — splash + bf16 CE + b=3](../../torchax/experiments/2026-04-23-exp15-splash-bf16ce-batch3-accepted.md) — the analog on the sibling stack.
- [Torchax exp 18 — fused_bwd + b=3 (torchax session-best progenitor)](../../torchax/experiments/2026-04-23-exp18-fused-bwd-batch3-accepted.md).
- [Torchax exp 25 — splash-block1024 (torchax session-best, 33,372 TPS)](../../torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md) — now surpassed.
- [program.md](../../program.md) — protocol.
- [`../model/pallas_attention.py`](../model/pallas_attention.py) — the splash wrapper.
- [`../train.py`](../train.py) — the native-JAX trainer (unchanged for this experiment).

## Sources

- `/tmp/gemma4_jax_exp36.log` — full 20-step run log.
- `/tmp/gemma4_jax_exp36_smoke.log` — 5-step smoke test.
- Profile directory: `raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/` — xprof run `2026-04-23-gemma4-jax-exp36-splash-batch3/2026_04_24_03_01_54` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp36-splash-batch3
- GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp36-splash-batch3/`
- [exp 35 page](2026-04-23-exp35-jax-splash-potential.md) — baseline comparison this entry stacks on.
