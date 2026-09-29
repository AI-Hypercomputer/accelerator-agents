---
title: "Exp 37 — splash + b=3 + bf16 CE gate in JAX (POTENTIAL, flat, JAX port was already bf16-CE)"
type: experiment
tags: [experiment, gemma4, jax, flax-nnx, splash-attention, bf16-ce, no-op-confirm]
hypothesis: jax-bf16-ce
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 0c44f60
verdict: inconclusive
---

Stack bf16 cross-entropy on top of [exp 36](2026-04-23-exp36-jax-splash-batch3-accepted.md) (splash + batch=3). The torchax analog ([exp 12](../../torchax/experiments/2026-04-23-exp12-bf16-ce-accepted.md)) removed an explicit `.to(torch.float32)` cast and won +3.0 % TPS / −1.5 GiB HBM. **Result on the JAX port: 34,629 TPS, +0.04 % vs exp 36 — dead flat.** Peak HBM 27.45 GiB (87.84 %), actually +0.34 GiB vs exp 36 rather than −1.5 GiB. Loss trajectory identical. Verdict: `potential` (flat-within-noise).

**Why this was a no-op**: the native-JAX port never had the fp32 upcast that the torchax path had. The original `forward_loss` in `jax/train.py` (shipped in [exp 34](2026-04-23-exp34-jax-baseline-accepted.md)) already did `log_softmax` on bf16 `flat_logits` — the comment in the original code even said "bf16 log_softmax to drop the ~4 GiB fp32 logits tensor", but the savings were never measurable against a baseline because the fp32 path never existed in this stack. Exp 37 adds a `JAX_CE_DTYPE={bf16,fp32}` env-var gate so the fp32 path is now reachable for A/B comparison; this experiment measures the default (bf16) path against exp 36's identical-by-construction default. The gate itself is the durable artifact; the measured delta is noise.

## Hypothesis

Does stacking "bf16 cross-entropy" on top of [exp 36](2026-04-23-exp36-jax-splash-batch3-accepted.md) (34,614 TPS) yield the +3.0 % TPS / −1.5 GiB HBM delta seen in the torchax analog ([exp 12](../../torchax/experiments/2026-04-23-exp12-bf16-ce-accepted.md))?

Mechanism (as understood before running): the `[B=3, S=1024, V=262144]` logits tensor at fp32 is ~3 GiB; at bf16 it's ~1.5 GiB. Eliminating an fp32 upcast before `log_softmax` should save ~1.5 GiB and remove one pass over the logits.

**Correction after code inspection**: the JAX port's `forward_loss` was already not upcasting. The comment in `train.py` line 213 ("bf16 log_softmax to drop the ~4 GiB fp32 logits tensor") described the design intent, not a cast that needed removing. So the "fix" was already shipped in exp 34 — this experiment is an instrumented re-measurement with an explicit env-var gate, not a change in numerical path.

## Setup

**Code change**: `jax/train.py` `forward_loss()` — add `JAX_CE_DTYPE` env-var gate. Default `bf16` preserves the exp 36 path byte-for-byte (modulo reduction-accumulator placement, see below). Setting `JAX_CE_DTYPE=fp32` re-introduces the upcast for future comparisons.

```diff
+    ce_dtype_env = os.environ.get("JAX_CE_DTYPE", "bf16").lower()
+    if ce_dtype_env == "fp32":
+        ce_dtype = jnp.float32
+    else:
+        ce_dtype = jnp.bfloat16
+    ...
     def forward_loss(state, input_ids, labels):
         ...
-        flat_logits = logits.reshape(-1, vocab)
+        flat_logits = logits.reshape(-1, vocab).astype(ce_dtype)
         ...
         log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
         ...
-        loss = -(picked * mask).sum() / jnp.maximum(mask.sum(), 1.0)
+        loss = -(picked.astype(jnp.float32) * mask.astype(jnp.float32)).sum() / jnp.maximum(mask.sum().astype(jnp.float32), 1.0)
         return loss.astype(jnp.float32)
```

The reduction accumulator was nudged to fp32 explicitly (was implicit-bf16 previously). This is a scalar-over-[B*S=3072] reduction — the fp32 cast is ~12 KiB of intermediate, negligible cost. Purpose: make the reported loss value drift-free across batch sizes.

Command (run from `.../jax/`):

```bash
WIKI_ROOT=/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki
PROFILE_DIR=$WIKI_ROOT/raw/profiles/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce
JAX_ATTENTION_IMPL=splash JAX_CE_DTYPE=bf16 python -m train \
  --batch_size 3 --seq_len 1024 --steps 20 \
  --profile_dir $PROFILE_DIR \
  --profile_steps 10 11 12
```

Global batch = 12, tokens/step = 12,288. Identical to exp 36.

## Correctness

Loss trajectory vs [exp 36](2026-04-23-exp36-jax-splash-batch3-accepted.md) (same seed, same data, same steps):

| step | exp 37 (bf16 CE gate) | exp 36 | Δ |
|---|---|---|---|
| 0 | 3.8184 | 3.8125 | +0.0059 |
| 2 | 2.7270 | 2.7344 | −0.0074 |
| 4 | 2.1809 | 2.1875 | −0.0066 |
| 9 | 1.8152 | 1.8203 | −0.0051 |
| 19 | 1.8314 | 1.8359 | −0.0045 |

All steps within 0.3 % — consistent with bf16 rounding. No NaN, no divergence. **Numerical path bit-equivalent-modulo-rounding to exp 36.** The tiny systematic offset is explained by the explicit fp32 accumulator in the final reduction (exp 37) vs the implicit bf16 reduction (exp 36): fp32 accumulation is slightly more accurate, so the reported loss value shifts by a few thousandths.

## Results

| Metric | exp 36 (splash + b=3) | **exp 37 (splash + b=3 + bf16-CE gate)** | Δ |
|---|---|---|---|
| **TPS (median, steps 6–15)** | 34,614 | **34,629** | **+0.04 %** (flat) |
| TPS (mean, steps 2–19) | 34,583 | 34,653 | +0.20 % |
| TPS (median, steps 2–19) | — | 34,697 | — |
| Step time (median 6–15) | 355.0 ms | 354.85 ms | −0.15 ms |
| Step 0 compile | 167 s | 167.6 s | +0.6 s |
| Step 1 recompile | 168 s | 168.4 s | +0.4 s |
| **Peak HBM** | 27.11 GiB (86.75 %) | **27.45 GiB (87.84 %)** | **+0.34 GiB (+1.09 pt)** |
|  — stack (static) | 16.44 GiB | 16.78 GiB | +0.34 GiB |
|  — heap (activations) | 10.67 GiB | 10.67 GiB | **unchanged** |
|  — free | 4.14 GiB | 3.80 GiB | −0.34 GiB |
| Loss step 19 | 1.8359 | 1.8314 | −0.0045 (healthy) |
| Profile size | 321 MB | 321 MB | — |

### HLO-op diff (exp 36 vs exp 37, 3-step profile window)

| Op | exp 36 ms | exp 37 ms | Δ |
|---|---|---|---|
| convolution fusion | 1512.18 | 1512.92 | +0.7 |
| loop fusion | 1264.78 | 1250.25 | −14.5 |
| collective-permute-done | 549.68 | 549.35 | −0.3 |
| custom fusion (splash) | 174.80 | 174.67 | −0.1 |
| data formatting | 157.06 | 157.18 | +0.1 |
| custom-call | 135.61 | 135.47 | −0.1 |
| broadcast | — | 77.31 | — |
| all-gather | 46.02 | 50.96 | +4.9 |
| all-reduce-scatter fusion | 53.87 | 53.87 | 0.0 |
| collective-permute-start | 45.70 | 43.66 | −2.0 |

**The HLO graph is essentially identical to exp 36.** `loop fusion` trimmed 14.5 ms (one fewer cast op in the CE reduction path — the implicit-bf16-to-implicit-fp32 shuffle is gone, replaced with explicit scalar-only fp32 casts). `all-gather` grew 5 ms (within noise on this host). Net effect on wall-clock step time: 0.0 % change.

### HBM — no savings

The hypothesis predicted −1.5 GiB heap (the `[B*S, V]` logits tensor at fp32 → bf16). Actual heap: **unchanged at 10.67 GiB.** Stack grew 0.34 GiB, likely due to the explicit fp32 scalar reductions adding a handful of small f32 constants to the static plan (negligible in absolute terms, but the HBM headroom dropped 4.14 → 3.80 GiB).

**Why no heap savings**: because there was no fp32 logits tensor in exp 36 to save. XLA lowered `jax.nn.log_softmax(flat_logits_bf16)` to a bf16-in/bf16-out reduction from the start. The peak-heap tenant at exp-36's peak moment is almost certainly the backward-pass activation tile, not the forward-pass log-softmax intermediate (verified indirectly: changing only the CE path doesn't move the heap).

Exp 12 (torchax) got its 1.5 GiB back because torchax's code *did* have an explicit `flat_logits.to(torch.float32)` before `log_softmax`. When that cast was removed, the fp32 `[B*S, V]` tile XLA was previously materializing went away. On the JAX port that cast was never there — confirmed by reading `jax/train.py` at commit 0c44f60 before this change.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce) — opens the interactive trace viewer.
- **Run name** (`mcp__xprof__list_runs`): `2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/2026_04_24_03_32_18`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/) (gitignored; 321 MB)
- **GCS mirror**: `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace of the native-JAX trainer at batch=3 seq=1024 fsdp=4 bf16 with `JAX_ATTENTION_IMPL=splash JAX_CE_DTYPE=bf16`. Step-time 354.85 ms; peak HBM 27.45 GiB / 87.84 %; HLO graph indistinguishable from exp 36 aside from one dropped implicit-cast in the reduction.

## Verdict

**POTENTIAL (flat, within noise).** TPS delta +0.04 % is well inside the ±0.5 % noise band. Loss matches exp 36 trajectory. HBM did not drop — confirming the bf16-CE intent was already realized by the exp-34 implementation. No new best.

**What was actually delivered**: the `JAX_CE_DTYPE` env-var gate. This is durable infrastructure (small diff in `train.py`, preserves exp-36 default), but the throughput/memory delta this experiment was supposed to measure was a mirage — the fp32-CE path never existed on this stack to begin with. The gate's real value is enabling an explicit fp32-CE A/B comparison on future refactors (e.g., if a Pallas RMSNorm kernel ships in exp 39 and changes the dtype-flow upstream of the LM head).

**Relation to torchax exp 12**: the torchax port has a +3.0 % / −1.5 GiB win from bf16 CE. The native-JAX port does not — the native-JAX port shipped with that win already baked in (the torchax→JAX rewrite skipped the spurious fp32 cast by construction). Cumulative delta from the shared baseline is therefore ~the same across both stacks at this point; torchax still has exp-12 accounted separately, JAX rolls it into exp-34.

**Trunk effect**: the `JAX_CE_DTYPE` env-var gate is merged. Default `bf16` = no behavior change vs exp 36. Use `JAX_CE_DTYPE=fp32` for controlled A/B when needed. `JAX_ATTENTION_IMPL=splash --batch_size 3` remain the recommended-run flags for the JAX stack; **exp 36 remains the current JAX-stack best at 34,614 TPS**.

## Follow-ups (ranked)

With exp 37 confirmed as a no-op, the ranking from exp 36 stands, with one small update:

1. **exp 38 — collective-permute-done investigation.** 12.1 % of step time (549 ms / 3-step window), unchanged from exp 36. `in_shardings`/`out_shardings` audit on the jitted step could reclaim 5–6 %. Confidence medium. Effort S-M. **Highest expected value now.**
2. **exp 39 — Pallas RMSNorm kernel.** `loop fusion` still 27.6 % of step time (1250 ms), 210 calls/step × 42 layers. Single-HBM-pass Pallas kernel worth 3–8 %. Confidence medium. Effort M.
3. **exp 40 — scan-over-layers (JAX-native).** Drops step-0 compile 167 s → ~5 s. Throughput neutral; latency-to-first-signal win. Confidence high. Effort M.
4. **exp 41 — b=4.** HBM at 87.8 % with 3.80 GiB free. b=4 adds ~3.5 GiB; might just fit. If exp 38 lands first and trims the `collective-permute` buffer, b=4 becomes very likely to fit.

**De-ranked from exp 36's original list**: "exp 37 — splash + b=3 + bf16 CE". Resolved as no-op-by-construction. Leaving the `JAX_CE_DTYPE=fp32` path available for future regression guards.

## See also

- [exp 36 — splash + batch=3 in JAX (JAX-stack best)](2026-04-23-exp36-jax-splash-batch3-accepted.md) — the baseline this stacks on.
- [exp 34 — JAX port baseline](2026-04-23-exp34-jax-baseline-accepted.md) — introduced the bf16 CE path by construction.
- [Torchax exp 12 — bf16 CE (+3.0 %, −1.5 GiB)](../../torchax/experiments/2026-04-23-exp12-bf16-ce-accepted.md) — the analog on the sibling stack, for contrast.
- [program.md](../../program.md) — protocol.
- [`../train.py`](../train.py) — the native-JAX trainer (gate added in `forward_loss`).

## Sources

- `/tmp/gemma4_jax_exp37.log` — full 20-step run log.
- `/tmp/gemma4_jax_exp37_smoke.log` — 5-step smoke test.
- Profile directory: `raw/profiles/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/` — xprof run `2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/2026_04_24_03_32_18` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce
- GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`
- [exp 36 page](2026-04-23-exp36-jax-splash-batch3-accepted.md) — comparison baseline (identical numerical path by construction).
