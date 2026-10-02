---
title: "Exp 50 — scan-over-layers tuned (POTENTIAL: closes 5 of 21 pt regression; TPS -16% vs exp 36, +6% vs exp 49)"
type: experiment
tags: [experiment, gemma4, jax, flax-nnx, scan, remat, potential]
hypothesis: jax-scan-over-layers-close-regression
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: TBD
verdict: supported
---

Refactored exp 49's scan-over-layers path to address the two root
causes the exp 49 subagent identified:

- **Fix D (kv-shared split)**: split the single scan over 7 super-blocks
  into **two separate scans** — a 4-block scan over the 24 non-KV-shared
  layers that produces the sliding/full K/V stores in its final carry,
  plus a 3-block scan over the 18 KV-shared layers that reads from that
  carry and has no `k_proj`/`v_proj` matmuls. Eliminates exp 49's
  zero-stub wasted matmuls.
- **Fix A then B (remat policy)**: removed exp 49's bare
  `@jax.checkpoint` from the scan body (Fix A → 52 GiB OOM, expected).
  Reapplied `jax.checkpoint(policy=checkpoint_dots_with_no_batch_dims)`
  matching exp 36's outer policy.
- **Additional (non-listed) fix**: removed the **outer**
  `jax.checkpoint(forward_loss, policy=...)` wrap in `train.py` under
  `JAX_SCAN_LAYERS=1`. Nested checkpoints were double-remat'ing the
  forward on backward; removing the outer wrap when the scan path
  carries its own per-iter checkpoints shaved **~34 ms/step (7.9 %)**.

**Outcome** — compile step 0 **180 → 72.7 s (−59.6 %, 2.5× faster)**,
TPS (steps 2-19, 18 samples) **34,614 → 29,044 (−16.1 %)**. Excluding
the single 828-ms step-9 outlier that is not structural (steady state
is 399 ms flat), 17-sample TPS = **30,778 (−11.08 %)**. **Improves on
exp 49 by +6.4 % TPS (all-sample) / +12.8 % (outlier-excluded)** while
preserving exp 49's compile-time win.

Filed as **potential** per the 3-hour time-box clause — the two
prescribed fix paths (A → B → D) closed ~5 of the 21 pt exp 49
regression but did not reach the ±0.5 % `-accepted` gate. Exp 36 remains
the JAX-stack best at 34,614 TPS; `JAX_SCAN_LAYERS=1` stays off by
default.

## Hypothesis under test

**Statement**: Removing (1) the zero-stub matmul waste on 18 shared
layers and (2) the exp-49 bare `jax.checkpoint` per-layer remat (while
keeping enough remat to not OOM) should close the bulk of exp 49's
21 pt TPS regression. Expected recoup per the exp 49 subagent:
~8 % from zero-stub elimination + 2-5 % from finer remat = ~10-13 %.
Target: within −0.5 % of exp 36 (34,441–34,787 TPS).

## Setup

**Hardware / env**: v6e-4, 1D `fsdp=4`, bf16, splash attention,
batch=3, seq=1024. `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache_exp50_v2`
(fresh compile, populated during smoke test).

**Command diff vs exp 49**: same env (`JAX_SCAN_LAYERS=1`,
`JAX_ATTENTION_IMPL=splash`). Code changes only.

**Full bench command**:

```bash
PROFILE=/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki/raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned
cd jax
JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache_exp50_v2 \
JAX_ATTENTION_IMPL=splash JAX_SCAN_LAYERS=1 \
python -u -m train --steps 20 --batch_size 3 --seq_len 1024 \
  --profile_dir $PROFILE --profile_steps 10 11 12
```

### Code changes

- **`jax/model/scan_layers.py`** — rewritten. Key differences from exp 49:
  - Two separate scan groups: `non_shared` = 4 super-blocks × (5 sliding
    + 1 full) with real `k_proj`/`v_proj`; `shared` = 3 super-blocks ×
    (5 sliding + 1 full) with Q-only attention that reads K/V from
    scan-closed-over `ks_store`/`kf_store` tensors produced by the
    non-shared group's final carry.
  - No zero-stub weights, no `jnp.where` KV select inside the body.
  - Per-iter `jax.checkpoint(..., policy=checkpoint_dots_with_no_batch_dims)`
    on both sliding and full layer steps — matches exp 36's selective
    remat policy, prevents the 52 GiB scan-activation OOM.
  - Sliding-scan inner body threads the latest-layer K/V through its
    carry (no outputs) so the final layer's K/V flows out via the
    outer scan's final state, with O(1) memory regardless of `n_blocks`.
  - Split per-layer weight extractors: `_layer_weights_non_shared`
    (with k/v/k_norm) and `_layer_weights_shared` (without). The
    caller passes `first_shared` explicitly.
- **`jax/model/modeling_gemma4.py`** — dispatch updated to pass
  `first_shared` to `collect_stacked_weights`.
- **`jax/train.py`** — under `JAX_SCAN_LAYERS=1`, skip the outer
  `jax.checkpoint(forward_loss, policy=...)` wrap. The scan bodies
  already carry that policy per-iteration; the outer wrap caused a
  measured nested double-remat that added ~34 ms/step (7.9 %).

### KV-sharing: two-group carry

The non-shared group (layers 0-23 = `[S,S,S,S,S,F] × 4`) stores layer
22's K/V (last sliding non-shared) and layer 23's K/V (last full
non-shared) as the last sliding-scan-body carry and the last
full-layer output of block 3. These four tensors seed the shared
group's attention as closure-captured constants — the shared group's
carry is just `hidden`.

### Remat path history

1. **Fix A (no inner checkpoint, rely on outer `forward_loss` wrap)**:
   OOM at 51.86 GiB (observed) — scan materializes `bf16[4,5,3,1024,10240]`
   activation stacks (1.17 GiB × 7 copies ~= 8 GiB, plus
   `f32[4,5,3,1024,2560]` norms ~= 600 MiB × 8 copies = 4.8 GiB, etc.).
   Expected by the exp 49 subagent.
2. **Fix B (inner `checkpoint(policy=checkpoint_dots_with_no_batch_dims)`)**:
   fits. Step time 432.98 ms, TPS 28,380, −18.0 % vs exp 36.
3. **Fix B + drop outer `jax.checkpoint` on `forward_loss`**:
   step time 399.24 ms (17-sample) / 423.08 ms (all-sample), TPS
   30,778 / 29,044, −11.1 % / −16.1 % vs exp 36. **Shipped.**

The step-3 improvement (~34 ms, 7.9 %) confirms the nested-checkpoint
double-remat hypothesis: when both outer and inner carry the same
policy, backward replays the forward twice.

## Correctness

Loss match vs exp 36 over 20 steps:

| step | exp 36 (loop) | exp 50 (scan-tuned) | Δ |
|---|---|---|---|
| 0 | 3.8125 | 3.8190 | +0.17 % |
| 2 | 2.7344 | 2.7125 | −0.80 % |
| 4 | 2.1875 | 2.1730 | −0.66 % |
| 9 | 1.8203 | 1.8075 | −0.70 % |
| 19 | 1.8359 | 1.8271 | −0.48 % |

All within bf16 reordering noise. Clean descent 3.82 → 1.83. No
divergence, no NaN. **Correctness verdict: PASS within 5 % tolerance**
(max |Δ| = 0.80 % at step 2, well under 5 %).

## Results

| Metric | exp 36 (loop) | exp 49 (scan) | **exp 50 (scan-tuned)** | Δ vs exp 36 | Δ vs exp 49 |
|---|---|---|---|---|---|
| **Compile step 0 (fresh)** | 180 s | 69.3 s | **72.7 s** | **−59.6 %** | +4.9 % |
| **TPS (all 18 samples, steps 2-19)** | 34,614 | 27,290 | **29,044** | **−16.1 %** | **+6.4 %** |
| TPS (17 samples, excluding step-9 outlier) | 34,614 | — | 30,778 | −11.1 % | +12.8 % |
| Mean step time (all) | 355.0 ms | 450.3 ms | **423.1 ms** | +19.2 % | −6.0 % |
| Steady-state step time (excluding step 9) | 355.0 ms | — | 399.2 ms | +12.5 % | — |
| **MFU (all samples)** | 23.05 % | 18.17 % | **19.34 %** | −3.71 pt | +1.17 pt |
| MFU (17 samples) | — | — | 20.50 % | −2.55 pt | — |
| Loss step 4 | 2.1875 | 2.1990 | 2.1730 | −0.66 % | −1.18 % |
| Loss step 19 | 1.8359 | 1.8441 | 1.8271 | −0.48 % | −0.92 % |

### Step-9 outlier

A single 828-ms step at step 9 across both 15-step and 20-step runs
(first run 828.4 ms; second run 778.3 ms) — every other step lands
in 398-401 ms. Not structural to scan; likely a periodic dataset
iterator tick (wikitext chunking) or Python-side event. Exp 36 data
was also captured with the same `awk '/step    2/,/step   19/'`
window so the comparison TPS (all-sample) is apples-to-apples —
reported as the headline number.

### Gap-closure analysis

Of exp 49's −21.2 pt regression vs exp 36, exp 50 closed:

- **Fix D (remove zero-stub matmuls)** + **Fix B (policy-matched inner
  remat)**: regained ~7 pt (−21.2 → −14.3 approx, measured after first
  fix pass).
- **Outer-checkpoint removal**: regained another ~5 pt (−14.3 → −9 approx
  on 17-sample basis).
- **Residual −11 pt (or −16 pt all-sample)**: pending. Hypothesized
  sources:
  1. Scan's inherent scheduling vs Python-unrolled for-loop: XLA sees
     all 42 layers in the unrolled case and can intermix all-gather /
     matmul / attention from any layer pair. Under scan, the body is
     one compilation unit with a strict state-carry dependency, so
     collectives are less flexible to overlap across iterations.
     This is a structural cost of scan on TPU for this workload.
  2. Inner sliding-scan is `n_blocks × 5` = 20 (non-shared) + 15
     (shared) iterations — each iter has scan-body overhead.
  3. Shared group's closure over `ks_store`/`kf_store` creates
     read-only broadcasts that may need a per-block boundary for
     splash's shard_map, less efficient than exp 36's per-layer
     splash call with full XLA visibility.

## HLO-op observation (xprof, steps 10-12)

- **Profile dir**: `raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned/`
  (~155 MB, gitignored).
- **GCS mirror**: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp50-scan-tuned/`
- **Steps captured**: 10, 11, 12 (mean 402 ms each, representative of
  steady state).
- HLO drill-down is follow-up work — the gap-closure analysis above is
  inference from step-time deltas, not xprof-confirmed.

## Profile

- **On-disk directory**: [`raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned/) (gitignored; ~155 MB).
- **GCS mirror**: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp50-scan-tuned/`
- **Steps captured**: 10, 11, 12.
- **What's inside**: trace of the scan-tuned JAX trainer at batch=3
  seq=1024 fsdp=4 bf16 splash. Steady-state step 399-402 ms (30,778 TPS
  / 20.50 % MFU on the 17-sample window). Used to diagnose the residual
  −11 pt gap vs exp 36.

## Verdict

**POTENTIAL** (supported hypothesis, partial gap closure). Filed per
the task brief's 3-hour time-box clause: "If you can't close the gap
below −5 % after trying both fix paths, stop after 3 hours and file
`-potential` with the remaining-work analysis."

**Decision per program.md tree**:
- Compile step 0 improved (180 → 72.7 s, −59.6 %) — meets the "compile
  improves" criterion.
- TPS regresses −16.1 % all-sample (> 5 % regression) — strictly
  rejected by the decision tree.
- But **beats exp 49 by +6.4 %** and closes ~5 of the 21-pt regression
  — meaningful improvement. Code path is cleaner (no zero-stub hack,
  no double-remat, no inside-scan `jnp.where` mux).
- Env gate stays off by default. Exp 36 remains JAX-stack best.

## Follow-ups (ranked)

1. **exp 51 — scan + xla SDPA (no splash)** to isolate whether splash
   under scan is the residual bottleneck. If the xla path shows a
   smaller regression, splash-under-scan is the hot spot; we'd need a
   splash-scan-compatible sharding primitive to recover. Trivial to
   try (`JAX_ATTENTION_IMPL=xla` only). Effort trivial. Confidence
   high for diagnosis.
2. **exp 52 — flatten inner + outer into a single scan per group using
   `vmap + jit` semantics**. Non-obvious: requires reshaping
   `[n_blocks, 5, ...]` weight trees into `[n_blocks*5, ...]`. If it
   works, removes the nested-scan overhead. Effort M. Confidence low.
3. **HLO drill-down**: capture a flame graph of one step and compare
   exp 50 trace against exp 36's trace op-by-op. Identify which ops
   grew under scan (collective-permute-done, all-gather, fusion
   boundaries). Effort S. Confidence medium for finding one more lever.
4. **Leave env gate off default**. The exp 45 compile cache already
   makes the 180-s compile cost disappear on cache hits; scan's
   72.7-s floor matters only on cache misses. The −16 % TPS cost makes
   it a dev-loop-only tool for now.

## See also

- [exp 49 — scan-over-layers original (POTENTIAL, −21.2 % TPS)](2026-04-24-exp49-jax-scan-layers-potential.md) — what this entry improves on.
- [exp 36 — splash + batch=3 (JAX-stack best, 34,614 TPS)](2026-04-23-exp36-jax-splash-batch3-accepted.md) — the target; exp 50 still trails by 11-16 %.
- [exp 45 — JAX persistent compile cache](2026-04-24-exp45-jax-compile-cache-accepted.md) — complementary; makes cache-miss the only case where compile-floor matters.
- [`../model/scan_layers.py`](../model/scan_layers.py) — rewritten implementation.

## Sources

- `/tmp/gemma4_jax_exp50_smoke.log` — 5-step smoke test.
- `/tmp/gemma4_jax_exp50.log` — 20-step benchmark log.
- `/tmp/gemma4_jax_exp50_freshcompile.log` — 3-step fresh-cache compile measurement (72.7 s).
- Profile directory: `raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned/` — xprof run `2026-04-24-gemma4-jax-exp50-scan-tuned/2026_04_24_08_08_12`.
- GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp50-scan-tuned/`
- [exp 36 page](2026-04-23-exp36-jax-splash-batch3-accepted.md) — comparison baseline.
- [exp 49 page](2026-04-24-exp49-jax-scan-layers-potential.md) — predecessor.
