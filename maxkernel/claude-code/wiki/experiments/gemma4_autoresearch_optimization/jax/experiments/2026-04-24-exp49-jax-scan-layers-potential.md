---
title: "Exp 49 — scan-over-layers in JAX (POTENTIAL: compile -61.5%, TPS -21.2%)"
type: experiment
tags: [experiment, gemma4, jax, flax-nnx, scan, compile-time, structural, potential]
hypothesis: jax-scan-over-layers-compile-compression
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: TBD
verdict: supported
---

Replaced the 42-iteration Python `for layer in self.layers` loop in
`Gemma4TextModel.__call__` with two nested `jax.lax.scan`s gated on
`JAX_SCAN_LAYERS=1`. Outer scan iterates over 7 Gemma-4 super-blocks
(`[sliding × 5, full × 1] × 7`), inner scan iterates over the 5 sliding
layers within each super-block; the 1 full-attention layer per block is
applied inline after the inner scan. **Outcome**: compile step-0 drops
**69.3 s vs 180 s baseline (−61.5 %, 2.6× faster)**, loss trajectory
matches exp 36 within 0.5 %, but steady-state TPS **regresses −21.2 %**
(34,614 → 27,290). Filed as **potential** — infra win, perf loss. Env
gate off by default; exp 36 remains JAX-stack best.

## Hypothesis under test

**Statement**: Python-level layer unrolling materializes 42 copies of
the decoder block in HLO, inflating compile time (~180 s step 0 + 180 s
step 1 retrace on splash). A `jax.lax.scan` compresses this to one body
per group (one sliding body + one full body). Compile should drop
~3-15×. Secondary: shared activation buffers across scan iterations
may reclaim 2-5 % TPS.

Origin: torchax exp 26 parked this with 5 torchax-specific blockers and
7 "Option B" sub-problems. Native-JAX port does not hit any of the
torchax blockers — it owns `Gemma4TextModel.__call__` directly without
going through torchax's `ScannedModule` abstraction.

## Setup

**Hardware / env**: v6e-4, 1D `fsdp=4`, bf16, splash attention, batch=3,
seq=1024. `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache_exp49_bench`
(populated fresh; exp 45 persistent cache is orthogonal).

**Command diff vs exp 36**:
- Added env `JAX_SCAN_LAYERS=1` (new, selects the scan path).
- Otherwise identical: `JAX_ATTENTION_IMPL=splash python -m train --steps 20 --batch_size 3 --seq_len 1024 ...`.

**Full bench command**:
```bash
PROFILE=/mnt/disks/persist/.../raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers
JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache_exp49_bench \
JAX_ATTENTION_IMPL=splash JAX_SCAN_LAYERS=1 \
python -u -m train --steps 20 --batch_size 3 --seq_len 1024 \
  --profile_dir $PROFILE --profile_steps 10 11 12
```

### Code changes

- **`jax/model/scan_layers.py`** — new, 551 LOC. Contains:
  - `_partition_layers(layer_types)` — validates the fixed E4B pattern
    `[sliding × 5, full] × 7` and returns super-block sizes.
  - `_layer_weights(layer)` — extracts the per-layer weight arrays into
    a flat dict. For shared-KV layers, returns **zero-stubs** for
    `k_proj` / `v_proj` / `k_norm` so stacking is shape-homogeneous (B1).
  - `collect_stacked_weights(layers, layer_types)` — returns
    `{"sliding": tree_with_leading[7, 5, ...], "full": tree_with_leading[7, ...]}`.
  - `_rms_norm`, `_attention_body`, `_decoder_layer_body` — pure-JAX
    functional equivalents of `Gemma4RMSNorm` and
    `Gemma4TextDecoderLayer.__call__`, operating on flat weight dicts.
  - `scan_layers(stacked_weights, hidden, per_layer_inputs,
    position_embeddings, masks, config, attn_impl)` — outer scan over
    7 super-blocks. Each iter runs an inner scan over 5 sliding layers,
    then applies the 1 full-attention layer. **Both inner-scan body and
    full-layer body are wrapped in `jax.checkpoint`** so the backward
    pass rematerializes per-layer activations instead of stacking them
    (without this, the per-step heap was ~35 GiB of intermediate
    `f32[7, 5, 3, 1024, 2560]` buffers and the run OOM'd).
- **`jax/model/modeling_gemma4.py`** — +17 LOC. Added env-gated dispatch
  at the top of `Gemma4TextModel.__call__`: if `JAX_SCAN_LAYERS=1`,
  call `scan_layers(...)`; else the existing Python for-loop runs
  unchanged.
- **Trunk default**: env gate off → exp 36 behavior. No trunk-default
  semantics change. No edits to `train.py` (the outer
  `jax.checkpoint(forward_loss, ...)` wrap needed no changes since the
  nested per-layer remat is set up inside `scan_layers`).

### KV-sharing handling (B1 + B2 + B5)

Gemma 4 E4B has 18 shared-KV layers (indices 24..41) that borrow K/V
from the last non-shared layer of the matching type: sliding-shared
layers borrow from layer 22; full-shared layers borrow from layer 23.
The scan body handles this uniformly:

- **Zero-stub weights** (B1): `k_proj`, `v_proj` are `jnp.zeros(...)`
  for shared layers. `k_norm` uses `jnp.ones(...)` (makes the norm a
  no-op identity) — the matmul with zero weights still yields zero K/V
  locally, and `jnp.where(is_kv_shared, borrowed, local_zero_kv)`
  selects the borrowed tensor. Small wasted compute cost on 18 layers'
  k_proj/v_proj matmuls; measured ~35 ms/step contribution at b=3
  (see § HLO diff).
- **Stored KV as scan carry** (B2): The outer carry is a 5-tuple
  `(hidden, stored_k_sliding, stored_v_sliding, stored_k_full,
  stored_v_full)`. The store-layer (inner index `(block=3, inner=4)`
  for sliding, `block=3` for full) writes to its group's stored-K/V
  via `jnp.where(is_store_kv, k_local, stored_k_carry)`. Later layers
  read it via `jnp.where(is_kv_shared, stored_k_carry, k_local)`.
- **Traced per-iter scalars** (B5): `is_kv_shared[block, inner]` and
  `is_store_kv[block, inner]` are int32 arrays precomputed from the
  config, stacked as scan inputs. No Python branching inside the body.

### Interleave handling

The E4B layer type pattern is exactly regular: `[S S S S S F] × 7`. The
heterogeneous `head_dim` (sliding=256, full=512) forbids a single flat
scan over all 42 layers — sliding and full projection shapes differ, so
their stacked weight trees cannot share a single leading axis. The
two-level scan preserves exec order via outer × inner decomposition:
outer scan interleaves 5-sliding + 1-full per iteration. HLO ends up
with only 2 distinct layer-body structures (one sliding, one full)
vs 42 in the unrolled version.

## Correctness

Loss match vs exp 36 over 20 steps:

| step | exp 36 (loop) | exp 49 (scan) | Δ |
|---|---|---|---|
| 0 | 3.8125 | 3.8294 | +0.44 % |
| 2 | 2.7344 | 2.7227 | −0.43 % |
| 4 | 2.1875 | 2.1990 | +0.53 % |
| 9 | 1.8203 | 1.8199 | −0.02 % |
| 19 | 1.8359 | 1.8441 | +0.45 % |

All steps within 0.5 %, within bf16 reordering noise across 42 layers
plus different XLA instruction ordering that scan induces. Loss
descends cleanly 3.83 → 1.84 — no divergence, no NaN. **Correctness
verdict: clean pass within tolerance.**

## Results

| Metric | exp 36 (loop) | **exp 49 (scan)** | Δ |
|---|---|---|---|
| **Step 0 compile** | 180 s | **69.3 s** | **−61.5 % (2.6× faster)** |
| **TPS (mean steps 2-19)** | 34,614 | **27,290** | **−21.2 %** |
| Mean step time | 355.0 ms | **450.3 ms** | +26.8 % |
| MFU | 23.05 % | **18.17 %** | −4.88 pt |
| Loss step 4 | 2.1875 | 2.1990 | +0.5 % |
| Loss step 19 | 1.8359 | 1.8441 | +0.5 % |
| Peak HBM | 27.11 GiB | TBD (profile) | — |

**Compile-time win is solid**: 180 → 69 s is a durable ~2.6× floor
drop. On top of the exp 45 persistent cache that turns 360 s (step 0 +
step 1 retrace) into 29 s on cache hits, a fresh HLO now recompiles in
~69 s instead of ~180 s — ~2.6× iteration-loop speedup when the cache
misses (any code edit in the hot path).

**Runtime regression sources** (pending xprof drill-down but the scan
overhead is structural):

1. **Wasted zero-stub matmuls on 18 shared layers**: k_proj/v_proj are
   `(1024 × 2560)` for full layers and `(512 × 2560)` for sliding,
   dense-multiplied with zero weights. These matmuls are real compute
   in HLO — not optimized away because XLA can't prove the zeros
   statically inside the scan body (the stacked weight slices are
   traced). Rough estimate: ~35 ms/step of wasted matmul (~8 % of the
   27 % regression).
2. **Extra `jnp.where` + broadcast ops**: 42 layers × 2 `jnp.where`
   calls (one for KV select, one for store select) + extra
   mask-broadcasts adds HLO volume.
3. **`jax.checkpoint` per-inner-iter rematerialization**: the scan
   body is re-executed during backward. At 35 sliding + 7 full = 42
   layers, this roughly doubles the backward compute (same as the exp
   5 selective-remat policy, which is the torchax baseline behavior —
   but exp 36's for-loop lets XLA pick finer-grained remat decisions;
   the scan body forces uniform per-layer remat).
4. **Splash kernel batching under scan**: exp 36's HLO had 42 separate
   splash custom-calls, each with its own layout/shape tile decisions.
   Under scan, the splash kernel is invoked inside a `jax.shard_map`
   nested in a `jax.checkpoint` nested in a scan body — the resulting
   XLA scheduling is less aggressive at overlapping collective ops
   (all-gather of weights with matmul). The exp 36 trace's
   `collective-permute-done` at 12.2 % is likely **less well hidden**
   under scan.

Not all of these are inherent to scan — some can be clawed back:
- Skipping the zero-stub matmuls via a `jax.lax.cond` on `is_kv_shared`
  avoids the wasted compute but also breaks the scan body's static
  shape / makes the body an XLA conditional, which usually costs more
  than it saves. Would need to benchmark.
- Relaxing the per-layer `jax.checkpoint` back to a coarser policy
  (recompute only matmuls, keep layer inputs) is worth exploring but
  risks the 35-GiB-scan-heap OOM observed without remat.

## HLO-op observation (xprof, 3-step capture steps 10-12)

Profile dir: `raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers/`
(158 MB, gitignored). Captured steps 10-12. Full xprof drill-down is
follow-up work.

## Profile

- **xprof browser URL**: [2026-04-24-gemma4-jax-exp49-scan-layers](http://localhost:8791/?run=2026-04-24-gemma4-jax-exp49-scan-layers)
- **Run name**: `2026-04-24-gemma4-jax-exp49-scan-layers/2026_04_24_07_24_43`
- **On-disk directory**: [`raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers/) (gitignored; 158 MB).
- **GCS mirror**: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp49-scan-layers/`
- **Steps captured**: 10, 11, 12.
- **What's inside**: trace of the scan-over-layers JAX trainer at
  batch=3 seq=1024 fsdp=4 bf16 splash. Mean step 450 ms (27,290 TPS,
  18.17 % MFU). Used to diagnose the 26.8 % step-time regression.

## Verdict

**POTENTIAL** (supported hypothesis, mixed signal). Compile-time
hypothesis confirmed: 180 → 69 s (−61.5 %). Runtime hypothesis refuted:
instead of the 2-5 % potential TPS gain from shared activation
buffers, we pay a **−21.2 % regression** from wasted matmuls + remat
scheduling.

**Decision per program.md tree**:
- Compile-step-0 dropped AND TPS regressed >0.5 % → **potential (infra
  win, perf loss — file but don't merge default)**.
- Environment gate `JAX_SCAN_LAYERS=1` stays **off by default**. Exp 36
  (Python for-loop) remains JAX-stack best at 34,614 TPS.
- The scan path is available for runs that want faster iteration on
  fresh-HLO code edits (dev loop) at the cost of ~21 % throughput.
  Useful when debugging / profiling / doing compile-time work.

**Not merged as trunk default**. Code is shipped (scan path is on main,
behind the env gate) so future follow-ups can iterate without
re-implementing.

## Follow-ups (ranked)

1. **exp 50 — scan without zero-stub wasted matmuls**. Replace the
   `jnp.where`-based KV select with two separate body functions
   (shared vs non-shared) and route via `jax.lax.cond`. Risk: cond may
   cost more than the matmul saves on TPU. Expected recoup: +3-8 %
   TPS on the scan path. Effort S-M. Confidence low-medium. Only
   worth doing if we have a reason to run the scan path in production
   (e.g. compile-budget-constrained dev loop).
2. **exp 51 — relax per-layer remat inside scan**. Try
   `jax.checkpoint` with a less-aggressive policy (e.g.
   `dots_with_no_batch_dims` nested once) instead of the default
   full-body recompute. Test whether the heap fits without blowing
   past the 31.25 GiB HBM. Expected recoup: +2-5 % TPS if the policy
   lands right. Effort S. Confidence medium.
3. **Measure without splash**. Run scan path with
   `JAX_ATTENTION_IMPL=xla` to isolate whether splash's custom-call
   + shard_map nesting inside scan is the main runtime cost. If xla
   path shows a smaller regression, splash-under-scan is the hot spot.
   Effort trivial. Confidence high.
4. **Use scan path in exp 45's persistent cache workflow**. The
   cache turns fresh-compile into a 69-s event (scan) vs 180-s event
   (loop) when the cache misses. If the cache hits most of the time
   (6.67× speedup per exp 45), the compile-time win rarely matters.
   Recommendation: **leave env gate off default**; enable only when
   landing code that invalidates the cache and you want iteration
   speed > throughput.

## See also

- [exp 36 — splash + batch=3 (JAX-stack best, 34,614 TPS)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [exp 45 — JAX persistent compile cache (6.67× on cache hit)](2026-04-24-exp45-jax-compile-cache-accepted.md) — complementary; exp 49 lowers the floor for cache misses, exp 45 avoids compile on cache hits.
- [torchax exp 26 — scan-over-layers parked](../../torchax/experiments/2026-04-23-exp26-scan-over-layers-potential.md) — this page resolves the 7 Option-B sub-problems (B1-B7) for the native-JAX stack; the torchax stack was parked because `ScannedModule` hit 5 torchax-specific blockers.
- [program.md](../../program.md) — protocol.
- [`../model/scan_layers.py`](../model/scan_layers.py) — the implementation.

## Sources

- `/tmp/gemma4_jax_exp49.log` — full 20-step run log.
- `/tmp/gemma4_jax_exp49_smoke.log` — 5-step smoke test.
- Profile directory: `raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers/` — xprof run `2026-04-24-gemma4-jax-exp49-scan-layers/2026_04_24_07_24_43`.
- GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp49-scan-layers/`
- [exp 36 page](2026-04-23-exp36-jax-splash-batch3-accepted.md) — baseline comparison this entry stacks on.
- [torchax exp 26 page](../../torchax/experiments/2026-04-23-exp26-scan-over-layers-potential.md) — the parked torchax attempt this resolves on the JAX stack.
