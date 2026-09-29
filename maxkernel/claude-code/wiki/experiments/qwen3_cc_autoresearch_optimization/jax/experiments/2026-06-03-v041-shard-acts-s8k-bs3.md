---
title: "Qwen3-8B jax v041 — activation sharding constraints @ seq8192 bs3"
type: experiment
hypothesis: qwen3-jax-activation-sharding-constraints
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-shard-acts-2026-06-03
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, sharding, activation-sharding, with-logical-constraint, maxtext, scan, splash, seq8192, bs3, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3-8B jax v041 — activation sharding constraints @ seq8192 bs3

## Hypothesis under test

**Hypothesis**: the obvious code gap vs MaxText — we apply **no** activation sharding constraints; MaxText pins
every layer-boundary activation. Adding `--shard_acts` (pin activations to `P("fsdp", ...)` at the layer input,
both post-norms, attention output, and MLP intermediate, mirroring MaxText's `with_logical_constraint`) cuts the
spurious layout conversions (our loop-fusion 17.4% vs MaxText 7.2%, data-formatting 3.8% vs 1.8%) and lifts the
frontier above v035 (6,030).

**Mechanism**: image `qwen3-8b-jax:v041-shard-acts` (FROM v033-maxtext-ce, swaps modeling_qwen3.py + __init__ +
train.py: flag-gated `_sac()` applying `jax.lax.with_sharding_constraint`). v035 config + `--shard_acts`:
`--use_remat --use_splash --use_scan --use_maxtext_ce --shard_acts --batch_size=3 --seqlen=8192` + base 12-flag
bundle. NO offload (isolate the constraint variable vs v035). equivalence PASS; shard_acts path validated on the
Auto-axis mesh.

**Predicted signal**: tok/s/chip > v035 6,030 (loop-fusion + data-formatting buckets drop toward MaxText); loss
parity (with_sharding_constraint = identity on values); exit 0.

**Falsification criterion**: ≤ 6,030 beyond noise — in pure FSDP XLA already infers the batch-sharded layout, so
the constraints are no-ops (then the inflated buckets are intrinsic recompute, not layout conversion).

## Setup

- Image `qwen3-8b-jax:v041-shard-acts`, v035 config + `--shard_acts=True` (no offload). `[shard] activation
  sharding constraints ON (P(fsdp, ...) at layer boundaries)` confirmed (14 constraints applied, 0 errors);
  `[scan]/[attn]/[remat:nothing_saveable]/[ce]` ON; `per_chip_batch=3 global_batch=24`.

## Results — PARITY (no-op in pure FSDP)

| run | activation constraints | tok/s/chip | MFU | steady step (ms) |
|-----|------------------------|-----------|-----|------------------|
| v035 | none | 6,030 | 34.6% | 4,075 |
| **v041** | **MaxText-style (P(fsdp,…))** | **6,039** | 34.6% | 4,070 |

Δ = **+0.15% (parity)** — within noise; step time identical (tightly converged ±4 ms). Loss 12.0994 →
12.0513 monotone, no NaN, exit 0. Compile 29.4 s (constraints didn't lengthen compile).

## Profile

- **Run name**: `2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3` · xprof [`http://localhost:8791/?run=2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3`](http://localhost:8791/?run=2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3) · GCS `.../plugins/profile/2026_06_03_00_28_22/`. On-disk: [`raw/profiles/2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3/`](../../../../../raw/profiles/2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3/). (No deep re-attribution — parity + identical step time imply the loop-fusion/data-formatting buckets are unchanged.)

## HLO Dump

- Not dumped (GCS path no-ops). Not material for a parity result.

## Verdict

**Refuted** (parity, not a win). The activation-sharding-constraint code gap vs MaxText is **real but
inconsequential for us**: the constraints fire cleanly but don't change throughput or step time. In **pure 1D
FSDP** (batch-sharded input + replicated weights, tp=1), XLA's SPMD partitioner already keeps activations
batch-sharded throughout — there is no layout ambiguity for explicit `with_sharding_constraint` to resolve, so
they are no-ops. Therefore our inflated **loop-fusion (17.4%)** and **data-formatting (3.8%)** buckets are
**intrinsic** — the RMSNorm recompute under `nothing_saveable` + the [QKV→splash transpose](../../../../observations/qwen3-jax-qkv-splash-transpose-intrinsic.md)
(already known intrinsic) — **not** layout-conversion overhead that constraints could remove. MaxText's
`with_logical_constraint` pays off under complex parallelism (tensor/sequence/expert); at our pure-FSDP
config it is redundant. **seq8192 frontier stays v035 (34.6% / 6,030).**

This closes the last identifiable **code**-level config difference vs MaxText. Combined with the config search
(v038–v040) and the sharding match, **the residual ~13% to MaxText is confirmed to be its host-offload
pipelining** (recompute-avoidance via overlapped host-DMA), which our stack does not reproduce (v039) — a
runtime/build capability, out of config/compile scope.

## Next hypotheses

- None on the config/compile axis — activation constraints (this), sharding, flags (v038), offload (v039),
  splash blocks (v040) all explored. The residual is MaxText's host-offload pipelining (runtime wall). Possible
  but lower-prior: save-on-device (not host) of the attention path at bs2 (avoids both recompute and host
  transfer) — but the dominant recompute is the **norms**, which are too large to save on-device (~10-14G).
  Catalogued. See [retrospective #5 outcome](../../../../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md).
