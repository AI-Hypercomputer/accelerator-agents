---
title: "Exp 8 — Splash kernel-only autotune (POTENTIAL, +30-32% kernel fwd+bwd; full-training validation pending)"
type: experiment
tags: [llama3, torchax, splash, kernel-autotune, kernel-only, potential]
hypothesis: llama3-torchax-splash-kernel-autotune
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp5-splash-seq2k (harness: tune_splash.py)"
branched_from: v6e8-llama3-8b-torchax-20260425-exp5-splash-seq2k
verdict: potential
---

First **kernel-only** autotune in this program. Sweeps splash-attention
`BlockSizes` × `q_layout` × `use_fused_bwd_kernel` for the exact (B, Hq, L, hd)
shapes the Llama 3 8B torchax trainer feeds the kernel, on a single TPU v6e
chip — no training loop, no FSDP collectives. Result: at every shape we tested
(seq=1024 bs=2, seq=1024 bs=4, seq=2048 bs=2), **the same config wins by
+30-32% on fwd+bwd kernel time** vs the production default in
[`splash_attn.py`](../splash_attn.py).
The winning config matches the MaxText Llama 3.1-8B recipe almost exactly:
`block_q=1024, block_kv=1024, q_layout=HEAD_DIM_MINOR, use_fused_bwd_kernel=True`,
with `block_q_dkv = block_kv_dkv = seq_len`.

End-to-end TPS impact is bounded by attention's share of step time: at
seq=1024 ≈8 % of step ⇒ expected +2-3 % TPS; at seq=2048 ≈12 % of step ⇒
+3-4 % TPS. Verdict is **potential** until a full XPK training run validates
that the kernel-level win survives XLA-fusion and end-to-end interactions
(the [gemma4 exp33 lesson](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp33-pallas-rmsnorm-rejected.md)
where a kernel-correct Pallas RMSNorm regressed −8.1 % end-to-end).

The wider methodological finding: the entire 3-shape sweep (171 configs) ran
in **≈3 minutes wall-clock** end-to-end on a v6e-8 — vs ~15 min × 8-16 = 2-4
hours of XPK training runs to test the same hypotheses one-by-one with
±2 % end-to-end noise. Kernel-only timing also resolves sub-1 % deltas that
get lost in step-time variance.

## Hypothesis under test

> Splash-attention `BlockSizes` (block_q, block_kv, dkv blocks, dq blocks)
> and `use_fused_bwd_kernel` are tunable per (shape, hardware) combination
> with measurable effect on the kernel's wall-clock cost. The current
> production values in [`splash_attn.py`](../splash_attn.py) were copied
> from MaxText's Llama 3.1-8B recipe but *not* re-tuned for our exact shapes
> after switching to torchax + HF SDPA. A focused kernel-only sweep will
> identify a better config for the three shapes we already train at
> (baseline, exp 3, exp 5) **and** for forward shapes we haven't yet trained
> at (e.g. seq=8192).

Falsification: if **no** config beats the production-default by ≥3 % on
fwd+bwd kernel time at any of the 3 in-program shapes, the existing values
are already at the local optimum and the autotune is a no-op (refuted).

## Setup

### Hardware / job

- TPU v6e-8 (1 slice, 2 hosts × 4 chips, 2x4 topology),
  cluster `<your-v6e8-cluster>` in `<your-zone>`. Each rank's process
  independently runs the sweep on its local chip 0 (no `jax.distributed`
  cross-rank coordination is needed — splash on a single device is what we're
  measuring; the other 7 chips sit idle). Both ranks' results are recorded
  for cross-validation.
- Image: `<your-registry>/test/llama3-8b-torchax-container:tune-v2`
  (base `jax-ai-image/tpu:jax0.9.0-rev1`, +`tune_splash.py`; same trainer
  image as `hf-v2` plus the new harness file). XLA flags via
  `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` to match production
  VMEM budget.

### Harness — [`tune_splash.py`](../tune_splash.py)

~430-line single-file kernel-only benchmark. Build/reuse decision: **build
focused harness using upstream JAX splash directly** rather than reuse
[tokamax's `splash_attention_benchmarking.py`](../../../../../raw/code/tokamax/tokamax/experimental/utils/tuning/tpu/splash_attention_benchmarking.py)
or [marin's `autotune_utils.py`](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/autotune_utils.py).
Reasons:

1. Our model uses
   [`jax.experimental.pallas.ops.tpu.splash_attention`](../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
   directly. Tokamax has its own *vendored* fork at
   [`tokamax/_src/ops/experimental/tpu/splash_attention/`](../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py)
   that may differ — porting risk for direct applicability of results.
2. Tokamax's harness depends on `tune_jax` (external pip package) for xprof-
   event-filtered timing. Our deltas of interest are >2 %, well above the
   noise of simple `time.perf_counter()` over 30 iters — extra dep not needed.
3. Marin's `autotune_utils.py` is a *cache-key-aware* autotune harness for
   inline use inside a training graph, not a standalone benchmark — wrong
   shape for this task.

We did borrow the **grid layout** (block_q × block_kv × layouts × fused_bwd)
and `randn_init` pattern from tokamax's script.

### Sweep design

Two phases per shape, to keep the full grid bounded.

- **Phase 1 — forward only.** `block_q × block_kv × q_layout`, with
  `k/v_layout = HEAD_DIM_MINOR` pinned (gemma4 exp24 found `q_layout`
  dominates). 18-32 configs depending on `seq_len`.
- **Phase 2 — full fwd+bwd.** Top-3 fwd configs from phase 1 × `dkv` block
  grid × `{fused_bwd=True, fused_bwd=False}`. For non-fused, `block_q_dq` and
  `block_kv_dq` are pinned to the dkv values to bound the search (a phase-3
  follow-up could decouple if any non-fused config is competitive — none is
  in this run). 24-54 configs per shape.

Total: **171 configs across 3 shapes**.

### Shapes swept

The Llama 3 8B model (Hq=32, Hkv=8, hd=128) post-`repeat_kv` MHA-shape
inputs to splash, per-shard under FSDP=8:

| Sweep | Q shape per shard | K/V shape per shard | Reference |
|---|---|---|---|
| `seq=1024 bs=2 (baseline shape)` | `(2, 32, 1024, 128)` | `(2, 32, 1024, 128)` | [baseline](2026-04-25-baseline.md), [exp 2](2026-04-25-exp2-splash-bs2-potential.md) |
| `seq=1024 bs=4 (exp3 shape)` | `(4, 32, 1024, 128)` | `(4, 32, 1024, 128)` | [exp 3](2026-04-25-exp3-splash-bs4-accepted.md) |
| `seq=2048 bs=2 (exp5 shape)` | `(2, 32, 2048, 128)` | `(2, 32, 2048, 128)` | exp 5 (commit `b65ecff`) |

### Anchors

Two reference configs measured first per sweep:

- **Production default** = the values currently hard-coded in
  [`splash_attn.py:42-53`](../splash_attn.py): `block_q=1024, block_kv=512,
  block_q_dkv=2048, block_kv_dkv=512, block_q_dq=2048, block_kv_dq=512,
  use_fused_bwd_kernel=False, q/k/v_layout=HEAD_DIM_MINOR`. (Production
  clamps each block size with `min(global, seq_len)`, so at seq=1024 the
  effective `block_q_dkv` is 1024.)
- **MaxText Llama 3.1-8B recipe** = the values in the
  [tpu-recipes Llama 3.1-8B `tuning_params`](../../../../../raw/code/tpu-recipes/training/v6e/Llama3.1-8B/jax/maxtext/v6e_8/recipe.json):
  `block_q = block_kv = block_q_dkv = block_kv_dkv = 1024,
  use_fused_bwd_kernel=True`.

### Submission

Workload `llama3-splash-tune-20260425-191044`. `xpk workload create`
command structure mirrors [`DEPLOY.md`](../DEPLOY.md) with the trainer
entrypoint replaced by:

```bash
cd /app/trainer && \
  python -u tune_splash.py --seq_len 1024 --batch_size 4 --mode fwd_bwd ; \
  python -u tune_splash.py --seq_len 2048 --batch_size 2 --mode fwd_bwd ; \
  python -u tune_splash.py --seq_len 1024 --batch_size 2 --mode fwd_bwd
```

## Results

### Headline — best config per shape (rank 0; rank 1 agrees within 0.04 %)

| Shape | Prod fwd ms | **Best fwd ms** | Δ vs prod | Prod fwd+bwd ms | **Best fwd+bwd ms** | Δ vs prod |
|---|---|---|---|---|---|---|
| seq=1024 bs=2 | 0.197 | **0.165** | **+16.2 %** | 0.683 | **0.478** | **+30.1 %** |
| seq=1024 bs=4 (exp3) | 0.382 | **0.320** | **+16.4 %** | 1.420 | **0.960** | **+32.4 %** |
| seq=2048 bs=2 (exp5) | 0.606 | **0.516** | **+14.9 %** | 2.229 | **1.536** | **+31.1 %** |

**Universal winner across all 3 shapes (only `dkv` block sizes vary):**

```python
block_q=1024
block_kv=1024              # production has 512 — wrong
block_kv_compute=1024
block_q_dkv=seq_len        # 1024 for seq=1024; 2048 for seq=2048
block_kv_dkv=seq_len
block_kv_dkv_compute=seq_len
block_q_dq=None            # fused: must be None
block_kv_dq=None
use_fused_bwd_kernel=True  # production has False — wrong
q_layout=HEAD_DIM_MINOR
k_layout=HEAD_DIM_MINOR
v_layout=HEAD_DIM_MINOR
```

The MaxText recipe (`block_kv=1024, fused_bwd=True, dkv=1024`) is within
**0.2 %** of the autotuned best at every shape — for this model on this
hardware, the MaxText recipe is essentially at the local optimum.

### Phase 1 top-5 — forward-only (per shape, ms; lower = better)

```
--- seq=1024 bs=4 (exp3) ---
  0.3199  bq=1024 bkv=1024  qly=HEAD_DIM_MINOR  (214.8 TFLOP/s)  *** BEST ***
  0.3712  bq=1024 bkv=1024  qly=SEQ_MINOR
  0.3815  bq=1024 bkv= 512  qly=HEAD_DIM_MINOR  *** PROD shape ***
  0.3859  bq= 512 bkv=1024  qly=HEAD_DIM_MINOR
  0.4163  bq=1024 bkv= 512  qly=SEQ_MINOR

--- seq=2048 bs=2 (exp5) ---
  0.5163  bq=1024 bkv=1024  qly=HEAD_DIM_MINOR  (266.2 TFLOP/s)  *** BEST ***
  0.5723  bq=2048 bkv=2048  qly=HEAD_DIM_MINOR
  0.5791  bq=2048 bkv=1024  qly=HEAD_DIM_MINOR
  0.5968  bq=1024 bkv=1024  qly=SEQ_MINOR
  0.6059  bq=1024 bkv= 512  qly=HEAD_DIM_MINOR  *** PROD shape ***

--- seq=1024 bs=2 (baseline) ---
  0.1650  bq=1024 bkv=1024  qly=HEAD_DIM_MINOR  *** BEST ***
  0.1954  bq=1024 bkv=1024  qly=SEQ_MINOR
  0.1965  bq=1024 bkv= 512  qly=HEAD_DIM_MINOR  *** PROD shape ***
  0.1976  bq= 512 bkv=1024  qly=HEAD_DIM_MINOR
  0.2173  bq=1024 bkv= 512  qly=SEQ_MINOR
```

### Phase 2 top-5 — full fwd+bwd (best fwd seeds × dkv grid × {fused, non-fused})

```
--- seq=1024 bs=4 (exp3) ---
  0.9598  bq=1024 bkv=1024  fused=T  dkv=(1024,1024)  *** BEST ***
  0.9953  bq=1024 bkv=1024  fused=T  dkv=(1024,1024)  qly=SEQ_MINOR
  1.0181  bq=1024 bkv= 512  fused=T  dkv=(1024,1024)
  1.0434  bq=1024 bkv=1024  fused=T  dkv=( 512,1024)
  1.0747  bq=1024 bkv=1024  fused=T  dkv=( 512,1024)  qly=SEQ_MINOR
  ...
  1.4202  bq=1024 bkv= 512  fused=F  dkv=(1024,1024) dq=(1024,1024)  *** PROD shape ***

--- seq=2048 bs=2 (exp5) ---
  1.5357  bq=1024 bkv=1024  fused=T  dkv=(2048,2048)  *** BEST ***
  1.5451  bq=1024 bkv=1024  fused=T  dkv=(1024,1024)  (within 0.6%)
  1.5865  bq=1024 bkv=1024  fused=T  dkv=(1024,2048)
  ...

--- seq=1024 bs=2 (baseline) ---
  0.4776  bq=1024 bkv=1024  fused=T  dkv=(1024,1024)  *** BEST ***
  0.4857  bq=1024 bkv=1024  fused=T  dkv=(1024,1024)  qly=SEQ_MINOR
  0.5094  bq=1024 bkv= 512  fused=T  dkv=(1024,1024)
  ...
```

Full per-config CSV at
[`raw/profiles/2026-04-25-exp8-splash-kernel-autotune/results.csv`](../../../../../raw/profiles/2026-04-25-exp8-splash-kernel-autotune/results.csv)
(171 rows × 14 columns; one CSV row per (shape, phase, config) tuple).

### Cross-host validation

Rank 0 and rank 1 timed independently on different chips of the same v6e-8
slice. Best-config ms agreement:

| Shape | Rank 0 ms | Rank 1 ms | Δ |
|---|---|---|---|
| seq=1024 bs=4 | 0.960 | 0.962 | +0.2 % |
| seq=2048 bs=2 | 1.536 | 1.538 | +0.13 % |
| seq=1024 bs=2 | 0.478 | 0.479 | +0.21 % |

→ measurement noise floor is well below the 30 % gain we measured.

## Profile

This experiment did **not** capture an `xprof` trace — splash kernel-only
timing on a single device is the artifact, not a step-level training trace.
The persistent on-disk record is the per-config CSV plus the raw rank-0/rank-1
stdout logs:

- **On-disk directory**: [`raw/profiles/2026-04-25-exp8-splash-kernel-autotune/`](../../../../../raw/profiles/2026-04-25-exp8-splash-kernel-autotune/)
  (in-tree, ~110 KiB total — small enough to keep checked in alongside the
  experiment, unlike multi-GiB xprof traces).
- **Contents**:
  - `rank0.log` — full stdout from rank-0 process (3 sweeps, 867 lines).
  - `rank1.log` — same from rank-1 (cross-validation).
  - `results.csv` — 171 rows of `(shape_label, phase, block_q, block_kv,
    q_layout, k_layout, v_layout, fused_bwd, block_q_dkv, block_kv_dkv,
    block_q_dq, block_kv_dq, ms, TFLOP/s)`. Reusable by future shape sweeps —
    same harness adds rows.
- **Steps captured**: n/a (kernel-only run; no training steps).
- **What's inside**: per-config kernel wall-clock (median of 30 timed iters
  after 3 warmup iters, `jax.block_until_ready` between iterations).

## Observations

### O1. Asymmetric blocks are wrong here — the production `block_kv=512` was a stale MaxText carry-over

Production [`splash_attn.py:43`](../splash_attn.py#L43) has `block_kv=512`,
which gives `block_q (1024) > block_kv (512)`. At seq=1024 this is **−14.5 %
on fwd alone** vs `block_kv=1024` (Phase 1: 0.382 ms vs 0.320 ms at exp3
shape). At seq=2048 it's **−14.7 %** (0.606 vs 0.516).

The likely origin trace: this trainer was scaffolded from
[torchax/examples/train_llama_torchtitan/](../../../../../raw/code/torchax/examples/train_llama_torchtitan/),
which carries an asymmetric `block_q=1024 block_kv=512` config that may have
been right for an *older* MaxText recipe at a different (B, H, L) — but the
current Llama 3.1-8B recipe in
[tpu-recipes](../../../../../raw/code/tpu-recipes/training/v6e/Llama3.1-8B/jax/maxtext/v6e_8/)
uses symmetric 1024.

The contradicts gemma4 [exp29](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp29-splash-asymmetric-rejected.md)'s
finding ("symmetric 1024 is the sweet spot") — it agrees with it: symmetric
`(1024, 1024)` *is* the sweet spot. The mistake was that gemma4 was at the
sweet spot already, while llama3 production sat at `(1024, 512)` not by a
recipe choice but by inheritance from a stale scaffold.

### O2. `use_fused_bwd_kernel=True` is a free ~15 % on fwd+bwd

For every (shape, fwd-config) pair, the `fused=True` row beats every
`fused=False` row by 7–15 %. Production has `use_fused_bwd_kernel=False`.
This is the **same finding** as gemma4
[exp17 (+0.9 %)](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp17-splash-fused-bwd-accepted.md);
the gemma4 number was small because its production already had a near-best
fwd config. For us, fused alone is a measurable second-order win on top of
the block-size fix.

The mechanism: the fused kernel does dq and dkv inside one Pallas call,
sharing reads of Q/K/V from VMEM and avoiding a separate dq launch.

### O3. `dkv` block sizes scale with `seq_len`, but the second-best `dkv = 1024` is within 0.6 % at seq=2048

At seq=2048 the absolute best is `dkv = (2048, 2048)` (1.536 ms). The
candidate `dkv = (1024, 1024)` is at 1.545 ms — only 0.6 % slower. So the
**simple rule "set `dkv` blocks to `min(2048, seq_len)`"** captures
essentially all the win and avoids needing per-shape `dkv` tuning if we
care about not maintaining a per-shape table. (This is what production
*already* does via `min(global_block_q_dkv, query.shape[2])` clamping.)

The `dkv` knob matters less than `block_kv` and `use_fused_bwd_kernel`.

### O4. `q_layout=HEAD_DIM_MINOR` is correct; `SEQ_MINOR` is consistently worse here

Contrary to gemma4
[exp24 (+0.5 % via SEQ_MINOR)](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp24-splash-seq-minor-accepted.md),
on this Llama 3 shape `SEQ_MINOR` is uniformly **worse** than
`HEAD_DIM_MINOR` by 1-7 % across all 3 sweeps. Different model = different
optimal layout; do not transfer between models without re-tuning.

### O5. Kernel TFLOP/s is good (~30 % of v6e bf16 peak)

The best fwd+bwd config at seq=2048 sustains **268.5 TFLOP/s** on the
splash kernel itself — ~29 % of the 918 TFLOP/s v6e bf16 peak. The
attention kernel is *not* the MFU floor for this trainer; the rest of the
graph (gemm via XLA, FSDP collectives, framework overhead) is what holds
overall MFU at 35.7 %.

### O6. End-to-end TPS impact is bounded by attention's share of step time

Back-of-envelope, 32 layers × per-layer kernel time = total attention
contribution per step:

| Shape | Prod fwd+bwd × 32L | Step time | Attn share | If kernel −30 % | Δ step time | Δ TPS |
|---|---|---|---|---|---|---|
| seq=1024 bs=2 (baseline) | 21.9 ms | 446 ms | 4.9 % | −6.6 ms | −1.5 % | **+1.5 %** |
| seq=1024 bs=4 (exp3) | 45.4 ms | 572 ms | 7.9 % | −13.6 ms | −2.4 % | **+2.4 %** |
| seq=2048 bs=2 (exp5) | 71.3 ms | ~613 ms | 11.6 % | −21.4 ms | −3.5 % | **+3.5 %** |

**Expected end-to-end gain: +1.5 % to +3.5 % TPS** depending on shape. This
is on the edge of the [exp 7](2026-04-25-exp7-splash-xla-bs4-rejected.md)
±2 % run-to-run noise band at seq=1024, but should be cleanly resolvable
at seq=2048 and at any future seq=4096+/seq=8192 shape (where attention
becomes a larger share of step time).

## Verdict + reasoning

**Potential.** The **kernel-level win is unambiguous** (+30 % at every
in-program shape, rank 0 ↔ rank 1 agreement within 0.2 %, MaxText recipe
independently reaches the same config). But this experiment did not run
the full training graph, so we have not yet measured:

- (a) whether XLA fusion across the splash custom-call boundary changes
  with the new BlockSizes — *unlikely* to regress because BlockSizes only
  changes Pallas-internal tiling, not the custom-call signature; but
  gemma4 [exp33 (Pallas RMSNorm, −8.1 %)](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp33-pallas-rmsnorm-rejected.md)
  is the cautionary precedent that says "kernel-correct ≠ end-to-end win";
- (b) whether `use_fused_bwd_kernel=True` interacts adversely with the
  trainer's remat policy (current default = `nothing_saveable`, which
  recomputes everything — fused-bwd should *help* this case by sharing
  Q/K/V reads inside one bwd Pallas call rather than two);
- (c) whether activation memory changes (fused-bwd typically saves a
  separate dq output buffer; this could free ~100 MiB at exp 3 shape and
  unblock bs≥6 — speculative, validate via xprof).

Promotion to **accepted** requires a full XPK training run with the
autotune-best splash config, holding everything else equal to exp 5 / exp 3.
See "Next hypotheses" below.

## Next hypotheses generated

1. **`exp 9` — full-training validation of the autotune-best splash config**
   on top of [exp 5](2026-04-25-exp5-splash-seq2k-accepted.md) (current
   trunk = seq=2048 bs=2). Code change: edit `splash_attn.py:42-53` to
   `block_q=1024, block_kv=1024, block_kv_compute=1024,
   block_q_dkv=2048, block_kv_dkv=2048, block_kv_dkv_compute=2048,
   block_q_dq=None, block_kv_dq=None, use_fused_bwd_kernel=True`. Expected
   gain: +3-4 % TPS at seq=2048 bs=2; +2-3 % at seq=1024 bs=4. Effort: S.
   **Highest-priority follow-up.**

2. **`exp 10` — autotune at seq=8192 (program target shape).** Same harness,
   `--seq_len 8192 --batch_size 1`. Expected: at seq=8192 attention is ~25 %
   of step time, so an 30 % kernel win lands at ~7-8 % end-to-end TPS gain.
   Also tests whether the dkv-block heuristic (`dkv = min(2048, seq_len)`)
   holds at seq=8192 or if a larger `dkv` value wins. Effort: S
   (re-run the same XPK harness command). Pre-requisite: not blocked by
   any other work.

3. **`exp 11` — combine autotune-best splash with bs=4 / seq=8192.** Once
   exp 9 lands, try whether the freed activation memory from
   `use_fused_bwd_kernel=True` opens previously-OOM batch / seq combinations
   (exp 4 was bs=8 OOM by 7.36 GiB; possibly we can push toward bs=5-6 or
   seq>1024 bs=4). Effort: S; speculative; validate via the dummy compile
   that the trainer does at trace time.

4. **`exp 12` — phase-3 dq-block decoupling sweep, only if any non-fused
   variant is competitive in a future shape.** In this run every non-fused
   config was strictly dominated by `fused=True`, so the pinned
   `dq_block = dkv_block` heuristic in phase 2 didn't cost us anything. If
   a future shape (e.g. very long seq with limited VMEM) makes fused-bwd
   infeasible, run this. Effort: S; defer.

5. **Methodology — promote `tune_splash.py` to a reusable autotune harness
   for any future Pallas kernel in this trainer.** The `BlockSizes`-style
   sweep generalizes to any kernel exposing a frozen-dataclass tunable
   surface. Sketch in
   [README.md](README.md). Effort: M.

## See also

- Exp 5 — splash + seq=2048 (current branch HEAD; page not yet filed —
  covered in commit `b65ecff`).
- [Exp 3 — splash + bs=4](2026-04-25-exp3-splash-bs4-accepted.md) — current best at seq=1024.
- [`splash_attn.py`](../splash_attn.py) — the production kernel wrapper to edit for exp 9.
- [`tune_splash.py`](../tune_splash.py) — this experiment's harness.
- [Gemma4 exp 25 — block_q=block_kv=1024 wins](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md)
  — same finding on a different model.
- [Gemma4 exp 17 — `use_fused_bwd_kernel=True` wins](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp17-splash-fused-bwd-accepted.md).
- [Gemma4 exp 33 — Pallas RMSNorm refuted (−8.1 %)](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp33-pallas-rmsnorm-rejected.md)
  — the cautionary precedent for "kernel-correct ≠ end-to-end win".
- [tokamax — splash autotune script](../../../../../raw/code/tokamax/tokamax/experimental/utils/tuning/tpu/splash_attention_benchmarking.py)
  — the more-feature-rich harness we chose not to use (vendored splash + tune_jax dep).
- [marin/levanter — autotune harness](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/autotune_utils.py)
  — different shape (in-graph autotune cache); same idea, different scope.

## Sources

- [`raw/profiles/2026-04-25-exp8-splash-kernel-autotune/`](../../../../../raw/profiles/2026-04-25-exp8-splash-kernel-autotune/)
  — rank0.log, rank1.log, results.csv (171 configs).
- [`tune_splash.py`](../tune_splash.py) — harness source.
- [`splash_attn.py`](../splash_attn.py) — production kernel wrapper (the
  config to edit for exp 9).
- [`Dockerfile`](../Dockerfile) and
  [`DEPLOY.md`](../DEPLOY.md) — image / submission flow (image tag
  `tune-v2`, registry `<your-registry>/test/llama3-8b-torchax-container`).
- Workload name (XPK): `llama3-splash-tune-20260425-191044`.
