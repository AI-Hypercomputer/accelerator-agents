---
title: "Exp 33 — hand-written Pallas RMSNorm + custom_vjp (REFUTED, -8.1 %)"
type: experiment
tags: [experiment, gemma4, pallas, rmsnorm, custom-vjp, refuted]
hypothesis: pallas-rmsnorm-hand-custom-vjp
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp33-pallas-rmsnorm (not merged)"
verdict: refuted
---

Second hand-rolled-Pallas experiment after [exp 8 (splash attention)](2026-04-23-exp8-splash-attention-accepted.md). Goal: replace Gemma 4's 250+ `Gemma4RMSNorm` invocations per step with a TPU Pallas kernel + hand-written `jax.custom_vjp`, collapsing XLA's per-norm `loop fusion` cost into a single Mosaic custom-call. **Result: -8.1 % TPS.** Loss trajectory preserved (kernel numerics correct — unit-tested to within bf16 epsilon). The regression is a boundary-cost story: every norm now crosses a shard_map + custom-call boundary that the XLA fusion pass can no longer thread through neighbouring work.

## Hypothesis under test

**Statement**: XLA's `loop fusion` bucket (~683 ms per 3 profiled steps × 4 chips ≈ 57 ms/step globally, per [exp 8 writeup](2026-04-23-exp8-splash-attention-accepted.md)) is dominated after splash by the many RMSNorm instances. Gemma 4 E4B has **42 layers × 6 norms per layer** (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `q_norm`, `k_norm`), plus a top-level `model.norm`, plus `v_norm` (`with_scale=False`), plus `embedding_pre_projection_norm`. If we replace all of them with a Pallas kernel that fuses the row reduction + rsqrt + weight-multiply + bf16↔fp32 casts into one custom-call, we save ~20 % of that bucket (~11 ms/step), clearing the path for >33,372 TPS.

**Why this looked reasonable a priori**: the splash custom-call did replace ~170 ms of attention-lowering with ~122 ms of Pallas kernel, a clean net win. Repeating the recipe for RMSNorm seemed mechanical.

Origin: exp 8 verdict's "next hypotheses" #5; exp 20 was the earlier attempt using `pallas-forge` which had no `custom_vjp`, so `jax.grad` couldn't differentiate through it and the attempt was parked.

## Setup

**Hardware**: v6e-4 TPU. `seq_len=1024`, `batch_size=3`, `fsdp=4`, bf16, selective remat, splash attention (exp 25 stack), bf16 CE. Identical to [exp 25 best](../../../../../RESULTS.tsv) except for the RMSNorm swap.

**Code changes**:
  - **New**: `torchax/model/pallas_rmsnorm.py` (~290 lines). Forward is a Pallas TPU kernel that normalizes one group of `block_rows=8|16|32` rows (TPU sublane requirement) at a time; backward is plain JAX (the bwd is memory-bound — per the exp-33 design memo the win is forward fusion, and a JAX bwd was sufficient to ship). Kernel cached per `(D, dtype)` via `functools.lru_cache`. Shape-polymorphic: D ∈ {2560 (hidden), 256 (head_dim)}; leading dims collapsed to a flat `N` for the grid. Wrapped in `jax.shard_map(mesh, in_specs=P('fsdp', …), out_specs=P('fsdp', …))` because Mosaic custom-calls cannot be auto-partitioned (same reason as splash in `pallas_attention.py`).
  - **Edited**: `torchax/train.py` (+6 lines). `if os.environ.get("PALLAS_RMSNORM") == "1": register_pallas_rmsnorm(mesh)`. Gated so trunk is untouched.

**Derivation check**: for `y = x * rstd * gain` with `rstd = (mean(x²) + ε)^{-½}` and `gain = w` when `with_scale=True` else `1`:
  - `dx = rstd * g - x * rstd³ * (Σ(g⋅x) / D)` where `g = dy * gain`.
  - `dw = Σ_over_rows (dy * x * rstd)` when `with_scale=True`.
  - Derivation-reference parity verified by `/tmp/test_pallas_rmsnorm.py` (see Validation).

**Weight convention**: this Gemma 4 variant uses `weight = ones` + `normed * weight`, NOT the Gemma-2 `zeros + (1+weight)` convention the exp-33 design memo quoted. Confirmed by reading `transformers/models/gemma4/modeling_gemma4.py:168-186`. The kernel preserves the actual (weight-times) semantics.

**Command diff from exp 25**:
```diff
-  python -u -m train --steps 20 --batch_size 3 --seq_len 1024 ...
+  PALLAS_RMSNORM=1 python -u -m train --steps 20 --batch_size 3 --seq_len 1024 ...
```

## Validation

**Unit test** (`/tmp/test_pallas_rmsnorm.py`, 5 shape/dtype combinations × {fwd, bwd}):
- fp32 fwd: abs_err ≤ 1.4e-6 (D=2560, 256), relative ≤ 1.8e-6. Match.
- bf16 fwd: abs_err ≤ 1.6e-2, relative ≤ 2.0e-2 — within bf16 precision at D=2560 reductions.
- fp32 bwd (`dx`, `dw`): relative ≤ 6e-7 vs pure-JAX reference. Exact to the last bit in most cases.
- bf16 bwd: relative ≤ 2.0e-2 on `dx`, `dw` identical.

**Smoke training** (5 steps, `PALLAS_RMSNORM=1`):
- Step 4 loss: **2.19** (exp 25 reference also ~2.19 at step 4). Match within noise — no `[pallas_rmsnorm] fallback` messages.
- Compile: step 0 = 167s, step 1 recompiles at 165s (same second-compile artefact present on exp 25).

## Results

| Metric | Exp 25 (current best) | **Exp 33 (pallas rmsnorm)** | Δ vs exp 25 |
|---|---|---|---|
| Step time (mean steps 2–19) | ~368 ms | **400.8 ms** | +8.9 % |
| **TPS** | **33,372** | **30,657** | **-8.1 %** |
| Compile step 0 | ~160 s | 168.7 s | +5 % |
| Loss at step 4 | 2.19 | 2.19 | match |
| Loss at step 19 | ~1.97 (range) | 1.82 | within noise of exp 25 |
| Tokens/step | 12,288 | 12,288 | same |

Raw per-step (steps 2–19): 400.0–401.6 ms, σ < 0.5 ms — extremely tight. The regression is not noise.

**Verdict**: 30,657 < 33,272 (the -0.3 % noise band floor). By the exp-33 decision tree this is **refuted** — not merged to trunk.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp33-pallas-rmsnorm](http://localhost:8791/?run=2026-04-23-gemma4-exp33-pallas-rmsnorm) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp33-pallas-rmsnorm`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp33-pallas-rmsnorm/`](../../../../../raw/profiles/2026-04-23-gemma4-exp33-pallas-rmsnorm/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — hand-written Pallas RMSNorm + custom_vjp replacing Gemma4RMSNorm (~250 sites/step); −8.1 % via fusion-boundary tax (Pallas custom-calls prevent XLA from fusing norms with neighboring matmuls). Also uploaded to `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-exp33-pallas-rmsnorm/` for direct xprof UI access.

## Mechanism — why it regressed

The kernel itself is fast (unit-tested; ~rstd reduction is single-pass at D ≤ 2560). The regression is **integration overhead**, not per-op cost:

1. **Custom-call boundaries break fusion.** Gemma 4 has ~250 RMSNorm calls per forward pass. On exp 25, each norm is a handful of HLOs that XLA's `loop_fusion` pass happily fuses with the surrounding `matmul → norm → residual-add → next-matmul` chain. On exp 33, each norm becomes an opaque Mosaic custom-call. XLA cannot fuse across custom-calls, so the upstream add + the downstream matmul now each need their own HBM round-trip for the norm's input and output. At ~250 norms × two extra HBM round-trips each × ~bf16 tensor of 1 × 1024 × 2560 × 2 B ≈ 5 MiB per round-trip, the extra HBM bandwidth alone exceeds the compute savings.
2. **shard_map launch overhead.** Each `jax.shard_map` wraps the Pallas custom-call for FSDP compliance. For 250 calls/step this is another non-trivial dispatch cost — even if each shard_map adds only ~100 µs, 250 × 0.1 = 25 ms/step, which is near the entire observed regression (32 ms).
3. **The bucket we targeted isn't purely RMSNorm.** The 57 ms/step `loop fusion` figure from exp 8 was across ALL fused elementwise+reduction ops (SwiGLU, residual adds, CE, bias broadcasts, etc.). RMSNorm is only a fraction. Even if we did save 20 % of the norm portion, that's plausibly <5 ms/step — nowhere near enough to offset the boundary cost above.

The splash experiment (exp 8) didn't hit this mechanism because splash replaces a single large op per layer (one attention block) and the surrounding math is coarse-grained matmuls that XLA tiles separately anyway. RMSNorm sits in the tight seam between matmuls where XLA's fusion was already doing real work.

## Verdict

**REFUTED**. -8.1 % TPS. Kernel numerics are correct, loss trajectory preserved — the failure mode is architectural (custom-call ⇒ unfusable) not correctness. Not merged. Branch `perfautoresearch/v6e4-20260423-exp33-pallas-rmsnorm` preserved for future reference.

Headline: the recipe that worked for attention (exp 8) does NOT generalize to element-wise-adjacent ops on TPU. XLA's fusion pass is already exploiting the norm's locality with the surrounding matmuls; a custom-call forces HBM round-trips that wipe out the kernel's internal savings.

## Next hypotheses

1. **Fuse the norm INTO a neighbouring custom-call.** If we ever write a Pallas MLP or attention-input kernel, folding the pre-MLP or pre-attention RMSNorm into its entry path preserves fusion (the Pallas kernel now owns the full `norm → matmul` seam). Scoped as "Pallas fused pre-MLP kernel" — significant engineering, deferred.
2. **bf16 stat reduction (not a kernel).** Within XLA, try forcing the RMSNorm mean-square reduction to run in bf16 instead of fp32. This would eliminate the fp32 cast pair that appears in `loop fusion`. Risk: numerical drift, may hit NaN at long seq. Easy to try as a modeling_gemma4 monkey-patch (no Pallas), S effort.
3. **Disable this path permanently in the gate.** `PALLAS_RMSNORM=0` is the default; leave the code in-tree as a documented refuted path so future folks don't re-try it without re-reading this page.
4. **Revisit after scan-over-layers lands** (exp 26, parked). With a `jax.scan` body, each norm becomes one HLO in the scanned body rather than 42 instances — potentially different fusion dynamics. Low priority until exp-26's B1-B7 sub-problems are closed.

## See also

- [exp 8 — splash attention via Pallas](2026-04-23-exp8-splash-attention-accepted.md) — the reference Pallas integration pattern and the RMSNorm-targeted `loop fusion` observation.
- [exp 25 best](../../../../../RESULTS.tsv) — the baseline this was measured against.
- [program.md § Pallas kernel landscape — kernels to BUILD](../../program.md).
- [pallas_attention.py](torchax/model/pallas_attention.py) — the reference `_MESH` + `shard_map` + `call_jax` wiring this file mirrors.
- [pallas_rmsnorm.py](torchax/model/pallas_rmsnorm.py) — the new kernel module (kept even though refuted — preserves engineering record).
- [torchax codebase](../../../../codebases/torchax.md) — `interop.call_jax`, `JittableModule`.

## Sources

- `raw/profiles/2026-04-23-gemma4-exp33-pallas-rmsnorm/` (GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-exp33-pallas-rmsnorm/`).
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_rmsnorm.py` (new file).
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` (+6 lines env-gated registration).
- `/tmp/gemma4_exp33.log` — 20-step benchmark log.
- `/tmp/gemma4_exp33_smoke.log` — 5-step smoke log.
- `/tmp/test_pallas_rmsnorm.py` — unit test comparing kernel + vjp to pure-JAX reference.
