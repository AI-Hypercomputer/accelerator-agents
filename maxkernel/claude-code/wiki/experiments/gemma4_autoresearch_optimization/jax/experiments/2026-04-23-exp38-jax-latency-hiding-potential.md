---
title: "Exp 38 — latency-hiding scheduler on JAX splash+b3 (POTENTIAL, +0.06% flat)"
type: experiment
tags: [experiment, gemma4, jax, xla-flags, scheduler, potential]
hypothesis: latency-hiding-scheduler-on-jax-stack
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: pending
verdict: inconclusive
---

Same JAX stack as exp 36 (splash + batch=3 + bf16 CE by construction), with `--xla_tpu_enable_latency_hiding_scheduler=true` added via `LIBTPU_INIT_ARGS`. **Result: 34,634 TPS vs exp 36's 34,614 — +0.06 %, flat.** No-op, mirroring the torchax exp 30 result on the same flag.

## Hypothesis

The JAX-stack profile at exp 36 showed a ~12.1 % `collective-permute-done` bucket. On the torchax stack we had ruled out `latency_hiding_scheduler` as a regressor in exp 30, but the JAX port has a different op graph (splash invoked differently, no torchax dispatch). Retest whether latency-hiding can recover some of the collective wait on this stack.

## Setup

- Branch: `main`.
- Zero code changes. `LIBTPU_INIT_ARGS="--xla_tpu_enable_latency_hiding_scheduler=true"` + `JAX_ATTENTION_IMPL=splash` + `--batch_size 3 --seq_len 1024 --steps 20`.

## Results

| Metric | Exp 36 (splash+b3) | **Exp 38 (+latency-hiding)** | Δ |
|---|---|---|---|
| TPS | 34,614 | **34,634** | **+0.06 %** |
| Step time (mean, steps 2–19) | ~355.0 ms | **354.80 ms** | −0.06 % |
| Compile step 0 | ~132 s | 166.58 s | +26 % |
| Loss descent | clean | match | identical |

Compile time bumped as expected (scheduler has more search space). Runtime identical within noise.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp38-latency-hiding](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp38-latency-hiding) — opens the interactive trace viewer.
- **Run name**: `2026-04-23-gemma4-jax-exp38-latency-hiding`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp38-latency-hiding/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp38-latency-hiding/) (gitignored; also mirrored at `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp38-latency-hiding/`).
- **Steps captured**: 10, 11, 12.
- **What's inside**: xprof trace of splash+b3 with latency-hiding-scheduler enabled. Use to compare the `collective-permute-done` bucket against exp 36's profile — if the wait duration dropped but other buckets grew to absorb it, the scheduler IS reordering but the reorder has no net benefit at this scale.

## Verdict

**POTENTIAL / INCONCLUSIVE.** Flat +0.06 %. Same result as torchax exp 30. Confirms that `latency_hiding_scheduler` alone is neither a regressor nor a winner on our workload — on EITHER stack.

## Conclusion — collective-permute-done is a floor, not a lever

Combined with torchax exp 30 + 31 (both flat on latency-hiding and overlap-compute-collective-tc in isolation), and exp 32 (2D TP mesh is 2.4× slower at this chip count), the **~12 % collective-permute-done bucket on v6e-4 1D fsdp=4 appears to be an ICI-bandwidth floor**, not a lever we can pull with XLA flags alone. Moving that bucket would require either:
- structural topology change (more chips, which changes collective-permute cost scaling), or
- recomputation-vs-communication tradeoff (e.g., replicate some weights to skip a gather — unlikely to win given our existing batch-dominated profile).

Park this investigation. Focus remaining effort on the `loop fusion` bucket (RMSNorm/SwiGLU/residual) and the convolution/matmul bucket.

## Next hypotheses

- **exp 39 — Pallas RMSNorm hand-rolled custom_vjp on JAX stack.** Torchax exp 33 failed because XLA was already fusing RMSNorm with neighboring matmuls and the Pallas custom-call broke the fusion. The JAX port has a slightly different op graph — worth retesting whether the same fusion-boundary tax applies. Bounded downside.
- **exp 40 — scan-over-layers.** Still the biggest compile-time lever; on native JAX the Gemma-4 blockers torchax exp 26 hit (ScannedModule assertions) don't apply — we own the forward.
- **exp 41 — persistent JAX compile cache.** 30× iteration-speed win, not TPS.

## See also

- [exp 36 — splash + batch=3 (current JAX-stack best)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [torchax exp 30 — same flag, same result](../../torchax/experiments/2026-04-23-exp30-latency-hiding-solo-potential.md)
- [torchax exp 32 — 2D mesh refuted at this chip count](../../torchax/experiments/2026-04-23-exp32-2d-mesh-tp2-rejected.md)

## Sources

- `/tmp/gemma4_jax_exp38.log`
- `raw/profiles/2026-04-23-gemma4-jax-exp38-latency-hiding/` — xprof run `2026-04-23-gemma4-jax-exp38-latency-hiding` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp38-latency-hiding
