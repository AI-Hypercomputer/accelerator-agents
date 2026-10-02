---
title: "Exp 41 — JAX VMEM limit 512 MiB (REJECTED, −13.8%)"
type: experiment
tags: [experiment, gemma4, jax, xla-flags, vmem, rejected]
hypothesis: vmem-bump-helps-splash-block1024
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: pending
verdict: refuted
---

Bumped `--xla_tpu_scoped_vmem_limit_kib` from the default 131,072 KiB (128 MiB) to 524,288 KiB (512 MiB) on the JAX stack at exp 36's config. **Result: −13.8 % TPS (34,614 → 29,832), MFU 23.05 → 19.87 %.** Larger VMEM budget pessimizes — scheduler/cost-model decisions worsen, not better.

## Hypothesis

Splash's `block_q = block_kv = 1024` at seq=1024 uses larger tiles than typical. A bigger VMEM budget might allow more of those tiles in flight → better pipelining → faster attention. Simple env-flag test.

## Result

- Step time: 355 → 411.91 ms
- TPS: 34,614 → 29,832 (−13.8 %)
- MFU: 23.05 % → 19.87 %
- Loss: clean match (step 19 = 1.83; exp 36 was 1.84 — within noise)

## Mechanism — why bigger VMEM hurts

The XLA/Mosaic cost model likely uses the scoped VMEM limit as an input to its allocator. With more VMEM allowed, it tries larger-footprint schedules that:
- reduce the number of concurrent tiles (fewer simultaneous MXU streams ⇒ worse pipelining);
- or pick a fusion granularity that increases reload cost on HBM (the `convolution fusion` bucket is already sized just under 128 MiB for the default; expanding to 512 MiB lets the cost model over-commit per step).

Classic "more budget is not better" — the default 128 MiB is the sweet spot on v6e-4 for this model at this config.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp41-vmem512](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp41-vmem512) — opens the trace viewer.
- **Run name**: `2026-04-23-gemma4-jax-exp41-vmem512`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp41-vmem512/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp41-vmem512/) (gitignored; GCS mirror at `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp41-vmem512/`).
- **Steps captured**: 10, 11, 12.
- **What's inside**: xprof trace of exp 36 config with VMEM limit at 512 MiB — compare to exp 36 trace to see which op buckets grew.

## Verdict

**REJECTED.** Not merged. Exp 36 (b=3 splash, default VMEM 128 MiB) remains JAX-stack best at 34,614 TPS / 23.05 % MFU.

Informative data point: the XLA default is well-tuned for this workload. Don't revisit `scoped_vmem_limit_kib` on this stack.

## Next hypotheses

- **exp 42 — scan-over-layers.** Compile-time lever, not TPS. JAX port can do this cleanly.
- **exp 43 — Pallas RMSNorm retest**. Torchax exp 33 was −8.1 % due to fusion-boundary tax. JAX graph may differ; bounded downside.
- **exp 44 — persistent JAX compile cache**. 30× iteration-speed win, infra only.
- **Possibly done with TPS gains on this stack** — exp 36 may be the practical ceiling at v6e-4 + Gemma 4 E4B + 1D fsdp=4 + splash. Worth documenting as a second ceiling-analysis page.

## See also

- [exp 36 — JAX best (default VMEM)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- Torchax never tested this flag explicitly — first data point on either stack.

## Sources

- `/tmp/gemma4_jax_exp41.log`
- `raw/profiles/2026-04-23-gemma4-jax-exp41-vmem512/` — xprof run `2026-04-23-gemma4-jax-exp41-vmem512` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp41-vmem512
