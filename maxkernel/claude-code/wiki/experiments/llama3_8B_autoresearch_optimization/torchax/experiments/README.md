# Llama 3 8B — torchax stack experiments

Experiment writeups for the **torchax** (PyTorch-on-JAX) path through the
Llama 3 8B autoresearch program. Sibling of
[`../../jax/experiments/`](../../jax/experiments/README.md) (native-JAX
stack) and [`../../maxtext/experiments/`](../../maxtext/experiments/README.md)
(MaxText reference baseline).

## Contents

- `2026-04-25-baseline.md` — initial reference run on v6e-8 (HF transformers
  + JittableModule + torchax shard_map FSDP).
- `2026-04-25-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages.
  Verdict suffixes: `-accepted` / `-rejected` / `-potential` (see
  [program.md § Experiment verdict suffix](../../program.md)).
- (No `OBSERVATIONS.md` or `RESULTS.tsv` yet — to be added when this stack
  matures past first-batch experiments.)

## Current best (trunk, final)

🏆 **Exp 72a/74b** at **6,559 tok/s/chip / 36.8 % MFU** (bs=3 seq=8192 fsdp=8 v6e-8) — **+42.9 % per-chip vs the 2026-04-25 morning baseline (4,591/chip)**. Stack: scan-over-layers + AMP master fp32 weights / bf16 compute + tokamax CE chunked_xla + tokamax-shipped splash with `use_base2_exp + fuse_reciprocal + max_logit_const=30`. See [2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md](2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md) (the frontier writeup also chronicles exp 73-82 follow-ups, all refuted).

The torchax frontier is **converged**: every post-exp-74b knob (q_seq_shards, VMEM=131k/65k, dots-saveable variants, splash block-size sweeps, default-matmul-precision) was refuted (see exp 75-82 in the frontier writeup). The remaining ~7.8 pp MFU gap to MaxText (44.6 %) and the ~17 % per-chip gap to the JAX sibling (~7,700/chip 43.3 % MFU) come from torchax dispatch overhead at the framework level, not from kernel choices — both are addressable only by closing the framework-overhead gap or migrating to native JAX (which the [`../../jax/`](../../jax/) sibling did).

## Cross-stack comparison

| Stack | Best | tok/s/chip | MFU | Δ vs torchax |
|-------|------|-----------:|----:|-------------:|
| 🏆 native-JAX (Flax NNX) [exp 28b](../../jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) | bs=4 seq=8192 | ~7,700 (peak 7,768) | ~43.3 % | **+17.4 %** |
| MaxText reference [baseline](../../maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md) | bs=3 seq=8192 | 7,069 | 44.6 % | +7.8 % |
| **torchax frontier** (this stack) [exp 74b](2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md) | bs=3 seq=8192 | **6,559** | **36.8 %** | (anchor) |
| torchax morning baseline [2026-04-25](2026-04-25-baseline.md) | bs=2 seq=1024 | 4,591 | 22.9 % | -30.0 % |

## What was learned (kept)

- **AMP master pattern** (fp32 weights, bf16 compute, fp32 adamw mu/nu) is correct and stable at seq=8192 (fixes the original NaN-at-seq≥2048 issue).
- **Tokamax CE** with `chunked_xla` impl unlocks bs=3+ at seq=8192 by avoiding the materialised `[B*L, V]` logits buffer (saves ~6 GiB).
- **Tokamax-shipped splash attention** (`tokamax._src.ops.experimental.tpu.splash_attention`) with `use_base2_exp + fuse_reciprocal + max_logit_const=30` beats `jax.experimental.pallas.ops.tpu.splash_attention` by ~+1.3-1.8 % per-chip (the +4.4 % delta on the JAX side is the same effect at higher absolute throughput).
- **scan-over-layers** is mandatory at seq=8192 — without it, compile-time HBM peaks exceed device capacity even with full remat.
- **Splash block sizes** `bq=2048 bkv=1024` fwd, `bq_dkv=2048 bkv_dkv=2048` bwd, `fused_bwd=True` is the kernel-tune winner (validated by the 171-config sweep in exp 8 and confirmed at the JAX side via [JAX kernel-tune harness](../../jax/tune_kernels.py)).

## What was tried and refuted (highlights)

- **TP=2** — -14 % on torchax stack (refuted by exp 25); FSDP=8 is the only viable parallelism choice at this shape on v6e-8.
- **Recipe XLA flag bundles** (CF_FOR_ALL_GATHER, LAYOUT_FOR_ALL_REDUCE_SCATTER, DATA_PARALLEL_OVERLAP) — neutral on torchax (refuted by exp 1, exp 7); the **HOST_OFFLOAD_FLAGS bundle is the breakthrough** (revealed only on the JAX path, exp 12-18).
- **Q-sequence shards** (`q_seq_shards=2/4`) — INVALID; produces NaN loss from step 1 (needs context-parallel mesh axis we don't have).
- **VMEM≠98 KiB** — VMEM=131072 (-2.7 %) and VMEM=65536 (-3.1 %) both regressed.
- **`dots_saveable` / `dots_with_no_batch_dims_saveable` remat** — OOM at bs=2 (matmul-output saves don't fit at this shape).
- **Splash block-size variants** beyond the kernel-tune winner — all -0.1 to -0.4 %.
- **JAX_DEFAULT_MATMUL_PRECISION** — within noise / -0.5 %; default already does bf16 efficiently on TPU.

Full ablation chronicle in [exp 72a writeup §"Refuted"](2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md).

## See also

- [`../train.py`](../train.py) — the torchax trainer (primary runner).
- [`../splash_attn.py`](../splash_attn.py) — production splash kernel wrapper
  (the autotune target for exp 8).
- [`../tune_splash.py`](../tune_splash.py) — kernel-only autotune harness.
- [`../model/`](../model/) — sharding wiring.
- [Shared program protocol](../../program.md).
