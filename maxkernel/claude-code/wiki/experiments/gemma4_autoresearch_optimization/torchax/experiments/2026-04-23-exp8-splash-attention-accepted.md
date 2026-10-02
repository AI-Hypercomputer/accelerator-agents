---
title: "Exp 8 — splash attention via Pallas (KEEP, new best +2.7 %)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, shard-map, tps-win]
hypothesis: splash-attention-pallas-kernel
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD + new pallas_attention.py + register_splash_attention in train.py"
verdict: supported
---

First Pallas-kernel experiment. Registered a custom attention function in HF's `ALL_ATTENTION_FUNCTIONS` that dispatches through `jax.experimental.pallas.ops.tpu.splash_attention_kernel` via `torchax.interop.call_jax` + `jax.shard_map`. **Modest but real win: +2.7 % TPS vs baseline**, loss trajectory preserved.

## Hypothesis under test

**Statement**: The default attention path on this stack (HF PyTorch → torchax SDPA lowering → XLA) materializes the `[B, n_heads, S, S]` attention score tensor and misses TPU-Pallas-specific optimizations. Swapping to the splash Pallas kernel (which fuses softmax, avoids score-matrix materialization, and supports GQA natively) should reduce attention time by 15–40 % per the [tokamax splash-attention source](../../../../sources/2026-tokamax-splash-attention.md), and possibly free HBM via N² elimination.

Origin: program.md's "Pallas kernel landscape" section, first entry; exp 8 is the first experiment to actually execute a Pallas kernel on this program.

## Setup

- Config same as [exp 6](2026-04-23-exp6-selective-batch2-accepted.md): `--batch_size 2 --seq_len 1024`, selective remat (`checkpoint_dots_with_no_batch_dims`) retained.
- Code changes:
  - Added new file `torchax/model/pallas_attention.py` (~250 lines) with `splash_attention_fn` matching HF's `ALL_ATTENTION_FUNCTIONS` signature; wraps `jax.experimental.pallas.ops.tpu.splash_attention_kernel` inside `jax.shard_map(mesh=mesh, in_specs=P('fsdp',…), out_specs=P('fsdp',…), check_vma=False)` called via `torchax.interop.call_jax`.
  - Handles causal (full-attention) and sliding-window (sliding_window=512) layers via `splash_attention_mask.CausalMask` and `LocalMask(window_size=(W, 0))` respectively.
  - GQA via splash's native `num_q_heads != num_kv_heads` support (no `jnp.repeat`).
  - Block sizes 512 for all tile params (symmetric, power-of-two ≤ seq_len).
  - Kernel cached per `(seq_len, num_q_heads, sliding_window)` via `functools.lru_cache` — 2 entries at this config.
- `train.py`: added `register_splash_attention(mesh)` call + `model.config._attn_implementation = "splash_pallas"` on both `model.config` and `model.config.text_config` before `JittableModule` wrapping.

### First failure mode (fixed)

Initial run crashed with:
```
NotImplementedError: Mosaic kernels cannot be automatically partitioned.
Please wrap the call in a shard_map.
```
Root cause: the Pallas/Mosaic custom-call can't be auto-partitioned by GSPMD when inputs have sharded layouts. Fix: explicitly wrap the kernel call in `jax.shard_map` with the same `PartitionSpec('fsdp', None, None, None)` on Q/K/V and output. The per-shard body runs `jax.vmap(kernel)` over the per-chip batch dim.

## Results

| Metric | Baseline | Exp 6 (previous best) | **Exp 8 (splash)** | Δ vs baseline | Δ vs exp 6 |
|---|---|---|---|---|---|
| Step time (wall) | 134.4 ms | 264.9 ms | **261.0 ms** | +94 % | −1.5 % |
| Compile step 0 | 149.5 s | 164.5 s | 156.8 s | +5 % | −5 % |
| **TPS** | **30,570** | 30,925 | **31,387** | **+2.7 %** | +1.5 % |
| Per-token cost | 32.8 µs | 32.3 µs | **31.9 µs** | −2.7 % | −1.3 % |
| Peak HBM | 29.69 GiB (95 %) | 25.92 GiB (83 %) | 25.85 GiB (83 %) | −13 % | flat |
| Stack reservation | 17.4 GiB | 13.57 GiB | 13.53 GiB | −22 % | flat |
| Loss trajectory | 3.93 → 1.97 | 3.82 → 1.57 | 3.82 → 1.55 | match | match |

**HLO-op diff** (3 profiled steps, across 4 chips):

| Op | Exp 6 (ms, approx) | Exp 8 (ms) | Notes |
|---|---|---|---|
| convolution fusion | 1262 (approx) | 1206 | -56 ms — Q@K^T and attn@V matmuls moved out of XLA conv fusion |
| loop fusion | 730 (approx) | 683 | -47 ms — softmax moved into splash kernel |
| **custom-call** (splash kernel) | 0 | **122** | **new op** — splash's Pallas custom-call |
| custom fusion | ~310 | 278 | |
| all-gather | ~220 | 191 | |
| data formatting | ~90 | 120 | +30 ms — extra layout transforms for the custom-call boundary |

Net: splash's 122 ms custom-call replaces ~170 ms of attention-related convolution+loop fusion work. Save ~48 ms on the profiled span across 3 steps / 4 chips = ~4 ms / step, matching the +1.5 % exp-6-over-exp-8 delta.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp8-splash-attention](http://localhost:8791/?run=2026-04-23-gemma4-exp8-splash-attention) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp8-splash-attention`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp8-splash-attention/`](../../../../../raw/profiles/2026-04-23-gemma4-exp8-splash-attention/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — splash Pallas kernel via `call_jax` + `shard_map`; custom-call op appears, replacing parts of convolution+loop fusion.

## Mechanism

Splash attention replaces the XLA-lowered attention with a Pallas/Mosaic kernel that:
1. **Fuses** Q@K^T, softmax, and attn@V into one tiled kernel body. No intermediate HBM round-trip for the `[B, H, S, S]` score tensor.
2. **Handles GQA natively**: `num_kv_heads=2`, `num_q_heads=8` — splash's kernel reshapes internally to `(kv_heads, q_heads_per_kv, ...)` without `jnp.repeat`.
3. **Tiled over (block_q, block_kv)** with VMEM-resident accumulators — keeps the softmax-normalizer running sum in VMEM, avoids bf16-precision loss at long sequences.

At seq=1024 this is a marginal win because:
- The attention work is only ~5 % of total FLOPs at this seq:len/hidden ratio (MLP dominates).
- The `[B, H, S, S] = [2, 8, 1024, 1024] × bf16 = 32 MiB/chip` score tensor is too small to cause HBM pressure. Memory savings are negligible here.
- Splash's win scales with seq² — at seq=2048 it should be ~4×, at seq=4096 ~16×.

**Exp 9 (launched in parallel)** tests splash at seq=2048. That's where the real win lives, and splash's numerically-stable softmax may also fix the known NaN-at-seq≥2048 correctness bug.

## Verdict

**SUPPORTED.** +2.7 % TPS, new current best. Loss trajectory preserved. Pallas kernel integration pattern validated — the `ALL_ATTENTION_FUNCTIONS` + `call_jax` + `shard_map` wiring works and is reusable for future Pallas experiments (e.g. tokamax CE kernel).

## Next hypotheses

1. **Exp 9 — splash + seq=2048** (launched). Splash's memory-efficient attention may let seq=2048 finally work at batch=1, AND the more numerically-stable softmax may fix the NaN bug.
2. **Splash block-size autotune**: current block_q = block_kv = 512. The splash-attention doc suggests autotuning these — small room for improvement via `tokamax.autotune` wrapping.
3. **Async-collective flags revisited at batch=2 + splash**: scheduler has both more compute and the splash custom-call to hide collectives behind.
4. **tokamax memory-efficient CE kernel**: the next Pallas target — saves ~4 GiB fp32 logits at batch=2. Would further free HBM.
5. **Build TPU Pallas RMSNorm / SwiGLU** (from program.md's "Kernels to BUILD" table): lower priority without profile signal — `loop fusion` is still a dominant bucket but not directly actionable without building the kernels.

## See also

- [program.md § Pallas kernel landscape](../../program.md) — the design doc that motivated this experiment.
- [OBSERVATIONS.md § exp08](OBSERVATIONS.md).
- [2026-04-23-exp6-selective-batch2-accepted.md](2026-04-23-exp6-selective-batch2-accepted.md) — the baseline this builds on.
- [splash-attention concept](../../../../concepts/splash-attention.md), [pallas-kernel](../../../../concepts/pallas-kernel.md), [attention-block-sizes](../../../../concepts/attention-block-sizes.md).
- [tokamax splash-attention source](../../../../sources/2026-tokamax-splash-attention.md).
- [torchax codebase](../../../../codebases/torchax.md) (`interop.call_jax`).

## Sources

- `raw/profiles/2026-04-23-gemma4-exp8-splash-attention/`
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_attention.py` (256 lines, new)
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` (+13 lines for registration)
- `/tmp/gemma4_exp8.log`
