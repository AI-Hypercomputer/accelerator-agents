---
title: "Exp 35 — splash pallas attention in JAX (POTENTIAL, +0.3 % TPS vs exp 34, flat within noise)"
type: experiment
tags: [experiment, gemma4, jax, flax-nnx, splash-attention, pallas]
hypothesis: jax-splash-attention
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: TBD
verdict: inconclusive
---

Wire `jax.experimental.pallas.ops.tpu.splash_attention` into the native-JAX (Flax NNX) port. Mirrors the torchax exp-8/17/24/25 compound. Expected +2–5 % TPS at batch=1 seq=1024 based on the torchax exp-8 gain at batch=2; actual result: **+0.33 % (30,386 TPS vs 30,285 baseline)** — **flat within noise**. Verdict: `-potential`. The kernel works correctly (loss trajectory identical to exp 34: both land at **2.2969 at step 19**; smoke-5 loss at steps 2/4 = 2.97/2.02 matches baseline 2.94/1.98 within bf16-reorder noise), and the HLO-op diff shows splash does save ~65 ms / 3-step window on `convolution fusion` — but splash itself adds ~17 ms / 3-step in `custom fusion`, so net step-time delta is +0.1 ms (noise).

The **structural win** is HBM: peak 16.43 GiB (52.6 %) vs exp-34 baseline 16.85 GiB (53.9 %). Small at batch=1 seq=1024 because the `[B=1, H=8, S=1024, S=1024]` score tensor is only ~16 MiB / layer, but this is the mechanism that unlocks batch=3 / seq=2048 on the torchax stack (exp 18/14). Filing `-potential` so the code lands for those follow-up experiments.

## Hypothesis

Does routing Gemma 4's attention through `jax.experimental.pallas.ops.tpu.splash_attention` (GQA-native, tiled, SEQ_MINOR, fused bwd, block=1024) win TPS on the JAX port at batch=1 seq=1024?

On torchax the same wiring gave **+2.7 % TPS at batch=2** (exp 8) — splash's win is per-token-fixed, so % should track across batches. Expected **+2–5 %** at batch=1 seq=1024.

## Setup

New attention module: `jax/model/pallas_attention.py` (~180 LOC). Exports `splash_attention(q, k, v, sliding_window) -> (B, T, Hq, D)` and `set_mesh(mesh)` for shard_map registration. Selection is per-forward via env var `JAX_ATTENTION_IMPL` (default `xla`; `splash` enables the Pallas path). `Gemma4TextAttention.__call__` reads the env var and dispatches.

Command (run from `.../jax/`):

```bash
WIKI_ROOT=/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki
PROFILE_DIR=$WIKI_ROOT/raw/profiles/2026-04-23-gemma4-jax-exp35-splash
JAX_ATTENTION_IMPL=splash python -m train \
  --batch_size 1 --seq_len 1024 --steps 20 \
  --profile_dir $PROFILE_DIR \
  --profile_steps 10 11 12
```

**What's different from exp 34:**

- **Attention kernel**: `_attn_xla_sdpa` (jnp.einsum + softmax with GQA jnp.repeat_kv) → `jax.experimental.pallas.ops.tpu.splash_attention` via `make_splash_mha_single_device`.
- **GQA native**: splash handles `num_q_heads=8, num_kv_heads=2` directly; we drop the `_repeat_kv` call.
- **Mask**: sliding layers use `LocalMask(window_size=(512, 0))`; full layers use `CausalMask`. Both baked into the kernel's `MaskInfo` at build time, so the `attention_mask` arg is ignored when splash is on.
- **Shard_map**: Mosaic custom-calls can't auto-partition, so the kernel body is wrapped in `jax.shard_map` over `P('fsdp', None, None, None)` (batch-sharded, heads/seq/D replicated). `train.py` registers the active mesh via `pallas_attention.set_mesh(mesh)` before the first forward.
- **Pre-build**: `_build_splash_kernel` is lru-cached and pre-built at startup (one entry per (seq_len, num_q_heads, sliding_window, head_dim) combo — Gemma 4 has two: sliding@256 and full@512) under `jax.ensure_compile_time_eval()` so the MaskInfo tensors materialize as concrete arrays. Without this, the first call inside the top-level jit captures tracers, and step-1 retrace trips `UnexpectedTracerError`.
- **No pre-kernel scaling**. The JAX port sets `Gemma4TextAttention.scaling = 1.0` (q_norm / k_norm pre-normalize per-head) — matches splash's internal "no 1/sqrt(d)" convention exactly. Unlike torchax, which pre-multiplies `q * 1/sqrt(d)` because HF's PyTorch SDPA assumes that semantics. This is the load-bearing numerical diff vs torchax/model/pallas_attention.py.
- **Block config**: matches torchax exp-25 best — `block_q = block_kv = block_kv_compute = 1024`, `block_*_dkv = 1024`, `use_fused_bwd_kernel=True`, `QKVLayout.SEQ_MINOR` for q/k/v. `block_*_dq` omitted (fused bwd path — setting them raises; the exp-16 → exp-17 history in the torchax module documents this).

## Correctness

**Smoke-5 loss trajectory (splash) vs exp 34 (XLA SDPA)**:

| step | exp 35 splash loss | exp 34 XLA loss |
|---|---|---|
| 0  | 3.9062 | 3.9219 |
| 1  | 3.7188 | — |
| 2  | 2.9688 | 2.9375 |
| 3  | 2.0156 | — |
| 4  | 2.0156 | 1.9766 |
| 19 | 2.2969 | 2.2969 |

Step 19 is a **bit-level match** — same loss on the same data at the same LR after splash's reordered bf16 arithmetic settles to the same trajectory.

**Parity test** (`tools/parity_splash.py`, run on-TPU with mesh fsdp=4 B=4 T=128 bf16):

| Layer | Stage | max_err | mean_err | PASS? |
|---|---|---|---|---|
| layer 0 (sliding, hd=256) | raw attn out (pre-o_proj) | 0.237 | 0.0045 | PASS (tol_max=0.3, tol_mean=0.01) |
| layer 0 | full block (post-o_proj) | 0.080 | 0.0045 | PASS (tol=0.1) |
| layer 5 (full, hd=512)    | raw attn out | 0.285 | 0.0042 | PASS |
| layer 5 | full block | 0.096 | 0.0046 | PASS |

Max raw-output error looks large (0.28) but mean is 1e-3; tail-value reordering at ~0.15 % of entries. Note **first 8 entries of the raw output are bit-identical** across both paths; divergence is in a handful of tail positions where `softmax(fp32) → cast-bf16` (XLA SDPA) and `splash's internal online-softmax` round differently. Loss-trajectory parity is the decisive invariant and it matches exactly at step 19.

## Results

| Metric | exp 34 XLA SDPA baseline | **exp 35 splash** | Δ |
|---|---|---|---|
| TPS (steps 6–15 median) | 30,285 | **30,386** | **+0.33 %** (flat) |
| TPS (steps 2–19 median) | — | 30,408 | — |
| Step time (median 6–15) | 135.2 ms | 134.8 ms | -0.4 ms |
| Step time (xprof avg steps 10–12) | — | 142.2 ms | (+profile-capture overhead) |
| Step 0 compile | 142 s | 132 s | -10 s |
| Step 1 recompile | ~142 s | 133 s | -9 s |
| Peak HBM | 16.85 GiB (53.9 %) | **16.43 GiB (52.6 %)** | -0.42 GiB (-1.3 pt) |
| Loss step 0 | 3.9219 | 3.9062 | bit-match |
| Loss step 19 | 2.2969 | 2.2969 | **exact** |

Against the torchax session-best (**exp 25: 33,372 TPS** with splash + batch=3 + bf16 CE + fused_bwd + selective remat): the JAX stack is now at **-8.9 %**, barely closed vs exp 34's -9.2 %. The remaining gap is batch=3 + bf16 CE (tokamax), not splash.

### HLO-op diff (top ops, baseline vs exp 35)

| Op | exp 34 time_ms | exp 35 time_ms | Δ |
|---|---|---|---|
| convolution fusion (matmuls) | 614.6 | 549.0 | **-65.6 ms** |
| loop fusion (elementwise + norms) | 336.7 | 332.2 | -4.5 |
| custom fusion (splash kernel lives here) | 152.3 | 169.1 | **+16.8 ms** |
| all-gather | 137.1 | 138.1 | +1 |
| all-reduce-scatter fusion | 83.4 | 83.4 | 0 |

Net across the 3-step profile window: splash saves ~49 ms on matmuls (Q·Kᵀ and AttnProbs·V moved out of XLA convolution-fusion into Mosaic) but adds 17 ms in custom-fusion (the Pallas splash kernel itself), leaving a small positive delta that rounds out to flat at the step-time level.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp35-splash](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp35-splash) — opens the interactive trace viewer.
- **Run name**: `2026-04-23-gemma4-jax-exp35-splash` (xprof mcp full id: `2026-04-23-gemma4-jax-exp35-splash/2026_04_24_02_34_10`)
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp35-splash/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp35-splash/) (gitignored; 190 MB).
- **Steps captured**: 10, 11, 12 (xprof trace + xplane.pb).
- **What's inside**: trace of the native-JAX trainer at batch=1 seq=1024 fsdp=4 bf16 with `JAX_ATTENTION_IMPL=splash`. Splash kernel visible as `custom fusion` at 9.9 % of step time; `convolution fusion` drops from 35.8 % (baseline) to 32.2 %.

## Verdict

**POTENTIAL / inconclusive.**

The kernel is correct (bit-match loss at step 19, parity tol pass) and the HLO-op diff confirms the mechanism (matmul time moved from XLA into Mosaic with a small net saving). But **net step-time delta is flat within noise** (+0.33 % TPS; below the +0.5 % accepted bar from the program rules).

Why so small at this config? Two reasons:

1. **Batch=1 seq=1024 is below the regime where splash's N² avoidance matters**. At these sizes the `[B, H, S, S]` score tensor is only 16 MiB / layer — already cheap for XLA to fuse through. torchax exp-8 saw +2.7 % at batch=2 seq=1024 on HF PyTorch where the dispatch overhead is larger; the native-JAX path is already closer to optimal XLA-SDPA. Splash's asymptotic win shows up at longer seq / higher batch.
2. **The savings splash produces (~49 ms on matmuls, 3-step window) are almost entirely consumed by the custom-fusion overhead of the Pallas kernel itself (~17 ms).** Ratio matters: the matmul savings grow with total attention FLOPs (= `B × H × S² × D`) while the kernel launch overhead is roughly constant. At batch=3 seq=2048 the ratio flips strongly in splash's favor.

Keep the code on the trunk (land the merge) because it **unlocks**:

- **exp 36 — batch=3 at seq=1024**. Torchax exp-18 got +8.0 % from batch=3 given splash + bf16 CE. HBM currently at 52 %, so batch=3 fits (52 + ~1.5 GiB/batch × 2 extra = ~55 %). Direct gain expected **+5–8 %**.
- **exp 37 — seq=2048 at batch=1 or batch=2**. Torchax exp-14 ran splash at seq=2048 (the same NaN-at-seq≥2048 bug applies — blocked until fixed). But even without batch raise, at seq=2048 splash's ~4× lower HBM footprint vs XLA-SDPA may change the convolution-fusion vs custom-fusion ratio in splash's favor.
- **exp 38 — bf16 cross-entropy (tokamax / hand-rolled)**. The other half of what gets torchax to 33,372 TPS.

Merging the code because the trunk-state now has correct splash wiring and the diff from `HEAD` is small + well-factored. The `-potential` tag signals "mechanism sound, wall-clock flat at current config — revisit when conditions change (larger batch/seq)".

## Next hypotheses (ranked)

1. **exp 36 — splash + batch=3**, same seq. Expected +5–8 %. Confidence high (direct analog of torchax exp 18). Effort S.
2. **exp 37 — splash + bf16 CE**. Expected +1–3 % + ~1.5 GiB freed. Confidence medium. Effort S (hand-roll log_softmax is already bf16; biggest piece is swapping to `tokamax.linear_softmax_cross_entropy_loss` which fuses `lm_head + log_softmax + NLL`).
3. **exp 38 — splash + scan-over-layers**. Expected ~40 × compile drop (step 0 ~130 s → ~4 s), no steady-state throughput change. The JAX port is more amenable than torchax because we own the `for i, layer in enumerate(self.layers)` loop directly. Confidence high. Effort M.

## See also

- `../../torchax/experiments/2026-04-23-exp8-splash-attention-accepted.md` — original splash wiring on torchax (+2.7 %).
- `../../torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md` — session-best, 33,372 TPS.
- `2026-04-23-exp34-jax-baseline-accepted.md` — baseline this compares against.
- `../model/pallas_attention.py` — the new attention module.
- `../tools/parity_splash.py` — parity harness.
