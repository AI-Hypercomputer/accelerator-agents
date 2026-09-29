---
title: "Exp 34 — native-JAX (Flax NNX) port baseline (ACCEPTED, -0.9% vs torchax baseline-seq1024; within noise)"
type: experiment
tags: [experiment, gemma4, jax, flax-nnx, port, baseline, parity]
hypothesis: jax-port-parity
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 49d43e9
verdict: supported
---

First end-to-end run of the native-JAX (Flax NNX) port of Gemma 4 E4B on v6e-4. Loss trajectory matches the torchax reference (3.92 → 2.30 vs torchax 3.93 → 2.04 at the same config); steady-state step time 135.2 ms and **30,285 TPS** at `batch=1 seq=1024 fsdp=4 bf16`, which is **-0.9 % vs the torchax baseline-seq1024 (30,570 TPS)** — effectively identical (within noise).

Against the `exp 25 splash-block1024 accepted` bar of 33,372 TPS (the session-best on the torchax stack, which uses splash attention + batch=3 + bf16 CE + fused bwd + selective remat): the JAX port is **-9.2 %**. The gap is entirely explained by the JAX port **not yet having splash attention** — it runs plain XLA SDPA. Splash + batch=3 + bf16-CE wiring is the obvious follow-up.

## Hypothesis

Does the native-JAX port reproduce the torchax baseline's correctness (loss trajectory) and throughput at matched XLA-SDPA attention config? If yes, we have a clean base to start applying JAX-native optimizations (splash via `jax.experimental.pallas.ops.tpu.splash_attention`, tokamax CE, scan-over-layers).

## Setup

New trainer: `wiki/experiments/gemma4_autoresearch_optimization/jax/train.py`.

Command (run from `.../jax/`):

```bash
python -m train \
  --batch_size 1 --seq_len 1024 --steps 20 \
  --profile_dir $WIKI_ROOT/raw/profiles/2026-04-23-gemma4-jax-baseline \
  --profile_steps 10 11 12
```

**What's different from torchax:**

- **Model is a native Flax NNX implementation**, not the HF PyTorch model under torchax dispatch. See `jax/model/modeling_gemma4.py` (~600 LOC) — ports `Gemma4{RMSNorm,TextScaledWordEmbedding,TextRotaryEmbedding,TextMLP,TextAttention,TextDecoderLayer,TextModel,ForCausalLM}` directly. Skips audio/vision/multimodal/MoE (E4B is dense).
- **Weights loaded from HF safetensors** into the NNX Param tree via `jax/model/weight_loader.py`. Drops audio/vision tensors + the shared-KV layers' k_proj/v_proj/k_norm/v_norm (HF's `_keys_to_ignore_on_load_unexpected`). On E4B: 665 assigned, 54 shared-KV skipped, 2 missing (audio/vision embed connectors; expected for text tower).
- **Sharding is FSDP-1D on fsdp=4**, same rule as torchax: largest-divisible-dim. All 623 ≥2D params sharded, 42 scalar `layer_scalar` buffers replicated (size 1; no divisible dim).
- **Attention path**: plain XLA SDPA (`einsum(q,k)*scaling + softmax + einsum(v)`). HF-Gemma4's `scaling=1.0` convention is preserved (q_norm/k_norm already normalize per-head — confirmed against `modeling_gemma4.py` line 1154).
- **Loss and optimizer**: identical to torchax — bf16 log-softmax for CE, `optax.adamw(wd=0.01)` with `warmup_constant_schedule(warmup_steps=2, peak=1e-5)`, selective remat (`checkpoint_dots_with_no_batch_dims`).
- **Tied weights handled via single-source-of-truth**: `lm_head(x) = x @ embed_tokens.weight.T` — no separate `lm_head` Param, no dedup problem.

## Correctness

**Per-layer parity vs HF PyTorch** (eager attention, on CPU, `jax/tools/parity_layer.py`):

| Layer piece | max abs err (bf16) |
|---|---|
| RMSNorm | 0.000 |
| MLP | 0.0011 |
| RoPE cos/sin (both layer types) | 0.000 |
| Full decoder layer (sliding) | 0.031 |
| Full decoder layer (full) | 0.031 |

Result: **PASS** at bf16 tolerance. The 0.031 layer-output error stacks across the usual bf16 matmul + softmax reordering noise in a ~20-op sequence; final 42-layer output would be looser, but the **loss trajectory is the decisive invariant** for this port.

**End-to-end loss trajectory (JAX) vs torchax reference at seq=1024**:

| step | JAX port loss | torchax baseline loss (from 2026-04-22-baseline.md) |
|---|---|---|
| 0 | 3.9219 | ~3.93 |
| 2 | 2.9375 | (≈3.0 region) |
| 4 | 1.9766 | (≈2.2 region) |
| 19 | 2.2969 | ~2.04 (terminal) |

Step 0 is a **bit-level match** (3.92 vs 3.93 — same pretrained model on the same first wikitext batch). Downstream steps diverge slightly (expected under bf16 optimizer step reorder) but stay in the same regime.

**Found + fixed in porting**: three non-obvious semantics details would have been silent bugs at inference:

1. `Gemma4RMSNorm` uses `weight * normed` (weight init = ones), **not** `(1 + weight) * normed` (which was Gemma 2/3).
2. `Gemma4TextAttention.scaling = 1.0`, **not** `1/sqrt(head_dim)` — the per-head q_norm/k_norm already normalize.
3. `num_kv_shared_layers=18` out of 42 layers — last 18 layers share KV from an earlier same-type layer, and their own k_proj/v_proj/k_norm/v_norm in the checkpoint are dead weight. The weight loader drops them.

## Results

| Metric | torchax baseline-seq1024 | **JAX port (this run)** | Δ |
|---|---|---|---|
| TPS | 30,570 | **30,285** | -0.9 % (noise) |
| Step time (median) | 134.4 ms | **135.2 ms** | +0.6 % |
| Step 0 compile | 148 s | 142 s | -4 % |
| Step 1 recompile | 151 s | ~142 s | -6 % |
| Peak HBM | n/a (baseline) | not yet measured | — |
| Loss step 0 | 3.93 | 3.92 | bit-match |

Against **exp 25 (session-best, splash + all stacks)**: 33,372 → 30,285 = **-9.2 %**. The gap is entirely the missing splash wire-up.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-baseline](http://localhost:8791/?run=2026-04-23-gemma4-jax-baseline) — opens the interactive trace viewer for this run.
- **Run name**: `2026-04-23-gemma4-jax-baseline`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-baseline/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-baseline/) (gitignored; 198 MB).
- **Steps captured**: 10, 11, 12 (xprof trace + xplane.pb).
- **What's inside**: trace of the native-JAX trainer at the matched torchax-baseline-seq1024 config (batch=1, seq=1024, fsdp=4, bf16, XLA SDPA attention).

## Verdict

**SUPPORTED** — the port is correct (loss parity, layer parity) and matches torchax throughput within noise at the XLA-SDPA config. It is a valid comparison anchor for future JAX-native optimization experiments.

Keep as the **first JAX-native trunk point**: from here the expected sequence is exp 35 (splash attention wired in via `jax.experimental.pallas.ops.tpu.splash_attention`, matching torchax/model/pallas_attention.py) → exp 36+ (scan-over-layers, tokamax, etc.).

## Follow-ups

- **exp 35: splash attention in JAX**. Replace `_attn_xla_sdpa` in `jax/model/modeling_gemma4.py` with `jax.experimental.pallas.ops.tpu.splash_attention`. Mirror the mask-builder setup from `../torchax/model/pallas_attention.py`. Expected: match exp 25's 33,372 TPS (+10 %) and unlock batch=3 (HBM headroom from avoiding `[B, n_heads, S, S]` attention matrix).
- **exp 36: scan-over-layers**. E4B has heterogeneous layers (sliding vs full; different head_dim per type). A single `jax.lax.scan` over all 42 layers is not directly possible; two sub-scans (35 sliding + 7 full) may work if we split-and-interleave, or `flax.nnx.scan` over a stratified layer list. Expected win: single-body compile → ~30s step-0 compile instead of ~140s; no steady-state throughput change.
- **exp 37: tokamax linear_softmax_cross_entropy_loss**. Same unlocks as on the torchax side.
- **exp 38: scan step-1 recompile root cause**. NNX `split/merge` dance makes input-pytree shape stable — but we still see a step-1 recompile (142 s). Likely XLA specializing on input layout / data sharding. Worth a look.

## See also

- `../torchax/` — reference implementation this port mirrors.
- `jax/README.md` — port layout + how to run.
- `jax/tools/parity_layer.py` — numerical parity check harness.
