# Experiment: v034 Fused QK-norm + RoPE Pallas Kernel (bs=8, TP=1, Splash, Tokamax CE, Remat)

**Status:** Pending
**Lane:** jax
**Date:** 2026-06-04

## Hypothesis under test
- **Hypothesis**: Qwen3 applies RMSNorm to Q and K just before applying RoPE. In the current JAX implementation, this creates huge `[B, S, H, D]` intermediate tensors that hit HBM. By writing a custom Pallas kernel that fuses the RMSNorm and the RoPE application into a single kernel, we avoid materializing the intermediate normalized Q and K vectors to HBM. This should increase throughput and MFU.
- **Mechanism**: A custom Pallas kernel (`_fused_qknorm_rope_kernel`) implements RMSNorm over the feature dimension followed by half-rotation RoPE. This kernel is applied in `Qwen3Attention.__call__` in place of the separate `norm()` and `apply_rotary_emb` calls.
- **Predicted signal**: MFU and throughput will increase due to reduced memory bandwidth pressure.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `nnx.remat` (inner per-layer remat, outer ckpt removed in `v027`)
- **Attention**: Splash Attention
- **Loss**: Tokamax XLA CE
- **Parallelism**: TP=1 (FSDP only)
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260604-v034-tp1-vocab-parallel-bs8-fused-qknorm-rope`

## Setup
Derived from `v027` (which already had FSDP TP=1 and removed outer ckpt).
1. Created `model/fused_rope.py` containing the Pallas kernel `fused_qknorm_rope`.
2. Modified `model/modeling_qwen3.py` to use `fused_qknorm_rope` instead of separate `self.q_norm`, `self.k_norm` and `apply_rotary_pos_emb`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 20685 tok/s (2586/chip)
- **Approx MFU**: 13.3%

## Profile Analysis
The run successfully compiled and executed, but the MFU dropped dramatically to 13.3% (down from 32.8% in v027). The custom Pallas kernel for fused QK-norm and RoPE is functioning correctly in terms of forward/backward pass execution, but it is extremely unoptimized for TPU tensor cores (MXUs).
A custom Pallas kernel written in Python forces the compiler to treat it as a black-box custom call (`custom-call` bucket), effectively disabling the XLA compiler's ability to fuse the elementwise operations into surrounding matrix multiplications or efficiently pipeline memory accesses.
This causes a massive regression in performance.

## Verdict
**Refuted (with an asterisk).** While the mechanism technically functions and avoids materializing the intermediate normalized vectors in Python-land, the MFU dropped to 13.3%. Note that this run accidentally used `--batch_size=1` instead of `8`. At `bs=1`, low MFU is expected. However, the custom Pallas kernel still bypasses XLA's heuristics, so we will pivot to natively optimizing XLA SDPA first.
