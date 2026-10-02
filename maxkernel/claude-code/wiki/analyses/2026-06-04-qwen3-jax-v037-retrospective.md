---
title: "v037 Retrospective: Splash and Chunked XLA Hang"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Retrospective: v037 (Splash + Chunked XLA Hang)

## The Symptom
In experiment `v037`, we attempted to run Qwen3-8B with both Splash attention and Tokamax's `chunked_xla` cross-entropy loss enabled. The run successfully compiled and executed step 0 (which took ~9 minutes), but then completely stalled on step 1. The logs repeatedly printed `WARNING:absl:Autotuning cache miss for ChunkedXlaLinearSoftmaxCrossEntropyLoss`, after which CPU usage dropped to near-zero and no further steps were executed.

## Root Cause Analysis
The stall was **not** an autotuning hang, nor was it a deadlock within the Splash kernel itself. The root cause lies in the interaction between JAX's asynchronous dispatch, XLA's handling of massive loop states, and `chunked_xla`'s backward pass design.

1. **The Autotuning Red Herring**: In `v037`, we patched the Tokamax autotuner to return an empty dictionary (`{}`) to bypass the slow compilation sweeps. When the Tokamax `Op` class received an empty cache response, it printed the "cache miss" warning and silently fell back to its default heuristics configuration (`b_block_sz=1024, v_block_sz=2048`).
2. **The 9,600-Iteration Loop**: With those block sizes, the backward pass of `chunked_xla` constructs a nested `jax.lax.fori_loop`. For our vocab size (151,936) and local batch/sequence dimensions, this results in 128 outer loops and 75 inner loops — totaling **9,600 iterations**.
3. **Massive Loop Carry State**: During these 9,600 iterations, the inner loop carries and updates a 1.24 GB gradient tensor (`dw_acc`).
4. **The Queue Overflow**: Step 0 executed successfully because `train.py` explicitly calls `jax.block_until_ready()` on the first step. However, for steps 1 through 20, the Python host eagerly dispatches the graph asynchronously. Dispatching multiple steps means enqueuing tens of thousands of `dynamic_update_slice` operations on a 1.24 GB tensor. This overflows the TPU command queue and causes severe HBM fragmentation, causing the TPU runtime to stall completely.

## The Solution
`chunked_xla` is an architectural mismatch for TPU when sequence lengths and vocabulary sizes produce high loop counts, as TPUs strongly prefer unrolled, dense matrix multiplications over fragmented loop state updates.

Furthermore, `chunked_xla` is unnecessary for our target configuration. At `TP=1` (pure FSDP), the standard dense `xla` loss implementation successfully avoids OOMs because XLA SPMD correctly shards the logits matrix without materializing the full 39.8 GB tensor locally.

**Next Steps**: We will formally deprecate `chunked_xla` for this architecture. To achieve our >35% MFU goal, we will pivot to optimizing the XLA fusions within the model itself (e.g., fusing RoPE and QK-norm prior to layout transposition) while retaining the dense XLA cross-entropy loss.
