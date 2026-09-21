<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "TPU Optimization — Running Large Models in Dummy/Random Weight Mode"
type: analysis
tags: [analysis, tpu, dummy-weights, compile-time, profiling, roofline, autotune, maxperf]
created: 2026-05-12
updated: 2026-05-12
---

## Headline Comparison

| Aspect | Real Checkpoint serving | Dummy/Random Weight serving | Impact |
|---|---|---|---|
| **Model Checkpoint Size** | ~480 GB (FP8) / ~960 GB (BF16) | **0 Bytes** | **Bypasses 480 GB GCS/HuggingFace download** |
| **TPU Dev Warmup time** | ~10–15 minutes (weights streaming) | **< 1 minute** | **Significant developer velocity increase** |
| **XLA/HLO compilation** | Shape-based compilation | Shape-based compilation | **Identical HLO graph fusions and machine code** |
| **HBM allocation** | Memory allocated by parameter shapes | Memory allocated by parameter shapes | **Identical VMEM/HBM memory footprints and OOM triggers** |
| **Throughput (TPS/chip)** | Model runs | Model runs | **Mathematically identical roofline & memory bandwidth** |
| **Evaluation Accuracy** | Normal accuracy benchmarks | Gibberish output | Cannot run HumanEval or MBPP |
| **Numeric Equivalence** | Normal hash outputs | Gibberish hash outputs | Cannot pass exact numeric-equivalence tests |

---

## Why This Works in JAX/Flax & XLA

JAX and the Cloud TPU runtime have a **highly functional, shape-driven architecture**:
1. **JIT Tracing (`jax.jit`)**: The compiler traces and compiles model operations using abstract shapes and data types (e.g., `ShapeDtypeStruct(shape=(64, 8192, 2048), dtype=jnp.bfloat16)`) rather than the actual weight values.
2. **Memory Layout & Alignments**: TPU vector registers (VREG) and High-Bandwidth Memory (HBM) layouts are aligned to dimension multiples (such as batch size, attention heads, vector lanes) to optimize systolic Matrix Unit (MXU) hardware utilization.
3. **Execution Parity**: An operation like `matmul(X, W)` executes the exact same assembly instruction pattern regardless of whether `W` contains trained parameters or random numbers. Thus, register thrashing (VREG spills) and DMA idling are completely identical.

---

## How to Execute the Model in Dummy Mode

### 1. TPU Serving via vLLM Engine
To served or benchmark `Qwen3-Coder-480B-A35B-Instruct-FP8` without loading a checkpoint, use the `--load-format dummy` parameter in the launch command:

```bash
# Serving serving without checkpoint downloads
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --load-format dummy \
    --tensor-parallel-size 16 \
    --max-model-len 8192 \
    --max-num-seqs 64
```
*See standard deployment launcher template at `runs/run_dummy.sh`.*

### 2. Standard JAX/Flax Model Invocations
In Flax/MaxText codebases, initialize dummy weights natively using a JAX PRNG key:

```python
import jax
import jax.numpy as jnp
from qwen3next_jax.config import Qwen3NextConfig
from qwen3next_jax.maxtext.modeling import ShardedQwen3NextForCausalLM

# Load metadata configuration
config = Qwen3NextConfig.from_pretrained("~/checkpoints/qwen3next/")
model = ShardedQwen3NextForCausalLM(config, use_remat=True)

# Random weight initialization directly across the sharded TPU device mesh
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
params = model.init(rng, dummy_input, deterministic=True)
```

---

## Integration with the MaxPerf Sub-Agents

This dummy mode is strategically integrated into the 12-step optimization loop:
* **TPUDiagnoseAgent**: When parsing xprof timeline traces, TPU unit utilization, or collective communication stalls (all-reduce, reduce-scatter), dummy mode allows capturing the identical roofline metrics.
* **MaxTile Sub-Agent**: Running search/autotuning sweeps over block sizes can be done instantly with dummy weights to isolate compilation and tile-level efficiencies.
* **MaxKernel**: Can perform the preliminary HLO scope test and VMEM boundary cost tests prior to attempting a full trained-weight validation loop.

---

## Related Documents & References

* **Model Executor Source Code (vLLM)**: [qwen3_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py) (High-level model execution flow in vLLM, mapping variables and sharding logic prior to calling JAX/TPU Pallas custom kernels).
* **Launch Script Template**: [runs/run_dummy.sh](../runs/run_dummy.sh)
* **TPU Execution & Monitoring Script**: [tpu_run_model.sh](../../tpu_run_model.sh) (Detailed TPU production launcher containing serving setups, FP8 KV-cache options, sharded ray executor configurations, and standard bench_serving client benchmark loop).
* **Central program contract**: [program.md](../program.md) (Target, Baseline and fixed bindings)
* **Wiki Index**: [wiki/index.md](../wiki/index.md)
