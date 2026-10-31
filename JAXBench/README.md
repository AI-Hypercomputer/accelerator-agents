# JAXBench

**JAXBench** is a benchmark suite of 50 curated JAX/TPU kernel workloads with a
production-ready evaluation harness. It is designed for benchmarking and
evaluating AI-generated TPU kernel optimizations.

## Workloads

The suite contains 50 workloads organized into two tiers:

- **17 Priority Kernels** (`1p`–`17p`): Production operators from models like
  Llama-3.1, DeepSeek-V3, Mixtral-8x7B, Mamba-2, RetNet, and AlphaFold2.
  Includes attention variants, MoE, GEMM, normalization, and more. 8 of these
  have hand-optimized Pallas TPU kernel variants.

- **33 KernelBench Workloads** (`18k`–`50k`): Fused operator patterns from
  [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 2,
  covering matmul+activation chains, convolutions with normalization, and
  multi-op fusions.

Each workload follows a consistent interface:

```python
CONFIG = {...}                              # Metadata (model, operator, dimensions)
def create_inputs(dtype=jnp.bfloat16): ...  # Generate input tensors
def workload(*inputs): ...                  # The computation to benchmark
def get_flops(): ...                        # Optional: manual FLOP count
```

## Quick Start

From the repository root (`accelerator-agents/`):

```bash
# List all workloads
python -m JAXBench list

# Run a single workload on TPU (baseline + optimized if present)
python -m JAXBench run --workload 8p_GEMM --tpu v6e

# Run all 50 workloads
python -m JAXBench run --all --tpu v6e

# Evaluate a candidate kernel against a workload
python -m JAXBench evaluate --workload 1p_Flash_Attention --kernel path/to/my_kernel.py --json
```

### Requirements

- Python 3.10+
- JAX with TPU support (`jax[tpu]`)
- A TPU VM (v5e or v6e)

## Evaluation Harness

The harness provides two primary interfaces:

### 1. Agent Evaluation (`evaluate`)

Evaluate a candidate kernel file against any workload. The kernel file only
needs to export a `workload(*inputs)` function — inputs are generated from the
baseline's `create_inputs()`.

```bash
python -m JAXBench evaluate \
    --workload 1p_Flash_Attention \
    --kernel path/to/my_kernel.py \
    --tpu v6e \
    --json
```

The evaluation pipeline:
1. **Compile** — JIT-compiles the candidate kernel
2. **Correctness** — Checks integer outputs for exact equality and floating outputs
   against baseline with `np.allclose` (atol=1e-2, rtol=1e-2). Integer outputs
   retain their precision, including int64/uint64 values; integer and non-integer
   output dtypes cannot be mixed. Equal integer values may use different widths
   or signedness, and floating-point tolerances do not apply to integer outputs.
3. **Benchmark** — Device-side profiling via `jax.profiler.trace()` (Perfetto traces)
4. **Compare** — Reports speedup vs baseline XLA and Pallas reference (if present)

Output (with `--json`):

```json
{
  "workload": "8p_GEMM",
  "status": "correct",
  "correctness": {"correct": true, "max_diff": 0.0003, "reason": "ok"},
  "baseline": {"median_ms": 5.32, "tflops": 723.1, "utilization_pct": 78.8},
  "kernel": {"median_ms": 4.10, "tflops": 938.1, "utilization_pct": 90.2},
  "pallas_reference": {"median_ms": 5.41, "tflops": 711.9, "utilization_pct": 77.6},
  "speedup_vs_baseline": 1.30,
  "speedup_vs_pallas": 1.32,
  "tpu": "v6e"
}
```

### 2. Benchmark Runner (`run`)

Reproduce benchmark numbers for all workloads:

```bash
python -m JAXBench run --all --tpu v6e
```

This runs all 50 baselines and 8 optimized variants, producing
`results.json` and `results.csv` with timing, TFLOPS, and MXU utilization.

### Python API

```python
from JAXBench.harness import evaluate_kernel, run_workload, run_all

# Evaluate a kernel
result = evaluate_kernel("8p_GEMM", "path/to/kernel.py", tpu="v6e")

# Run a single workload
result = run_workload("8p_GEMM", variant="baseline", tpu="v6e")

# Run all workloads
results = run_all(tpu="v6e")
```

## Writing a Kernel for JAXBench

To optimize a workload, create a Python file that exports a single function:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(x, weight):
    # Your optimized implementation here
    # Receives the same inputs as the baseline's create_inputs()
    return result
```

Then evaluate it:

```bash
python -m JAXBench evaluate --workload 8p_GEMM --kernel my_kernel.py --json
```

## Timing Methodology

All benchmarks use **device-side timing** via `jax.profiler.trace()`:

- Captures Perfetto JSON traces during benchmark iterations
- Extracts `jit_*()` wrapper event durations (actual TPU kernel execution time)
- Excludes host dispatch overhead, Python overhead, and data transfer
- Falls back to wall-clock timing only for eager workloads (2 of 50)

## TPU Support

| TPU     | Peak TFLOPS (bf16) | HBM Bandwidth |
|---------|-------------------|----------------|
| v5e     | 197               | 819 GB/s       |
| v6e     | 918               | 1,600 GB/s     |

Use `--tpu auto` (default) to auto-detect from `jax.devices()`.

## Correctness Regression Tests

The correctness regression tests run on CPU without a TPU. From the repository
root, use an environment with NumPy and a CPU-capable JAX installation:

```bash
JAX_PLATFORMS=cpu python -B -m unittest discover -s JAXBench/tests -v
```

The tests cover exact integer comparison, integer dtype compatibility, large
values and signed/unsigned boundaries, overflow-safe error reporting, and
floating-point tolerance behavior. The 64-bit boundary cases use NumPy arrays
to preserve their values independently of JAX's `jax_enable_x64` setting.
Evaluator integration tests execute JAX computations on CPU and verify that
incorrect candidates are rejected before benchmarking. Workload discovery, TPU
metadata, and timing are stubbed in those tests; they do not measure TPU
performance or validate TPU kernel execution.

## Directory Structure

```
JAXBench/
├── __init__.py
├── __main__.py              # CLI entry point
├── README.md
├── benchmark/               # 50 workloads
│   ├── __init__.py          # Workload discovery
│   ├── 1p_Flash_Attention/
│   │   ├── baseline.py      # Reference JAX implementation
│   │   └── optimized.py     # Hand-tuned Pallas kernel
│   ├── 8p_GEMM/
│   │   ├── baseline.py
│   │   └── optimized.py
│   ├── 19k_Matmul_.../
│   │   └── baseline.py      # No optimized variant yet
│   └── ...
└── harness/
    ├── __init__.py
    ├── correctness.py       # Exact integers; floats: atol=1e-2, rtol=1e-2
    ├── evaluator.py         # Agent evaluation pipeline
    ├── loader.py            # Dynamic module loading
    ├── profiler.py          # Device-side profiling (Perfetto)
    ├── runner.py            # Benchmark runner
    └── tpu_specs.py         # TPU hardware specs
```
