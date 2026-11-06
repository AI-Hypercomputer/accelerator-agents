# MaxKernel Custom Kernel Onboarding Guide

This guide walks you through onboarding a custom kernel into **MaxKernel** for autonomous parallel search and continuous hill-climbing optimization on Cloud TPUs.

---

## 1. Overview

MaxKernel optimizes kernels without requiring modifications to standard benchmark suites (such as JAXBench). By providing a minimal, self-contained directory with a reference baseline and sample input shapes, the MaxKernel search engine will autonomously explore Pallas kernel architectures, memory layouts, tiling strategies, and pipelining schedules.

### Supported Modes:
* **Greenfield Optimization**: MaxKernel writes a Pallas kernel from scratch based on a pure JAX reference function.
* **Continuous Hill-Climbing**: MaxKernel takes an existing Pallas kernel (e.g., an MLPerf baseline or experimental implementation) as a starting seed (`initial_kernel.py`) and focuses exclusively on improving throughput, memory footprint, and latency.

---

## 2. Directory Structure

A custom kernel workload is defined by a directory containing at least `reference.py`:

```
my_custom_kernel/
├── reference.py             # REQUIRED: Ground-truth reference implementation and input generator
├── initial_kernel.py        # OPTIONAL: Existing Pallas/JAX seed kernel to hill-climb from
└── kernel_task.yaml         # OPTIONAL: Numerical verification tolerances and task metadata
```

---

## 3. The `reference.py` Contract

The `reference.py` file serves as the oracle for both input shape generation and numerical correctness verification. It must define two functions:

### 1. `get_inputs()`
Generates concrete sample input tensors with the exact dimensions, batch sizes, sequence lengths, and data types (e.g. `bfloat16`, `float8_e4m3fn`) you want to target.

### 2. `computation(*dynamic_args, *static_args)`
The reference functional computation. This does not need to be performant; standard JAX operations (`jax.lax`, `jnp.matmul`, `jnp.einsum`) are preferred. The search engine tests candidate Pallas kernels against this function to verify numerical equivalence.

### Copy-Paste Template:
```python
"""Reference implementation and input generator for custom kernel optimization."""

import jax
import jax.numpy as jnp


def get_inputs():
  """Generates sample inputs representing production workloads.

  Returns:
    tuple: (dynamic_args, static_args)
      - dynamic_args: List of jax.Array tensors passed to the kernel.
      - static_args: List of compile-time static parameters (integers, strings, booleans).
  """
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key)

  # Define your production shapes:
  batch_size = 8
  hidden_dim = 4096
  out_dim = 4096

  lhs = jax.random.normal(k1, (batch_size, hidden_dim), dtype=dtype)
  rhs = jax.random.normal(k2, (hidden_dim, out_dim), dtype=dtype)

  dynamic_args = [lhs, rhs]
  static_args = []
  return dynamic_args, static_args


def computation(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Reference functional computation used as the correctness oracle."""
  return jnp.matmul(lhs, rhs)
```

---

## 4. Seeding with an Existing Candidate (`initial_kernel.py`)

If you already have a functional or partially optimized Pallas kernel (e.g. from a prior run, research branch, or baseline library) and want MaxKernel to optimize it further, include `initial_kernel.py`.

* **How It Works**: The orchestrator injects `initial_kernel.py` directly into the search graph at depth 0 with strategy `seed_kernel`.
* **Worker Prompting**: Autonomous agent workers will read `optimized_kernel.py` (your seed), understand the existing tile sizes, buffers, and pipeline stages, and hill-climb from that baseline rather than starting from a blank canvas.

---

## 5. Optional Configuration: `kernel_task.yaml`

To customize numerical validation tolerances, place a `kernel_task.yaml` file in the same directory:

```yaml
task_id: my_moe_fused_gmm
description: Fused token-sorting grouped matmul for MoE
rtol: 0.01
atol: 0.05
```

If omitted, MaxKernel uses standard default tolerances (`rtol=1e-2`, `atol=5e-2` for `bfloat16`).

---

## 6. Running Parallel Search

MaxKernel's `auto_search.run_search` executes parallel tree search across concurrent worker agents.

### Key Search Knobs:

| Parameter | Flag | Description | Recommended |
|---|---|---|---|
| **Max Iterations** | `--agent_config '{"max_iterations": N}'` | Number of refinement cycles each agent runs per task | `5` to `10` |
| **Backend Model** | `--agent_config '{"model_name": "..."}'` | Generative model driving code synthesis | `gemini-3.6-flash` |
| **Parallel Runs** | `--num_parallel_runs` | Total candidate branches to explore in the search graph | `6` to `20` |
| **Concurrency** | `--max_concurrency` | Number of agent sessions running concurrently | `2` to `4` |
| **Seed Kernel** | `--initial_kernel_path` | Path to existing Pallas kernel to hill-climb from | `/path/to/initial_kernel.py` |
| **Output Path** | `--optimized_file_path` | Path to save the best performing candidate kernel | `/path/to/optimized_kernel.py` |

### CLI Execution Command:
```bash
python -m auto_search.run_search \
  --reference_file_path=/path/to/my_custom_kernel/reference.py \
  --initial_kernel_path=/path/to/my_custom_kernel/initial_kernel.py \
  --algorithm=parallel \
  --num_parallel_runs=10 \
  --max_concurrency=2 \
  --agent_config='{"max_iterations": 5, "model_name": "gemini-3.6-flash"}' \
  --optimized_file_path=/path/to/my_custom_kernel/optimized_kernel.py \
  --graph_db_path=/path/to/my_custom_kernel/search_graph.json
```

---

## 7. Cloud TPU Cluster Execution

When running on managed cluster infrastructure:

1. **Upload your custom kernel folder to Google Cloud Storage (GCS)**:
   ```bash
   gcloud storage cp -r ./my_custom_kernel/ gs://my-bucket/custom_kernels/my_kernel/
   ```

2. **Launch the optimization workload with overrides**:
   * Set `custom_kernel_gcs=gs://my-bucket/custom_kernels/my_kernel/`
   * Set `initial_kernel=gs://my-bucket/custom_kernels/my_kernel/initial_kernel.py` (optional)
   * Configure search iterations: `max_iterations=5` and `num_parallel_runs=10`
   * Select hardware topology: Cloud TPU v7x (`v7x-2x2x1`), Cloud TPU v6e (`v6e-1`), or TPU v5p (`v5p-2x2x1`).

The automated job will download the reference inputs, initialize the search frontier, stream lineage telemetry, and persist the highest speedup kernel back to GCS.
