# MaxKernel Evaluation Module

The evaluation module in MaxKernel provides tools to benchmark and evaluate JAX kernels on remote TPU VMs (or locally). It allows you to compare a reference implementation with an optimized implementation, verify correctness, and measure performance (speedup).

## Overview

The module consists of two main components:
1.  **`jax_kernel_evaluator.py`**: A Python module containing the `JAXKernelEvaluator` class, which handles the orchestration of running evaluations on a remote TPU VM or locally.
2.  **`benchmark.py`**: A Command Line Interface (CLI) tool that automates the evaluation process across a dataset of problems.

---

## JAXKernelEvaluator

The `JAXKernelEvaluator` class is responsible for setting up the environment, uploading necessary files to a remote TPU VM (if applicable), executing the evaluation harness, and parsing the results.

### Key Features
-   Supports both **remote** evaluation (on a TPU VM via SSH) and **local** evaluation.
-   Automates file transfer and command execution on remote TPUs.
-   Includes an optional **adaptation** step using the Gemini API to refactor code or generate task configurations if they are missing.
-   Handles timeouts and attempts to clean up runaway processes on remote VMs.

### Programmatic Usage

Here is an example of how to use `JAXKernelEvaluator` in a Python script:

```python
from evaluation.jax_kernel_evaluator import JAXKernelEvaluator

# Initialize the evaluator for remote execution
evaluator = JAXKernelEvaluator(
  local=False,
  tpu_name="your-tpu-name",
  project="your-gcp-project",
  zone="your-tpu-zone",
  venv_path="/path/to/venv/on/tpu",
)

# Run evaluation
result = evaluator.evaluate(
  reference_code_path="path/to/reference.py",
  optimized_code_path="path/to/optimized.py",
  task_yaml_path="path/to/kernel_task.yaml",
  atol=1e-3,
  rtol=1e-3,
)

print(f"Task ID: {result.task_id}")
print(f"Correctness: {result.correctness}")
print(f"Speedup: {result.speedup}x")
if result.error_trace:
  print(f"Error: {result.error_trace}")
```

### Initialization Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `local` | `bool` | If `True`, runs evaluation locally. Default is `False`. |
| `tpu_name` | `str` | Name of the TPU VM to connect to for remote evaluation. |
| `project` | `str` | Google Cloud project ID. |
| `zone` | `str` | Google Cloud zone. |
| `venv_path` | `str` | Path to the Python virtual environment locally or on the remote TPU. |
| `remote_base_dir` | `str` | Base directory on remote TPU for storing temporary files. |

### CLI Usage

> [!NOTE]
> All CLI commands should be run from the project root directory (`MaxKernel`).

You can also use `jax_kernel_evaluator.py` directly from the command line to evaluate a single kernel without setting up a dataset.

**Example:**

```bash
python3 -m evaluation.jax_kernel_evaluator \
    --reference_code_path path/to/reference.py \
    --optimized_code_path path/to/optimized.py \
    --task_yaml_path path/to/kernel_task.yaml \
    --local
```

**Arguments:**

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--reference_code_path` | `str` | (Required) Local path to the reference JAX script. |
| `--optimized_code_path` | `str` | (Required) Local path to the optimized JAX script. |
| `--task_yaml_path` | `str` | Path to the kernel_task.yaml file. |
| `--local` | Flag | Run evaluation locally instead of on a remote TPU. |
| `--tpu_name` | `str` | Name of the TPU VM (required if not local). |
| `--project` | `str` | Google Cloud project ID. |
| `--zone` | `str` | Google Cloud zone. |
| `--venv_path` | `str` | Path to the Python virtual environment. |
| `--remote_base_dir` | `str` | Base directory on remote TPU for temp files. |
| `--adapt` | `str` list | Components to adapt via LLM (`reference_code`, `optimized_code`, `kernel_task`). |
| `--timeout_seconds` | `int` | Maximum time in seconds to wait for execution (default: 300). |
| `--no_cleanup` | Flag | Skip cleanup of temporary files on the remote VM. |
| `--atol` | `float` | Absolute tolerance for correctness check (default: 1e-3). |
| `--rtol` | `float` | Relative tolerance for correctness check (default: 1e-3). |

---

## Benchmark CLI (`benchmark.py`)

The `benchmark.py` script is a command-line tool that iterates through a dataset of problems and evaluates them using `JAXKernelEvaluator`.

### Dataset Structure

To use the benchmark CLI, your dataset directory should have the following structure:

```text
dataset_dir/
├── problem_1/
│   ├── reference.py
│   ├── optimized.py
│   └── kernel_task.yaml
├── problem_2/
│   ├── reference.py
│   ├── optimized.py
│   └── kernel_task.yaml
...
```

### `kernel_task.yaml` fields

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Task identifier (should match the directory name). |
| `input_gen_code` | `str` | Source of `get_inputs()`, returning `(dynamic_args, static_args)` or a list of such pairs. |
| `atol`, `rtol` | `float` or `list[float]` | Correctness tolerances, optionally one per input config. |
| `sort_before_compare` | `bool` | Sort every output leaf along its last axis before comparing (for order-free outputs such as top-k index sets). |
| `donate_args` | `{int: int}` | Buffers the kernel updates in place, as `dynamic_arg_index: output_leaf_index`. See below. |

#### In-place buffers (`donate_args`)

Kernels that update a buffer in place (a paged KV cache, a recurrent state)
return the updated buffer as one of their outputs. Without buffer donation, XLA
has to leave the input intact and materialize the output in a fresh buffer,
i.e. every call pays a full copy of the cache that a production caller never
pays. `donate_args` tells the harness which inputs may be donated and which
output leaf returns them:

```yaml
# kv_cache is dynamic arg 3 and comes back, updated, as output 1.
donate_args:
  3: 1
```

The harness then donates those inputs on the outer `jax.jit` for **both** the
reference and the optimized kernel (so the comparison stays fair), and after
every benchmark run feeds the returned buffer back in as the next run's input.
The map is validated against the kernel's real outputs: the input and the
output leaf must have identical shape and dtype. Only list buffers that are
returned; inputs that production code donates merely to free memory early
(queries, new keys/values) alias no output and change nothing in a benchmark.
An optimized kernel benefits from the contract by updating the buffer in place
(e.g. `pallas_call(..., input_output_aliases=...)`) rather than copying it.
A `donate_argnames` on the reference's own `jax.jit` has no effect inside the
harness, because inner-jit donation is dropped when traced under another jit.

### CLI Usage Examples

> [!NOTE]
> All CLI commands should be run from the project root directory (`MaxKernel`).

**Run benchmark on a remote TPU:**

```bash
python3 -m evaluation.benchmark \
    --tpu_name my-tpu \
    --zone us-central1-a \
    --project my-project \
    --dataset_dir /path/to/dataset \
    --venv_path /path/to/venv/on/tpu
```

**Run benchmark locally:**

```bash
python3 -m evaluation.benchmark \
    --local \
    --dataset_dir /path/to/dataset \
    --venv_path /path/to/local/venv
```

**Run with code adaptation (using Gemini API):**

If your dataset is missing `kernel_task.yaml` or requires code adaptation to fit the harness, you can use the `--adapt` flag. Note that this requires setting the `GEMINI_API_KEY` environment variable.

```bash
export GEMINI_API_KEY="your-api-key"

python3 -m evaluation.benchmark \
    --tpu_name my-tpu \
    ... \
    --adapt reference_code optimized_code kernel_task
```

### CLI Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--local` | Flag | Run evaluation locally instead of on a remote TPU. |
| `--tpu_name` | `str` | The name of the target TPU VM. |
| `--zone` | `str` | The Google Cloud zone where the TPU is located. |
| `--project` | `str` | The Google Cloud project ID. |
| `--venv_path` | `str` | Absolute path to Python venv to use. |
| `--dataset_dir` | `str` | Local directory containing the benchmark dataset. |
| `--adapt` | `str` list | Components to adapt via LLM (`reference_code`, `optimized_code`, `kernel_task`). |
| `--reference_file_name`| `str` | Filename for reference code (default: `reference.py`). |
| `--optimized_file_name`| `str` | Filename for optimized code (default: `optimized.py`). |
| `--task_file_name` | `str` | Filename for kernel task yaml (default: `kernel_task.yaml`). |
| `--atol` | `float` | Absolute tolerance for correctness check (default: 1e-3). |
| `--rtol` | `float` | Relative tolerance for correctness check (default: 1e-3). |

### Results

The CLI will save execution results to a JSON file in the `dataset_dir` named `evaluation_results_<tpu_name>.json`. It will also automatically resume from previous runs if this file exists. At the end of the run, a summary of results will be printed.
