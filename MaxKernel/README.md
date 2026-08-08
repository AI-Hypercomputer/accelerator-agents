# MaxKernel: Intelligent TPU Kernel Engineering Agents

**MaxKernel** is a suite of AI-powered agents designed for generating, optimizing, testing, profiling, and autotuning high-performance TPU kernels using [JAX/Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html).

Whether you need interactive, step-by-step kernel development or a fully autonomous optimization pipeline, MaxKernel provides dedicated agents tailored to your workflow.

---

## 🚀 Choose Your Agent

MaxKernel provides two primary agent workflows:

| Feature | 🧑‍💻 **HITL Agent** (`hitl_agent`) | 🤖 **Auto Agent** (`auto_agent`) |
| :--- | :--- | :--- |
| **Workflow** | **Interactive / Human-in-the-Loop** | **Fully Autonomous Pipeline** |
| **Best For** | Exploratory development, custom kernels, interactive debugging, learning Pallas | Batch optimization, benchmark evaluation, unattended end-to-end kernel synthesis |
| **User Interaction** | Reviews & approves plans, guides step-by-step code generation | Automated improvement loop with closed-loop feedback |
| **Execution Backends**| Local (TPU VM / CPU) | Local (Recommended), GCE / GKE (In progress) |
| **Setup Script** | `bash prepare_maxkernel.sh` | `bash prepare_maxkernel.sh` |
| **Run Script** | `bash run_hitl_agent.sh` | `bash run_auto_agent.sh` |
| **Interfaces** | Interactive CLI & Web UI | Interactive CLI & Web UI |

---

## ✨ Core Capabilities

Both agents share a rich set of specialized tools and capabilities:

* 📐 **Optimization Planning**: Formulates tiling strategies, memory layouts (VMEM/SMEM/HBM), pipelining, and grid configurations tailored to your TPU generation.
* 💻 **Kernel Implementation**: Generates clean, idiomatic JAX/Pallas kernels adhering to hardware best practices and approved plans.
* 🛡️ **Compilation Validation**: Fast pre-validation to catch shape mismatches, syntax errors, and compilation failures before execution.
* 🧪 **Automated Testing**: Generates and executes comprehensive `pytest` suites validating compilation, numerical correctness (against reference implementations), and execution metrics.
* ⚡ **Performance Profiling**: Pinpoints hardware bottlenecks, DMA/memory transfer overheads, and compute-to-memory ratios.
* 🎛️ **Automated Autotuning**: Performs grid search over block sizes, grid layouts, and hyperparameters to maximize TPU throughput.
* 🔄 **GPU-to-JAX Conversion (only in HITL)**: Automatically ports existing CUDA, Triton, and PyTorch kernels into JAX/Pallas equivalents.
* 🔒 **Scoped & Safe**: Restricts agent file operations to a user-defined work directory (`WORKDIR`).

---

## 🏁 Quickstart

### 1. Prerequisites

* **Python 3.11+** with pip installed
* **Google Cloud Project** with Vertex AI / Gemini API access
* **TPU Access** (or CPU testbed) for executing and profiling kernels

### 2. Environment Setup

Run the automated setup script (compatible with both **Linux** and **macOS**):

```bash
bash prepare_maxkernel.sh
```

**What the setup script does automatically:**
1. **Python Environment Setup**: Prompts to create and activate a `.venv` virtual environment (or uses your active Python environment).
2. **Dependency Installation**: Installs all required packages from `dependency/main_requirements.txt` and `dependency/agent_requirements.txt`, verifies Node.js/npx, and installs the repository in editable mode (`pip install -e .`).
3. **Environment Configuration**: Prompts and creates the `.env` configuration file.
4. **Evaluation Server Config**: Resolves your local IP address across Linux/macOS and generates `eval_config.yaml` for both `auto_agent` and `hitl_agent`.

### 3. Environment Variables (`.env`)

The setup script generates a `.env` file at the repository root. You can view or customize it at any time:

```bash
# Model & Cloud Configuration
GOOGLE_CLOUD_PROJECT="your-project-id"   # Required: GCP project ID
GOOGLE_GENAI_USE_VERTEXAI=TRUE           # Vertex AI authentication
GOOGLE_CLOUD_LOCATION="global"           # GCP location (e.g. global, us-central1)
GEMINI_API_KEY="your-api-key"            # Optional: Gemini API key if using API key auth
RAG_CORPUS="projects/.../ragCorpora/..." # Optional: Vertex AI RAG corpus for Pallas docs
INCLUDE_THOUGHTS="true"                  # Optional: Show agent reasoning traces

# Hardware & Workspace Settings
WORKDIR="/path/to/your/workdir"          # Scoped directory for kernel inputs/outputs
TPU_VERSION="TPU v5e"                    # Options: TPU v4, TPU v5e, TPU v5p, TPU v6e, TPU 7x
```

---

## 🤖 Using the Auto Agent (`auto_agent`)

The Auto Agent runs an end-to-end autonomous loop:
`Base Kernel Preparation` ➔ `Planning` ➔ `Implementation` ➔ `Compilation Validation` ➔ `Testing` ➔ `Autotuning` ➔ `Profiling` ➔ `Iterative Improvement`.

> [!TIP]
> **Recommended Execution Backend:** We currently encourage using the default **`local` backend directly on a TPU VM**. Remote backend setup for GCE and GKE is currently under development.

### Run in CLI Mode

```bash
# Start autonomous agent on local backend (Recommended on TPU VM)
bash run_auto_agent.sh

# Stop running background servers & processes
bash run_auto_agent.sh stop
```

### Run in Web UI Mode

```bash
bash run_auto_agent.sh --ui
```

### CLI Options

```text
Usage: ./run_auto_agent.sh [command] [options]

Commands:
  start                   Start the agent and background servers (default)
  stop                    Stop all running agent sessions, servers, and tunnels

Options:
  -b, --backend <type>    Execution backend: 'local' (default/recommended), 'gce', or 'gke'
  --ui                    Start with the web UI on port 1430 (default: CLI mode)
  -h, --help              Show help message
```

---

## 🧑‍💻 Using the HITL Agent (`hitl_agent`)

The Human-in-the-Loop agent is designed for interactive collaboration. It prompts you to review optimization plans and guides each phase of kernel generation.

### Run in CLI Mode

```bash
# Start a new interactive session
bash run_hitl_agent.sh

# Reset and restart existing processes
bash run_hitl_agent.sh --reset
```

### Run in Web UI Mode

```bash
bash run_hitl_agent.sh --ui
```
* Access the web interface at **`http://localhost:1430`**
* Manage multiple sessions and review visual conversation history.


---

## 🔍 Using Auto-Search (`auto_search`)

**Auto-Search** orchestrates autonomous worker agents to systematically search for optimized TPU kernel implementations across single reference problems or entire benchmark datasets.

### Search Algorithms

1. **Parallel Search (`--algorithm parallel`)**: Dispatches multiple independent worker agents concurrently and returns the candidate with the highest speedup.
2. **Beam Search (`--algorithm beam`)**: A tree-search approach that explores multiple optimization branches hierarchically, evaluates speedups, and retains the top `beam_size` candidates at each depth.

### 1. Optimize a Single Problem (`run_search.py`)

```bash
# Run Parallel Search on a single reference kernel
python -m auto_search.run_search \
  --reference_file_path /path/to/problem_dir/reference.py \
  --algorithm parallel \
  --num_parallel_runs 4 \
  --max_concurrency 2

# Run Beam Search on a single reference kernel
python -m auto_search.run_search \
  --reference_file_path /path/to/problem_dir/reference.py \
  --algorithm beam \
  --beam_size 2 \
  --branches_per_node 2 \
  --max_depth 3 \
  --max_concurrency 2
```

### 2. Run Batch Search on a Dataset (`run_batch_search.py`)

Optimize an entire dataset containing multiple problem subdirectories (each containing a `reference.py`):

```bash
# Run Batch Parallel Search across dataset problems
python -m auto_search.run_batch_search \
  --data_dir /path/to/dataset_dir \
  --algorithm parallel \
  --num_parallel_runs 4 \
  --num_problem_concurrency 2 \
  --max_concurrency 2

# Run Batch Beam Search across dataset problems
python -m auto_search.run_batch_search \
  --data_dir /path/to/dataset_dir \
  --algorithm beam \
  --beam_size 2 \
  --branches_per_node 2 \
  --max_depth 3 \
  --num_problem_concurrency 2 \
  --max_concurrency 2
```

### Key Auto-Search Options

* `--reference_file_path`: Path to single reference kernel file (`run_search.py`).
* `--data_dir`: Path to dataset directory containing benchmark problem subdirectories (`run_batch_search.py`).
* `--algorithm`: Search algorithm to use (`parallel` or `beam`, default: `parallel`).
* `--max_concurrency`: Maximum concurrent worker agents (default: `2`).
* `--num_parallel_runs`: Number of independent agents spawned per problem in parallel search (default: `2`).
* `--beam_size`: Top candidates retained at each tree depth in beam search (default: `2`).
* `--branches_per_node`: New optimization strategies explored per candidate node (default: `2`).
* `--max_depth`: Maximum depth of the beam search tree (default: `2`).
* `--strategies`: (Optional) Space-separated list of explicit strategies to explore (e.g. `--strategies "Fuse ops" "Tiling"`).
* `--graph_db_path`: Path to a `search_graph.json` to resume an interrupted search.

---

## 📁 Work Directory & File Safety

All agents operate within a **scoped work directory** (`WORKDIR` in `.env`):

* **Inputs**: Place your reference code, GPU kernels, or problem specifications in `WORKDIR`. For AutoSearch, the inputs are reference_file_path and data_dir and they can be placed outside the `WORKDIR` as the content of the reference files will be read to the agent first.
* **Outputs**: Generated plans (`*.md`), JAX/Pallas kernels (`*.py`), test scripts (`test_*.py`), and profile logs are saved directly in `WORKDIR` under their session sub directories.
* **Isolation**: The agent cannot read or modify files outside `WORKDIR`, protecting your host environment.
