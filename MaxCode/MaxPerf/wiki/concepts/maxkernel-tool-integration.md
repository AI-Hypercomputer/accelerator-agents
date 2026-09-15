<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# MaxKernel Tool Integration Guide

This guide details how the local **MaxKernel Agent** interacts with the **MaxKernel Tool** running on the TPU VM to automate compiler checks, correctness testing, profiling, and tile-size autotuning.

---

## 1. Remote Architecture on TPU VM

The `MaxKernel` tool runs as a background service on the TPU VM HEAD node (`Worker 3`).

### 1.1 TPU VM Daemon Setup
The remote service is defined in `tpu_server.py` using FastAPI. It runs on port `5463` and executes Pallas kernels inside isolated Python subprocesses on the TPU VM host.

**To start the server on Worker 3:**
```bash
# SSH into Worker 3 and launch the server in the background
./runs/ssh_tpu.sh "source ~/vllm_env/bin/activate && nohup python3 -m uvicorn auto_agent.server_utils.tpu_server:app --host 0.0.0.0 --port 5463 > ~/tpu_server.log 2>&1 &"
```

**To verify the server is running:**
```bash
./runs/ssh_tpu.sh "curl -s http://localhost:5463/health"
# Expected response: {"status": "healthy"}
```

---

## 2. Agent Interaction Protocol

Since the local workspace does not have direct network access to the TPU VM's private ports, the local agent communicates with the FastAPI daemon via **SSH standard input redirection**.

```
[Local MaxKernel Agent]
        │
        ▼ writes payload to local JSON file: payload.json
        │
        ▼ runs local gcloud alpha compute tpus tpu-vm ssh command:
        │  ./runs/ssh_tpu.sh 'curl -s -X POST -H "Content-Type: application/json" -d @- http://localhost:5463/<endpoint>' < payload.json
        │
   [SSH Tunnel (gcloud)]
        │
        ▼ (Worker 3 / HEAD Node)
   [FastAPI Daemon (port 5463)]
        │
        ▼ Executes compiler/tests in sandboxed subprocess on TPU
        │
        ▼ Returns JSON response back through stdout
        │
[Local MaxKernel Agent] Parses JSON stdout response
```

---

## 3. Tool API Endpoints & Payloads

### 3.1 Verification and Correctness Testing (`/correctness_test`)
Tests if a given kernel compiles and produces matching numeric outputs compared to a reference baseline.

*   **Endpoint**: `POST http://localhost:5463/correctness_test`
*   **Payload Schema (`payload.json`)**:
    ```json
    {
      "code": "import jax\nimport jax.numpy as jnp\nfrom jax.experimental import pallas as pl\n...",
      "dependencies": {
        "reference_impl.py": "# Reference code for comparison\ndef ref_matmul(a, b):\n    return jnp.dot(a, b)\n"
      },
      "timeout": 60
    }
    ```
*   **Response Schema**:
    ```json
    {
      "output": "CORRECTNESS: True\nRESULT_TIME: 1.42 ms",
      "error": null,
      "exit_code": 0
    }
    ```

### 3.2 Automated Tile-Size Autotuning (`/autotune`)
Runs a grid-search sweep over a specified hyperparameter search space to find the configuration with the lowest execution latency on physical TPU cores.

*   **Endpoint**: `POST http://localhost:5463/autotune`
*   **Payload Schema (`payload.json`)**:
    ```json
    {
      "code_template": "import jax\n...\n# Autotuned parameters\nBLOCK_M = {BLOCK_M}\nBLOCK_N = {BLOCK_N}\n...",
      "search_space": {
        "BLOCK_M": [16, 32, 64],
        "BLOCK_N": [32, 64, 128]
      },
      "dependencies": {
        "correctness_impl.py": "<validation code verifying result and printing 'CORRECTNESS: True'>"
      },
      "timeout": 300,
      "total_timeout": 1800
    }
    ```
*   **Response Schema**:
    ```json
    {
      "output": "{\"best_cfg\": {\"BLOCK_M\": 32, \"BLOCK_N\": 64}, \"best_time\": 0.84, \"best_output\": \"...\", \"all_results\": [...]}",
      "exit_code": 0
    }
    ```

### 3.3 Roofline Profiling (`/profile`)
Executes the kernel while capturing XLA HLO traces and extracts the compute vs. memory bandwidth utilization ratio.

*   **Endpoint**: `POST http://localhost:5463/profile`
*   **Payload Schema (`payload.json`)**:
    ```json
    {
      "code": "import jax\n# profiling execution script...",
      "dependencies": {},
      "timeout": 120
    }
    ```
*   **Response Schema**:
    ```json
    {
      "output": "{\"ratio\": 0.35, \"xplane_path\": \"/tmp/tmp_dir/plugins/profile/2026_...xplane.pb\"}",
      "error": null,
      "exit_code": 0
    }
    ```

---

## 4. Local Agent Integration Instructions

To use the tool, the `MaxKernel` agent must follow these steps:

1.  **Generate Payload**: Save the template code and search space parameters to a temporary file: `scratch/autotune_payload.json`.
2.  **Execute Command**: Run the SSH redirect command using the `run_command` tool:
    ```bash
    ./runs/ssh_tpu.sh 'curl -s -X POST -H "Content-Type: application/json" -d @- http://localhost:5463/autotune' < scratch/autotune_payload.json
    ```
3.  **Parse & File Results**: Extract the `best_cfg` and execution times from the JSON stdout response. Incorporate them directly into the experiment page under `experiments/<slug>/experiment.md`.
