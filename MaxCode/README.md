# MaxCode Extension

This extension provides development tools for the MaxCode project,
including tools for AI-powered code migration between ML frameworks.

## Quick Start

Want to try MaxCode without the full Gemini CLI setup? The standalone demo
converts a PyTorch repo to JAX in five steps:

```bash
cd MaxCode/examples/demo
pip install -r requirements.txt
export GOOGLE_API_KEY=<your-key>    # Windows CMD: set GOOGLE_API_KEY=<your-key>

python step1_clone_repo.py          # Clone a PyTorch repo from GitHub
python step2_populate_rag.py        # Build the RAG reference database
python step3_merge.py               # Merge model + utility files (MergeAgent)
python step4_convert.py             # Convert to JAX with validation + repair
python step5_verify.py              # Verify conversion quality (VerificationAgent)
```

See [examples/demo/README.md](examples/demo/README.md) for full setup
instructions and details on what each step does.

## Prerequisites

This extension uses the Google AI API, which requires an API key. You can get
an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
Once you have a key, set it as an environment variable:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

## Setup

First, run the setup script to create a Python virtual environment and install
dependencies:

```bash
./setup_env.sh
```

Activate the environment before proceeding:
```bash
source $HOME/maxcode_venv/bin/activate
```

Then, register the extension with the Gemini CLI:

```bash
gemini extension link MaxCode/mcp_server
```

**Important:** If you are running on an installation cloned from GitHub, you
**must** run the `gemini` command from within the `MaxCode` directory. This
is necessary for the CLI to locate and launch the agent server correctly.

After linking the extension and starting `gemini-cli` for the first time,
`gemini list extensions` may show the server for `maxcode-dev` as
disconnected (red). If this happens, try restarting `gemini-cli`.

## Usage

### Verify Installation

You can verify that the extension is registered by running:

```bash
gemini list extensions
```

You should see `maxcode-dev` in the list. If the server is
running correctly, it should have a connected (green) status.

### Running Tools

**Make sure your virtual environment is activated (`source
$HOME/maxcode_venv/bin/activate`) before running Gemini**, as the
toolchain needs access to packages like JAX and PyTorch to run equivalence
tests.

**You do not need to start the server manually.**

To run tools, first start Gemini in interactive mode. We also recommend
enabling `--debug` to get detailed logs:

```bash
gemini --debug
```

Once in interactive mode, you can run tools using `dev-server <tool_name>
<args>`.

The Gemini CLI automatically starts and manages the MCP server process. When you
call a tool for the first time in a session (e.g., `dev-server
run_migration_workflow "<your prompt>"`) and the server is not already running,
Gemini CLI will launch it using the command defined in
`gemini-extension.json`:
`python3.11 -m mcp_server.primary_agent_server`.

The first tool call will start the server if it is not running. Subsequent tool calls will be
faster as they connect to the already-running server. If you start a new
terminal session, the first call in that session may again be slow if it
needs to relaunch the server.

### Debugging

If you run Gemini with `gemini --debug` as recommended above, it will
display verbose logs, including RPC calls made to the MCP server and server-side
logs, which can be helpful for debugging.

You can also review the full session history, which is saved in JSON format in
the `~/.gemini/chats` directory. These files contain all prompts, responses, and
tool interactions, allowing you to reconstruct the session for debugging. You
can save a session file during an interactive session using the `/chat save
<filename>` command.

#### Monitoring Agent Progress

Because the ADK agents orchestrate their multi-step workflow (like migration,
evaluation, and test generation) behind a single synchronous MCP tool call,
intermediate steps and tool executions are not natively displayed in the
`gemini-cli` UI. If you do not tail the log file as described below, no
progress will be displayed in your terminal until the migration task is complete
or an error occurs. This is because `gemini-cli` waits for a tool to return
its final response and does not display the in-progress status updates (sent
via `ctx.info()`) that a dedicated graphical UI or streaming interface would
show.

To watch the agents work in real-time, open a separate terminal window and tail
the FastMCP server log by running the following command:

```bash
tail -f /tmp/agent_server.log
```

#### Code Migration

To migrate a Python module or package from one framework (e.g., PyTorch) to
JAX, use the `run_migration_workflow` tool with a prompt describing the
migration task. The agent will perform the end-to-end migration, which
includes:
1.  Converting the PyTorch code to JAX.
2.  Generating model configurations from the PyTorch code.
3.  Generating oracle data (inputs, outputs, weights) from the PyTorch models.
4.  Generating equivalence tests to validate the migrated JAX code against the
    oracle data.

**Disclaimer:** This toolchain is under active development and should be
considered experimental. Testing has primarily focused on single-file or
small-module migrations containing standard PyTorch layers. While it may
attempt to process directories or more complex models, successful migration is
not guaranteed for large repositories or highly customized PyTorch models
(e.g., models relying heavily on custom C++ extensions or specialized
libraries like vLLM). For best results, we recommend migrating one file or
small module at a time.

**Example:**

In interactive mode, you can use natural language directly:

```
Migrate /path/to/your/pytorch_module.py to /tmp/migrated_jax_mlp using API key YOUR_API_KEY.
```

Alternatively, you can explicitly call the agent from the shell or in
interactive mode using the following syntax:

```bash
# From your shell:
gemini call dev-server run_migration_workflow --prompt "Migrate /path/to/your/pytorch_module.py to /tmp/migrated_jax_mlp using API key YOUR_API_KEY."
# From inside interactive mode:
dev-server run_migration_workflow --prompt "Migrate /path/to/your/pytorch_module.py to /tmp/migrated_jax_mlp using API key YOUR_API_KEY."
```

This command will create a timestamped subdirectory inside
`/tmp/migrated_jax_mlp/` (e.g., `/tmp/migrated_jax_mlp/YYYYMMDD_HHMMSS/`)
containing the following:

*   `original_source/`: A copy of the input PyTorch file(s).
*   Migrated JAX code (e.g., `pytorch_module.py`).
*   `mapping.json`: Logs the correspondence between source and migrated files.
*   `evaluation/`: A directory containing:
    *   `model_configs.json`: Configurations for instantiating PyTorch models.
    *   `data/`: Directory with `.pkl` files containing oracle data.
    *   `tests/`: Directory with `test_*.py` files for equivalence testing.

You can run a generated test file using `python
/tmp/migrated_jax_mlp/YYYYMMDD_HHMMSS/evaluation/tests/test_pytorch_module.py`.
By default, it looks for `.pkl` files in the same directory, but you can
specify a different path using the `--pickle_path` flag: `python
/tmp/migrated_jax_mlp/YYYYMMDD_HHMMSS/evaluation/tests/test_pytorch_module.py
--pickle_path=/tmp/migrated_jax_mlp/YYYYMMDD_HHMMSS/evaluation/data/pytorch_module.pkl`.

### Generating only Evaluation Data or Tests

If you only want to generate model configs, oracle data, or equivalence tests
without running a full migration, you can prompt the `evaluation_agent`
specifically:

```bash
# Example for generating only configs:
dev-server run_evaluation_workflow --prompt "Generate model configs for PyTorch files in /path/to/torch_models and save to /path/to/model_configs.json using API key YOUR_API_KEY."

# Example for generating only oracle data:
dev-server run_evaluation_workflow --prompt "Generate oracle data for PyTorch files in /path/to/torch_models, using configs in /path/to/model_configs.json, and save data to /path/to/data_dir using API key YOUR_API_KEY."

# Example for generating only tests:
dev-server run_evaluation_workflow --prompt "Generate an equivalence test for JAX file /path/to/jax_model.py and PyTorch file /path/to/torch_model.py, save to /path/to/test_output.py using API key YOUR_API_KEY."

# Example for running equivalence tests:
dev-server run_evaluation_workflow --prompt "Run equivalence tests for migration defined in /path/to/mapping.json, using data in /path/to/data_dir and saving tests to /path/to/tests_dir, using API key YOUR_API_KEY."
```

## Architecture

The migration pipeline: **Clone -> Index -> Merge -> Convert -> Verify**.

Key agents in `agents/migration/`:
- **MergeAgent** — Pure-logic preprocessing: file discovery, filtering, import
  graph analysis, and merging (no LLM calls).
- **PrimaryAgent** — Orchestrates conversion: routes to model or utility
  conversion agents, fills missing components, validates and repairs.
- **VerificationAgent** — Post-processing quality scoring: AST-based
  completeness + optional LLM-based correctness.

For more details on the project architecture and agent structure, see
[ARCHITECTURE.md](ARCHITECTURE.md).
