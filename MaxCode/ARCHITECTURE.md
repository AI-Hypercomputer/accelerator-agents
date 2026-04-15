# MaxCode Architecture

This document outlines the architecture of the MaxCode project and
the end-to-end flow for code migration using the Gemini CLI extension.

## End-to-End Flow

The migration process is initiated from the Gemini CLI, which interacts with a
local MCP server invoking ADK agents and tools to perform the migration.

### 1. Gemini Extension Configuration

The `MaxCode/mcp_server/gemini-extension.json` file
configures the Gemini CLI to use a local agent server. When a tool is invoked
for which this server is registered, Gemini CLI uses the command
`python3.11 -m mcp_server.primary_agent_server`
to start the server if it's not running, as mentioned in `README.md`.

### 2. MCP Server and ADK Runtime

`mcp_server/primary_agent_server.py` implements the MCP (Model Calling Platform)
server runtime using `google.adk` (Agent Development Kit) and
`mcp.server.fastmcp`. It exposes tools like `run_migration_workflow` and
`run_evaluation_workflow`, which act as entry points for ADK's `Runner` to
execute the `migration_agent` and `evaluation_agent`, respectively.

### 3. ADK Agent Definitions

`mcp_server/adk_agents.py` defines ADK agents:

*   **`migration_agent`**: Responsible for end-to-end migration tasks. It orchestrates multiple tools to perform migration and validation.
*   **`evaluation_agent`**: Responsible for running specific evaluation tasks in isolation (e.g., only generating oracle data).

### 4. ADK Tools

`tools/migration_tool.py`, `tools/evaluation_tool.py`, and
`tools/verification_tool.py` define ADK `FunctionTool`s that wrap specific
Python functions for code conversion, quality verification, config generation,
data generation, and testing.

### 5. Migration Pipeline

For **directory inputs**, `PrimaryAgent` uses `MergeAgent`
(`agents/migration/merge_agent.py`) to preprocess the repository before
conversion. The merge step:
- Discovers all nn.Module files and builds an import dependency graph
- Filters infrastructure files (fused kernels, CUDA wrappers, etc.)
- Merges model files into a single file in topological order
- Discovers and merges utility files separately
- Filters infrastructure classes from merged output

For **single-file inputs**, the existing direct conversion path is used.

After conversion, `migration_tool.convert_code` automatically runs
`VerificationAgent` (`agents/migration/verification_agent.py`) to produce
a completeness scorecard (AST-based, no LLM). The verification tool is
also available standalone via `tools/verification_tool.py`.

### 6. ADK Agent Orchestration

The `migration_agent` orchestrates the end-to-end migration and validation
workflow by calling tools in sequence:
1.  **`migration_tool.convert_code`**: Merges, converts, and verifies
    PyTorch code to JAX using `PrimaryAgent` (which delegates to
    `MergeAgent` for directories). Copies the original source code and
    saves results to a timestamped output directory.
2.  **`verification_tool.verify_conversion`** (optional): Standalone
    quality verification with completeness and correctness scores.
3.  **`evaluation_tool.generate_model_configs`**: Generates configuration
    files from the original PyTorch code.
4.  **`evaluation_tool.generate_oracle_data`**: Generates oracle data
    (.pkl files) from the PyTorch code using the generated configurations.
5.  **`evaluation_tool.run_equivalence_tests`**: Generates test scripts
    that compare JAX outputs against PyTorch oracle data, and then runs these
    tests using `subprocess`.

The result is a destination directory containing the migrated JAX code, a
`mapping.json` file, a `verification_scorecard.json`, and an `evaluation`
subdirectory with configurations, oracle data, and test scripts.

## Summary

The overall flow for migration is:

```
Gemini CLI -> mcp_server:primary_agent_server -> adk_agents:migration_agent ->
  1. tools:migration_tool:convert_code
     (Merge -> Convert -> Validate/Repair -> Verify)
  2. tools:verification_tool:verify_conversion (optional, standalone)
  3. tools:evaluation_tool:generate_model_configs (Config Gen)
  4. tools:evaluation_tool:generate_oracle_data (Data Gen)
  5. tools:evaluation_tool:run_equivalence_tests (Test Gen & Run)
```

The internal flow within `convert_code` for directory inputs:

```
MergeAgent.run(repo_dir)          # Preprocessing: discover, filter, merge
  -> PrimaryAgent._convert_file() # LLM conversion (model + utils)
  -> PrimaryAgent._fill_missing() # Gap-filling pass
  -> PrimaryAgent._validate()     # Validation + repair loop
  -> VerificationAgent.verify()   # Quality scorecard
```

## Agent Structure and Extension

The project separates agent implementation logic from ADK agent/tool
definitions:

*   **`agents/<domain>/`**: Contains agent classes with core implementation logic (e.g., `agents/migration/primary_agent.py`, `agents/migration/merge_agent.py`, `agents/migration/verification_agent.py`).
*   **`tools/`**: Contains ADK `FunctionTool` wrappers that call agent logic or other Python functions (e.g., `tools/migration_tool.py`, `tools/verification_tool.py`).
*   **`mcp_server/adk_agents.py`**: Defines the ADK agent hierarchy, instructions, and tool mappings.

### How to Add a New Capability
If you want to add a new capability (e.g., "code profiling"), follow these
steps:
1.  **Implement Logic:** Create new agent logic in a relevant domain, e.g.,
    `agents/profiling/profiler.py`.
2.  **Create Tool:** Create a new tool in `tools/`, e.g.,
    `tools/profiling_tool.py`, that imports `agents/profiling/profiler.py`,
    defines a function like `run_profiler`, and wraps it in a `FunctionTool`:
    `profiling_tool = FunctionTool(run_profiler)`.
3.  **Create ADK Sub-Agent:** In `mcp_server/adk_agents.py`, create a new ADK
    agent definition: `profiling_agent = Agent(..., tools=[profiling_tool])`.
4.  **Integrate:** If the new agent represents a distinct workflow, expose it
    as a new tool in `mcp_server/primary_agent_server.py` (e.g.,
    `run_profiling_workflow`). If it's part of an existing workflow, add its
    tool to the `tools` list of an existing agent like `migration_agent` or
    `evaluation_agent`.

### How to Modify an Existing Capability
To modify migration behavior, edit the implementation logic in
`agents/migration/primary_agent.py`, `agents/migration/single_file_agent.py`, or
other related files. If the function signature exposed to the ADK tool needs to
change, update the wrapper function in `tools/migration_tool.py`.
