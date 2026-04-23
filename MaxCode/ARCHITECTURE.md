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

`tools/migration_tool.py` and `tools/evaluation_tool.py` define ADK
`FunctionTool`s that wrap specific Python functions for code conversion,
config generation, data generation, and testing.

### 5. Migration and Validation Logic

The `migration_agent` orchestrates the end-to-end migration and validation
workflow by calling tools in sequence:
1.  **`migration_tool.convert_code`**: Converts PyTorch code to JAX using
    `agents.migration.primary_agent.PrimaryAgent`, copies the original source
    code, and saves the results to a timestamped output directory. Returns
    paths to the migrated code, original code, and mapping file.
2.  **`evaluation_tool.generate_model_configs`**: Generates configuration
    files from the original PyTorch code.
3.  **`evaluation_tool.generate_oracle_data`**: Generates oracle data
    (.pkl files) from the PyTorch code using the generated configurations.
4.  **`evaluation_tool.run_equivalence_tests`**: Generates test scripts
    that compare JAX outputs against PyTorch oracle data, and then runs these
    tests using `subprocess`.

The result is a destination directory containing the migrated JAX code, a
`mapping.json` file, and an `evaluation` subdirectory with configurations,
oracle data, and test scripts.

## Summary

The overall flow for migration is:

```
Gemini CLI -> mcp_server:primary_agent_server -> adk_agents:migration_agent ->
  1. tools:migration_tool:convert_code (Migration)
  2. tools:evaluation_tool:generate_model_configs (Config Gen)
  3. tools:evaluation_tool:generate_oracle_data (Data Gen)
  4. tools:evaluation_tool:run_equivalence_tests (Test Gen & Run)
```

## Agent Structure and Extension

The project separates agent implementation logic from ADK agent/tool
definitions:

*   **`agents/<domain>/`**: Contains agent classes with core implementation logic (e.g., `agents/migration/primary_agent.py`).
*   **`tools/`**: Contains ADK `FunctionTool` wrappers that call agent logic or other Python functions (e.g., `tools/migration_tool.py`).
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
