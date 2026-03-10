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
`mcp.server.fastmcp`. It exposes a single tool, `run_agent`, which acts as the
entry point for ADK's `Runner` to execute the `master_agent`.

### 3. ADK Agent Definitions

`mcp_server/adk_agents.py` defines the hierarchy of ADK agents:

*   **`master_agent`**: The top-level agent invoked by `run_agent`. Its role is to understand the user's intent and delegate to the appropriate sub-agent (e.g., `migration_agent`).
*   **`migration_agent`**: A sub-agent responsible for migration tasks. It uses `migration_tool` to perform actions.

### 4. ADK Tools

`tools/migration_tool.py` defines ADK `FunctionTool`s that wrap Python
functions. `migrate_module_tool` wraps the `migrate_module` function, which
contains the logic for orchestrating an end-to-end migration and validation
task.

### 5. Migration and Validation Logic

The `migrate_module` function in `tools/migration_tool.py` orchestrates the
end-to-end migration and validation workflow:
1.  It instantiates and runs `agents.migration.primary_agent.PrimaryAgent`,
    which contains the core logic for repository migration (e.g.,
    `PytorchToJaxSingleFileAgent`).
2.  After migration, it calls
    `tools.evaluation_tool.generate_model_configs` to generate configuration
    files from the original PyTorch code.
3.  It calls `evaluation.make_data.generate_data` to generate oracle data
    (.pkl files) from the PyTorch code using the generated configurations.
4.  It calls `tools.evaluation_tool.generate_equivalence_tests` to generate
    test scripts for each migrated file, comparing the JAX output against the
    PyTorch oracle data.

The result is a destination directory containing the migrated JAX code, a
`mapping.json` file, and an `evaluation` subdirectory with configurations,
oracle data, and test scripts.

## Summary

The overall flow for migration is:

```
Gemini CLI -> mcp_server:primary_agent_server -> adk_agents:master_agent -> adk_agents:migration_agent -> tools:migration_tool:migrate_module ->
  1. agents/migration:PrimaryAgent (Migration)
  2. tools/evaluation_tool:generate_model_configs (Config Gen)
  3. evaluation/make_data:generate_data (Data Gen)
  4. tools/evaluation_tool:generate_equivalence_tests (Test Gen)
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
4.  **Integrate:** Add `profiling_agent` to `master_agent`'s `sub_agents` list
    in `mcp_server/adk_agents.py` and update `master_agent`'s instructions to
    delegate profiling tasks to `profiling_agent`.

### How to Modify an Existing Capability
To modify migration behavior, edit the implementation logic in
`agents/migration/primary_agent.py`, `agents/migration/single_file_agent.py`, or
other related files. If the function signature exposed to the ADK tool needs to
change, update the wrapper function in `tools/migration_tool.py`.
