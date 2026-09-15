---
title: "xla-compiler-helper"
type: codebase
tags: [mcp, compiler, agent-tooling, xla, pallas, active-documentation]
commit: experimental
created: 2026-06-02
updated: 2026-06-02
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->


`xla-compiler-helper` (XLA Compiler Assistant / XLA ADE) is an MCP (Model Context Protocol) server that equips AI agents and developers with a unified interface to query XLA compiler details, author custom Pallas kernels, and inspect compiler flags.

It runs in **Proxy Mode**, acting as a stateless client that forwards tool invocations to a central backend host containingGemini-powered research agents, Google3 codebase indices, and an empirical TPU experiment database.

---

## 🏗️ Architecture

```
Agent (MCP Client: editor, CLI, or autonomous agent)
   |
   | MCP (stdio transport via Blaze subprocess)
   v
//experimental/users/chrishjones/xla_ade/mcp:server (FastMCP server)
   |
   | HTTP POST / GET (corp-only network)
   v
http://chrishjones.c.googlers.com:8080 (Central Backend Server)
   |
   +-- XLA codebase search / Gemini API key rotation pool
   +-- Empirical TPU Experiment Database (flag outcomes & metrics)
   +-- Pallas compiler guides & Active Documentation wiki
```

---

## 📋 Prerequisites

1. **Corp Network Connectivity:** You must be connected to the Google corporate network (e.g., via gTransit, VPN, or running inside a workstation) to resolve and reach the central backend server.
2. **SSO Authentication:** You must have active `gcert` credentials (LOAS2) on your workstation or host system.
3. **Workspace Access:** You must be able to run blaze commands inside your Google3 client/workspace.

---

## ⚙️ Registration & Integration

### 1. VS Code (GoogleCode / OpenCode)
Add the following configuration to your user settings JSON file (`settings.json`) under the `mcp.servers` block:

```json
"mcp.servers": {
  "xla-compiler-helper": {
    "command": "blaze",
    "args": [
      "run",
      "//experimental/users/chrishjones/xla_ade/mcp:server"
    ]
  }
}
```

### 2. Cider-V (Web IDE)
1. Open the **Settings Panel** (gear icon) -> select **MCP Servers**.
2. Click **Add Server** and populate:
   - **Name:** `xla-compiler-helper`
   - **Command:** `blaze`
   - **Arguments:** `run //experimental/users/chrishjones/xla_ade/mcp:server`
3. Click **Save & Restart**.

### 3. Jetski Agent (Autonomous Coding Agent)
Jetski requires registering the server and exporting the exact tool schemas so that the planner can reason about the tools:

1. **Add Server Definition:** Open `~/.gemini/config/mcp_config.json` and add:
   ```json
   "xla-compiler-helper": {
     "$typeName": "exa.cascade_plugins_pb.CascadePluginCommandTemplate",
     "command": "blaze",
     "args": [
       "run",
       "//experimental/users/chrishjones/xla_ade/mcp:server"
     ],
     "cwd": "/google/src/cloud/<username>/<workspace>/google3"
   }
   ```
2. **Export Tool Schemas:** Execute the schema dumper command from your google3 directory:
   ```bash
   blaze run //experimental/users/chrishjones/xla_ade/mcp:server -- --dump_schemas ~/.gemini/jetski/mcp/xla-compiler-helper/
   ```
3. Restart your Jetski session to load the new tools.

> [!TIP]
> **One-Click Automated Setup:** You can automate all of the above Jetski and Gemini CLI configurations by running the provided install script from your google3 directory:
> ```bash
> ./experimental/users/chrishjones/xla_ade/mcp/install.sh
> ```

---

## 🛠️ Tools Exposed

### 1. `Question`
Answers complex compilation, flag-tuning, or structural Pallas implementation questions.

- **Parameters:**
  - `message` (string, required): The compiler/compilation query.
  - `session_id` (string, optional): A session identifier returned from a previous query to sustain chat/follow-up thread context.
- **Example Usage:**
  ```python
  Question(message="explain how stablehlo is lowered to llo")
  ```

### 2. `FlagSearch`
Returns detailed descriptions, origin, safety rationale, and outcomes/performance metrics of prior TPU runs using that flag.

- **Parameters:**
  - `flag_name` (string, required): The target compiler flag (e.g., `--allow_spmd_sharding_propagation_to_output`).
- **Example Usage:**
  ```python
  FlagSearch(flag_name="--allow_spmd_sharding_propagation_to_output")
  ```

---

## 🤖 Recommended Discovery Flow

When diagnosing a TPU performance problem or compiler optimization opportunity, follow this flow:
1. **Ask XLA Helper:** Send a natural language query via the `Question` tool to establish initial ideas or explanation of compiler stages.
2. **Refine Context:** Ask follow-up queries using the same `session_id` to refine strategies (e.g., write a candidate kernel outline).
3. **Investigate Flags:** Query potential compiler flags via `FlagSearch` to examine their empirical performance impact and default settings from previous runs.
