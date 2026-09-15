<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# XLA Compiler Assistant - MCP Server (Proxy Mode)

This is a Model Context Protocol (MCP) server that equips coding agents (like Jetski, Cider-V assistants, or VS Code agents) with a unified interface to query XLA compiler details, write custom Pallas kernels, and research XLA flags.

## 🏗️ Architecture (Strategy B - Hosted Proxy)
This MCP server runs in **Proxy Mode**.
*   **Stateless Client:** The local MCP binary running in your workspace is extremely lightweight and has zero API key requirements. It simply forwards tool queries to a central Flask web server over HTTP.
*   **Centralized Server:** The host server (`http://chrishjones.c.googlers.com:8080`) holds the Gemini API keys and rotation pools, manages conversation sessions, parses compilation flags, and runs the deep-dive wiki search and subagent logic in the background.

---

## 📋 Prerequisites
1.  You must be connected to the Google internal corp network (or via gTransit/VPN) to communicate with the central host.
2.  You must sync this package to your CitC workspace.

---

## 📥 1. Download / Sync

To sync the MCP server package to your CitC workspace, run:
```bash
# From your google3 workspace directory:
hg sync
```
Ensure you can see the files under:
`//experimental/users/chrishjones/xla_ade/mcp/`

---

## ⚙️ 2. Installation in Editors

### A. VS Code (GoogleCode / OpenCode)
To register this MCP server globally in your VS Code editor:

1.  Open VS Code.
2.  Open your user settings file (`settings.json`) via the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` -> `Preferences: Open User Settings (JSON)`).
3.  Add the following block under `mcp.servers`:
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
4.  Save the file. VS Code will launch the server in the background, and your editor agent will immediately gain access to the XLA helper tools.

### B. Cider-V (Web IDE)
To register this MCP server in Cider-V:

1.  Open Cider-V.
2.  Open the **Settings Panel** (gear icon in the bottom-left corner) -> select **MCP Servers**.
3.  Click **Add Server** and fill out the fields:
    *   **Name:** `xla-compiler-helper`
    *   **Command:** `blaze`
    *   **Arguments:** `run //experimental/users/chrishjones/xla_ade/mcp:server`
4.  Click **Save & Restart**. The tools will immediately appear in Cider-V's AI assistant panel.

---

## 🛠️ Tools Exposed

This MCP server exposes two core tools:

### 1. `Question(message: str, session_id: Optional[str] = None)`
Answers any complex compiler, structural, or implementation questions about XLA and Pallas.
*   **How it works:** The backend server automatically searches the wiki, invokes subagents to synthesize multi-page results, and returns a dense, verified compiler response.
*   **Follow-up Questions:** To ask a follow-up, simply call `Question` again and pass the returned `session_id` from the previous turn. The server will resolve your follow-up inside the exact same conversation history natively!

### 2. `FlagSearch(flag_name: str)`
Looks up a specific XLA compiler flag (e.g., `--allow_spmd_sharding_propagation_to_output`).
*   **Returns:** The flag's description, default/available values, safety rationale, origin CL, **and all matching outcomes and metrics from previous empirical TPU server experiments** (status, execution time, memory usage, XManager/Sponge runs, and notes).
*   *Note: You can pass flags prefixed with or without hyphens (`--`).*

---

## 🔄 Configuration Overrides
By default, the MCP client connects to the central backend at `http://chrishjones.c.googlers.com:8080`. If the backend address changes or you wish to run a local backend, you can override the destination by passing the `XLA_ADE_SERVER` environment variable in your editor settings:

```json
"env": {
  "XLA_ADE_SERVER": "http://new-host.c.googlers.com:9090"
}
```

---

## 🤖 3. Installation in Agents & CLI Tools

### 🚀 Option 1: Automated One-Click Installation (Recommended)
You can automate the entire configuration and schema generation process with a single script execution from your `google3` workspace directory:

```bash
./experimental/users/chrishjones/xla_ade/mcp/install.sh
```

This script will automatically:
1.  Create and safely append the `xla-compiler-helper` configuration to your Jetski config (`~/.gemini/config/mcp_config.json`), including the required `$typeName` field.
2.  Export the exact JSON tool schemas (`Question.json` and `FlagSearch.json`) directly to the Jetski schema store at `~/.gemini/jetski/mcp/xla-compiler-helper/`.
3.  Append the configuration to your Gemini CLI settings (`~/.gemini/settings.json`) if available.

---

### ⚙️ Option 2: Manual Installation

### A. Jetski Agent
To manually register the MCP server in your Jetski configuration:

1.  Open your global Jetski MCP configuration file: `~/.gemini/config/mcp_config.json`.
2.  Add the following server definition inside your `mcpServers` block, making sure to include the required **`$typeName`** field:
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
3.  **Export Tool Schemas:** Jetski requires the exact JSON schemas of your tools to reason about and execute them. You must dump the schemas into your local Jetski mcp folder:
    ```bash
    blaze run //experimental/users/chrishjones/xla_ade/mcp:server -- --dump_schemas ~/.gemini/jetski/mcp/xla-compiler-helper/
    ```
4.  Restart Jetski. The tools will be successfully loaded and visible in the side panel!

### B. Gemini CLI
To manually add this MCP server to the Google-internal **Gemini CLI** (`gemini`):

#### Method 1: Using the Command Line
Run this command to register the Blaze target in your user scope:
```bash
gemini mcp add --scope=user xla-compiler-helper blaze run //experimental/users/chrishjones/xla_ade/mcp:server
```

#### Method 2: Manual Configuration
1.  Open or create the Gemini CLI settings file in your home directory: `~/.gemini/settings.json`.
2.  Add the following entry inside your `mcpServers` object:
    ```json
    "xla-compiler-helper": {
      "command": "blaze",
      "args": [
        "run",
        "//experimental/users/chrishjones/xla_ade/mcp:server"
      ],
      "cwd": "/google/src/cloud/<username>/<workspace>/google3"
    }
    ```
3.  Verify the installation by running `gemini` and typing **`/mcp`** in the interactive shell.

---

## 💡 Note on Running the Server

**⚠️ DO NOT run `python3 server.py` directly in your shell!**

This MCP server is designed to be executed exclusively via **Blaze** using `blaze run //experimental/users/chrishjones/xla_ade/mcp:server`.
*   Running via Blaze guarantees that the Google3 Python environment, the `mcp` Python SDK, and all transport configurations are hermetically packaged and executed on the user's machine without requiring any manual `pip install` or local Python package installations.
*   When registered in editors or agents, they will automatically spawn and manage the `blaze run` subprocess via standard I/O (`stdio` transport).
