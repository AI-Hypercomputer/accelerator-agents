#!/bin/bash
# Installation script for XLA ADE MCP Server in Jetski and Gemini CLI

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
GOOGLE3_DIR=$(echo "$SCRIPT_DIR" | grep -o '.*/google3')
export GOOGLE3_DIR

echo "=================================================="
echo "Installing XLA Compiler Assistant MCP..."
echo "=================================================="

# 1. Setup Jetski Configuration
JETSKI_CONFIG_DIR="$HOME/.gemini/config"
JETSKI_CONFIG="$JETSKI_CONFIG_DIR/mcp_config.json"
mkdir -p "$JETSKI_CONFIG_DIR"

if [ ! -f "$JETSKI_CONFIG" ]; then
  echo '{"mcpServers": {}}' > "$JETSKI_CONFIG"
fi

echo "Adding server entry to Jetski config ($JETSKI_CONFIG)..."
python3 - <<EOF
import json
import os

config_path = "$JETSKI_CONFIG"
google3_dir = os.environ.get("GOOGLE3_DIR")
with open(config_path, "r") as f:
    data = json.load(f)

if "mcpServers" not in data:
    data["mcpServers"] = {}

data["mcpServers"]["xla-compiler-helper"] = {
    "\$typeName": "exa.cascade_plugins_pb.CascadePluginCommandTemplate",
    "command": os.path.join(google3_dir, "blaze-bin/experimental/users/chrishjones/xla_ade/mcp/server"),
    "args": [],
    "cwd": google3_dir
}

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
EOF

# 2. Export Schemas for Jetski
JETSKI_SCHEMA_DIR="$HOME/.gemini/jetski/mcp/xla-compiler-helper"
echo "Generating Jetski tool schema files in $JETSKI_SCHEMA_DIR..."
blaze run //experimental/users/chrishjones/xla_ade/mcp:server -- --dump_schemas "$JETSKI_SCHEMA_DIR"

# 3. Setup Gemini CLI Configuration
GEMINI_CLI_CONFIG="$HOME/.gemini/settings.json"
if [ -f "$GEMINI_CLI_CONFIG" ]; then
  echo "Adding server entry to Gemini CLI config ($GEMINI_CLI_CONFIG)..."
  python3 - <<EOF
import json
import os
config_path = "$GEMINI_CLI_CONFIG"
google3_dir = os.environ.get("GOOGLE3_DIR")
with open(config_path, "r") as f:
    data = json.load(f)

if "mcpServers" not in data:
    data["mcpServers"] = {}

data["mcpServers"]["xla-compiler-helper"] = {
    "command": os.path.join(google3_dir, "blaze-bin/experimental/users/chrishjones/xla_ade/mcp/server"),
    "args": [],
    "cwd": google3_dir
}

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
EOF
else
  echo "Gemini CLI config not found at $GEMINI_CLI_CONFIG. Skipping. (You can install it manually later or run 'gemini mcp add')"
fi

echo "=================================================="
echo "🎉 XLA MCP Server Installation Complete!"
echo "Please restart Jetski or Gemini CLI to load the new tools."
echo "=================================================="
