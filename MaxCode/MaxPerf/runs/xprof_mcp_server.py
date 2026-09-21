# pylint: skip-file
#!/usr/bin/env python3
"""xprof_mcp_server.py — Standalone FastMCP Server for the TPU XPROF Profiler.

This script runs on your TPU VM and proxies xprof TensorBoard plugin endpoints
as clean MCP tools.

Prerequisites (on TPU VM):
    pip install mcp requests

Usage:
    # Start the server in streamable HTTP mode (port 8792):
    python3 runs/xprof_mcp_server.py --transport sse --port 8792

    # Or start in stdio mode (spawns inside an assistant session):
    python3 runs/xprof_mcp_server.py --transport stdio
"""

import argparse
import sys
from mcp.server.fastmcp import FastMCP
import requests

# Initialize FastMCP
mcp = FastMCP("xprof-mcp")

# Configuration constants
XPROF_HOST = "localhost"
XPROF_PORT = 8791
XPROF_BASE_URL = f"http://{XPROF_HOST}:{XPROF_PORT}/data/plugin/profile"

def query_xprof_api(tag: str, run: str, params: dict = None) -> dict:
  """Helper to fetch data from the xprof TensorBoard plugin HTTP endpoints."""
  url = f"{XPROF_BASE_URL}/data"
  query_params = {"tag": tag, "run": run}
  if params:
    query_params.update(params)

  try:
    response = requests.get(url, params=query_params, timeout=15)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.ConnectionError:
    return {
        "status": "error",
        "message": (
            "Could not connect to the backend xprof server at"
            f" {XPROF_BASE_URL}. Verify that 'xprof --logdir=<dir>"
            f" --port={XPROF_PORT}' is running."
        ),
    }
  except Exception as e:
    return {"status": "error", "message": str(e)}

@mcp.tool()
def list_runs() -> str:
  """List all available profiling runs collected in the logdir."""
  url = f"{XPROF_BASE_URL}/runs"
  try:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    runs = response.json()
    return f"Available Profiling Runs:\n" + "\n".join(
        f"- {run}" for run in runs
    )
  except Exception as e:
    return f"Error listing runs: {str(e)}. Ensure backend xprof is running."

@mcp.tool()
def get_overview(run: str) -> str:
  """Get high-level TPU performance overview (step time, MXU utility, HBM BW, idle %)."""
  res = query_xprof_api("overview_page", run)
  return str(res)

@mcp.tool()
def get_memory_profile(run: str) -> str:
  """Get peak HBM usage, memory allocation, and fragmentation details."""
  res = query_xprof_api("memory_profile", run)
  return str(res)

@mcp.tool()
def get_top_hlo_ops(run: str) -> str:
  """Get top HLO operations ranked by execution time, FLOPs, and bytes accessed."""
  res = query_xprof_api("hlo_stats", run)
  return str(res)

@mcp.tool()
def get_op_profile(run: str) -> str:
  """Get hierarchical operation breakdown (useful for serving decode steps)."""
  res = query_xprof_api("op_profile", run)
  return str(res)

@mcp.tool()
def get_device_information(run: str) -> str:
  """Get details about accelerator hardware limits and critical intensity points."""
  res = query_xprof_api("roofline_model", run)
  return str(res)

@mcp.tool()
def list_hlo_modules(run: str) -> str:
  """List all compiled HLO program module names in the run."""
  res = query_xprof_api("module_list", run)
  return str(res)

@mcp.tool()
def get_hlo_module_content(run: str, module: str) -> str:
  """Get the full HLO text representation for a compiled module."""
  res = query_xprof_api(
      "graph_viewer", run, {"type": "long_txt", "module": module}
  )
  return str(res)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="xprof-mcp FastMCP Server")
  parser.add_argument(
      "--transport",
      choices=["stdio", "sse"],
      default="sse",
      help="MCP transport layer to use (default: sse)",
  )
  parser.add_argument(
      "--port",
      type=int,
      default=8792,
      help="Port to run the SSE server on (default: 8792)",
  )
  args = parser.parse_args()

  print(
      f"Starting xprof-mcp server using {args.transport} transport...",
      file=sys.stderr,
  )
  if args.transport == "sse":
    print(f"SSE host running on port {args.port}...", file=sys.stderr)
    mcp.settings.port = args.port
    mcp.run(transport="sse")
  else:
    mcp.run(transport="stdio")
