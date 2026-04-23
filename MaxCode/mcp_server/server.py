"""MCP server for Accelerator Agents."""

from absl import app
from mcp.server import fastmcp

mcp = fastmcp.FastMCP("Accelerator Agents")


@mcp.tool()
def hello(name: str) -> str:
  """Returns a greeting to the given name."""
  return f"Hello {name}!"


def main(argv):
  del argv  # Unused.
  mcp.run()


if __name__ == "__main__":
  app.run(main)
