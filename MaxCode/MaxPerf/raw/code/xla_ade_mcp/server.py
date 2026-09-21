"""MCP server for XLA ADE (Active Documentation Engine)."""

import asyncio
import json
import os
from absl import app
from absl import flags
from mcp import types
from mcp.server import fastmcp
import requests

SERVER_URL = os.environ.get(
    "XLA_ADE_SERVER", "http://chrishjones.c.googlers.com:8080"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dump_schemas",
    None,
    "If specified, writes out the JSON schemas of all registered tools to this"
    " directory and exits.",
)

mcp = fastmcp.FastMCP("XlaAde")


@mcp.tool(name="Question")
def ask_question(message: str, session_id: str | None = None) -> str:
  """Answers any complex XLA, compiler, flag, or Pallas question.

  Automatically searches the XLA wiki, XLA flags database, and Pallas guide
  pages, and synthesizes a complete, dense technical response. Use this tool as
  the primary interface for any XLA compiler related queries. Supports
  follow-up questions by passing back the returned session_id.

  Args:
      message: The technical question or follow-up question about XLA/Pallas.
      session_id: Optional. Pass the session_id returned from a previous call to
        continue the conversation in-context.

  Returns:
      A formatted string containing the synthesized response and the session_id.
  """
  try:
    body = {"message": message, "session_id": session_id}
    response = requests.post(
        f"{SERVER_URL}/api/mcp/chat", json=body, timeout=300
    )
    response.raise_for_status()
    data = response.json()

    reply = data.get("reply", "No response received.")
    new_session_id = data.get("session_id", "")

    return f"Reply:\n{reply}\n\n[Session ID: {new_session_id}]"
  except Exception as e:  # pylint: disable=broad-except
    return f"Error calling XLA Helper API: {e}"


@mcp.tool(name="FlagSearch")
def flag_search(flag_name: str) -> str:
  """Queries details and empirical experiment results for an XLA compiler flag.

  Args:
      flag_name: The name of the XLA flag, optionally prefixed with hyphens
        (e.g., '--allow_spmd_sharding_propagation_to_output' or
        'allow_spmd_sharding_propagation_to_output').

  Returns:
      A formatted markdown string containing the flag's description, category,
      default values, and detailed outcomes/metrics from previous empirical TPU
      experiments.
  """
  try:
    response = requests.get(
        f"{SERVER_URL}/api/mcp/flag", params={"name": flag_name}, timeout=15
    )
    if response.status_code == 404:
      return f"Flag '{flag_name}' not found."
    response.raise_for_status()
    data = response.json()

    meta = data.get("metadata", {})
    results = data.get("results", [])

    output = []
    output.append(f"# Flag: {flag_name.lstrip('-')}")
    output.append("")
    output.append("## Flag Metadata")
    for key, val in sorted(meta.items()):
      if isinstance(val, list):
        val_str = ", ".join([str(x) for x in val])
      elif isinstance(val, dict):
        val_str = str(val)
      else:
        val_str = str(val)
      output.append(f"* **{key.replace('_', ' ').title()}:** {val_str}")

    output.append("")
    output.append("## Empirical TPU Experiment Results")
    if not results:
      output.append("No empirical experiment runs found for this flag.")
    else:
      for idx, exp in enumerate(results):
        output.append(f"### Run {idx+1}: {exp.get('topic_id', 'N/A')}")
        for key, val in sorted(exp.items()):
          if key == "topic_id":
            continue
          if isinstance(val, list):
            val_str = ", ".join([str(x) for x in val])
          else:
            val_str = str(val)
          output.append(f"  * **{key.replace('_', ' ').title()}:** {val_str}")
        output.append("")

    return "\n".join(output)
  except Exception as e:  # pylint: disable=broad-except
    return f"Error calling flag API: {e}"


@mcp.prompt()
def discovery_flow() -> list[types.PromptMessage]:
  """Returns the recommended Discovery Flow for XLA ADE."""
  return [
      types.PromptMessage(
          content=types.TextContent(
              type="text",
              text=(
                  "To research an XLA compiler issue, draft a Pallas kernel, or"
                  " analyze a flag, follow this Discovery Flow:\n\n1. **Ask the"
                  " XLA Helper**: Call the `Question` tool with your query"
                  " (e.g."
                  " 'explain how stablehlo is lowered to llo').\n2. **Follow-up"
                  " Interactively**: If you need clarification or deeper"
                  " details"
                  " on their response, call the `Question` tool again with your"
                  " follow-up, making sure to pass the returned `session_id`"
                  " from the previous turn to maintain full context.\n3. **Flag"
                  " Deep Dive**: If you want a detailed lookup and empirical"
                  " test outcomes of a specific XLA flag, call the"
                  " `FlagSearch` tool (e.g.,"
                  " `FlagSearch(flag_name='--allow_spmd_sharding_propagation_to_output')`).\n"
              ),
          ),
          role="user",
      )
  ]


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.dump_schemas:
    target_dir = FLAGS.dump_schemas
    os.makedirs(target_dir, exist_ok=True)

    mcp_tools = asyncio.run(mcp.list_tools())

    for tool in mcp_tools:
      tool_dict = {
          "name": tool.name,
          "description": tool.description,
          "inputSchema": tool.inputSchema,
      }

      schema_path = os.path.join(target_dir, f"{tool.name}.json")
      with open(schema_path, "w") as f:
        json.dump(tool_dict, f, indent=2)
      print(f"Exported schema to {schema_path}")
    return

  mcp.run(transport="stdio")


if __name__ == "__main__":
  app.run(main)
