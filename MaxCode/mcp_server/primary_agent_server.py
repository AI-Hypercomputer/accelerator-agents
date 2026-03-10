"""MCP server for Primary Agent."""

import logging
import os
import re

from absl import app
from mcp_server import adk_agents
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from mcp.server import fastmcp

logging.basicConfig(level=logging.INFO)
mcp = fastmcp.FastMCP("Primary Agent")

# Configure file logging
try:
  file_handler = logging.FileHandler("/tmp/agent_server.log", mode="a")
  file_handler.setLevel(logging.INFO)
  formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  )
  file_handler.setFormatter(formatter)
  logging.getLogger().addHandler(file_handler)
  logging.info("File logging configured to /tmp/agent_server.log")
except OSError as e:
  logging.error("Failed to configure file logging: %s", e)


@mcp.tool()
async def run_agent(
    prompt: str, api_key: str = "", *, ctx: fastmcp.Context
) -> str:
  """Runs the Primary Agent using ADK primitives.

  Args:
    prompt: The prompt to pass to the agent.
    api_key: The Google AI API key to use for migration.
    ctx: The MCP context for streaming progress.
  """
  handler_to_reset = None
  for h in logging.getLogger().handlers:
    if hasattr(h, "baseFilename") and h.baseFilename == os.path.abspath(
        "/tmp/agent_server.log"
    ):
      handler_to_reset = h
      break
  if handler_to_reset:
    handler_to_reset.close()
    handler_to_reset.mode = "w"
  logging.info("Job started, clearing agent server log.")
  if handler_to_reset:
    handler_to_reset.mode = "a"
  logging.info(
      "Received run_agent call. Prompt: %s, API Key: %s", prompt, api_key
  )
  await ctx.info("Starting ADK Agent task...")
  logging.info("Starting ADK Agent task...")
  try:
    effective_api_key = api_key
    if not effective_api_key:
      match = re.search(r"using API key (.+)", prompt, re.IGNORECASE)
      if match:
        effective_api_key = match.group(1).strip().rstrip(".")
        logging.info("API Key extracted from prompt: %s", effective_api_key)

    # 1. Handle API Key (Set in env so ADK model client finds it)
    if effective_api_key:
      os.environ["GOOGLE_API_KEY"] = effective_api_key
    session_service = InMemorySessionService()
    runner = Runner(
        agent=adk_agents.master_agent,
        app_name="maxcode",
        session_service=session_service,
    )
    await session_service.create_session(
        app_name="maxcode", session_id="session", user_id="user"
    )
    response_text = []
    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
      if calls := event.get_function_calls():
        for call in calls:
          logging.info("[%s] Executing tool: '%s'", event.author, call.name)
          await ctx.info(f"[{event.author}] Executing tool: '{call.name}'")
      if responses := event.get_function_responses():
        for resp in responses:
          logging.info("[%s] Tool '%s' finished.", event.author, resp.name)
          await ctx.info(f"[{event.author}] Tool '{resp.name}' finished.")
      if (
          not event.is_final_response
          and event.content
          and event.content.parts
          and event.content.parts[0].text
      ):
        logging.info("[%s] %s", event.author, event.content.parts[0].text)
        await ctx.info(f"[{event.author}] {event.content.parts[0].text}")
      if (
          event.is_final_response
          and event.content
          and event.content.parts
          and event.content.parts[0].text
      ):
        response_text.append(event.content.parts[0].text)
    await ctx.info("ADK Agent task completed.")
    logging.info("ADK Agent task completed.")
    return "\n".join(response_text)
  except (
      ValueError,
      TypeError,
      IndexError,
      RuntimeError,
  ) as err:
    logging.exception("Exception in run_agent")
    await ctx.error(f"Error during agent execution: {err}")
    return f"Error during agent execution: {err}"


def main(argv):
  del argv  # Unused.
  mcp.run()


if __name__ == "__main__":
  app.run(main)
