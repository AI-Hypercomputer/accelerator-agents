import asyncio
import json
import logging

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.tools import BaseTool


class DummyTool(BaseTool):
  def __init__(self):
    super().__init__(name="dummy_tool", description="Returns the number 42.")

  def __call__(self, arg: str, ctx) -> str:
    return "42"


async def main():
  leaf_agent = LlmAgent(
    name="LeafAgent",
    model="gemini-2.5-flash",
    instruction="Say 42.",
    tools=[DummyTool()],
  )

  agent = LoopAgent(
    name="MockPipelineAgent",
    sub_agents=[leaf_agent],
    max_iterations=1,
  )

  # Inject hooks
  from auto_agent.agent import _inject

  _inject(agent)

  from auto_agent.agent_client.auto_agent_client import AutoAgentClient

  client = AutoAgentClient(
    user_id="mock_user",
    session_id="mock_session",
    query="Execute the dummy tool.",
    agent=agent,
    events_compaction=False,
  )

  await client.create_session({})

  import time

  client.session.state["iteration"] = 0
  client.session.state["pipeline_start_time"] = time.time()

  await client.run_async()

  print("\n--- TIMING METRICS ---")
  print(json.dumps(client.session.state.get("timing_metrics", {}), indent=2))


if __name__ == "__main__":
  logging.basicConfig(level=logging.WARNING)
  asyncio.run(main())
