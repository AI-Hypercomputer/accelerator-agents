import asyncio
import json
import logging
import os
import time

from auto_agent.agent import create_root_agent
from auto_agent.agent_client.auto_agent_client import AutoAgentClient


async def run_fast():
  session_dir = "/home/stingram_google_com/timing/accelerator-agents/MaxKernel/auto_agent/mock_session"
  os.makedirs(session_dir, exist_ok=True)
  agent = create_root_agent(session_dir=session_dir, max_iterations=1)

  with open(os.path.join(session_dir, "base_kernel.py"), "w") as f:
    f.write("def pallas_kernel(): pass")

  client = AutoAgentClient(
    user_id="mock_user",
    session_id="mock_session",
    query="Optimize the generic kernel base code given.",
    agent=agent,
    events_compaction=False,
  )

  await client.create_session({})
  client.session.state["iteration"] = 0
  client.session.state["pipeline_start_time"] = time.time()

  print(
    "Starting pipeline (letting it run for 15s to capture at least 1 LLM call)..."
  )
  task = asyncio.create_task(client.run_async())

  await asyncio.sleep(15)
  task.cancel()
  try:
    print("Aborting early to check metrics...")
    await task
  except asyncio.CancelledError:
    pass

  print("\n--- CAPTURED TIMING METRICS (partial run) ---")
  print(json.dumps(client.session.state.get("timing_metrics", {}), indent=2))


if __name__ == "__main__":
  logging.getLogger("google.adk").setLevel(logging.CRITICAL)
  asyncio.run(run_fast())
