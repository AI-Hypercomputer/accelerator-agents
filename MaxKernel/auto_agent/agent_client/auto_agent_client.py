import argparse
import asyncio
import json
import logging
from typing import Any, Optional

# Load environment variables from .env file if available
try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  logging.warning(
    "dotenv not installed, skipping loading environment variables"
  )

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import Content, Part

from auto_agent.agent import root_agent

logger = logging.getLogger(__name__)


class AutoAgentClient:
  user_id: str
  session_id: str
  query: str
  app_name: str = "auto_agent"

  def __init__(
    self,
    user_id: str,
    session_id: str,
    query: str,
    agent: Optional[Any] = None,
  ):
    self.user_id = user_id
    self.session_id = session_id
    self.query = query
    self.agent = agent or root_agent
    self.session_service = InMemorySessionService()
    self.session = None

  async def create_session(self) -> None:
    self.session = await self.session_service.create_session(
      app_name=self.app_name,
      user_id=self.user_id,
      session_id=self.session_id,
    )

  def get_session_data(self) -> dict:
    if not self.session:
      raise ValueError("Session has not been created yet.")

    return json.loads(
      self.session.model_dump_json(by_alias=True, exclude_none=True)
    )

  def get_state(self, key: Optional[str] = None) -> Any:
    if not self.session:
      raise ValueError("Session has not been created yet.")

    state = self.session.state
    if key is None:
      return state

    if key not in state:
      raise ValueError(f"Key '{key}' not found in state")
    return state.get(key)

  async def run_async(self) -> None:
    if not self.session:
      await self.create_session()

    runner = Runner(
      app_name=self.app_name,
      agent=self.agent,
      session_service=self.session_service,
    )

    new_message = Content(parts=[Part(text=self.query)])

    logger.info(f"Starting in-process agent run for session {self.session_id}")
    try:
      async for event in runner.run_async(
        user_id=self.user_id,
        session_id=self.session_id,
        new_message=new_message,
      ):
        pass
      logger.info(
        f"Finished in-process agent run for session {self.session_id}"
      )
    finally:
      # Retrieve the updated session from the session service, even if the run crashed
      self.session = await self.session_service.get_session(
        app_name=self.app_name,
        user_id=self.user_id,
        session_id=self.session_id,
      )


def read_query_from_file(file_path: str) -> str:
  with open(file_path, "r") as file:
    return file.read().strip()


def main():
  # Customizable variables
  parser = argparse.ArgumentParser(description="Auto Kernel Agent Client")
  parser.add_argument(
    "--user-id", default="u_123", help="User ID for the session"
  )
  parser.add_argument(
    "--session-id", default="s_123", help="Session ID for the session"
  )
  parser.add_argument(
    "--query-file",
    default="client_query.txt",
    help="File containing the query to send",
  )
  args = parser.parse_args()

  user_id = args.user_id
  session_id = args.session_id

  query = read_query_from_file(args.query_file)

  # Create client instance
  client = AutoAgentClient(user_id, session_id, query)

  logger.info(
    f"Generating script for user {user_id} in session {session_id} with query: {query}"
  )
  asyncio.run(client.run_async())


if __name__ == "__main__":
  main()
