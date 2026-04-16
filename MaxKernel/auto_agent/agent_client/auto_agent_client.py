import argparse
import json
import logging

import requests

REQUEST_TIMEOUT = 60 * 60 * 2


# Configure logging
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutoAgentClient:
  user_id: str
  session_id: str
  query: str
  base_url: str
  app_name: str = "auto_agent"

  def __init__(
    self,
    user_id: str,
    session_id: str,
    query: str,
    base_url: str = "http://localhost:8000",
  ):
    self.user_id = user_id
    self.session_id = session_id
    self.query = query
    self.base_url = base_url

  def create_session(self):
    session_url = f"{self.base_url}/apps/{self.app_name}/users/{self.user_id}/sessions/{self.session_id}"
    response = requests.post(
      session_url, headers={"Content-Type": "application/json"}
    )
    return response

  def send_query(self):
    run_url = f"{self.base_url}/run"
    payload = {
      "appName": self.app_name,
      "userId": self.user_id,
      "sessionId": self.session_id,
      "newMessage": {"role": "user", "parts": [{"text": self.query}]},
    }
    response = requests.post(
      run_url,
      headers={"Content-Type": "application/json"},
      data=json.dumps(payload),
      timeout=REQUEST_TIMEOUT,
    )
    return response

  def _get_session_data(self) -> dict:
    session_url = f"{self.base_url}/apps/{self.app_name}/users/{self.user_id}/sessions/{self.session_id}"
    response = requests.get(
      session_url, headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
      raise ValueError(f"Failed to get state: {response.status_code}")
    try:
      return response.json()
    except json.JSONDecodeError:
      raise ValueError("Failed to decode JSON response from server")

  def get_state(self, key: str = None):
    state_data = self._get_session_data()
    state = state_data.get("state")
    if not state:
      raise ValueError("State not found in response")

    if key is None:
      return state

    if key not in state:
      raise ValueError(f"Key '{key}' not found in state")
    return state.get(key)


def run_agent(client: AutoAgentClient):
  # Create session
  session_response = client.create_session()
  if session_response.status_code != 200:
    raise ValueError(f"Failed to create session: {session_response.text}")

  # Send query
  query_response = client.send_query()
  if query_response.status_code != 200:
    raise ValueError(f"Failed to send query: {query_response.text}")


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
  parser.add_argument(
    "--already-generated",
    action="store_true",
    help="Whether to use an already generated script",
  )
  args = parser.parse_args()

  user_id = args.user_id
  session_id = args.session_id

  query = read_query_from_file(args.query_file)

  # Create client instance
  client = AutoAgentClient(user_id, session_id, query)

  if not args.already_generated:
    # Generate and return script
    logger.info(
      f"Generating script for user {user_id} in session {session_id} with query: {query}"
    )
    run_agent(client)
  else:
    logger.info(
      f"Using already generated script for user {user_id} in session {session_id}"
    )


if __name__ == "__main__":
  main()
