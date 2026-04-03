import argparse
import json
import logging

import requests

from tpu_kernel_gen.agents.kernel_gen_agent.constants import REQUEST_TIMEOUT

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class KernelGenClient:
  user_id: str
  session_id: str
  query: str
  base_url: str
  app_name: str = "kernel_gen_agent"

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
    response = requests.post(session_url, headers={"Content-Type": "application/json"})
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

  def get_results(self):
    session_url = f"{self.base_url}/apps/{self.app_name}/users/{self.user_id}/sessions/{self.session_id}"
    response = requests.get(session_url, headers={"Content-Type": "application/json"})
    try:
      state_data = response.json()
      state = state_data.get("state")
      base_kernel_compilation_result = state.get("base_kernel_compilation_result")
      base_kernel_correctness_result = state.get("base_kernel_correctness_result")
      tiled_kernel_compilation_result = state.get("tiled_kernel_compilation_result")
      tiled_kernel_correctness_result = state.get("tiled_kernel_correctness_result")

      fix_jax_base_code_loop_iter = state.get("fix_jax_base_code_loop_iter", 0)
      fix_base_kernel_loop_iter = state.get("fix_base_kernel_loop_iter", 0)
      fix_tiled_kernel_loop_iter = state.get("fix_tiled_kernel_loop_iter", 0)

      kernel_tiling_optimization_result = state.get("kernel_tiling_optimization_result", None)

      return {
        "fix_jax_base_code_loop_iter": fix_jax_base_code_loop_iter,
        "fix_base_kernel_loop_iter": fix_base_kernel_loop_iter,
        "fix_tiled_kernel_loop_iter": fix_tiled_kernel_loop_iter,
        "base_kernel_compilation_result": base_kernel_compilation_result,
        "base_kernel_correctness_result": base_kernel_correctness_result,
        "tiled_kernel_compilation_result": tiled_kernel_compilation_result,
        "tiled_kernel_correctness_result": tiled_kernel_correctness_result,
        "kernel_tiling_optimization_result": kernel_tiling_optimization_result,
      }
    except:
      status_code = response.status_code
      if status_code == 200:
        raise ValueError("State or result not found in response")
      else:
        raise ValueError(f"Failed to get state: {status_code}")

  def get_state(self, key: str):
    session_url = f"{self.base_url}/apps/{self.app_name}/users/{self.user_id}/sessions/{self.session_id}"
    response = requests.get(session_url, headers={"Content-Type": "application/json"})
    try:
      state_data = response.json()
      state = state_data.get("state")
      return state.get(key)
    except:
      status_code = response.status_code
      if status_code == 200:
        raise ValueError(f"Key '{key}' not found in state")
      else:
        raise ValueError(f"Failed to get state: {status_code}")


def generate_script(client: KernelGenClient):
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


def save_script_to_file(script: str, file_path: str):
  if script.startswith("```python"):
    script = script[9:]
  if script.endswith("```"):
    script = script[:-3]
  script = script.strip()
  with open(file_path, "w") as file:
    file.write(script)
  logger.info(f"Script saved to {file_path}")


def main():
  # Customizable variables
  parser = argparse.ArgumentParser(description="Kernel Generation Client")
  parser.add_argument("--user-id", default="u_123", help="User ID for the session")
  parser.add_argument("--session-id", default="s_123", help="Session ID for the session")
  parser.add_argument(
    "--query-file",
    default="client_query.txt",
    help="File containing the query to send",
  )
  parser.add_argument(
    "--already_generated",
    default=False,
    help="Whether to use an already generated script",
  )
  parser.add_argument(
    "--output-file",
    default="base_kernel_code.py",
    help="File to save the generated script",
  )
  args = parser.parse_args()

  user_id = args.user_id
  session_id = args.session_id

  query = read_query_from_file(args.query_file)

  # Create client instance
  client = KernelGenClient(user_id, session_id, query)

  if not args.already_generated:
    # Generate and return script
    logger.info(f"Generating script for user {user_id} in session {session_id} with query: {query}")
    generate_script(client)
  else:
    logger.info(f"Using already generated script for user {user_id} in session {session_id}")

  base_kernel_code = client.get_state("base_kernel_code")

  logger.info("Results: %s", client.get_results())

  save_script_to_file(base_kernel_code, args.output_file)


if __name__ == "__main__":
  main()
