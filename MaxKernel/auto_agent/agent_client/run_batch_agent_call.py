import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from auto_agent.agent_client.auto_agent_client import (
  AutoAgentClient,
  run_agent,
)

# Configure logging
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Optimize dataset kernels using AutoAgent"
  )
  parser.add_argument(
    "--data_dir", type=str, required=True, help="Path to dataset directory"
  )
  parser.add_argument(
    "--num_concurrent", type=int, default=1, help="Number of concurrent calls"
  )
  parser.add_argument(
    "--port", type=int, default=8000, help="Port for AutoAgentClient"
  )
  parser.add_argument(
    "--max_retries",
    type=int,
    default=2,
    help="Maximum number of retries for failed tasks",
  )
  parser.add_argument(
    "--log_file", type=str, default=None, help="File to save logs to"
  )
  return parser.parse_args()


def get_optimized_kernel_path(session_file_path: str):
  """Gets the optimized kernel path from the saved session JSON file."""
  with open(session_file_path, "r") as f:
    session_data = json.load(f)

  state = session_data.get("state", {})
  workdir = state.get("workdir")
  optimized_kernel_path = state.get("optimized_kernel_path")

  if optimized_kernel_path:
    if not os.path.isabs(optimized_kernel_path) and workdir:
      optimized_kernel_path = os.path.join(workdir, optimized_kernel_path)
  return optimized_kernel_path


def save_session_to_file(client: AutoAgentClient, file_path: str):
  """Fetches full session data and saves it to a local JSON file."""
  session_data = client._get_session_data()
  with open(file_path, "w") as f:
    json.dump(session_data, f, indent=2)
  logger.info(f"Saved session data to {file_path}")


def process_problem(
  problem_id: str, data_dir: str, port: int, max_retries: int
):
  problem_dir = os.path.join(data_dir, problem_id)
  reference_file = os.path.join(problem_dir, "reference.py")

  if not os.path.exists(reference_file):
    logger.error(f"reference.py not found in {problem_dir}")
    return problem_id, "Failed: reference.py missing"

  with open(reference_file, "r") as f:
    reference_code = f.read()

  query = (
    "Optimize the code below for peak performance with pallas kernel\n\n"
    + reference_code
  )

  # Retry logic
  for attempt in range(1, max_retries + 1):
    logger.info(f"Processing {problem_id} - Attempt {attempt}/{max_retries}")
    user_id = "user_0"
    session_id = f"session_{problem_id}_attempt_{attempt}_{int(time.time())}"

    client = AutoAgentClient(
      user_id=user_id,
      session_id=session_id,
      query=query,
      base_url=f"http://localhost:{port}",
    )

    try:
      run_agent(client)

      # Save session file
      session_file = os.path.join(problem_dir, "session.json")
      save_session_to_file(client, session_file)

      optimized_path = get_optimized_kernel_path(session_file)
      if not optimized_path:
        logger.error(
          f"Optimized kernel path not found in state for {problem_id}"
        )
        continue  # Try next attempt

      # Read the code from the path
      with open(optimized_path, "r") as f:
        optimized_code = f.read()

      # Paste the code to the problem folder in the dataset
      output_file = os.path.join(problem_dir, "optimized.py")
      with open(output_file, "w") as f:
        f.write(optimized_code)

      logger.info(f"Successfully saved optimized kernel to {output_file}")
      return problem_id, "Success"

    except Exception as e:
      logger.error(f"Error on attempt {attempt} for {problem_id}: {e}")
      if attempt == max_retries:
        return problem_id, f"Failed after {max_retries} attempts: {e}"

  return problem_id, "Failed"


def main():
  args = parse_args()

  if args.log_file:
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(
      logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

  if not os.path.exists(args.data_dir):
    logger.error(f"Data directory not found: {args.data_dir}")
    return

  problems = [
    d
    for d in os.listdir(args.data_dir)
    if os.path.isdir(os.path.join(args.data_dir, d))
  ]
  problems.sort()

  logger.info(f"Found {len(problems)} problems to process.")
  logger.info(
    f"Concurrent workers: {args.num_concurrent}, Max retries: {args.max_retries}"
  )

  with ThreadPoolExecutor(max_workers=args.num_concurrent) as executor:
    future_to_problem = {
      executor.submit(
        process_problem, problem, args.data_dir, args.port, args.max_retries
      ): problem
      for problem in problems
    }

    completed = 0
    for future in as_completed(future_to_problem):
      problem = future_to_problem[future]
      try:
        prob_id, status = future.result()
        completed += 1
        logger.info(
          f"[{completed}/{len(problems)}] Problem {prob_id}: {status}"
        )
      except Exception as e:
        logger.error(f"Problem {problem} raised an exception: {e}")


if __name__ == "__main__":
  main()
