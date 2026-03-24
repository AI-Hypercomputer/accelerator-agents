import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tpu_kernel_gen.agents.kernel_gen_agent.client import (
  KernelGenClient,
  generate_script,
  read_query_from_file,
  save_script_to_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
  parser = argparse.ArgumentParser(description="Kernel benchmark evaluation")
  parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory")
  parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
  parser.add_argument("--num_concurrent", type=int, default=1, help="Number of concurrent evaluations")
  parser.add_argument("--level", type=int, help="KernelBench level of evaluation")
  parser.add_argument(
    "--subset",
    type=str,
    help="Subset of problems to generate",
  )
  parser.add_argument("--port", type=int, default=8000, help="Port for KernelGenClient")
  parser.add_argument(
    "--max_retries",
    type=int,
    default=2,
    help="Maximum number of retries for failed tasks",
  )
  return parser.parse_args()


def save_result_to_json(problem_id: str, result: str, eval_json_path: str, json_lock: threading.Lock):
  """Thread-safe function to save results to JSON file"""
  with json_lock:
    with open(eval_json_path, "r") as f:
      eval_data = json.load(f)

    eval_data[f"{problem_id}"] = result

    with open(eval_json_path, "w") as f:
      json.dump(eval_data, f, indent=2)


def e2e_single_script(
  user_id: str,
  session_id: str,
  query: str,
  exp_dir: str,
  level: int,
  problem_id: str,
  port: int,
  eval_json_path: str = None,
  json_lock: threading.Lock = None,
):
  client = KernelGenClient(
    user_id=user_id,
    session_id=session_id,
    query=query,
    base_url=f"http://localhost:{port}",
  )
  try:
    generate_script(client)
  except Exception as e:
    logging.error(f"[{user_id=}] [{session_id=}] Error generating script: {e}")
    if not ("Read timed out" in str(e) or "HTTPConnectionPool" in str(e)):
      raise e

  try:
    if eval_json_path and json_lock:
      save_result_to_json(problem_id, client.get_results(), eval_json_path, json_lock)

    problem_dir = os.path.join(exp_dir, f"level_{level}_problem_{problem_id}")
    os.makedirs(problem_dir, exist_ok=True)

    jax_base_code = client.get_state("jax_base_code")
    save_script_to_file(
      jax_base_code,
      os.path.join(problem_dir, f"level_{level}_problem_{problem_id}_sample_0_jax_base.py"),
    )

    base_kernel_code = client.get_state("base_kernel_code")
    save_script_to_file(
      base_kernel_code,
      os.path.join(problem_dir, f"level_{level}_problem_{problem_id}_sample_0_base_kernel.py"),
    )

    base_kernel_correctness_test_code = client.get_state("base_kernel_correctness_test_code")
    save_script_to_file(
      base_kernel_correctness_test_code,
      os.path.join(
        problem_dir,
        f"level_{level}_problem_{problem_id}_sample_0_base_kernel_correctness_test.py",
      ),
    )

    tiled_kernel_code = client.get_state("tiled_kernel_code")
    save_script_to_file(
      tiled_kernel_code,
      os.path.join(
        problem_dir,
        f"level_{level}_problem_{problem_id}_sample_0_tiled_kernel.py",
      ),
    )

    tiled_kernel_correctness_test_code = client.get_state("tiled_kernel_correctness_test_code")
    save_script_to_file(
      tiled_kernel_correctness_test_code,
      os.path.join(
        problem_dir,
        f"level_{level}_problem_{problem_id}_sample_0_tiled_kernel_correctness_test.py",
      ),
    )

    # performance_test_code = client.get_state(
    #     "performance_test_code"
    # )
    # save_script_to_file(
    #     performance_test_code,
    #     os.path.join(
    #         problem_dir,
    #         f"level_{level}_problem_{problem_id}_sample_0_kernel_performance_test.py",
    #     ),
    # )

    tile_tuning_script = client.get_state("tile_tuning_script")
    save_script_to_file(
      tile_tuning_script,
      os.path.join(
        problem_dir,
        f"level_{level}_problem_{problem_id}_sample_0_kernel_tile_tuning_script.py",
      ),
    )

  except Exception as e:
    logging.error(f"[{user_id=}] [{session_id=}] Error saving scripts or getting results: {e}")
    raise e


def main():
  args = parse_args()

  files = [
    f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f)) and f.split("_")[0].isdigit()
  ]
  files.sort(key=lambda x: int(x.split("_")[0]))

  if args.subset:
    start, end = map(int, args.subset.strip("()").split(","))
    files = [f for f in files if start <= int(f.split("_")[0]) <= end]

  num_files = len(files)

  os.makedirs(args.exp_dir, exist_ok=True)
  eval_json_path = os.path.join(args.exp_dir, "eval.json")
  if not os.path.exists(eval_json_path):
    with open(eval_json_path, "w") as f:
      f.write("{}")

  # Lock for thread-safe JSON file writing
  json_lock = threading.Lock()

  def process_single_file(file_path: str, retry_count: int) -> str:
    """Process a single file and return problem_id and result"""
    problem_id = file_path.split("_")[0]
    logging.info(f"Processing {problem_id} attempt {retry_count}/{args.max_retries}")

    user_id = "user_0"
    session_id = f"session_level{args.level}_problem{problem_id}_attempt{retry_count}"
    query = read_query_from_file(os.path.join(args.data_dir, file_path))
    dst_file = os.path.join(
      args.exp_dir,
      f"level_{args.level}_problem_{problem_id}_sample_0_kernel.py",
    )

    e2e_single_script(
      user_id,
      session_id,
      query,
      args.exp_dir,
      args.level,
      problem_id,
      args.port,
      eval_json_path,
      json_lock,
    )
    logging.info(f"Completed processing {problem_id}")
    return problem_id

  logger.info(f"Total files: {num_files}, Concurrent workers: {args.num_concurrent}, Max retries: {args.max_retries}")

  # Use ThreadPoolExecutor for dynamic concurrent processing
  with ThreadPoolExecutor(max_workers=args.num_concurrent) as executor:
    # Track retry counts for each file
    retry_counts = {file_path: 0 for file_path in files}

    # Submit all tasks initially
    future_to_file = {
      executor.submit(process_single_file, file_path, retry_counts[file_path]): file_path for file_path in files
    }

    completed_count = 0
    total_attempts = len(files)

    # Process completed tasks as they finish
    while future_to_file:
      completed_futures = []

      for future in as_completed(future_to_file.copy()):
        file_path = future_to_file.pop(future)

        try:
          # This is non-blocking since as_completed only yields completed futures
          problem_id = future.result(timeout=0)

          completed_count += 1
          logger.info(f"Completed {completed_count}/{num_files}: Problem {problem_id}")

        except Exception as e:
          problem_id = file_path.split("_")[0]
          retry_counts[file_path] += 1

          if retry_counts[file_path] <= args.max_retries:
            # Check if the exception is from script generation
            if "Internal Server Error" in str(e):
              # Re-submit the failed task only for script generation errors
              new_future = executor.submit(process_single_file, file_path, retry_counts[file_path])
              future_to_file[new_future] = file_path
              total_attempts += 1

              logger.warning(
                f"Retrying problem {problem_id} (attempt {retry_counts[file_path]}/{args.max_retries}) for script generation error: {e}"
              )
            else:
              # Non-generation error, don't retry
              logger.error(f"Failed problem {problem_id} with non-generation error (not retrying): {e}")
              completed_count += 1
              logging.info(f"Completed {completed_count}/{num_files}: Problem {problem_id}")

          else:
            # Max retries exceeded, log final failure
            logger.error(f"Failed problem {problem_id} after {args.max_retries} retries: {e}")
            completed_count += 1
            logging.info(f"Completed {completed_count}/{num_files}: Problem {problem_id}")

        # Process one completed future at a time to maintain responsiveness
        break

  logger.info(f"All tasks completed. Total attempts: {total_attempts}")


if __name__ == "__main__":
  main()
