import argparse
import asyncio
import json
import logging
import os
from typing import Any, Tuple

from auto_search.run_search import run_search, setup_logging

logger = logging.getLogger(__name__)


async def process_problem(
  problem_dir: str,
  algorithm: str,
  sem: asyncio.Semaphore,
  **kwargs: Any,
) -> Tuple[str, str]:
  """Executes the search algorithm for a single benchmark problem in the batch."""
  async with sem:
    try:
      return await run_search(
        problem_dir=problem_dir,
        algorithm=algorithm,
        **kwargs,
      )
    except Exception as e:
      problem_id = os.path.basename(os.path.normpath(problem_dir))
      logger.error(
        f"Error executing search on {problem_id}: {e}", exc_info=True
      )
      return problem_id, f"Failed with exception: {e}"


async def run_batch_search(
  data_dir: str,
  algorithm: str = "parallel",
  num_problem_concurrency: int = 1,
  **kwargs: Any,
):
  """Coordinates concurrent problem execution across the dataset."""
  if not os.path.isdir(data_dir):
    logger.error(
      f"Dataset directory not found or not a directory: {data_dir}"
    )
    return

  data_dir_valid = [
    os.path.join(data_dir, d)
    for d in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, d, "reference.py"))
  ]
  data_dir_valid.sort()

  if not data_dir_valid:
    logger.warning(
      f"No valid benchmark problems (directories containing reference.py) found in {data_dir}"
    )
    return

  logger.info(f"Found {len(data_dir_valid)} problems to process.")
  max_concurrency = kwargs.get("max_concurrency", 2)
  logger.info(
    f"Algorithm: {algorithm}, Problem Concurrency:"
    f" {num_problem_concurrency}, Worker Concurrency: {max_concurrency}"
  )

  sem = asyncio.Semaphore(num_problem_concurrency)
  tasks = [
    process_problem(
      problem_dir=problem_dir,
      algorithm=algorithm,
      sem=sem,
      **kwargs,
    )
    for problem_dir in data_dir_valid
  ]

  completed = 0
  results = []
  for future in asyncio.as_completed(tasks):
    try:
      prob_id, status = await future
      completed += 1
      results.append((prob_id, status))
      logger.info(
        f"[{completed}/{len(data_dir_valid)}] Problem {prob_id}: {status}"
      )
    except Exception as e:
      logger.error(f"A task raised an exception: {e}")

  logger.info("\n--- Search Execution Summary ---")
  for prob_id, status in sorted(results):
    logger.info(f"{prob_id}: {status}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run Auto-Search algorithms against batch dataset."
  )

  # General & Orchestration Arguments
  orch_group = parser.add_argument_group(
    "General & Orchestration Arguments",
    "Arguments shared across all algorithms and orchestrators.",
  )
  orch_group.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to benchmark dataset directory",
  )
  orch_group.add_argument(
    "--algorithm",
    type=str,
    choices=["parallel", "beam", "agentic"],
    default="parallel",
    help="Search algorithm to execute",
  )
  orch_group.add_argument(
    "--max_concurrency",
    type=int,
    default=2,
    help="Max concurrent worker expansions",
  )
  orch_group.add_argument(
    "--num_problem_concurrency",
    type=int,
    default=1,
    help="Number of dataset problems to run concurrently",
  )
  orch_group.add_argument(
    "--log_file",
    type=str,
    default=None,
    help="File to save logs to",
  )
  # Parallel Search Arguments
  parallel_group = parser.add_argument_group(
    "Parallel Search Arguments",
    "Parameters specific to the 'parallel' search algorithm.",
  )
  parallel_group.add_argument(
    "--num_parallel_runs",
    type=int,
    default=2,
    help="Number of parallel runs",
  )
  parallel_group.add_argument(
    "--max_retries",
    type=int,
    default=1,
    help="Max worker retries per expansion task",
  )
  parallel_group.add_argument(
    "--strategies",
    nargs="+",
    type=str,
    default=None,
    help="List of strategy strings to explore",
  )
  parallel_group.add_argument(
    "--agent_config",
    type=str,
    default=None,
    help="JSON string of agent config parameters (e.g. '{\"max_iterations\": 5}')",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  setup_logging(args.log_file)

  parsed_agent_config = None
  if args.agent_config:
    try:
      parsed_agent_config = json.loads(args.agent_config)
    except json.JSONDecodeError as e:
      logger.error(f"Invalid JSON string for --agent_config: {e}")
      return

  kwargs = {
    "max_concurrency": args.max_concurrency,
    "num_parallel_runs": args.num_parallel_runs,
    "max_worker_retries": args.max_retries,
    "strategies": args.strategies,
    "agent_config": parsed_agent_config,
  }

  asyncio.run(
    run_batch_search(
      data_dir=args.data_dir,
      algorithm=args.algorithm,
      num_problem_concurrency=args.num_problem_concurrency,
      **kwargs,
    )
  )


if __name__ == "__main__":
  main()
