import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Optional, Tuple

try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  logging.warning(
    "dotenv not installed, skipping loading environment variables"
  )

from auto_search.algorithms.parallel_search import (
  SimpleParallelSearchOrchestrator,
)
from auto_search.orchestrator import SearchOrchestrator

logger = logging.getLogger(__name__)


def get_orchestrator(
  algorithm: str,
  problem_id: str,
  reference_code: str,
  graph_db_path: str,
  **kwargs: Any,
) -> SearchOrchestrator:
  """Factory function to instantiate the requested search orchestrator."""
  if algorithm == "parallel":
    return SimpleParallelSearchOrchestrator(
      problem_id=problem_id,
      reference_code=reference_code,
      graph_db_path=graph_db_path,
      max_concurrency=kwargs.get("max_concurrency", 2),
      num_parallel_runs=kwargs.get("num_parallel_runs", 2),
      strategies=kwargs.get("strategies"),
      max_worker_retries=kwargs.get("max_worker_retries", 1),
      agent_config=kwargs.get("agent_config"),
    )
  elif algorithm in ("beam", "agentic"):
    raise NotImplementedError(
      f"Algorithm '{algorithm}' is currently a placeholder."
    )
  else:
    raise ValueError(f"Unknown algorithm: {algorithm}")


def save_optimized_kernel(
  orchestrator: SearchOrchestrator,
  problem_dir: str,
  algorithm: str,
) -> Optional[Tuple[str, float]]:
  """Saves the best kernel code found during search to the problem directory."""
  best_id = orchestrator.graph.best_node_id
  if not best_id or best_id not in orchestrator.graph.nodes:
    return None

  best_node = orchestrator.graph.nodes[best_id]
  if not best_node.code:
    return None

  output_file = os.path.join(problem_dir, f"optimized_{algorithm}.py")
  with open(output_file, "w") as f:
    f.write(best_node.code)

  logger.info(f"Saved best kernel ({best_id}) to {output_file}")
  latency = best_node.evaluation.latency_ms or -1.0
  return best_id, latency


async def run_search(
  problem_dir: str,
  algorithm: str = "parallel",
  graph_db_path: Optional[str] = None,
  **kwargs: Any,
) -> Tuple[str, str]:
  """Executes the search algorithm asynchronously for a single problem directory."""
  problem_id = os.path.basename(os.path.normpath(problem_dir))
  reference_file = os.path.join(problem_dir, "reference.py")

  if not os.path.exists(reference_file):
    logger.error(f"reference.py not found in {problem_dir}")
    return problem_id, "Failed: reference.py missing"

  with open(reference_file, "r") as f:
    reference_code = f.read()

  if not graph_db_path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_subdir = os.path.join(
      os.environ.get("WORKDIR"),
      "search_runs",
      f"{problem_id}_run_{algorithm}_{timestamp}",
    )
    os.makedirs(run_subdir, exist_ok=True)
    graph_db_path = os.path.join(run_subdir, "search_graph.json")
  else:
    os.makedirs(os.path.dirname(os.path.abspath(graph_db_path)), exist_ok=True)

  logger.info(
    f"Starting '{algorithm}' search on {problem_id} (graph: {graph_db_path})"
  )

  try:
    orchestrator = get_orchestrator(
      algorithm=algorithm,
      problem_id=problem_id,
      reference_code=reference_code,
      graph_db_path=graph_db_path,
      **kwargs,
    )

    if os.path.exists(graph_db_path):
      logger.info(f"Existing graph found for {problem_id}. Resuming...")
      orchestrator.resume()

    await orchestrator.run()

    best_result = save_optimized_kernel(orchestrator, problem_dir, algorithm)
    if best_result:
      best_id, latency = best_result
      return problem_id, f"Success (Best: {best_id}, Latency: {latency} ms)"

    return problem_id, "Completed (No valid candidate found)"

  except Exception as e:
    logger.error(f"Error executing search on {problem_id}: {e}", exc_info=True)
    return problem_id, f"Failed: {e}"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run Auto-Search algorithm on a single problem directory."
  )
  # General & Orchestration Arguments
  orch_group = parser.add_argument_group(
    "General & Orchestration Arguments",
    "Arguments shared across all algorithms and orchestrators.",
  )
  orch_group.add_argument(
    "--problem_dir",
    type=str,
    required=True,
    help="Path to problem directory containing reference.py",
  )
  orch_group.add_argument(
    "--algorithm",
    type=str,
    choices=["parallel", "beam", "agentic"],
    default="parallel",
    help="Search algorithm to execute",
  )
  orch_group.add_argument(
    "--graph_db_path",
    type=str,
    default=None,
    help="Explicit path to existing graph JSON to resume from (or write to)",
  )
  orch_group.add_argument(
    "--max_concurrency",
    type=int,
    default=2,
    help="Max concurrent worker expansions",
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


def setup_logging(log_file: Optional[str]):
  """Configures console and file logging."""
  log_format = "%(asctime)s - %(levelname)s - %(message)s"
  if log_file:
    # 1. Configure the main runner logger (write only main script progress to the main log file)
    main_logger = logging.getLogger("__main__")
    main_logger.propagate = False

    batch_handler = logging.FileHandler(log_file)
    batch_handler.setFormatter(logging.Formatter(log_format))
    main_logger.addHandler(batch_handler)
    main_logger.setLevel(logging.INFO)

    # 2. Configure the root logger (write all internal and dependency logs to a separate file)
    base, ext = os.path.splitext(log_file)
    agent_log_file = f"{base}_agent{ext}"
    logging.basicConfig(
      filename=agent_log_file,
      level=logging.INFO,
      format=log_format,
      force=True,
    )
  else:
    # Route all logs to the console/terminal
    logging.basicConfig(
      level=logging.INFO,
      format=log_format,
      force=True,
    )


def main():
  args = parse_args()
  setup_logging(args.log_file)

  agent_config = None
  if args.agent_config:
    try:
      agent_config = json.loads(args.agent_config)
    except json.JSONDecodeError as e:
      logger.error(f"Invalid JSON string for --agent_config: {e}")
      return

  kwargs = {
    "max_concurrency": args.max_concurrency,
    # Parallel Search Arguments
    "num_parallel_runs": args.num_parallel_runs,
    "max_worker_retries": args.max_retries,
    "strategies": args.strategies,
    "agent_config": agent_config,
  }

  prob_id, status = asyncio.run(
    run_search(
      problem_dir=args.problem_dir,
      algorithm=args.algorithm,
      graph_db_path=args.graph_db_path,
      **kwargs,
    )
  )
  logger.info(f"Result for {prob_id}: {status}")


if __name__ == "__main__":
  main()
