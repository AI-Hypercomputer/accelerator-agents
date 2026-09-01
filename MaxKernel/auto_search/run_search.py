import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Optional, Tuple

try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  logging.warning(
    "dotenv not installed, skipping loading environment variables"
  )

from auto_search.algorithms.beam_search import BeamSearchOrchestrator
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
      events_compaction=kwargs.get("events_compaction", False),
    )
  elif algorithm == "beam":
    strategies_kwargs = {}
    if kwargs.get("strategies"):
      strategies_kwargs["strategies"] = kwargs.get("strategies")

    return BeamSearchOrchestrator(
      problem_id=problem_id,
      reference_code=reference_code,
      graph_db_path=graph_db_path,
      max_concurrency=kwargs.get("max_concurrency", 2),
      max_worker_retries=kwargs.get("max_worker_retries", 2),
      beam_size=kwargs.get("beam_size", 2),
      branches_per_node=kwargs.get("branches_per_node", 2),
      max_depth=kwargs.get("max_depth", 2),
      keep_factor=kwargs.get("keep_factor", 1.0),
      agent_config=kwargs.get("agent_config"),
      events_compaction=kwargs.get("events_compaction", False),
      **strategies_kwargs,
    )
  elif algorithm == "agentic":
    raise NotImplementedError(
      f"Algorithm '{algorithm}' is currently a placeholder."
    )
  else:
    raise ValueError(f"Unknown algorithm: {algorithm}")


def save_optimized_kernel(
  orchestrator: SearchOrchestrator,
  optimized_file_path: str,
) -> Optional[Tuple[str, float]]:
  """Saves the best kernel code found during search to the optimized file path."""
  best_id = orchestrator.graph.best_node_id
  if not best_id or best_id not in orchestrator.graph.nodes:
    return None

  best_node = orchestrator.graph.nodes[best_id]
  if not best_node.code:
    return None

  os.makedirs(
    os.path.dirname(os.path.abspath(optimized_file_path)), exist_ok=True
  )
  with open(optimized_file_path, "w") as f:
    f.write(best_node.code)

  logger.info(f"Saved best kernel ({best_id}) to {optimized_file_path}")
  speedup = best_node.evaluation.speedup or 0.0
  return best_id, speedup


async def run_search(
  reference_file_path: str,
  optimized_file_path: Optional[str] = None,
  algorithm: str = "parallel",
  graph_db_path: Optional[str] = None,
  problem_id: Optional[str] = None,
  **kwargs: Any,
) -> Tuple[str, str]:
  """Executes the search algorithm asynchronously for a single reference file."""
  problem_dir = os.path.dirname(os.path.abspath(reference_file_path))
  default_problem_id, ext = os.path.splitext(
    os.path.basename(reference_file_path)
  )
  if not problem_id:
    problem_id = default_problem_id

  if not optimized_file_path:
    optimized_file_path = os.path.join(
      problem_dir, f"{problem_id}_optimized_{algorithm}{ext}"
    )
    logger.info(f"No optimized file path provided. Using {optimized_file_path}")

  if not os.path.exists(reference_file_path):
    logger.error(f"{reference_file_path} not found")
    return problem_id, f"Failed: {reference_file_path} missing"

  with open(reference_file_path, "r") as f:
    reference_code = f.read()

  run_subdir = None
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
    graph_exist = os.path.exists(graph_db_path)

    orchestrator = get_orchestrator(
      algorithm=algorithm,
      problem_id=problem_id,
      reference_code=reference_code,
      graph_db_path=graph_db_path,
      **kwargs,
    )

    if graph_exist:
      logger.info(f"Existing graph found for {problem_id}. Resuming...")
      orchestrator.resume()

    await orchestrator.run()

    best_result = save_optimized_kernel(orchestrator, optimized_file_path)

    if run_subdir:
      import shutil

      from auto_search.utils.analyze_timing import analyze_path

      dest_dir = os.path.join(problem_dir, os.path.basename(run_subdir))
      shutil.copytree(run_subdir, dest_dir, dirs_exist_ok=True)
      logger.info(f"Copied artifacts to {dest_dir}")

      try:
        logger.info("Generating timing summary...")
        summary_text = analyze_path(dest_dir)
        out_file = os.path.join(dest_dir, "timing_summary.md")
        with open(out_file, "w") as f:
          f.write("```text\n" + summary_text + "\n```\n")
        logger.info(f"Saved metric summary to: {out_file}")

        logger.info("Generating token summary...")
        try:
          from auto_search.utils.analyze_tokens import (
            analyze_path as analyze_token_path,
          )

          token_summary_text = analyze_token_path(dest_dir)
          if token_summary_text:
            token_out_file = os.path.join(dest_dir, "token.md")
            with open(token_out_file, "w", encoding="utf-8") as f:
              f.write(token_summary_text)
            print(
              f"\n\n====== TOKEN METRICS ======\n{token_summary_text}\n===========================\n"
            )
        except Exception as e:
          logger.error(f"Failed to generate token summary: {e}")

      except Exception as analyze_err:
        logger.error(f"Failed to generate timing summary: {analyze_err}")

    if best_result:
      best_id, speedup = best_result
      return problem_id, f"Success (Best: {best_id}, Speedup: {speedup}x)"

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
    "--reference_file_path",
    type=str,
    required=True,
    help="Path to the reference kernel file",
  )
  orch_group.add_argument(
    "--optimized_file_path",
    type=str,
    default=None,
    help=(
      "Path to save the optimized kernel. "
      "Defaults to <problem_id>_optimized_{algorithm}.py"
      "in the reference file's directory."
    ),
  )
  orch_group.add_argument(
    "--problem_id",
    type=str,
    default=None,
    help=(
      "Explicitly assign a problem ID to this search. "
      "Defaults to the reference file name."
    ),
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
  orch_group.add_argument(
    "--max_worker_retries",
    type=int,
    default=1,
    help="Max worker retries per expansion task",
  )
  orch_group.add_argument(
    "--strategies",
    nargs="+",
    type=str,
    default=None,
    help="List of strategy strings to explore",
  )
  orch_group.add_argument(
    "--agent_config",
    type=str,
    default=None,
    help="JSON string of agent config parameters (e.g. '{\"max_iterations\": 5}')",
  )
  orch_group.add_argument(
    "--events_compaction",
    action="store_true",
    help="Enable event compaction",
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
  # Beam Search Arguments
  beam_group = parser.add_argument_group(
    "Beam Search Arguments",
    "Parameters specific to the 'beam' search algorithm.",
  )
  beam_group.add_argument(
    "--beam_size",
    type=int,
    default=2,
    help="Size of the beam (number of candidates to keep per depth)",
  )
  beam_group.add_argument(
    "--branches_per_node",
    type=int,
    default=2,
    help="Number of branches/strategies to explore per node in the beam",
  )
  beam_group.add_argument(
    "--max_depth",
    type=int,
    default=2,
    help="Maximum depth of the beam search",
  )
  beam_group.add_argument(
    "--keep_factor",
    type=float,
    default=1.0,
    help="Factor of parent speedup to keep candidates (e.g. 1.0 means must not be worse than parent)",
  )
  return parser.parse_args()


def setup_logging(log_file: Optional[str]):
  """Configures console and file logging."""
  log_format = "%(asctime)s - %(levelname)s - %(message)s"
  formatter = logging.Formatter(log_format)

  if log_file:
    base, ext = os.path.splitext(log_file)
    agent_log_path = f"{base}_agent{ext}"
    main_log_path = log_file
  else:
    agent_log_path = "agent.log"
    main_log_path = None

  # 1. Catch the rest of the logs in the Root Logger and route to agent_log_path
  logging.basicConfig(
    filename=agent_log_path,
    level=logging.INFO,
    format=log_format,
    force=True,
  )

  # 2. Explicitly route only auto_search and __main__ to the main_log_path
  auto_search_loggers = [
    logging.getLogger("auto_search"),
    logging.getLogger("__main__"),
  ]

  if main_log_path:
    auto_search_handler = logging.FileHandler(main_log_path)
  else:
    auto_search_handler = logging.StreamHandler(sys.stdout)

  auto_search_handler.setFormatter(formatter)

  for logger in auto_search_loggers:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
      logger.handlers.clear()
    logger.addHandler(auto_search_handler)


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
    "max_worker_retries": args.max_worker_retries,
    "strategies": args.strategies,
    "agent_config": agent_config,
    "events_compaction": args.events_compaction,
    # Parallel Search Arguments
    "num_parallel_runs": args.num_parallel_runs,
    # Beam Search Arguments
    "beam_size": args.beam_size,
    "branches_per_node": args.branches_per_node,
    "max_depth": args.max_depth,
    "keep_factor": args.keep_factor,
  }

  prob_id, status = asyncio.run(
    run_search(
      reference_file_path=args.reference_file_path,
      optimized_file_path=args.optimized_file_path,
      algorithm=args.algorithm,
      graph_db_path=args.graph_db_path,
      problem_id=args.problem_id,
      **kwargs,
    )
  )
  logger.info(f"Result for {prob_id}: {status}")


if __name__ == "__main__":
  main()
