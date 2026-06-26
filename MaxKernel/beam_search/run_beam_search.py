"""Local dry-run script to verify Beam Search Orchestrator functionality."""

import asyncio
import logging
import sys
from pathlib import Path

# Add MaxKernel root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from beam_search.orchestrator import AgenticSearchOrchestrator

# Configure logging to show pipeline steps clearly in console
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  handlers=[logging.StreamHandler(sys.stdout)],
  force=True
)
logger = logging.getLogger("run_beam_search")


async def main():
  import argparse
  import time

  # Parse arguments
  parser = argparse.ArgumentParser(description="Run Beam Search Orchestrator")
  parser.add_argument("--run_id", type=str, default=None, help="Descriptive Run ID")
  parser.add_argument("--task_id", type=str, default="12p_RMSNorm", help="Task/Kernel ID (e.g. 12p_RMSNorm)")
  parser.add_argument("--rounds", type=int, default=2, help="Number of search rounds")
  parser.add_argument("--beam_size", type=int, default=2, help="Beam size")
  parser.add_argument(
      "--use_beam_worker",
      type=str,
      default="true",
      help="Use BeamWorkerPipeline (correctness-only): true/false",
  )
  args = parser.parse_args()

  # Convert string boolean
  use_beam_worker = args.use_beam_worker.lower() == "true"

  # Paths to resources
  import os
  project_root = Path(__file__).resolve().parents[2]
  baseline_path = str(project_root / f"JAXBench/benchmark/{args.task_id}/baseline.py")
  reference_path = str(project_root / f"JAXBench/benchmark/{args.task_id}/baseline.py")
  task_yaml = str(project_root / f"MaxKernel/evaluation/jaxbench_adapted_dataset/{args.task_id}/kernel_task.yaml")
  output_dir = project_root / "MaxKernel/beam_search/output"

  # Validate file existence
  if not os.path.exists(baseline_path):
    logger.error(f"Baseline code not found at: {baseline_path}")
    sys.exit(1)
  if not os.path.exists(task_yaml):
    logger.error(f"Task YAML configuration not found at: {task_yaml}")
    sys.exit(1)

  # Build run_id from naming convention if not specified by user
  if args.run_id:
    run_id = args.run_id
  else:
    kernel_name = Path(task_yaml).parent.name
    worker_short = "bw" if use_beam_worker else "aa"
    timestamp_short = time.strftime("%m%d_%H%M")
    run_id = f"{kernel_name}_{worker_short}_r{args.rounds}_b{args.beam_size}_{timestamp_short}"

  run_output_dir = output_dir / run_id

  logger.info("="*80)
  logger.info(f"               STARTING RUN: {run_id}")
  logger.info(f"               BEAM SIZE = {args.beam_size}, ROUNDS = {args.rounds}")
  logger.info("="*80)
  
  orchestrator_mode2 = AgenticSearchOrchestrator(
    baseline_code_path=baseline_path,
    reference_code_path=reference_path,
    task_yaml_path=task_yaml,
    output_dir=run_output_dir,
    mock_mode=True,
    use_beam_worker=use_beam_worker
  )

  res_mode2 = await orchestrator_mode2.run_search(
      num_rounds=args.rounds,
      beam_size=args.beam_size,
      dropout_menu_options=0.5,
      keep_factor=1.0
  )
  logger.info(f"Search Completed. Result: {res_mode2}")


if __name__ == "__main__":
  asyncio.run(main())
