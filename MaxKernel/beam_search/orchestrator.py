"""Top-level Beam Search orchestrator for MaxKernel."""

import asyncio
import json
import logging
import shutil
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# ADK imports
from google.adk import Runner
from google.adk.agents import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

# MaxKernel imports
from auto_agent.agent import root_agent
from auto_agent.beam_worker import beam_worker_agent
from auto_agent.config import WORKDIR
from beam_search.utils import normalize_ast_code
from evaluation.fake_kernel_evaluator import FakeKernelEvaluator
from evaluation.jax_kernel_evaluator import JAXKernelEvaluator

logger = logging.getLogger(__name__)

# Complete TPU/Pallas optimization strategy list from autocomp
TPU_PALLAS_OPTIMIZATION_STRATEGIES = [
    "Reduce data movement",
    "Overlap data movement and compute",
    "Cache reused data in local memory instead of reloading from main memory",
    "Loop tiling",
    "Loop reordering and restructuring",
    "Loop unrolling",
    "Fuse operations",
    "Use lower precision",
    "Double buffering",
    "Software pipelining",
    "Hoist redundant operations out of loops",
    "Eliminate redundant computation",
    "Simplify or remove unnecessary code",
    "Try new parameter values",
    "Rewrite the algorithm to reduce total work",
    "Place reduction axis last in grid to enable in-place SRAM accumulation without HBM round-trips",
    "Align block dimensions to 8x128 tile boundaries to avoid wasted padding and register spills",
    "Use scratch_shapes=[pltpu.VMEM(...)] for persistent high-precision accumulators during reduction loops",
    "Maximize block sizes up to ~16 MB VMEM capacity to increase arithmetic intensity per pipeline step",
    "Use scalar prefetch via PrefetchScalarGridSpec to load indices/metadata into SMEM without stalling vector core",
    "Upcast bf16/int8 to float32 before elementwise ops, downcast only on final output write",
    "Fuse transpose into lax.dot_general contraction dimensions instead of materializing transposed operands",
    "Arrange grid iteration order so consecutive invocations reuse already-resident input slices",
    "Increase pipeline buffer count beyond double buffering to hide memory latency for bandwidth-bound kernels",
    "Generate random numbers inside kernel via hardware PRNG with key in SMEM instead of passing precomputed arrays",
    "Avoid singleton dimensions in last two array axes to prevent full-tile waste per element",
    "Reduce along second-to-last dimension rather than last dimension when possible",
    "Prefer add/multiply over exp/tanh/division; restructure math to minimize expensive elementwise ops",
    "Tune block sizes jointly — systematically vary BM, BN, BK together under the VMEM budget constraint (16 MiB including double-buffering)",
    "Compute arithmetic intensity accounting for tiling amplification to predict compute-bound vs memory-bound regime",
    "Minimize control flow inside kernels; consolidate into single basic blocks to avoid unrolling overhead",
    "Pass all data as explicit kernel inputs with BlockSpec instead of closing over constants",
    "Use pltpu.VMEM scoped scratch buffers for temporary storage within kernel lifetime",
    "Balance block size against pipeline depth to amortize startup/drain bubble cost over enough iterations",
    "Explicitly initialize accumulator buffers to zero on first reduction iteration since SRAM starts undefined"
]


# Load API key configuration
local_env = Path(__file__).resolve().parents[1] / ".env"
if local_env.exists():
  load_dotenv(local_env)


class AgenticSearchOrchestrator:
  """Orchestration layer that manages candidate exploration loops."""

  def __init__(
    self,
    baseline_code_path: str,
    reference_code_path: str,
    task_yaml_path: str,
    output_dir: Path,
    mock_mode: Optional[bool] = None,
    use_beam_worker: bool = True,  # True = BeamWorkerPipeline, False = AutonomousPipelineAgent
  ):
    self.baseline_code_path = baseline_code_path
    self.reference_code_path = reference_code_path
    self.task_yaml_path = task_yaml_path
    self.output_dir = output_dir
    self.use_beam_worker = use_beam_worker

    # Configure evaluator backend (defaulting to MOCK_COMPILER env var)
    if mock_mode is None:
      mock_mode = os.environ.get("MOCK_COMPILER", "true").lower() == "true"
    self.mock_mode = mock_mode

    if self.mock_mode:
      logger.info("[Orchestrator] Running in MOCK mode. Using FakeKernelEvaluator.")
      self.evaluator = FakeKernelEvaluator()
    else:
      logger.info("[Orchestrator] Running in PRODUCTION mode. Using JAXKernelEvaluator.")
      self.evaluator = JAXKernelEvaluator(local=True)

    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Rebind WORKDIR in config and all modules that imported it to ensure consistency
    workdir_str = str(self.output_dir)
    os.environ["WORKDIR"] = workdir_str

    import auto_agent.config
    auto_agent.config.WORKDIR = workdir_str

    import auto_agent.subagents.pipeline_agent
    auto_agent.subagents.pipeline_agent.WORKDIR = workdir_str

    import auto_agent.callbacks
    auto_agent.callbacks.WORKDIR = workdir_str

    import auto_agent.tools.file_tools
    auto_agent.tools.file_tools.WORKDIR = workdir_str

    # Update local reference in orchestrator module
    global WORKDIR
    WORKDIR = workdir_str

    # Read baseline source code
    with open(self.baseline_code_path, "r") as f:
      self.baseline_code = f.read()

  # BE AWARE: the current implementation is broken. It simply concatenates all candidate codes into one python file, thus the
  # output python contains repeated main functions and won't execute all candidates.
  #
  # TODO(ligh-svg): Fix the grouped kernel harness to allow executing all candidates. 
  def _prepare_grouped_harness(self, candidate_paths: List[str]) -> str:
    """Inlines all candidate implementations into a single file."""
    inlined_solutions = []
    for idx, path in enumerate(candidate_paths):
      if not os.path.exists(path):
        continue
      try:
        with open(path, "r") as f:
          code = f.read()
        # Rename function computation() to solution_{idx}() to prevent symbol collision and match mock evaluator expectations
        renamed_code = re.sub(r"def computation\(", f"def solution_{idx}(", code)
        inlined_solutions.append(
          f"# --- Candidate {idx} ---\n{renamed_code}"
        )
      except Exception as e:
        logger.error(f"Failed to inline candidate {path}: {e}")

    return "\n\n".join(inlined_solutions)

  async def run_worker_session(
    self,
    worker_name: str,
    session_id: str,
    prompt_focus: str,
    base_code: str,
    parent_latency: float,
  ) -> Optional[dict]:
    """Spawns an isolated worker session and extracts its optimized kernel path."""
    logger.info(f"[Orchestrator] Spawning {worker_name} (session_id={session_id})...")

    # Select the agent class based on mode flag
    if self.use_beam_worker:
      agent_to_run = beam_worker_agent
      logger.info(f"[{worker_name}] Mode: BeamWorkerPipeline (correctness-only).")
    else:
      agent_to_run = root_agent
      # Allow up to 5 iterations for correctness repairs, but stop on first valid profiled candidate
      agent_to_run.max_iterations = 5
      agent_to_run.stop_on_first_valid = True
      logger.info(
          f"[{worker_name}] Mode: AutonomousPipelineAgent (stop_on_first_valid=True, max_iterations=5)."
      )

    session_service = InMemorySessionService()
    runner = Runner(
      app_name=worker_name,
      agent=agent_to_run,
      session_service=session_service
    )

    user_message = f"Optimize the JAX/Pallas kernel. Focus: {prompt_focus}"
    content = types.Content(
      role="user",
      parts=[types.Part.from_text(text=user_message)]
    )

    # Create the session and pre-inject the base kernel path so the worker can read it
    session = await session_service.create_session(
      app_name=worker_name,
      user_id="orchestrator",
      session_id=session_id
    )

    # Write the base candidate code to the expected session work directory
    session_dir = os.path.join(WORKDIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    session_base_path = os.path.join(session_dir, "base_kernel.py")
    with open(session_base_path, "w") as f:
      f.write(base_code)

    # Pre-populate state
    session.state["base_kernel_path"] = session_base_path
    session.state["parent_latency_ms"] = parent_latency
    
    # Save back to database
    session_service.sessions.setdefault(worker_name, {}).setdefault("orchestrator", {})[session_id] = session

    correct_code_path = None

    # Execute the worker pipeline
    async for event in runner.run_async(
      user_id="orchestrator",
      session_id=session_id,
      new_message=content,
      run_config=RunConfig()
    ):
      if event.actions and event.actions.state_delta:
        delta = event.actions.state_delta
        if "final_correct_code_path" in delta:
          correct_code_path = delta["final_correct_code_path"]
          logger.info(f"[{worker_name}] Reported correctness achieved at: {correct_code_path}")

    # Inspect final session state
    final_session = await session_service.get_session(
      app_name=worker_name,
      user_id="orchestrator",
      session_id=session_id
    )
    
    if not final_session:
      logger.warning(f"[{worker_name}] No final session found.")
      return None

    # Extract optimized code path based on Mode
    if not self.use_beam_worker:
      # Mode 1 (AutonomousPipelineAgent): Check if a best iteration was selected
      best_iter = final_session.state.get("best_iteration", -1)
      if best_iter != -1:
        opt_path = final_session.state.get("optimized_kernel_path")
        logger.info(f"[{worker_name}] Output tuned code from iteration {best_iter}: {opt_path}")
        return {
            "path": opt_path,
            "parent_latency_ms": parent_latency
        }
      logger.warning(f"[{worker_name}] Pipeline failed correctness tests in Mode 1.")
      return None
    else:
      # Mode 2 (BeamWorkerPipeline): Check worker success status
      if final_session.state.get("worker_status") == "Success":
        opt_path = correct_code_path or final_session.state.get("optimized_kernel_path")
        logger.info(f"[{worker_name}] Output correct code: {opt_path}")
        return {
            "path": opt_path,
            "parent_latency_ms": parent_latency
        }
      logger.warning(f"[{worker_name}] Pipeline failed correctness tests in Mode 2.")
      return None

  def _evaluate_candidate_performance(
      self, path: str, parent_latency: float, fallback_analysis: str = ""
  ) -> dict:
    """Evaluates a single candidate and parses its performance metrics."""
    opt_dir = os.path.dirname(path)

    # Try to load autotune results first (Mode 1 auto_agent output)
    autotune_json = os.path.join(opt_dir, "autotune_results.json")
    best_latency = None
    if os.path.exists(autotune_json):
      try:
        with open(autotune_json, "r") as f:
          data = json.load(f)
        if "best_time_ms" in data:
          best_latency = float(data["best_time_ms"])
          logger.info(f"[Orchestrator] Found autotune latency {best_latency} ms in {autotune_json}")
      except Exception as e:
        logger.warning(f"Failed to read autotune results from {autotune_json}: {e}")

    # Fallback to evaluation_metrics.json if autotune was missing but metrics exist
    metrics_json = os.path.join(opt_dir, "evaluation_metrics.json")
    hbm_util = 0.0
    comp_util = 0.0
    density = 0.0
    analysis = fallback_analysis or "Evaluated performance"

    if best_latency is None and os.path.exists(metrics_json):
      try:
        with open(metrics_json, "r") as f:
          meta = json.load(f)
        perf_analysis = meta.get("performance_analysis", "")
        match = re.search(r"latency:\s*([\d.]+)\s*ms", perf_analysis, re.IGNORECASE)
        if match:
          best_latency = float(match.group(1))
          logger.info(f"[Orchestrator] Parsed latency {best_latency} ms from metrics analysis text")
      except Exception as e:
        logger.warning(f"Failed to parse latency from existing metrics file: {e}")

    if best_latency is not None:
      logger.info(f"[Orchestrator] Bypassing evaluator for candidate: {path}. Using parsed latency: {best_latency} ms")
      latency_ms = best_latency
      if self.mock_mode:
        hbm_util = 10.0
        comp_util = 2.0
        density = 0.5
        analysis = f"Bypassed evaluator (mock metrics). Latency: {best_latency:.3f} ms"
      else:
        if os.path.exists(metrics_json):
          try:
            with open(metrics_json, "r") as f:
              meta = json.load(f)
            hbm_util = meta.get("estimated_hbm_utilization", 0.0)
            comp_util = meta.get("estimated_compute_utilization", 0.0)
            density = meta.get("estimated_computation_density_flops_per_byte", 0.0)
            analysis = meta.get("performance_analysis", "Bypassed evaluator")
          except Exception as e:
            logger.warning(f"Failed to read metrics from {metrics_json}: {e}")
    else:
      logger.info(f"[Orchestrator] No cached metrics found. Evaluating candidate: {path}")
      eval_res = self.evaluator.evaluate(
          reference_code_path=self.reference_code_path,
          optimized_code_path=path,
          task_yaml_path=self.task_yaml_path
      )
      latency_ms = eval_res.optimized_time_ms
      if os.path.exists(metrics_json):
        try:
          with open(metrics_json, "r") as f:
            meta = json.load(f)
          hbm_util = meta.get("estimated_hbm_utilization", 0.0)
          comp_util = meta.get("estimated_compute_utilization", 0.0)
          density = meta.get("estimated_computation_density_flops_per_byte", 0.0)
          analysis = meta.get("performance_analysis", "")
        except Exception as e:
          logger.warning(f"Failed to read evaluation metrics from {metrics_json}: {e}")

    with open(path, "r") as f:
      code_content = f.read()

    return {
        "path": path,
        "latency_ms": latency_ms,
        "parent_latency_ms": parent_latency,
        "code": code_content,
        "hbm_utilization_pct": hbm_util,
        "compute_utilization_pct": comp_util,
        "computation_density_flops_per_byte": density,
        "analysis": analysis
    }

  def _run_sequential_fallback_evaluation(
      self, valid_results: List[dict], fallback_analysis: str = "Sequential fallback prediction"
  ) -> List[dict]:
    """Evaluates candidates sequentially when batched evaluation fails or is skipped."""
    round_candidates = []
    for res_entry in valid_results:
      cand = self._evaluate_candidate_performance(
          path=res_entry["path"],
          parent_latency=res_entry["parent_latency_ms"],
          fallback_analysis=fallback_analysis
      )
      round_candidates.append(cand)
    return round_candidates

  def _apply_parent_regression_gate(
      self, candidates: List[dict], keep_factor: float
  ) -> List[dict]:
    """Filters out candidates whose latency regresses beyond the parent latency threshold."""
    valid_candidates = []
    for cand in candidates:
      parent_limit = cand["parent_latency_ms"] * keep_factor
      if cand["latency_ms"] >= parent_limit:
        logger.info(
            f"[Orchestrator] Pruning candidate {cand['path']} (latency {cand['latency_ms']:.3f} ms "
            f">= parent limit {parent_limit:.3f} ms)"
        )
        continue
      valid_candidates.append(cand)
    return valid_candidates

  def _deduplicate_candidates_ast(self, candidates: List[dict]) -> List[dict]:
    """Deduplicates candidates based on their normalized AST representations."""
    seen_ast_hashes = set()
    deduped_candidates = []
    for cand in candidates:
      normalized_code = normalize_ast_code(cand["code"])
      ast_hash = "".join(normalized_code.split())
      if ast_hash not in seen_ast_hashes:
        seen_ast_hashes.add(ast_hash)
        deduped_candidates.append(cand)
      else:
        logger.info(
            f"[Orchestrator] Pruning duplicate candidate {cand['path']} (AST duplicate of earlier candidate)"
          )
    return deduped_candidates

  def _log_round_beam_state(self, round_idx: int, beam: List[dict]) -> None:
    """Logs the beam ranking status at the end of a round."""
    print(f"\n==================================================")
    print(f"            ROUND {round_idx} BEAM STATE")
    print(f"==================================================")
    for rank, cand in enumerate(beam):
      print(f"  Rank {rank+1}: Latency={cand['latency_ms']:.3f} ms | Path={cand['path']}")
      if self.mock_mode:
        print(
            f"          HBM Util={cand['hbm_utilization_pct']}% | "
            f"Comp Util={cand['compute_utilization_pct']}% | "
            f"density={cand['computation_density_flops_per_byte']:.2f}"
        )
    print(f"==================================================\n")

  def _prepare_worker_tasks(
      self, round_idx: int, beam: List[dict], beam_size: int, dropout_menu_options: float
  ) -> List:
    """Prepares run_worker_session tasks for all worker beams in a round."""
    tasks = []
    for idx in range(beam_size):
      # Select parent candidate (branching if len(beam) < beam_size)
      parent_candidate = beam[idx % len(beam)]

      # Apply stochastic dropout to optimization strategies menu
      selected_opts = [
          opt for opt in TPU_PALLAS_OPTIMIZATION_STRATEGIES
          if random.random() < dropout_menu_options
      ]
      # Fallback: ensure at least one strategy focus is passed
      if not selected_opts:
        selected_opts = [random.choice(TPU_PALLAS_OPTIMIZATION_STRATEGIES)]

      focus_text = "\n".join(f"- {opt}" for opt in selected_opts)
      prompt_focus = (
          "Focus your optimization effort on applying the following strategies:\n"
          f"{focus_text}"
      )

      session_id = f"beam_r{round_idx}_w{idx}_{int(time.time())}"
      tasks.append(
          self.run_worker_session(
              worker_name=f"Round_{round_idx}_Worker_{idx}",
              session_id=session_id,
              prompt_focus=prompt_focus,
              base_code=parent_candidate["code"],
              parent_latency=parent_candidate["latency_ms"]
          )
      )
    return tasks

  def _evaluate_round_candidates(
      self, round_idx: int, valid_results: List[dict]
  ) -> List[dict]:
    """Evaluates the round candidates using either grouped harness or sequential fallback."""
    valid_paths = [r["path"] for r in valid_results]
    round_candidates = []

    if not self.use_beam_worker:
      # Mode 1: Evaluate candidates individually (since they have already run local autotuning)
      logger.info(f"[Round {round_idx}] Mode 1: Evaluating candidates individually...")
      round_candidates = self._run_sequential_fallback_evaluation(
          valid_results, fallback_analysis="Individual autotuned run"
      )

    else:
      # Mode 2: Dispatch a GROUPED harness containing all candidates
      logger.info(f"[Round {round_idx}] Mode 2: Compiling grouped harness for {len(valid_paths)} candidates...")
      grouped_code = self._prepare_grouped_harness(valid_paths)
      grouped_path = self.output_dir / f"grouped_harness_r{round_idx}.py"
      with open(grouped_path, "w") as f:
        f.write(grouped_code)

      # Dispatch grouped harness to the evaluation backend
      logger.info(f"[Round {round_idx}] Dispatching grouped harness to evaluation backend...")
      try:
        self.evaluator.evaluate(
            reference_code_path=self.reference_code_path,
            optimized_code_path=str(grouped_path),
            task_yaml_path=self.task_yaml_path
        )
      except Exception as e:
        logger.warning(f"[Round {round_idx}] Grouped evaluation failed with error: {e}")

      # Parse metrics from JSON file
      metrics_json = os.path.join(os.path.dirname(grouped_path), "evaluation_metrics.json")
      valid_paths_fallback = False
      if os.path.exists(metrics_json):
        try:
          with open(metrics_json, "r") as f:
            meta = json.load(f)
          
          candidates_data = meta.get("candidates", [])
          for idx, data in enumerate(candidates_data):
            cand_id = data.get("candidate_id", idx)
            if cand_id >= len(valid_results):
              continue
            res_entry = valid_results[cand_id]
            path = res_entry["path"]
            with open(path, "r") as f:
              code_content = f.read()

            round_candidates.append({
              "path": path,
              "latency_ms": data.get("latency_ms", self.evaluator.reference_time_ms),
              "parent_latency_ms": res_entry["parent_latency_ms"],
              "code": code_content,
              "hbm_utilization_pct": data.get("estimated_hbm_utilization", 0.0),
              "compute_utilization_pct": data.get("estimated_compute_utilization", 0.0),
              "computation_density_flops_per_byte": data.get("estimated_computation_density", 0.0),
              "analysis": data.get("analysis", "")
            })
        except Exception as e:
          logger.error(f"[Round {round_idx}] Failed to parse grouped evaluation metrics: {e}")
          valid_paths_fallback = True
      else:
        valid_paths_fallback = True

      # Fallback to sequential evaluation if file parsing was unsuccessful
      if valid_paths_fallback or not round_candidates:
        logger.warning(f"[Round {round_idx}] Grouped harness metrics missing. Falling back to sequential evaluation.")
        round_candidates = self._run_sequential_fallback_evaluation(valid_results)

    return round_candidates

  async def run_search(
    self,
    num_rounds: int = 3,
    beam_size: int = 2,
    dropout_menu_options: float = 0.5,
    keep_factor: float = 1.0,
  ) -> dict:
    """Executes the complete multi-round Beam Search optimization process."""
    t0 = time.perf_counter()
    logger.info(
      f"[Orchestrator] Starting Beam Search: {num_rounds} rounds, Beam Size = {beam_size}, "
      f"dropout_menu_options={dropout_menu_options}, keep_factor={keep_factor}, "
      f"Mode={'BeamWorker' if self.use_beam_worker else 'AutonomousAgent'}"
    )
    
    # 1. Establish initial baseline performance
    logger.info("[Orchestrator] Evaluating baseline kernel...")
    base_cand = self._evaluate_candidate_performance(
        path=self.baseline_code_path,
        parent_latency=0.0,
        fallback_analysis="Baseline reference"
    )
    # Override baseline metadata defaults
    base_cand["code"] = self.baseline_code
    base_cand["parent_latency_ms"] = base_cand["latency_ms"]
    beam = [base_cand]
    
    logger.info(f"[Orchestrator] Baseline Latency: {base_cand['latency_ms']:.3f} ms")

    for round_idx in range(1, num_rounds + 1):
      logger.info(f"\n=== STARTING SEARCH ROUND {round_idx}/{num_rounds} ===")
      
      # 2. Distribute candidates from the beam to workers and run them
      tasks = self._prepare_worker_tasks(round_idx, beam, beam_size, dropout_menu_options)
      worker_results = await asyncio.gather(*tasks)
      valid_results = [r for r in worker_results if r is not None]
      
      logger.info(f"[Round {round_idx}] Collected {len(valid_results)} correct candidate outputs.")
      if not valid_results:
        logger.warning(f"[Round {round_idx}] No valid candidates generated this round. Skipping evaluation.")
        continue

      # 3. Profile mutations
      round_candidates = self._evaluate_round_candidates(round_idx, valid_results)

      # 5. Apply Parent Regression Budget Gate (Incumbents are exempt)
      valid_round_candidates = self._apply_parent_regression_gate(round_candidates, keep_factor)

      # Merge survivors with incumbents
      all_candidates = beam + valid_round_candidates
      
      # 6. Deduplicate candidates using AST-based normalization
      deduped_candidates = self._deduplicate_candidates_ast(all_candidates)

      # Sort by latency (lower is better)
      deduped_candidates.sort(key=lambda x: x["latency_ms"])

      # 7. Prune beam to beam_size
      beam = deduped_candidates[:beam_size]
      
      self._log_round_beam_state(round_idx, beam)

    # Return final best candidate
    best_candidate = beam[0]
    elapsed = time.perf_counter() - t0
    logger.info(f"[Orchestrator] Beam Search finished. Duration: {elapsed:.2f} s. Best Latency: {best_candidate['latency_ms']:.3f} ms")

    # Save the final winning optimized code to output_dir
    best_output_path = self.output_dir / "best_optimized_kernel.py"
    try:
      shutil.copy(best_candidate["path"], best_output_path)
      logger.info(f"[Orchestrator] Saved winning optimized kernel to {best_output_path}")
      best_path_to_return = str(best_output_path)
    except Exception as e:
      logger.error(f"[Orchestrator] Failed to copy winning kernel to {best_output_path}: {e}")
      best_path_to_return = best_candidate["path"]

    return {
      "status": "success",
      "best_latency_ms": best_candidate["latency_ms"],
      "best_code_path": best_path_to_return
    }
