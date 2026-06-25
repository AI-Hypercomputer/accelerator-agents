# Implementation & Integration Plan: MaxKernel Beam Search

This document outlines the architectural design, refactoring strategy, and implementation phases to integrate the Beam Search optimization engine (from AutoComp) into the **MaxKernel** repository. 

---

## 1. Goal & Context
Our objective is to combine AutoComp's high-dimensional beam search engine with MaxKernel's modular ADK-based agent orchestration. 

To achieve this efficiently, we split the responsibilities:
1.  **Orchestrator Layer (`beam_search/`)**: Manages the beam repository, spawns worker agents in parallel, groups candidate implementations, and dispatches them to the evaluation backend for performance profiling.
2.  **Worker Layer (`auto_agent/` - Subagent)**: Autonomously ensures code correctness (local compilation and unit tests) using a self-correction loop. It exits early as soon as the code passes correctness checks, delegating latency profiling to the orchestrator.

---

## 2. Target Directory Structure
The orchestrator and worker agent will exist as sibling packages at the root of the `MaxKernel/` repository to maintain a clean separation of concerns:

```
MaxKernel/
├── beam_search/                  # Top-level Search Orchestrator (Main entry point)
│   ├── __init__.py
│   ├── orchestrator.py           # Coordinates the Beam Search loop & rounds
│   ├── tools.py                  # Orchestrator tools (e.g. grouped harness generator)
│   └── docs/                     # Documentation specific to Beam Search
│       ├── worker_integration_plan.md       # This design document
│       └── worker_integration_checklist.md  # Step-by-step progress checklist
│
├── auto_agent/                   # Autonomous Worker Agent (Spawned as subagents)
│   ├── agent.py                  # Entry point for standard single-agent runs
│   ├── beam_worker.py            # Entry point for correctness-only worker runs
│   ├── subagents/                # Planner, Implementer, Validator, Pipelines
│   │   ├── beam_worker_pipeline.py # Shorter correctness-only pipeline for search
│   │   └── pipeline_agent.py     # Production end-to-end tuning pipeline
│   └── tools/                    # Compiler & local test checks used by workers
│
├── evaluation/                   # Backend profiling files
│   ├── jax_kernel_evaluator.py   # Production TPUVM Pallas evaluator
│   ├── fake_kernel_evaluator.py  # Mock/Fake evaluator for local verification
│   └── ...                       
```

---

## 3. Architecture: Why `BaseAgent` and not `LlmAgent`?
A key detail of MaxKernel's design is that coordinator agents (like `AutonomousPipelineAgent` and the proposed `BeamWorkerPipeline`) subclass ADK's `BaseAgent` directly, rather than `LlmAgent`.

*   **`LlmAgent` (Leaf/Reasoning Agents)**: Designed for agents whose execution loop is driven by prompting an LLM (using system instructions, chat history, and tool calls). Examples: `plan_kernel_agent`, `implement_kernel_agent`.
*   **`BaseAgent` (Coordinator Agents)**: The abstract foundation for any agent that implements custom execution logic via Python code (`_run_async_impl`).
*   **The Coordinator Role**: The pipeline agent acts as a deterministic state machine that chains subagents together. It does not need its own LLM reasoning loop to decide its next step; the execution sequence is defined programmatically in Python. Therefore, it subclasses `BaseAgent` to run Python control flow while delegating the LLM-heavy "thinking" steps to its `LlmAgent` subagents.

---

## 4. Refactoring Design Options for the Worker Pipeline

We evaluated two options for implementing the correctness-only worker in MaxKernel.

### Option A: Subclass `AutonomousPipelineAgent` (Recommended)
We subclass `AutonomousPipelineAgent` and override the `_run_async_impl` method to shorten the loop.

#### **Implementation Sketch (`auto_agent/subagents/beam_worker_pipeline.py`):**
```python
from auto_agent.subagents.pipeline_agent import AutonomousPipelineAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
import logging

class BeamWorkerPipeline(AutonomousPipelineAgent):
  """Subclass of AutonomousPipelineAgent that exits early on correctness success."""

  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    iteration = 0
    yield self._initialize_state(ctx)

    while iteration < self.max_iterations:
      logging.info(f"[{self.name}] Iteration {iteration + 1}/{self.max_iterations}")

      # Reuse parent's subagent steps
      async for event in self.plan_agent.run_async(ctx): yield event
      async for event in self.implement_agent.run_async(ctx): yield event
      async for event in self.validate_agent.run_async(ctx): yield event

      if not ctx.session.state.get("kernel_compilation_status", {}).get("success", False):
        iteration += 1
        continue

      async for event in self.test_gen_agent.run_async(ctx): yield event
      if not ctx.session.state.get("validation_loop_status", {}).get("success", False):
        iteration += 1
        continue

      async for event in self.test_run_agent.run_async(ctx): yield event
      
      # Check correctness success and EXIT EARLY
      test_results = ctx.session.state.get("test_results", {})
      if test_results.get("success", False):
        optimized_code_path = ctx.session.state.get("optimized_kernel_path")
        logging.info(f"[{self.name}] Correctness achieved: {optimized_code_path}")
        yield Event(
          author=self.name,
          actions=EventActions(
            state_delta={
              "worker_status": "Success",
              "final_correct_code_path": optimized_code_path
            }
          )
        )
        return  # Stop the pipeline loop immediately

      logging.error(f"[{self.name}] Correctness tests failed. Looping back.")
      iteration += 1

    yield Event(
      author=self.name,
      actions=EventActions(state_delta={"worker_status": "Failed"})
    )
```

#### **Pros & Cons:**
*   **Pros**: 
    *   Clean separation: The execution loop logic for "tuning search" and "correctness validation" remain separate.
    *   No change to existing production pipeline code, minimizing regression risk.
*   **Cons**: Requires maintaining the overridden async loop.

---

### Option B: Optional Subagents in `AutonomousPipelineAgent`
We modify the base `AutonomousPipelineAgent` to accept `None` for `autotune_agent` and `profile_agent`, and adjust the control flow logic internally.

#### **Changes to `AutonomousPipelineAgent`:**
1.  Make subagents optional in constructor:
    ```python
    autotune_agent: BaseAgent | None = None
    profile_agent: BaseAgent | None = None
    ```
2.  Update the loop to check for `None` before executing steps 6 (Autotune) and 7 (Profile).
3.  Add an **early exit check** after correctness test validation if tuning is disabled.

#### **Pros & Cons:**
*   **Pros**:
    *   No subclassing required. Single class manages all variants.
*   **Cons**:
    *   Modifies core MaxKernel pipeline control logic, which requires careful testing.
    *   Introduces conditional checks inside a previously clean state machine loop.

### Recommendation
We recommend **Option A (Subclassing)**. The control flow of `AutonomousPipelineAgent` is highly tuned for iterative improvement (running multiple iterations to find the *best* candidate even if the first one is correct). A Beam Search worker wants to exit *immediately* on the first correct compilation to save time, as the outer orchestrator handles performance selection. Subclassing allows us to cleanly express this difference in loop invariants without adding complex conditional checks inside the production pipeline.

---

## 5. Detailed Implementation Phases

### Phase 1: Implement the Correctness Worker in `auto_agent`
We will reuse MaxKernel's existing subagents by implementing a specialized correctness pipeline.

1.  **Create `BeamWorkerPipeline`**:
    *   Implement `MaxKernel/auto_agent/subagents/beam_worker_pipeline.py` subclassing `AutonomousPipelineAgent` (Option A).
    *   Add an early-exit check: as soon as `test_results.get("success")` is `True`, yield a success event with the `optimized_kernel_path` and terminate the generator loop.
2.  **Expose the Worker Entrypoint**:
    *   Create `MaxKernel/auto_agent/beam_worker.py`.
    *   Instantiate `BeamWorkerPipeline` using the existing production instances: `plan_kernel_agent`, `implement_kernel_agent`, `validate_kernel_compilation_agent`, `validated_test_generation_agent`, and `unified_test_agent`.

### Phase 2: Add Fake Evaluation Backend
To support local testing of the orchestrator loop without requiring physical TPU/Trainium hardware, we will port the fake backend as a standalone Python file.

1.  **Implement `FakeKernelEvaluator`**:
    *   Create `MaxKernel/evaluation/fake_kernel_evaluator.py`.
    *   The evaluator should accept multiple inlined code candidates, simulate compilation, and return randomized (but decreasing across rounds) execution latencies in milliseconds.

### Phase 3: Implement the Beam Search Orchestrator
Implement the top-layer coordinator that manages the exploration loop.

1.  **Create Orchestrator Logic**:
    *   Create `MaxKernel/beam_search/orchestrator.py`.
    *   Implement `AgenticSearchOrchestrator` which:
        *   Accepts a starting baseline Pallas kernel and a target test harness.
        *   Spawns multiple `beam_worker_agent` instances in parallel (using `asyncio.gather`) with distinct optimization prompt directives (e.g. tiling-focus vs. memory-bound focus).
        *   Gathers the correct code paths from the completed worker sessions.
        *   Renames the entry point functions (e.g., `solution_0`, `solution_1`) and inlines them into a single grouped test harness file.
        *   Dispatches the grouped harness to the evaluation backend (`FakeKernelEvaluator` or real TPU backend `JAXKernelEvaluator`) for latency profiling.
        *   Ranks and filters the candidates to select the top candidates for the next round.
2.  **Configure Mock Mode**:
    *   In `beam_search/tools.py` (or directly in the orchestrator config), read the environment variable `MOCK_COMPILER=True` or a command-line flag `--mock` to toggle between calling the `FakeKernelEvaluator` (simulated runs) and the real hardware test runner.

### Phase 4: Local Verification
Verify the integrated orchestrator-worker loop.

1.  **Create local runner script**:
    *   Implement `MaxKernel/run_beam_search.sh` or a Python helper script to trigger `beam_search/orchestrator.py` with mock mode enabled.
2.  **Run a test round**:
    *   Verify that:
        *   The orchestrator correctly spawns multiple worker pipelines.
        *   The workers execute the MaxKernel implementation agents and compile locally.
        *   On correctness success, workers exit early.
        *   The orchestrator groups the outputs, calls the fake backend, and prints the latency summary.
