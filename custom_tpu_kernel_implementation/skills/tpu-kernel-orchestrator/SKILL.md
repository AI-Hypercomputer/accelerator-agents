---
name: tpu-kernel-orchestrator
description: This skill should be used when the user asks to "implement a custom TPU kernel", "bring up a Pallas TPU kernel", "wire a new model feature into tpu-inference", "add TPU kernel support for <model>", "port a GPU kernel to TPU", or wants to add attention, sparse-indexing, or quantization support to TPU Inference. Interactively collects setup details, tracks state in state.md across turns, formulates an exploration plan, and delegates kernel implementation and model wiring to the kernel-developer and model-wiring-enablement subagents.
---

# Custom TPU Kernel Implementation & Wiring Orchestrator

Use this skill to run the orchestrator ("the brain") role for bringing up a custom TPU kernel and wiring it into a model inference pipeline in `tpu-inference`. Act as the primary orchestrator: collect setup details, maintain `state.md`, formulate and get sign-off on an exploration plan, then delegate implementation to the `kernel-developer` and `model-wiring-enablement` subagents via the Agent tool (`subagent_type: kernel-developer` / `subagent_type: model-wiring-enablement`). Run each delegated agent in the foreground (`run_in_background: false`) — its output determines whether to proceed to the next phase.

## Phase 1: Interactive Information Collection & Setup

Prompt the user for the required setup information one question at a time. Collect all of the following before proceeding:

1. **Target Model & Feature Specs**: The target model and feature to implement (e.g., GLM-5.2 DSA, DeepSeek-v4, custom attention, quantization). Ask for a HuggingFace link, paper/doc link, or config reference.
2. **TPU Inference Path**: The directory containing the local `tpu-inference` repository.
3. **Reference JAX Implementation / Code Path**: The path to the reference JAX implementation or baseline JAX kernel file and function name. If none exists, ask whether one needs to be created from GPU reference code or a mathematical specification.
4. **Pallas TPU Kernel Path**: The path to the Pallas TPU kernel file to be created or modified.
5. **Reference / Baseline Kernel Code (if any)**: Ask whether any existing baseline, prototype, or reference kernel code exists (GPU reference code, prototype TPU kernel, related ops), and its path. Do not assume a reference kernel exists.
6. **Target Hardware**: The target TPU architecture (e.g., v5e, v6e, v7x/Ironwood, v7p).
7. **SSH Details**: An example SSH command for the TPU machine (e.g., `ssh <host>`). Confirm the machine matches the target hardware and is reachable.
8. **Python Environment on VM**: The exact path to the Python environment on the remote TPU machine.
9. **Remote TPU Inference Path & Reference Inference Command**: The path to `tpu-inference` on the VM and the reference offline inference command/script used for bring-up (runs a few dummy offline prompts and completions).
10. **Accuracy Verification Command**: The command or test script used to evaluate and verify model output accuracy (e.g., `lm-eval`, task evaluation scripts, accuracy tests).

## State Tracking (`state.md`)

Maintain a `state.md` file in the workspace directory and update it continuously so context survives across turns. It must include:
- **Current Objective & Feature**: Target model feature and kernel changes.
- **Current Execution Phase**: Setup, Exploration, Kernel Development, Model Wiring, Bring-Up Verification, or Accuracy Verification.
- **Active Run & Test Details**: Latest test/evaluation command executed on the TPU VM via SSH and its pass/fail status.
- **JAX Reference Checkpoint Status**: Verification status of the ground-truth JAX kernel.
- **Pallas TPU Kernel Status**: Verification status of the Pallas TPU kernel.
- **End-to-End Bring-Up Status**: Verification status of the offline reference inference command.
- **Accuracy Verification Status**: Verification status of the model accuracy evaluation.
- **Hypothesis Backlog & Next Steps**: Immediate next actions and remaining subagent delegations.

## Phase 2: Exploration & Plan Generation

1. **Search & explore similar TPU kernels & model features**: Search `tpu-inference` for existing similar kernel features or related model architectures (attention variants, sparse indexing ops, quantization formats, layer wrappers). Ask the user if any similar models or reference implementations exist and, if so, where. Analyze them for design, VMEM allocation patterns, and kernel usage to leverage established patterns.
2. **Evaluate extending vs. creating new TPU kernels**: Bias toward extending an existing kernel to minimize code duplication. Present findings and a concrete recommendation (extend vs. create new, with trade-offs and rationale) and explicitly ask for the user's preference before finalizing the approach.
3. **Synthesize plan & obtain sign-off**: Produce a step-by-step plan covering JAX reference implementation, Pallas TPU kernel implementation, layer wiring, bring-up tests, and accuracy verification. Present it to the user, incorporate feedback, and get explicit approval before implementation begins.

## Phase 3: Delegate Kernel Development

Invoke the `kernel-developer` agent (Agent tool, `subagent_type: kernel-developer`) to execute kernel implementation and targeted testing:

1. **JAX reference kernel & targeted test**: The agent implements/modifies the pure JAX reference kernel, builds a targeted unit test, and executes `pytest` on the TPU VM via SSH.
2. **Progress report & user checkpoint**: The agent reports whether the JAX reference kernel was implemented and verified. Get user feedback and confirmation before proceeding to Pallas.
3. **Pallas TPU kernel & optimization**: The agent implements the production Pallas TPU kernel (VMEM double-buffering `pltpu.VMEM`, DMA semaphores, scalar prefetches) and verifies numerical tolerance (`assert_allclose`) against the JAX reference kernel on the VM via SSH. Update `state.md`.

## Phase 4: Delegate Model Wiring & Verification

Invoke the `model-wiring-enablement` agent (Agent tool, `subagent_type: model-wiring-enablement`) to execute non-kernel model wiring, end-to-end bring-up, and accuracy verification:

1. **Model layer & interface wiring**: The agent implements non-kernel layer modifications via OOT registration and subclassing, threads kernel parameters through `shard_map`/layer interfaces, and connects router/indexer custom ops.
2. **End-to-end bring-up verification**: The agent executes the reference inference command on the TPU VM via SSH using the remote Python environment to verify dummy offline prompts and completions.
3. **Accuracy verification**: The agent executes the accuracy verification command on the TPU VM via SSH to verify model output accuracy and evaluation metrics. Update `state.md`.

## Phase 5: Bookkeeping, Documentation & Cleanup

1. Maintain `dev_log.md` and `state.md` in the root directory tracking decisions, subagent outputs, kernel modifications, and test results.
2. Clean up temporary scratch scripts, saved `.npy` tensors, and temporary runscripts.
3. Archive raw `.log` files into a `logs/` subfolder.
4. Generate a git diff, patch file, or clear summary of all modified and created files for final user review.

## Additional Resources

Detailed subagent instructions live in:
- **`../../agents/kernel-developer.md`** — JAX/Pallas kernel implementation and verification.
- **`../../agents/model-wiring-enablement.md`** — non-kernel layer wiring and bring-up/accuracy verification.

### Example Prompt

> "Invoke the tpu-kernel-orchestrator skill. I want to bring up Dynamic Sparse Attention (DSA) support for GLM-5.2 in TPU Inference."
