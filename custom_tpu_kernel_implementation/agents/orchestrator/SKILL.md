---
name: tpu-kernel-orchestrator
description: An orchestrator subagent skill that interactively collects setup details, manages continuous state tracking (state.md), formulates exploration plans, and delegates work to the kernel-developer and model-wiring-enablement subagent skills to bring up custom TPU kernels and wire them into model inference pipelines.
---

# Custom TPU Kernel Implementation & Wiring Orchestrator

When asked to implement custom TPU kernels or wire new model features, follow this structured orchestrator workflow.

## Subagent Delegation Overview

You act as the primary orchestrator ("the brain"). You will:
1. Interactively collect setup details and parameters from the user.
2. Initialize and continuously update `state.md` to track progress and context across turns.
3. Formulate high-level plans (exploring existing TPU kernels, evaluating extending vs creating new kernels, presenting options for user sign-off).
4. Delegate kernel implementation and verification to the **`kernel-developer`** subagent skill.
5. Delegate non-kernel layer wiring and end-to-end inference verification to the **`model-wiring-enablement`** subagent skill.


## Phase 1: Interactive Information Collection & Setup

Begin by interactively prompting the user to provide the required setup information. Ask these questions one at a time. Collect these details clearly before proceeding:

1. **Target Model & Feature Specs**: The target model and feature to implement (e.g., GLM-5.2 DSA, DeepSeek-v4, custom attention, quantization, etc.); ask the user for a HuggingFace link, paper/doc link, or config reference.
2. **TPU Inference Path**: The directory containing the local `tpu-inference` repository (e.g., `/usr/local/google/home/jacobplatin/repos/tpu-inference`).
3. **Reference JAX Implementation / Code Path**: The path to the reference JAX implementation or baseline JAX kernel file (ask the user for the exact file path and function name; if none exists, ask if one needs to be created from GPU reference code or mathematical specifications).
4. **Pallas TPU Kernel Path**: The path to the Pallas TPU kernel file to be created or modified (ask the user for the file path or candidate kernel files).
5. **Reference / Baseline Kernel Code (if any)**: Ask the user if any existing baseline, prototype, or reference kernel code (e.g., GPU reference code, prototype TPU kernel, or related ops) exists, and if so, ask for its path. Do not assume a reference kernel exists.
6. **Target Hardware**: The target TPU architecture (e.g., v5e, v6e, v7x/Ironwood, v7p).
7. **SSH Details**: Example SSH command for the TPU machine (e.g., `ssh jacobplatin-v7p8-1`). Ensure the machine matches the target hardware and you have SSH access.
8. **Python Environment on VM**: The exact path to the Python environment on the remote TPU machine (e.g., `/mnt/disks/jacobplatin/anaconda3/envs/new-jax/bin/python`).
9. **Remote TPU Inference Path & Reference Inference Command**: The path to `tpu-inference` on the VM (e.g., `/mnt/disks/jacobplatin/tpu-inference`) and the reference offline inference command or script used for model bring-up (e.g., `python examples/offline_inference.py ...` which runs a few dummy offline prompts and completions).
10. **Accuracy Verification Command**: The command or test script used to evaluate and verify model output accuracy and benchmark evaluation metrics (e.g., `lm-eval`, task evaluation scripts, or reference accuracy verification tests).


## State Tracking (`state.md`) Setup

To ensure resilience and allow context recovery across turns, you MUST maintain a `state.md` file in the workspace directory. Update this file continuously as you progress through the workflow. It must include:
- **Current Objective & Feature**: Target model feature and kernel changes.
- **Current Execution Phase**: (e.g., Setup, Exploration, Kernel Development, Model Wiring, Bring-Up Verification, Accuracy Verification).
- **Active Run & Test Details**: Latest test or evaluation command executed on TPU VM via SSH and pass/fail status.
- **JAX Reference Checkpoint Status**: Verification status of the ground-truth JAX kernel.
- **Pallas TPU Kernel Status**: Verification status of the Pallas TPU kernel.
- **End-to-End Bring-Up Status**: Verification status of the offline reference inference command.
- **Accuracy Verification Status**: Verification status of the model accuracy evaluation.
- **Hypothesis Backlog & Next Steps**: Immediate next actions and remaining subagent delegations.


## Phase 2: Exploration & Plan Generation

1. **Search & Explore Similar TPU Kernels & Model Features:**
   * Search across the `tpu-inference` repository to identify whether similar kernel features or related model architectures (e.g., similar attention variants, sparse indexing ops, quantization formats, or layer wrappers) already exist in the codebase.
   * Ask the user if any similar models or reference feature implementations exist in `tpu-inference`. If so, ask the user to point to them.
   * Analyze existing similar models/kernels to understand their design, layer structures, VMEM allocation patterns, and kernel usage so you can leverage established patterns.
2. **Evaluate Extending Existing vs. Creating New TPU Kernels:**
   * Evaluate whether an existing TPU kernel in `tpu-inference` can be extended/modified to support the new feature, or whether a new kernel must be created from scratch (bias towards extending existing kernels whenever possible to minimize code duplication).
   * Present findings and analysis to the user along with a concrete recommendation (e.g., extending an existing kernel vs creating a new one from scratch, explaining trade-offs and rationale). Explicitly ask for user input and preference before finalizing the approach.
3. **Synthesize Plan & Obtain Sign-off:**
   * Synthesize a clear step-by-step plan covering JAX reference implementation, Pallas TPU kernel implementation, layer wiring, bring-up tests, and accuracy verification tests. Present the plan to the user, incorporate feedback, and obtain explicit user approval before proceeding to implementation.


## Phase 3: Delegate Kernel Development (`kernel-developer`)

Invoke the **`kernel-developer`** subagent skill (via `invoke_subagent`) to execute kernel implementation and targeted testing:

1. **JAX Reference Kernel & Targeted Test**: `kernel-developer` implements/modifies the pure JAX reference kernel, builds a targeted unit test, and executes `pytest` on the TPU VM via SSH.
2. **Progress Report & User Checkpoint**: `kernel-developer` presents a progress report to the user summarizing whether the JAX reference kernel was successfully implemented and verified. Get user feedback and confirmation before proceeding to Pallas.
3. **Pallas TPU Kernel & Optimization**: `kernel-developer` implements the production Pallas TPU kernel (VMEM double-buffering `pltpu.VMEM`, DMA semaphores, scalar prefetches) and verifies numerical tolerance (`assert_allclose`) against the JAX reference kernel on the VM via SSH. Update `state.md`.


## Phase 4: Delegate Model Wiring & Verification (`model-wiring-enablement`)

Invoke the **`model-wiring-enablement`** subagent skill (via `invoke_subagent`) to execute non-kernel model wiring, end-to-end bring-up, and accuracy verification:

1. **Model Layer & Interface Wiring**: `model-wiring-enablement` implements non-kernel layer modifications using out-of-tree (OOT) registration and subclassing, threads kernel parameters through `shard_map` / layer interfaces, and connects router/indexer custom ops.
2. **End-to-End Bring-Up Verification**: `model-wiring-enablement` executes the reference inference command (e.g., `python examples/offline_inference.py`) on the TPU VM via SSH using the remote Python environment to verify dummy offline prompts and completions.
3. **Accuracy Verification**: `model-wiring-enablement` executes the accuracy verification command on the TPU VM via SSH using the remote Python environment to verify model output accuracy, numerical correctness, and evaluation metrics. Update `state.md`.


## Phase 5: Bookkeeping, Documentation & Cleanup

1. Maintain `dev_log.md` and `state.md` files in the root directory tracking decisions, subagent outputs, kernel modifications, and test results.
2. Clean up temporary scratch scripts, saved `.npy` tensors, and temporary runscripts.
3. Archive raw `.log` files into a subfolder named `logs/`.
4. Generate a git diff, patch file, or clear summary of all modified and created files for final user review.
