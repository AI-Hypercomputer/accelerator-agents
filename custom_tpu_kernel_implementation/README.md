# Custom TPU Kernel Implementation & Wiring Skill

This directory contains a suite of AI agent skills designed to automate, track, and verify the process of implementing custom TPU kernels (both pure JAX reference implementations and Pallas TPU kernels) and wiring them into LLM inference pipelines in `tpu-inference`.

## Overview of Skills

This project is powered by three primary agent skills located in the `agents/` directory, working in tandem (following the orchestrator-subagent architecture):

1. **`orchestrator`** (`agents/orchestrator/`): The core entrypoint skill (`tpu-kernel-orchestrator`) that interactively collects setup requirements, manages continuous state tracking (`state.md`), formulates exploration plans, presents trade-off recommendations to the user, and delegates specialized sub-tasks to the subagents.
2. **`kernel_developer`** (`agents/kernel_developer/`): A specialized subagent skill (`kernel-developer`) dedicated to:
   - Implementing ground-truth JAX reference kernels and running targeted unit tests on remote TPU VMs via SSH.
   - Presenting progress checkpoints to the user after JAX reference kernel verification.
   - Implementing production Pallas TPU kernels with hardware optimizations (VMEM double-buffering `pltpu.VMEM`, DMA semaphores, scalar prefetches) and verifying numerical tolerance (`assert_allclose`) on the TPU VM.
3. **`model_wiring_enablement`** (`agents/model_wiring_enablement/`): A specialized subagent skill (`model-wiring-enablement`) dedicated to:
   - Implementing non-kernel layer changes using out-of-tree (OOT) registration and subclassing.
   - Threading kernel parameters and router/indexer custom ops through layer wrappers (`shard_map`).
   - Executing end-to-end reference inference bring-up and accuracy verification on the remote TPU VM via SSH and debugging runtime or accuracy errors.

## How to Use

To begin a custom TPU kernel bring-up or model feature implementation:

1. Ask the agent to invoke the **`tpu-kernel-orchestrator`** skill (or `orchestrator` skill in `agents/orchestrator/`).
2. The agent will prompt you interactively (one question at a time) for setup parameters, SSH details, and reference commands.
3. The orchestrator will track state in `state.md` and delegate tasks to the `kernel-developer` and `model-wiring-enablement` subagents.

### Example Prompt

> "Invoke the tpu-kernel-orchestrator skill. I want to bring up Dynamic Sparse Attention (DSA) support for GLM-5.2 in TPU Inference."
