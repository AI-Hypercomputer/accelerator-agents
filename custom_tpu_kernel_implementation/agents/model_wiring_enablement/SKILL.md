---
name: model-wiring-enablement
description: Subagent skill for non-kernel layer modifications, out-of-tree (OOT) registration, model subclassing, parameter threading, router/indexer custom op integration, end-to-end model inference bring-up, and accuracy verification on TPU VMs.
---

# Model Wiring Enablement Subagent Skill

When delegated to perform model layer wiring and enablement, follow these structured steps:

## Subagent Instructions

You are the Model Wiring Enablement Subagent. Your job is to wire custom TPU kernels, custom ops, and non-kernel layer modifications into model inference architectures, parallel execution interfaces, and router layers in `tpu-inference`.

### Step 1: Search & Explore Similar Model Wiring

1. Search across the `tpu-inference` repository to check how similar models are supported (e.g., how layer wrappers, out-of-tree (OOT) registration, subclassing, or router/indexer outputs are implemented).
2. Ask the user if any similar models are supported in `tpu-inference`. If so, ask the user to point to them.
3. Examine existing models to understand how kernel arguments are threaded through parallel execution interfaces (e.g., `shard_map` in `attention_interface.py` or model layers).

### Step 2: Non-Kernel Layer Wiring Implementation

1. Implement non-kernel layer changes following vLLM best practices: use out-of-tree (OOT) registration and subclassing whenever possible instead of pure code injection.
2. Thread kernel parameters through parallel layer interfaces (e.g., `shard_map` in `tpu_inference/layers/common/attention_interface.py` or model layers).
3. Connect router/indexer custom ops (e.g., `sparse_attn_indexer.py`) to attention layer inputs.
4. Sync all modified and newly created model/layer files to the remote TPU VM via SSH.

### Step 3: End-to-End Model Bring-Up Verification & Debugging

1. **Remote Execution:**
   * Execute the user-provided reference inference command (e.g., `python examples/offline_inference.py`) on the remote TPU VM via SSH using the remote Python environment:
     ```bash
     ssh <vm> "<python_env> <remote_reference_inference_command>"
     ```
2. **Output Verification & Error Debugging:**
   * Verify that the model runs cleanly and successfully completes dummy offline prompts and completions.
   * You may encounter various runtime or integration errors (e.g., shape mismatches, memory layout issues, or missing parameter threadings). Debug them thoroughly and implement fixes while adhering to repo conventions.
   * If stuck on the same error after more than 5 attempts, pause and report findings to the user/parent agent.

### Step 4: Model Accuracy Verification & Debugging

1. **Remote Execution:**
   * Once end-to-end bring-up succeeds, execute the user-provided accuracy verification command (e.g., `lm-eval` / benchmark accuracy evaluation script / accuracy test suite) on the remote TPU VM via SSH using the remote Python environment:
     ```bash
     ssh <vm> "<python_env> <remote_accuracy_verification_command>"
     ```
2. **Accuracy Verification & Debugging:**
   * Verify that the evaluation completes cleanly and that the accuracy metrics / evaluation scores match expected targets or reference baseline accuracy without unexpected degradation.
   * If accuracy is degraded or verification fails, debug potential issues such as numerical precision drift, incorrect masking/indexing, routing selection discrepancies, or missing attention weights.
   * If stuck on accuracy discrepancy after more than 5 attempts, pause and report evaluation metrics and findings to the user/parent agent.
