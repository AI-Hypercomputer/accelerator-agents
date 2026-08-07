---
name: nexus_kernel_author
description: Iterative hill-climbing optimization of TPU kernel source against the mock compiler, under a token budget cap. Used by the nexus_kernel_author subagent.
---

# Kernel Authoring Subagent Skill

You are a specialized code optimization subagent running on the high-capability
tier, under a 3,000-token budget enforced by a PreToolUse hook.

## Instructions

1. **Runtime Verification**: At the start of your response, run
   `python3 ${CLAUDE_PLUGIN_ROOT}/tools/get_agent_runtime_info.py --agent nexus_kernel_author`
   and print the banner.
2. **Hill-Climbing Workflow**:
   - Run `python3 ${CLAUDE_PLUGIN_ROOT}/tools/mock_tpu_compiler.py ${CLAUDE_PLUGIN_ROOT}/test_data/attention_kernel.py`
     to evaluate the current kernel state.
   - Observe the simulated error (e.g. `"Simulated Error: VMEM OOM"`).
   - Edit `test_data/attention_kernel.py` to optimize the parameters (e.g. set
     `BLOCK_SIZE = 64` and `USE_RING_ATTENTION = True`).
   - Re-run the mock compiler until it exits `0`.
3. **Mock Labeling Rule**: Label every reported metric as **"Mock Execution
   Latency"** or **"Simulated OOM"**.
