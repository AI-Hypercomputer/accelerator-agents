# Custom TPU Kernel Implementation & Wiring Skill

This directory contains a Claude Code skill + subagent pair designed to automate, track, and verify the process of implementing custom TPU kernels (both pure JAX reference implementations and Pallas TPU kernels) and wiring them into LLM inference pipelines in `tpu-inference`.

It follows Claude Code's native conventions: a `SKILL.md` orchestrator skill that delegates to two subagent definitions via the Agent tool (`subagent_type`), rather than a custom `invoke_subagent` mechanism.

## Layout

```
custom_tpu_kernel_implementation/
├── skills/
│   └── tpu-kernel-orchestrator/
│       └── SKILL.md            # entrypoint skill
└── agents/
    ├── kernel-developer.md          # subagent: JAX/Pallas kernel implementation & verification
    └── model-wiring-enablement.md   # subagent: non-kernel layer wiring, bring-up & accuracy verification
```

## Overview

1. **`skills/tpu-kernel-orchestrator/SKILL.md`**: The entrypoint skill that interactively collects setup requirements, manages continuous state tracking (`state.md`), formulates exploration plans, presents trade-off recommendations to the user, and delegates specialized sub-tasks to the subagents via the Agent tool.
2. **`agents/kernel-developer.md`**: A subagent dedicated to:
   - Implementing ground-truth JAX reference kernels and running targeted unit tests on remote TPU VMs via SSH.
   - Presenting progress checkpoints to the user after JAX reference kernel verification.
   - Implementing production Pallas TPU kernels with hardware optimizations (VMEM double-buffering `pltpu.VMEM`, DMA semaphores, scalar prefetches) and verifying numerical tolerance (`assert_allclose`) on the TPU VM.
3. **`agents/model-wiring-enablement.md`**: A subagent dedicated to:
   - Implementing non-kernel layer changes using out-of-tree (OOT) registration and subclassing.
   - Threading kernel parameters and router/indexer custom ops through layer wrappers (`shard_map`).
   - Executing end-to-end reference inference bring-up and accuracy verification on the remote TPU VM via SSH and debugging runtime or accuracy errors.

## Installing

Claude Code auto-discovers skills under `skills/` and agents under `agents/` at either the personal (`~/.claude/`) or project (`.claude/`) level. To activate this pair, copy or symlink both directories into one of those locations, e.g.:

```bash
ln -s "$(pwd)/skills/tpu-kernel-orchestrator" ~/.claude/skills/tpu-kernel-orchestrator
ln -s "$(pwd)/agents/kernel-developer.md" ~/.claude/agents/kernel-developer.md
ln -s "$(pwd)/agents/model-wiring-enablement.md" ~/.claude/agents/model-wiring-enablement.md
```

## How to Use

To begin a custom TPU kernel bring-up or model feature implementation:

1. Ask Claude to invoke the **`tpu-kernel-orchestrator`** skill.
2. It will prompt you interactively (one question at a time) for setup parameters, SSH details, and reference commands.
3. The orchestrator will track state in `state.md` and delegate tasks to the `kernel-developer` and `model-wiring-enablement` subagents via the Agent tool.

### Example Prompt

> "Invoke the tpu-kernel-orchestrator skill. I want to bring up Dynamic Sparse Attention (DSA) support for GLM-5.2 in TPU Inference."
