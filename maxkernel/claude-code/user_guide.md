# MaxKernel User Guide

A short walkthrough of running MaxKernel end to end. For what the repo contains
and how the loop is wired, see [README.md](README.md).

## Prerequisites

Then follow the instructions in [README.md](README.md) to:

1. Register the agent in your Claude Code.
2. Launch Claude in fully auto mode:

   ```bash
   claude --dangerously-skip-permissions
   ```

## Prompt

````
Optimize this code into Pallas kernel:
```
<your code or code path>
```

Use this TPU VM when needed:
gcloud compute tpus tpu-vm ssh <your_VM> \
  --zone=<zone> \
  --project=<project>
````

You can skip the TPU VM part if you want to use the local TPU VM you are running
MaxKernel from. You can also provide it with a GKE cluster.

## What you can give it

`<your code or code path>` can be any of these, and you do not have to
say which — the run classifies it first and records the answer in `state.json`
as `input_language`:

- **JAX / Pallas** — used as the baseline directly.
- **CUDA** — a `.cu`/`.cuh` file, or CUDA sources embedded in a Python file for
  `torch.utils.cpp_extension.load_inline`. (Coming soon)
- **PyTorch** — an `nn.Module` reference.
- **Triton** — Triton kernels launched on torch tensors from a Python entry
  point (`Model`, `computation`, …) with a `get_inputs()`. The port is checked
  against outputs recorded by running it on CPU through Triton's interpreter,
  as for PyTorch.
- **cuTile** — cuTile kernels with a Python entry point (`Model`,
  `computation`, …).

You can also hand over one **reference kernel**: a GPU implementation of the
same computation in CUDA, Triton, cuTile, CuTe DSL or another GPU language, as
a file or a directory. Say so in the prompt ("use `kernel.cu` as a
reference"). A reference is never run or measured; the planner mines it for
design ideas, after a reconciliation step checks that it computes what your
primary computes.

## Watching a run

Observe `state.json` as it makes progress. The artifacts for each iteration will
be in `workspace/<id>/iter<n>/…`. The optimized code is `optimized.py`.

If the agent gets stuck, you can always stop it and re-prompt it to *"continue to
optimize this kernel"* — it will pick up where it left off from `state.json`.
