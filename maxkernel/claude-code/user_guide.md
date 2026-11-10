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

Observe `state.json` as it makes progress.

**The first thing that happens is not optimization.** After the base kernel and
test harness are prepared, MaxKernel characterizes the baseline into
`workspace/<id>/baseline/`. That phase runs several TPU jobs and produces no
faster kernel, which is expected: it is buying the map every later decision is
steered by. The two files worth opening yourself:

*   `baseline/roofline.md`: a per-case table of how fast this kernel *could*
    go, with a target range and a one-sentence statement of which direction the
    run is going to work in.
*   `baseline/attribution.md`: where the baseline's time actually goes, ranked.

Then the artifacts for each iteration appear in `workspace/<id>/iter<n>/…`. The
optimized code is `optimized.py`; `kernel_plan.md` opens with the hypothesis the
iteration is testing, and `keep_decision.md` says whether it survived and
whether its stated mechanism turned out to be right (those two can differ).

Some per-iteration files are deliberately absent: `floor.json` only exists if
the change was kept, `gap_probe.json` only if the gap was worth probing, and
`autotune_summary.md` only if a tuning trigger fired. A missing file is a
result, not a crash.

**The run has no fixed length.** It ends when it converges, however many
iterations that takes. If `state.stopping.recommend_stop` is `true`, it did: the
kernel is close to a floor that is itself close to the hardware, and several
experiments against distinct hypotheses came back under 1%. Read
`state.stopping.untested_hypotheses` before concluding the kernel is done: it
lists the explanations nobody got round to attacking.

If instead the run reaches `state.max_iterations` (default 5), it **stopped
without converging**: the loop was cut off mid-search, not finished, and
`state.stopping` will tell you which criterion was still unmet. That is the case
where re-prompting *"continue to optimize this kernel"* is most likely to pay.
You can also raise `max_iterations` in `state.json` before starting, or lower it
if you only want a quick look.

If the agent gets stuck, you can always stop it and re-prompt it to *"continue to
optimize this kernel"* — it will pick up where it left off from `state.json`. The baseline phase is
idempotent and will not be re-run.
