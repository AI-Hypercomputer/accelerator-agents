# maxtext code — Llama 3 8B autoresearch (reference baseline stack)

Reference-baseline runs of **Llama 3.1-8B via [MaxText](../../codebases/maxtext.md)** on TPU. This is the **reference** stack for the program — we don't develop optimizations *here*; we run the official AI-Hypercomputer recipes verbatim and use them as a measured ceiling for the torchax and native-JAX stacks to target.

Companion folders: [`../torchax/`](../torchax/README.md) (PyTorch-on-JAX, where the program started — final 6,559/chip 36.8 % MFU), [`../jax/`](../jax/README.md) (native-JAX port — final ~7,700/chip 43.3 % MFU, **+8.9 % per-chip vs MaxText**).

## Status (2026-04-27)

| Run | tok/s/chip | MFU | Date | Page |
|-----|-----------:|----:|------|------|
| MaxText `tpu-recipes-v0.1.4` recipe `llama3_1_8b_8192_no_collective_matmul`, bs=3 seq=8192 fsdp=8 v6e-8 | **7,069** | **44.6 %** | 2026-04-25 | [baseline](experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md) |

Reproduces the recipe-README's published Trillium numbers (~413 TFLOPs/dev, ~7,139 TPS) within −1 % on this cluster. Used as the published ceiling that the torchax + native-JAX stacks target.

**As of 2026-04-27 the native-JAX stack has exceeded this baseline by +8.9 % per chip** (mean of 3 reruns: ~7,700/chip 43.3 % MFU; peak run 7,768/43.6 %). The 1.0 pp reported MFU gap is FLOP-counter normalization difference (under MaxText's accounting our throughput translates to 49.0 % MFU, +4.4 pp above their reported 44.6 %).

## Convention

- **No source code in this folder** — we run upstream MaxText verbatim. The MaxText repo lives under [`raw/code/maxtext`](../../../../raw/code/maxtext) (submodule); each experiment pins a specific tag (e.g. `tpu-recipes-v0.1.4`) and a `jax-stable-stack` Docker base image (e.g. `jax0.6.1-rev1`). Recipes pulled from [`raw/code/tpu-recipes/training/trillium/Llama3.1-8B-MaxText/`](../../../../raw/code/tpu-recipes/training/trillium/Llama3.1-8B-MaxText/).
- **What lives in this folder**: `experiments/` — dated experiment pages (`<YYYY-MM-DD>-<slug>.md`) and any per-experiment override notes. No `train.py`, `model/`, `config.yaml` etc.
- **Working trees outside `raw/`**: when a tag must be checked out (e.g. `git worktree add /mnt/disks/persist/maxtext-tpu-recipes-v0.1.4 tpu-recipes-v0.1.4`), do it under `/mnt/disks/persist/` per SCHEMA rule 1; record the path in the experiment page.
- **Each experiment page** carries: the recipe path used, the exact MaxText commit, the BASE_IMAGE, the xpk workload command, the steady-state metrics table, and the profile path. Profiles land in `raw/profiles/<YYYY-MM-DD>-<exp-slug>/`.

## Why a separate stack folder

This wiki's `program-gke.md` calls MaxText "the reference stack" we benchmark our hand-tuned stacks against. Splitting it into `maxtext/` (separate from `torchax/` and `jax/`) makes that role explicit:

- A new optimization in `torchax/` is judged "did it improve over the prior `torchax/` run?" *and* "did it close the gap to the matched `maxtext/` ceiling?"
- The `maxtext/` runs themselves are not optimized — they are reproductions of published recipes, used to anchor the per-cluster ceiling.

If a `maxtext/` knob ablation does turn into an optimization study (e.g. "is the host-offload set in the recipe overkill at v6e-8?"), file it as a hypothesis page under `wiki/hypotheses/` with `origin: maxtext-recipe-v0.1.4`, not as a code change here.

## Layout

```
maxtext/
  README.md              this file
  experiments/           per-experiment pages (dated)
    2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md   ← first reference baseline
    ...                   future cross-cluster ports + ablations
```

## See also

- [MaxText codebase page](../../codebases/maxtext.md)
- [tpu-recipes codebase page](../../codebases/tpu-recipes.md)
- [llama3_8B_autoresearch_optimization program](../README.md)
