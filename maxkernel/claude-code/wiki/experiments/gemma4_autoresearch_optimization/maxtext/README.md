# maxtext code — Gemma 4 E4B autoresearch (reference baseline stack, **APPROXIMATION**)

Reference-baseline runs of **Gemma 4 E4B in [MaxText](../../codebases/maxtext.md)** on TPU. Mirrors the role of the [llama3_8B/maxtext/](../../llama3_8B_autoresearch_optimization/maxtext/README.md) stack: we don't develop optimizations here; we run a fixed config close to upstream and use the result as a measured ceiling for the `torchax/` and `jax/` stacks to target.

> ## ⚠️ Approximation caveat — read before using these numbers
>
> Unlike the Llama-3.1-8B MaxText baseline (which reproduces an upstream-published recipe verbatim), **MaxText does NOT ship a Gemma 4 E4B model**. The configs that exist upstream are `gemma4-26b.yml` (MoE) and `gemma4-31b.yml` (Dense) — both significantly larger and architecturally different. Worse, **Gemma 4 support landed on MaxText `main` after `tpu-recipes-v0.1.4`** — there is no released-tag stack for Gemma 4 of any size on Trillium.
>
> We author a **wiki-local `gemma4-e4b.yml`** (text-only, ports the HF `google/gemma-4-E4B` `text_config` dimensions into MaxText's config format). The runs are **architectural approximations of E4B**, not bit-exact ports:
>
> - **HF E4B has `num_kv_shared_layers=18`** — the last 18 of 42 layers reuse K/V projections from earlier same-type layers. **MaxText's `gemma4` decoder_block does not implement this feature.** The approximation has 18 extra k/v projection sets — about **+47M params (~0.6 % over true E4B)**. Marginal compute impact; not zero.
> - **MaxText `share_kv_projections` is unrelated** — it shares K/V projections within a layer (across attention modules), not across layers. We set it to `false` here.
> - **Layer attention pattern matches** — 7 cycles of `(SW, SW, SW, SW, SW, GLOBAL)` = 42 layers, same as HF E4B. ✅
> - **Dimensions match** — 42 layers, hidden=2560, heads=8, kv_heads=2, head_dim=256, FFN=10240, vocab=262144, SW=512, soft-cap=30.0. ✅
> - **Same Python entrypoint as upstream Gemma 4 26B/31B**: `python3 -m maxtext.trainers.pre_train.train` with `model_name=gemma4-e4b`.
>
> **What the throughput represents**: the FLOPs/s and TPS are valid measurements of the dense-shape model that this config builds; they're directly comparable across runs that use this same config. They are **not directly comparable to wall-clock numbers from a true HF E4B implementation** until either (a) MaxText adds `num_kv_shared_layers` support, or (b) we add a parity check showing the loss trajectories match. Treat as "E4B-shape ceiling, not E4B-correct ceiling."

Companion folders: [`../torchax/`](../torchax/README.md) (primary — PyTorch-on-JAX, where optimizations are explored), [`../jax/`](../jax/README.md) (secondary — native-JAX port).

## Convention

Same as the [Llama 3 8B maxtext stack folder](../../llama3_8B_autoresearch_optimization/maxtext/README.md):

- **No source code in this folder** — runs upstream MaxText. The MaxText repo lives under [`raw/code/maxtext`](../../../../raw/code/maxtext); each experiment pins a specific commit/tag and a base image.
- **What lives in this folder**: `experiments/` — dated experiment pages (`<YYYY-MM-DD>-<slug>.md`).
- **Working trees outside `raw/`**: when a tag must be checked out, do it under `/mnt/disks/persist/` per SCHEMA rule 1.
- **Each experiment page** carries: the recipe path used, the exact MaxText commit, the BASE_IMAGE, the xpk workload command, the steady-state metrics table, and the profile path.

## Layout

```
maxtext/
  README.md              this file (with the approximation caveat above)
  experiments/           per-experiment pages (dated)
    2026-04-25-maxtext-gemma4-e4b-v6e8-baseline.md   ← first reference baseline
    ...
```

## Wiki-local config files

The model config lives in the maxtext-main worktree:

- `/mnt/disks/persist/maxtext-main/src/maxtext/configs/models/gemma4-e4b.yml` (added on top of MaxText `main` @ `532c8b3d8`).

If MaxText upstream eventually adds an E4B config and `num_kv_shared_layers` support, retire the local config in favor of upstream and re-run to refresh the ceiling.

## See also

- [MaxText codebase page](../../../codebases/maxtext.md)
- [llama3_8B/maxtext/](../../llama3_8B_autoresearch_optimization/maxtext/README.md) — the stack pattern this folder mirrors.
- [gemma4_autoresearch_optimization program](../README.md)
