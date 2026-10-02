---
title: 'Module: src/maxdiffusion/models/ltx_video/gradient_checkpoint.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/gradient_checkpoint.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.gradient_checkpoint`/
symbols:
  GradientCheckpointType.to_jax_policy: GradientCheckpointType#to_jax_policy().
  GradientCheckpointType: GradientCheckpointType#
  GradientCheckpointType.apply: GradientCheckpointType#apply().
  GradientCheckpointType.from_str: GradientCheckpointType#from_str().
  SKIP_GRADIENT_CHECKPOINT_KEY: SKIP_GRADIENT_CHECKPOINT_KEY.
  GradientCheckpointType.NONE: GradientCheckpointType#NONE.
  GradientCheckpointType.FULL: GradientCheckpointType#FULL.
  GradientCheckpointType.MATMUL_WITHOUT_BATCH: GradientCheckpointType#MATMUL_WITHOUT_BATCH.
---
# Module: [`src/maxdiffusion/models/ltx_video/gradient_checkpoint.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py)

## Classes
### `GradientCheckpointType`  ·  implements/extends Enum
- def: [`src/maxdiffusion/models/ltx_video/gradient_checkpoint.py:26`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L26)
- doc: Defines the type of the gradient checkpoint we will have
- signature: `class GradientCheckpointType(Enum):`
- members:
  - `apply(self, module: nn.Module)` — [`L68`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L68) — Applies a gradient checkpoint policy to a module
  - `from_str(cls, s: Optional[str] = None)` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L42) — Constructs the gradient checkpoint type from a string
  - `to_jax_policy(self)` — [`L56`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L56) — Converts the gradient checkpoint type to a jax policy
  - `FULL` — [`L38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L38)
  - `MATMUL_WITHOUT_BATCH` — [`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L39)
  - `NONE` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L37)
- uses (calls/refs, reference-scoped): [`SKIP_GRADIENT_CHECKPOINT_KEY`](gradient_checkpoint.md#SKIP_GRADIENT_CHECKPOINT_KEY)

## Module values
- `SKIP_GRADIENT_CHECKPOINT_KEY` — [`L23`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/gradient_checkpoint.py#L23)

