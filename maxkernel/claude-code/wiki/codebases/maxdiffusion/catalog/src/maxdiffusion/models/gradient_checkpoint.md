---
title: 'Module: src/maxdiffusion/models/gradient_checkpoint.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/gradient_checkpoint.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.gradient_checkpoint`/
symbols:
  GradientCheckpointType.to_jax_policy: GradientCheckpointType#to_jax_policy().
  GradientCheckpointType: GradientCheckpointType#
  GradientCheckpointType.apply: GradientCheckpointType#apply().
  GradientCheckpointType.from_str: GradientCheckpointType#from_str().
  SKIP_GRADIENT_CHECKPOINT_KEY: SKIP_GRADIENT_CHECKPOINT_KEY.
  GradientCheckpointType.NONE: GradientCheckpointType#NONE.
  GradientCheckpointType.FULL: GradientCheckpointType#FULL.
  GradientCheckpointType.MATMUL_WITHOUT_BATCH: GradientCheckpointType#MATMUL_WITHOUT_BATCH.
  GradientCheckpointType.OFFLOAD_MATMUL_WITHOUT_BATCH: GradientCheckpointType#OFFLOAD_MATMUL_WITHOUT_BATCH.
  GradientCheckpointType.CUSTOM: GradientCheckpointType#CUSTOM.
  GradientCheckpointType.HIDDEN_STATE_WITH_OFFLOAD: GradientCheckpointType#HIDDEN_STATE_WITH_OFFLOAD.
---
# Module: [`src/maxdiffusion/models/gradient_checkpoint.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py)

## Classes
### `GradientCheckpointType`  ·  implements/extends Enum
- def: [`src/maxdiffusion/models/gradient_checkpoint.py:28`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L28) — documented in [maxdiffusion-models-wan-transformers-transformer_wan_animate](../../../../concepts/maxdiffusion-models-wan-transformers-transformer_wan_animate.md)
- doc: Defines the type of the gradient checkpoint we will have
- signature: `class GradientCheckpointType(Enum):`
- members:
  - `apply(self, module: nnx.Module, names_which_can_be_saved: list = [], names_which_can_be_offloaded: list = [], static_argnums=(), prevent_cse: bool = False)` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L90) — Applies a gradient checkpoint policy to a module
  - `from_str(cls, s: Optional[str] = None)` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L47) — Constructs the gradient checkpoint type from a string
  - `to_jax_policy(self, names_which_can_be_saved: list = [], names_which_can_be_offloaded: list = [])` — [`L61`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L61) — Converts the gradient checkpoint type to a jax policy — documented in [maxdiffusion-models-wan-transformers-transformer_wan_animate](../../../../concepts/maxdiffusion-models-wan-transformers-transformer_wan_animate.md)
  - `CUSTOM` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L43)
  - `FULL` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L40)
  - `HIDDEN_STATE_WITH_OFFLOAD` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L44)
  - `MATMUL_WITHOUT_BATCH` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L41)
  - `NONE` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L39)
  - `OFFLOAD_MATMUL_WITHOUT_BATCH` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L42)
- uses (calls/refs, reference-scoped): [`SKIP_GRADIENT_CHECKPOINT_KEY`](gradient_checkpoint.md#SKIP_GRADIENT_CHECKPOINT_KEY)
- used by: [`__call__`](wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel.__call__), [`__call__`](wan/transformers/transformer_wan_vace.md#WanVACEModel.__call__), [`_run_all_blocks`](wan/transformers/transformer_wan.md#WanModel._run_all_blocks), [`gradient_checkpoint`](wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel.gradient_checkpoint), [`gradient_checkpoint`](wan/transformers/transformer_wan_vace.md#WanVACEModel.gradient_checkpoint), [`gradient_checkpoint`](wan/transformers/transformer_wan.md#WanModel.gradient_checkpoint)

## Module values
- `SKIP_GRADIENT_CHECKPOINT_KEY` — [`L24`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py#L24)

