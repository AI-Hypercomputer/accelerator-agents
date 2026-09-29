---
title: 'Module: src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.utils.convert_torch_weights_to_jax`/
symbols:
  Checkpointer.save: Checkpointer#save().
  Checkpointer.state_dict_to_structure_dict: Checkpointer#state_dict_to_structure_dict().
  args: args.
  parser: parser.
  Checkpointer.train_state_handler: Checkpointer#train_state_handler.
  Checkpointer.save_buffer_size_bytes: Checkpointer#save_buffer_size_bytes().
  Checkpointer.wait: Checkpointer#wait().
  Checkpointer: Checkpointer#
  Checkpointer.save_buffer_size: Checkpointer#save_buffer_size.
  Checkpointer.manager: Checkpointer#manager.
  Checkpointer.STATE_DICT_SHAPE_KEY: Checkpointer#STATE_DICT_SHAPE_KEY.
  Checkpointer.STATE_DICT_DTYPE_KEY: Checkpointer#STATE_DICT_DTYPE_KEY.
  Checkpointer.TRAIN_STATE_FILE_NAME: Checkpointer#TRAIN_STATE_FILE_NAME.
  main: main().
  Checkpointer.__init__: Checkpointer#__init__().
  Checkpointer.use_zarr3: Checkpointer#use_zarr3.
  Checkpointer.restore_buffer_size: Checkpointer#restore_buffer_size.
---
# Module: [`src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py)

## Classes
### `Checkpointer`
- def: [`src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py:37`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L37)
- doc: Checkpointer - to load and store JAX checkpoints
- signature: `class Checkpointer:`
- members:
  - `__init__(self, checkpoint_dir: str, use_zarr3: bool = False, save_buffer_size: Optional[int] = None, restore_buffer_size: Optional[int] = None)` — [`L46`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L46) — Constructs the checkpointer object
  - `save(self, step: int, state: train_state.TrainState, config: Dict[str, Any])` — [`L101`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L101) — Saves the checkpoint asynchronously
  - `save_buffer_size_bytes(self)` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L81)
  - `state_dict_to_structure_dict(state_dict: Dict[str, Any])` — [`L87`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L87) — Converts a state dict to a dictionary stating the shape and dtype of the state_dict elements.
  - `wait(self)` — [`L131`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L131) — Waits for the checkpoint save operation to complete
  - `STATE_DICT_DTYPE_KEY` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L43)
  - `STATE_DICT_SHAPE_KEY` — [`L42`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L42)
  - `TRAIN_STATE_FILE_NAME` — [`L44`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L44)
  - `manager` — [`L74`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L74)
  - `restore_buffer_size` — [`L62`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L62)
  - `save_buffer_size` — [`L61`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L61)
  - `train_state_handler` — [`L64`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L64)
  - `use_zarr3` — [`L60`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L60)

## Functions
- `main(args)` — [`L145`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L145) — Convert a Torch checkpoint into JAX.

## Module values
- `args` — [`L258`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L258)
- `parser` — [`L230`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/convert_torch_weights_to_jax.py#L230)

