---
title: 'Module: src/maxtext/utils/train_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/train_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.train_utils`/
symbols:
  setup_train_loop: setup_train_loop().
  jit_train_and_eval_step: jit_train_and_eval_step().
  create_training_optimizer: create_training_optimizer().
  create_checkpoint_manager: create_checkpoint_manager().
  jit_train_step: jit_train_step().
  jit_eval_step: jit_eval_step().
  setup_train_loop.create_train_state_fn: setup_train_loop().create_train_state_fn().
  validate_completed_steps: validate_completed_steps().
  validate_train_config: validate_train_config().
  maybe_apply_dcn_throttling: maybe_apply_dcn_throttling().
  maybe_cleanup_dcn_throttling: maybe_cleanup_dcn_throttling().
---
# Module: [`src/maxtext/utils/train_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py)

## Functions
- `create_checkpoint_manager(config, mesh, init_state_fn)` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L50) — Creates the init_rng, optimizer, learning rate schedule, and checkpoint manager.
- `create_train_state_fn()` — [`L242`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L242)
- `create_training_optimizer(config, model)` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L42) — Creates the optimizer and learning rate schedule.
- `jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step)` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L156) — Returns a JIT-compiled eval step function.
- `jit_train_and_eval_step(config, model, mesh, state, state_mesh_shardings, train_step, eval_step=None, eval_data_iterator=None, params_shardings=None)` — [`L179`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L179) — Returns a JIT-compiled train and eval step function.
- `jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings, mesh=None)` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L116) — Returns a JIT-compiled train step function, which is loaded from a file if specified in the config.
- `maybe_apply_dcn_throttling(config)` — [`L426`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L426) — Applies programmatic traffic control (tc) bandwidth limit if configured.
- `maybe_cleanup_dcn_throttling(config)` — [`L457`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L457) — Cleans up traffic control (tc) rules.
- `setup_train_loop(config, recorder, devices=None)` — [`L205`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L205) — Set up prerequisites for the training loop -
- `validate_completed_steps(completed_steps: int, config_steps: int)` — [`L415`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L415) — Raises RuntimeError if training has already completed up to config_steps.
- `validate_train_config(config)` — [`L388`](../../../../../../../raw/code/maxtext/src/maxtext/utils/train_utils.py#L388) — Validates the configuration is set correctly for 'train.py'.

