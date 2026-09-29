---
title: 'Module: src/maxtext/trainers/pre_train/train.py'
type: catalog
provenance: extracted
module: src/maxtext/trainers/pre_train/train.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.trainers.pre_train.train`/
symbols:
  train_loop: train_loop().
  get_train_func: get_train_func().
  train_step: train_step().
  get_train_func.elastic_train_wrapper: get_train_func().elastic_train_wrapper().
  main: main().
  loss_fn: loss_fn().
  initialize: initialize().
  run: run().
  train_step.diff_wrapper: train_step().diff_wrapper().
  eval_step: eval_step().
  get_first_step: get_first_step().
  train_step.move: train_step().move().
  training_loop_iteration: training_loop_iteration().
  get_train_func.on_elastic_event: get_train_func().on_elastic_event().
  get_train_func.on_slices_ready: get_train_func().on_slices_ready().
  VertexTensorboardManager: VertexTensorboardManager.
  _vertex_tb_is_stub: _vertex_tb_is_stub.
---
# Module: [`src/maxtext/trainers/pre_train/train.py`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py)

## Functions
- `diff_wrapper(curr_params, custom_params, rest, config, data)` — [`L420`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L420)
- `elastic_train_wrapper(argv: Sequence[str])` — [`L961`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L961) — Wrapper for elastic training initializes variables and runs the train loop.
- `eval_step(model, config, state, data, dropout_rng=None)` — [`L599`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L599) — eval_step no backprop and new state compared with train_step.
- `get_first_step(model, state)` — [`L78`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L78)
- `get_train_func(config, recorder, argv)` — [`L950`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L950) — Returns the train function, wrapping in elastic_retry if elastic training is enabled.
- `initialize(argv: Sequence[str])` — [`L915`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L915) — Initialization of hyperparameters and utilities
- `loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True)` — [`L93`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L93) — loss_fn for both train and eval.
- `main(argv: Sequence[str])` — [`L982`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L982)
- `move(path, value)` — [`L470`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L470)
- `on_elastic_event()` — [`L955`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L955)
- `on_slices_ready()` — [`L958`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L958)
- `run(config, recorder)` — [`L944`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L944) — Run the job given hyperparameters and utilities.
- `train_loop(config, recorder, state=None)` — [`L751`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L751) — Main Training loop.
- `train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None)` — [`L338`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L338) — Training step for both Linen and NNX models.
- `training_loop_iteration(jax_device_state: dict[str, Any], python_vars: dict[str, Any], immutable_data: dict[str, Any])` — [`L639`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L639) — Executes a single iteration of the training loop.

## Module values
- `VertexTensorboardManager` — [`L75`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L75)
- `_vertex_tb_is_stub` — [`L75`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train.py#L75)

