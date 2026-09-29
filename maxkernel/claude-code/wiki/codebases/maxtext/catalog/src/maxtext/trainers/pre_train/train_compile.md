---
title: 'Module: src/maxtext/trainers/pre_train/train_compile.py'
type: catalog
provenance: extracted
module: src/maxtext/trainers/pre_train/train_compile.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.trainers.pre_train.train_compile`/
symbols:
  main: main().
  is_oom: is_oom().
  get_shaped_inputs: get_shaped_inputs().
  get_topology_mesh: get_topology_mesh().
  _collect_nnx_activation_shardings: _collect_nnx_activation_shardings().
  validate_config: validate_config().
  jit_and_compile: jit_and_compile().
  Transformer: Transformer.
  _collect_nnx_activation_shardings._nnx_forward: _collect_nnx_activation_shardings()._nnx_forward().
  get_shaped_inputs.create_train_state_fn: get_shaped_inputs().create_train_state_fn().
  save_compiled: save_compiled().
---
# Module: [`src/maxtext/trainers/pre_train/train_compile.py`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py)

## Functions
- `_collect_nnx_activation_shardings(create_model_fn, config, mesh)` — [`L108`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L108) — Runs an abstract NNX forward pass to populate `_ACTIVATION_SHARDINGS_DUMP`.
- `_nnx_forward(decoder_input_tokens, decoder_positions, decoder_segment_ids)` — [`L117`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L117)
- `create_train_state_fn()` — [`L149`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L149)
- `get_shaped_inputs(topology_mesh, config)` — [`L130`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L130) — Get shaped abstractions of inputs to train_step: state, batch and rng
- `get_topology_mesh(config)` — [`L69`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L69) — Get the target hardware devices, and create configured mesh with them
- `is_oom(argv: Sequence[str])` — [`L239`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L239) — Function returns a boolean indicating whether OOM happens
- `jit_and_compile(func, func_input_args, func_input_kwargs, mesh, in_shardings, out_shardings, static_argnums, donate_argnums, config, logical_axis_rules)` — [`L201`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L201) — Jit, lower, and compile func.
- `main(argv: Sequence[str])` — [`L313`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L313)
- `save_compiled(compiled, save_name)` — [`L232`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L232) — Serialize and save the compiled function.
- `validate_config(config)` — [`L61`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L61) — Validates the config is is setup correctly to compile, returning a useful error message if not.

## Module values
- `Transformer` — [`L58`](../../../../../../../../raw/code/maxtext/src/maxtext/trainers/pre_train/train_compile.py#L58)

