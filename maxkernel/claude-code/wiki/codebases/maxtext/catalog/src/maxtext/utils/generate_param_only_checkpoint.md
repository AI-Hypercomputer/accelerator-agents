---
title: 'Module: src/maxtext/utils/generate_param_only_checkpoint.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/generate_param_only_checkpoint.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.generate_param_only_checkpoint`/
symbols:
  generate_decode_checkpoint: generate_decode_checkpoint().
  _generate_lora_decode_checkpoints: _generate_lora_decode_checkpoints().
  _possibly_unroll_params.unroll_layer_group: _possibly_unroll_params().unroll_layer_group().
  _possibly_unroll_params_nnx.unroll_layer_group: _possibly_unroll_params_nnx().unroll_layer_group().
  _possibly_unroll_lora_params_nnx.unroll_layer_group: _possibly_unroll_lora_params_nnx().unroll_layer_group().
  _possibly_unroll_params: _possibly_unroll_params().
  _generate_lora_decode_checkpoints_nnx: _generate_lora_decode_checkpoints_nnx().
  _save_decode_checkpoint: _save_decode_checkpoint().
  _save_decode_checkpoint_nnx._wrap_value: _save_decode_checkpoint_nnx()._wrap_value().
  _possibly_unroll_params_nnx: _possibly_unroll_params_nnx().
  _read_train_checkpoint: _read_train_checkpoint().
  _save_decode_checkpoint_nnx: _save_decode_checkpoint_nnx().
  _possibly_unroll_lora_params_nnx: _possibly_unroll_lora_params_nnx().
  main: main().
  _read_train_checkpoint.init_state_fn: _read_train_checkpoint().init_state_fn().
  _possibly_unroll_params.unroll_layer_group.new_pspec: _possibly_unroll_params().unroll_layer_group().new_pspec().
  _possibly_unroll_params.unroll_layer_group.slice_ith: _possibly_unroll_params().unroll_layer_group().slice_ith().
  _possibly_unroll_params_nnx.unroll_layer_group.drop_scan_axis: _possibly_unroll_params_nnx().unroll_layer_group().drop_scan_axis().
  _possibly_unroll_params_nnx.unroll_layer_group.slice_ith: _possibly_unroll_params_nnx().unroll_layer_group().slice_ith().
  _possibly_unroll_lora_params_nnx.unroll_layer_group.new_pspec: _possibly_unroll_lora_params_nnx().unroll_layer_group().new_pspec().
  _possibly_unroll_lora_params_nnx.unroll_layer_group.slice_ith: _possibly_unroll_lora_params_nnx().unroll_layer_group().slice_ith().
  _save_lora_decode_checkpoint_nnx: _save_lora_decode_checkpoint_nnx().
---
# Module: [`src/maxtext/utils/generate_param_only_checkpoint.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py)

## Functions
- `_generate_lora_decode_checkpoints(config, mesh)` — [`L184`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L184) — Read lora checkpoints checkpoint at path defined by load_full_state_path.
- `_generate_lora_decode_checkpoints_nnx(config, mesh)` — [`L331`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L331) — NNX-shaped sibling of `_generate_lora_decode_checkpoints`.
- `_possibly_unroll_lora_params_nnx(config, lora_state, lora_state_annotations, mesh)` — [`L267`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L267) — Unroll scanned LoRA delta layers when force_unroll is set on the NNX path.
- `_possibly_unroll_params(config, training_state, training_state_annotations, mesh)` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L53) — Unroll scanned input layers when force_unroll is set.
- `_possibly_unroll_params_nnx(config, state, state_mesh_shardings, mesh)` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L100) — NNX equivalent of _possibly_unroll_params.
- `_read_train_checkpoint(config, checkpoint_manager, mesh)` — [`L149`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L149) — Read training checkpoint at path defined by load_full_state_path.
- `_save_decode_checkpoint(config, state, checkpoint_manager)` — [`L225`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L225) — Generate checkpoint for decode from the training_state.
- `_save_decode_checkpoint_nnx(config, state, checkpoint_manager)` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L239) — Save a bf16 NNX-format param-only decode checkpoint.
- `_save_lora_decode_checkpoint_nnx(config, lora_state, checkpoint_manager)` — [`L313`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L313) — Save a bf16 LoRA-only decode checkpoint (NNX path).
- `_wrap_value(node)` — [`L255`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L255)
- `drop_scan_axis(named_sharding)` — [`L118`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L118)
- `generate_decode_checkpoint(config)` — [`L366`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L366) — Generate an decode checkpoint from a given training checkpoint.
- `init_state_fn()` — [`L158`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L158)
- `main(argv: Sequence[str])` — [`L424`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L424)
- `new_pspec(x)` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L69)
- `new_pspec(x)` — [`L286`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L286)
- `slice_ith(input_layers)` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L77)
- `slice_ith(input_layers)` — [`L129`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L129)
- `slice_ith(input_layers)` — [`L294`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L294)
- `unroll_layer_group(num_layers, layer_name="layers")` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L61) — Helper function to unroll layers (e.g. dense or MoE) into individual layers.
- `unroll_layer_group(num_layers, layer_name="layers")` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L112)
- `unroll_layer_group(num_layers, layer_name="layers")` — [`L280`](../../../../../../../raw/code/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py#L280)

