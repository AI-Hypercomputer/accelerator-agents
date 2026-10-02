---
title: 'Module: src/maxtext/utils/model_creation_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/model_creation_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.model_creation_utils`/
symbols:
  from_pretrained: from_pretrained().
  _align_checkpoint_to_model_shapes: _align_checkpoint_to_model_shapes().
  _fuse_moe_weights._maybe_fuse: _fuse_moe_weights()._maybe_fuse().
  _fix_restore_args_for_shape_mismatch._fix_one: _fix_restore_args_for_shape_mismatch()._fix_one().
  from_pretrained._walk_align: from_pretrained()._walk_align().
  from_pretrained.to_dict: from_pretrained().to_dict().
  _zero_pad_axis: _zero_pad_axis().
  _fuse_moe_weights: _fuse_moe_weights().
  create_nnx_sharded_model_hybrid: create_nnx_sharded_model_hybrid().
  get_nnx_create_model_fn: get_nnx_create_model_fn().
  from_pretrained._adjust_target_for_moe_fusion: from_pretrained()._adjust_target_for_moe_fusion().
  setup_decode_state_from_nnx: setup_decode_state_from_nnx().
  _fix_restore_args_for_shape_mismatch: _fix_restore_args_for_shape_mismatch().
  from_pretrained._filter_to_model_keys: from_pretrained()._filter_to_model_keys().
  setup_decode_state_from_nnx._state_to_dict: setup_decode_state_from_nnx()._state_to_dict().
  _fix_restore_args_for_shape_mismatch._key_str: _fix_restore_args_for_shape_mismatch()._key_str().
  _fuse_moe_weights._lookup_model: _fuse_moe_weights()._lookup_model().
  _stored_shape_evenly_shardable: _stored_shape_evenly_shardable().
  _fix_restore_args_for_shape_mismatch._lookup_stored_meta: _fix_restore_args_for_shape_mismatch()._lookup_stored_meta().
  create_model: create_model().
  get_nnx_create_model_fn._create_model: get_nnx_create_model_fn()._create_model().
  create_nnx_abstract_model: create_nnx_abstract_model().
  _partition_size: _partition_size().
  from_config: from_config().
  create_models_and_meshes: create_models_and_meshes().
  _is_orbax_array_metadata: _is_orbax_array_metadata().
  _VLLM_REPEAT_AXES: _VLLM_REPEAT_AXES.
  _VLLM_ZERO_PAD_AXES: _VLLM_ZERO_PAD_AXES.
  _fuse_moe_weights._is_fusion_site: _fuse_moe_weights()._is_fusion_site().
  _fuse_moe_weights._maybe_fuse._pad_and_chunk: _fuse_moe_weights()._maybe_fuse()._pad_and_chunk().
  _normalize_logical_axes: _normalize_logical_axes().
  _zero_pad_axis._pad_local: _zero_pad_axis()._pad_local().
  _fuse_moe_weights._key_str: _fuse_moe_weights()._key_str().
  get_transformer_model: get_transformer_model().
  create_nnx_sharded_model_hybrid.create_sharded_state: create_nnx_sharded_model_hybrid().create_sharded_state().
  from_pretrained._build_value_target: from_pretrained()._build_value_target().
  from_pretrained._free_device_memory: from_pretrained()._free_device_memory().
  from_pretrained._unwrap_for_align: from_pretrained()._unwrap_for_align().
  setup_configs_and_devices: setup_configs_and_devices().
---
# Module: [`src/maxtext/utils/model_creation_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py)

## Functions
- `_adjust_target_for_moe_fusion(target, meta_tree, is_nnx)` — [`L935`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L935)
- `_align_checkpoint_to_model_shapes(ckpt_arr, model_arr, logical_axes=None)` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L147) — Align ckpt_arr to model_arr's shape and re-shard to model_arr's sharding.
- `_build_value_target(v)` — [`L1002`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1002)
- `_create_model()` — [`L555`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L555)
- `_filter_to_model_keys(ckpt, model)` — [`L1126`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1126) — Recursively keep only keys present in model, dropping checkpoint-only fields (e.g. to_nnx__rngs).
- `_fix_one(path, restore_arg)` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L414)
- `_fix_restore_args_for_shape_mismatch(restore_args, stored_metadata_tree, mesh)` — [`L346`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L346) — Adjust restore_args for arrays whose checkpoint shape differs from the model shape.
- `_free_device_memory(node)` — [`L1054`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1054)
- `_fuse_moe_weights(ckpt_tree, model_arrays_tree)` — [`L217`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L217) — Fuse separate wi_0/wi_1 checkpoint entries into a single wi when model uses fused layout.
- `_is_fusion_site(node)` — [`L226`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L226) — A ckpt-side dict that holds wi_0/wi_1 leaf siblings — the parent of a fusion.
- `_is_orbax_array_metadata(x)` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L62)
- `_key_str(key)` — [`L236`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L236)
- `_key_str(key)` — [`L373`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L373) — Extract string name from a JAX path key (DictKey, GetAttrKey, etc.).
- `_lookup_model(path)` — [`L245`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L245)
- `_lookup_stored_meta(path)` — [`L383`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L383) — Navigate stored_metadata_tree using path keys from the restore_args tree.
- `_maybe_fuse(path, ckpt_node)` — [`L255`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L255)
- `_normalize_logical_axes(axes)` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L81) — Coerce an axes value (PartitionSpec / tuple / None / leaf marker) to a plain tuple or None.
- `_pad_and_chunk(arr, target_total_size)` — [`L277`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L277)
- `_pad_local(x)` — [`L132`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L132)
- `_partition_size(partition, mesh)` — [`L313`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L313) — Total mesh-axis size used to shard a single tensor axis.
- `_state_to_dict(tree)` — [`L1202`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1202)
- `_stored_shape_evenly_shardable(restore_arg, stored_shape)` — [`L328`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L328) — Whether the restore_arg's NamedSharding evenly partitions the stored shape.
- `_unwrap_for_align(v)` — [`L1091`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1091)
- `_walk_align(ckpt, model_arr, axes)` — [`L1134`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1134)
- `_zero_pad_axis(arr, axis, extra)` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L98) — Append ``extra`` zeros at the end of ``axis``.
- `create_model(config, mesh, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None, *, quant_mode_str: str = "train")` — [`L540`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L540) — Instantiates and returns the model object, sharded across the mesh.
- `create_models_and_meshes(trainer_config, sampler_config, trainer_devices, sampler_devices, tokenizer_pad_id=None)` — [`L747`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L747) — Create reference and actor models and their respective meshes.
- `create_nnx_abstract_model(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, *, quant_mode_str: str = "train")` — [`L562`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L562) — Creates an abstract NNX model.
- `create_nnx_sharded_model_hybrid(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None)` — [`L599`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L599) — Creates a sharded model for hybrid NNX modules containing Linen sub-modules.
- `create_sharded_state()` — [`L629`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L629)
- `from_config(config: pyconfig.HyperParameters, devices: Sequence[jax.Device] | None = None, mesh: Mesh | None = None, *, model_mode: str = MODEL_MODE_TRAIN, rngs: None = None, quant_mode_str: str = "train")` — [`L483`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L483)
- `from_pretrained(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, wrap_with_tunix_adapter=False, tokenizer_pad_id=None, *, quant_mode_str: str = "train")` — [`L794`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L794) — Creates a NNX model with sharded parameters, possibly loading from a checkpoint.
- `get_nnx_create_model_fn(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, *, quant_mode_str: str = "train")` — [`L551`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L551)
- `get_transformer_model(config, mesh, quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None)` — [`L532`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L532) — Returns the transformer model based on the configuration.
- `setup_configs_and_devices(argv: list[str] | None = None, kwargs: dict | None = None, **extra_kwargs)` — [`L653`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L653) — Setup device allocation and configs for training and inference.
- `setup_decode_state_from_nnx(model, config, rng, mesh)` — [`L1178`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1178) — Setup decode state by loading an NNX or NNX-RL checkpoint into a linen TrainState.
- `to_dict(tree)` — [`L1112`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L1112)

## Module values
- `_VLLM_REPEAT_AXES` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L73)
- `_VLLM_ZERO_PAD_AXES` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/utils/model_creation_utils.py#L78)

