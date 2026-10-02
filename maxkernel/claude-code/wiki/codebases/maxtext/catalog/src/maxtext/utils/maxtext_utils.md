---
title: 'Module: src/maxtext/utils/maxtext_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/maxtext_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.maxtext_utils`/
symbols:
  calculate_tflops_training_per_device: calculate_tflops_training_per_device().
  calculate_ffn_mamtul_tflops_per_device: calculate_ffn_mamtul_tflops_per_device().
  calculate_gemma4_small_tflops_training_per_device: calculate_gemma4_small_tflops_training_per_device().
  calculate_routed_and_shared_ffn_tflops_per_device: calculate_routed_and_shared_ffn_tflops_per_device().
  load_compiled: load_compiled().
  setup_decode_state: setup_decode_state().
  setup_initial_state: setup_initial_state().
  get_abstract_state: get_abstract_state().
  calculate_mla_tflops_per_device: calculate_mla_tflops_per_device().
  calculate_vision_encoder_tflops: calculate_vision_encoder_tflops().
  update_state_param: update_state_param().
  init_initial_state: init_initial_state().
  create_learning_rate_schedule.make_cos_schedule: create_learning_rate_schedule().make_cos_schedule().
  OVERWRITE_WITH_GRADIENT: OVERWRITE_WITH_GRADIENT.
  calculate_llama4_attention_tflops: calculate_llama4_attention_tflops().
  get_reorder_callable: get_reorder_callable().
  apply_gradient_clipping: apply_gradient_clipping().
  setup_training_state: setup_training_state().
  get_prefill_kv_cache_annotations: get_prefill_kv_cache_annotations().
  get_kv_cache_annotations: get_kv_cache_annotations().
  get_prefill_kv_cache_annotations_nnx: get_prefill_kv_cache_annotations_nnx().
  get_kv_cache_annotations_nnx: get_kv_cache_annotations_nnx().
  create_learning_rate_schedule: create_learning_rate_schedule().
  maybe_dump_jaxpr: maybe_dump_jaxpr().
  get_mesh_from_config: get_mesh_from_config().
  calculate_mixed_attention_model_tflops_training_per_device: calculate_mixed_attention_model_tflops_training_per_device().
  get_shared_expert_mlp_dim: get_shared_expert_mlp_dim().
  init_decode_state: init_decode_state().
  _nnx_cache_partition_specs: _nnx_cache_partition_specs().
  shard_reorder_causal_load_balanced: shard_reorder_causal_load_balanced().
  get_shaped_batch: get_shaped_batch().
  load_compiled.load_serialized_compiled: load_compiled().load_serialized_compiled().
  load_compiled.get_train_input_output_trees: load_compiled().get_train_input_output_trees().
  calculate_gemma2_tflops_training_per_device: calculate_gemma2_tflops_training_per_device().
  calculate_gemma4_tflops_training_per_device: calculate_gemma4_tflops_training_per_device().
  calculate_gemma4_small_tflops_training_per_device.attention_flops_for: calculate_gemma4_small_tflops_training_per_device().attention_flops_for().
  calculate_gemma4_small_tflops_training_per_device.qo_flops_for: calculate_gemma4_small_tflops_training_per_device().qo_flops_for().
  calculate_gemma4_small_tflops_training_per_device.kv_flops_for: calculate_gemma4_small_tflops_training_per_device().kv_flops_for().
  _calculate_chunked_attention_flops_per_layer: _calculate_chunked_attention_flops_per_layer().
  calculate_indexer_mask_ratio: calculate_indexer_mask_ratio().
  calculate_indexer_tflops_per_device: calculate_indexer_tflops_per_device().
  get_dense_moe_layers: get_dense_moe_layers().
  calculate_gated_delta_net_flops_per_device: calculate_gated_delta_net_flops_per_device().
  calculate_gemma3_vision_layers_tflops_per_device: calculate_gemma3_vision_layers_tflops_per_device().
  calculate_llama4_vision_layers_tflops_per_device: calculate_llama4_vision_layers_tflops_per_device().
  calculate_engram_tflops: calculate_engram_tflops().
  update_state_param.create_jax_path: update_state_param().create_jax_path().
  update_state_param._apply_update: update_state_param()._apply_update().
  init_training_state: init_training_state().
  setup_initial_state._merge_params: setup_initial_state()._merge_params().
  get_abstract_state.move: get_abstract_state().move().
  get_abstract_state_nnx: get_abstract_state_nnx().
  get_prefill_kv_cache_annotations.init_kv_cache: get_prefill_kv_cache_annotations().init_kv_cache().
  get_kv_cache_annotations.init_kv_cache: get_kv_cache_annotations().init_kv_cache().
  create_device_mesh: create_device_mesh().
  create_learning_rate_schedule.make_cos_schedule.schedule: create_learning_rate_schedule().make_cos_schedule().schedule().
  maybe_dump_jaxpr.to_abstract: maybe_dump_jaxpr().to_abstract().
  get_input_data_sharding: get_input_data_sharding().
  assert_params_sufficiently_sharded: assert_params_sufficiently_sharded().
  add_data_to_sharding: add_data_to_sharding().
  maybe_update_params_sharding_with_opt: maybe_update_params_sharding_with_opt().
  all_gather_over_fsdp: all_gather_over_fsdp().
  get_functional_train_with_signature: get_functional_train_with_signature().
  get_functional_eval_with_signature: get_functional_eval_with_signature().
  should_prevent_cse_in_remat: should_prevent_cse_in_remat().
  calculate_tokens_training_per_device: calculate_tokens_training_per_device().
  calculate_prefill_tflops_per_device: calculate_prefill_tflops_per_device().
  get_nested_value: get_nested_value().
  collect_intermediates_by_suffix: collect_intermediates_by_suffix().
  get_intermediate_value: get_intermediate_value().
  get_abstract_param: get_abstract_param().
  get_logical_annotations: get_logical_annotations().
  save_quantized_checkpoint_if_configured: save_quantized_checkpoint_if_configured().
  add_config_to_summary_writer: add_config_to_summary_writer().
  print_shardings_params: print_shardings_params().
  prepare_kv_caches_for_scan: prepare_kv_caches_for_scan().
  update_kv_caches_after_scan: update_kv_caches_after_scan().
---
# Module: [`src/maxtext/utils/maxtext_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py)

## Functions
- `_apply_update(path, param)` — [`L1323`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1323)
- `_calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size)` — [`L478`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L478) — Calculates the non-causal FLOPs for a single layer of chunked attention.
- `_merge_params(p_raw, p_init)` — [`L1540`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1540)
- `_nnx_cache_partition_specs(abstract_model, config, mesh)` — [`L1758`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1758) — Per-leaf PartitionSpec tree for the abstract model's nnx.Cache vars.
- `add_config_to_summary_writer(config, summary_writer)` — [`L1796`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1796) — Writes config params to tensorboard
- `add_data_to_sharding(mesh, path, aval, shardings)` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L71)
- `all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode)` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L86)
- `apply_gradient_clipping(raw_grads, state, clipping_threshold)` — [`L1210`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1210) — Applies gradient clipping to raw gradients, with special handing for FLAX fp8 stats.
- `assert_params_sufficiently_sharded(params, mesh, tolerance)` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L63)
- `attention_flops_for(attn_type)` — [`L424`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L424)
- `calculate_engram_tflops(config)` — [`L890`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L890) — Calculate engram TFLOPs per device.
- `calculate_ffn_mamtul_tflops_per_device(config, mlp_dim, in_dim=None)` — [`L653`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L653) — Helper function to calculate matmul TFLOP in ffn based on MLP dimension.
- `calculate_gated_delta_net_flops_per_device(config)` — [`L721`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L721) — - Calculates the FLOPs for a single Gated Delta Net (Linear Attention) layer.
- `calculate_gemma2_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops)` — [`L232`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L232) — Calculate training TFLOP for Gemma2 as in Gemma2 we combine [local_attention, global_attention] into one decoder
- `calculate_gemma3_vision_layers_tflops_per_device(config)` — [`L772`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L772) — Estimate TFLOPs for Gemma3 vision encoder (ViT-style).
- `calculate_gemma4_small_tflops_training_per_device(config, embedding_flops)` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L394) — Training TFLOPs for Gemma 4 small (E2B / E4B).
- `calculate_gemma4_tflops_training_per_device(config, total_ffn_flops_all_layers, embedding_flops, attention_pattern_length)` — [`L304`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L304) — Calculate training TFLOPs for Gemma 4.
- `calculate_indexer_mask_ratio(indexer_topk, max_target_length)` — [`L523`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L523) — Calculates the sparse-to-dense ratio for Indexer TFLOPs.
- `calculate_indexer_tflops_per_device(config)` — [`L571`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L571) — Calculates TFLOPs for the DeepSeek Lightning Indexer (handles causal reduction).
- `calculate_llama4_attention_tflops(config)` — [`L489`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L489) — Calculates attention-only training TFLOPs for Llama4's specific architecture,
- `calculate_llama4_vision_layers_tflops_per_device(config)` — [`L827`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L827) — Estimate TFLOPs for Llama4 vision encoder (ViT-style).
- `calculate_mixed_attention_model_tflops_training_per_device(config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length)` — [`L259`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L259) — Calculate training TFLOPs for models with a mixed attention pattern of local
- `calculate_mla_tflops_per_device(config)` — [`L597`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L597) — Calculate Multi-Head Latent Attention TFLOP (handles causal reduction)
- `calculate_prefill_tflops_per_device(num_model_parameters, prefill_length, config, log=True)` — [`L1183`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1183) — Calculate training TFLOP
- `calculate_routed_and_shared_ffn_tflops_per_device(config)` — [`L680`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L680) — Helper function to calculate DeepSeek-style ffn TFLOP
- `calculate_tflops_training_per_device(config, log=True)` — [`L941`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L941) — Calculate training TFLOP
- `calculate_tokens_training_per_device(config)` — [`L227`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L227) — Calculate training Tokens per device
- `calculate_vision_encoder_tflops(config)` — [`L922`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L922) — Calculate vision encoder TFLOPs per prefill step per device.
- `collect_intermediates_by_suffix(intermediate_outputs, *suffix_keys: str)` — [`L1255`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1255) — Collects intermediate leaf values whose dict-key path ends with suffix_keys.
- `create_device_mesh(config, devices=None)` — [`L1809`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1809) — Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas
- `create_jax_path(target_path)` — [`L1317`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1317)
- `create_learning_rate_schedule(config)` — [`L1893`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1893) — Creates a learning rate schedule with warmup and decay.
- `get_abstract_param(model, config)` — [`L1375`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1375) — Get abstract model structure (name, shape) without materializing the weights to save memory
- `get_abstract_state(config, mesh, init_state_fn, is_training=True)` — [`L1563`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1563) — Get a shaped abstraction of the state (including optimizer)
- `get_abstract_state_nnx(config, mesh, nnx_init_trainstate_fn, is_training=True)` — [`L1611`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1611) — Calculates the abstract sharded state and memory placement for an NNX TrainState.
- `get_dense_moe_layers(config)` — [`L695`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L695) — Helper function to calculate number of dense and moe layers
- `get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config)` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L109) — Get the shardings (both state and data) for `eval_step`.
- `get_functional_train_with_signature(train_step, data_sharding, state_mesh_shardings, model, config, params_shardings=None)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L93) — Get the shardings (both state and data) for `train_step`.
- `get_input_data_sharding(config, mesh)` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L56)
- `get_intermediate_value(model, nested_key, default=None, clear=False)` — [`L1279`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1279) — Retrieves an intermediate value from an NNX model. This functions has context about
- `get_kv_cache_annotations(model, config, rng, mesh)` — [`L1728`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1728) — Get a shaped abstraction of the state (including optimizer)
- `get_kv_cache_annotations_nnx(abstract_model, config, mesh)` — [`L1777`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1777) — NNX equivalent of get_kv_cache_annotations.
- `get_logical_annotations(config, mesh, init_state_fn)` — [`L1554`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1554)
- `get_mesh_from_config(config: pyconfig.HyperParameters, devices: Sequence[jax.Device] | None = None)` — [`L2048`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L2048) — Geh mesh from the configuration.
- `get_nested_value(dictionary, nested_key, default=None)` — [`L1234`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1234) — Retrieves a value from a nested key in a dictionary.
- `get_prefill_kv_cache_annotations(model, config, rng, mesh)` — [`L1695`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1695) — Get a shaped abstraction of the state (including optimizer)
- `get_prefill_kv_cache_annotations_nnx(abstract_model, config, mesh)` — [`L1772`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1772) — NNX equivalent of get_prefill_kv_cache_annotations.
- `get_reorder_callable(cp_size, shard_mode, reorder_strategy=ReorderStrategy.DUAL_CHUNK_SWAP, hardware="tpu")` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L135) — Creates a callable that can be used with map() to reorder batches.
- `get_shaped_batch(config, batch_sharding=None)` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L146) — Return the shape of the batch - this is what eval_shape would return for the
- `get_shared_expert_mlp_dim(config)` — [`L672`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L672) — Returns the MLP dimension used by the shared expert.
- `get_train_input_output_trees(func, input_args, input_kwargs)` — [`L208`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L208)
- `init_decode_state(apply_fn, params)` — [`L1333`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1333) — Init train state with null opt state for decode.
- `init_initial_state(model, tx, config, is_training, key)` — [`L1345`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1345) — We pass in "static" objects like model, tx, config as JAX compares them by
- `init_kv_cache(model, config)` — [`L1698`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1698)
- `init_kv_cache(model, config)` — [`L1731`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1731)
- `init_training_state(apply_fn, params, tx)` — [`L1339`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1339) — Init train state with null opt state for decode.
- `kv_flops_for(attn_type)` — [`L437`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L437)
- `load_compiled(config, partial_train, state, execution_devices)` — [`L198`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L198) — # Loading a serialized compiled train step function.
- `load_serialized_compiled(save_name)` — [`L204`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L204)
- `make_cos_schedule(init_lr, final_lr, len_steps)` — [`L1910`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1910)
- `maybe_dump_jaxpr(config, p_train_step, train_step_inputs)` — [`L2007`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L2007) — Dump jaxpr to local then upload to GCS.
- `maybe_update_params_sharding_with_opt(config, state_mesh_shardings)` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L78)
- `move(path, x)` — [`L1591`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1591)
- `prepare_kv_caches_for_scan(kv_caches, scan_length, block_len, stack=False)` — [`L2072`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L2072) — Groups the flat list of KV caches into block-sized tuples and optionally stacks them.
- `print_shardings_params(params, params_sharding, mesh, logical_annotations=None)` — [`L1964`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1964) — Print state shardings comparing Logical Definition vs Physical Result.
- `qo_flops_for(attn_type)` — [`L431`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L431)
- `save_quantized_checkpoint_if_configured(config, params)` — [`L1782`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1782) — Save quantized checkpoint if configured
- `schedule(step)` — [`L1911`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1911)
- `setup_decode_state(config, mesh, checkpoint_manager, init_state_fn)` — [`L1401`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1401) — Setup decode state by loading params from a checkpoint.
- `setup_initial_state(data_iterator, config, mesh, checkpoint_manager, init_state_fn, is_training=True)` — [`L1449`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1449) — We initialize the model and optimizer state, and optionally load from a
- `setup_training_state(data_iterator, config, mesh, checkpoint_manager, init_state_fn)` — [`L1437`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1437)
- `shard_reorder_causal_load_balanced(batch, cp_size, shard_mode, reorder_strategy=ReorderStrategy.DUAL_CHUNK_SWAP, hardware="tpu")` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L123) — Shard the output of the reordered sequence.
- `should_prevent_cse_in_remat(config)` — [`L176`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L176) — Determines whether to prevent common subexpression elimination (CSE) in remat.
- `to_abstract(x)` — [`L2017`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L2017)
- `update_kv_caches_after_scan(kv_caches, returned_kv_cache, scan_length, block_len, stacked=False)` — [`L2089`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L2089) — Updates the original flat list of KV caches from the scanned outputs.
- `update_state_param(state, target_path, value)` — [`L1307`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L1307) — Updates a specific parameter in state.params at the given path.

## Module values
- `OVERWRITE_WITH_GRADIENT` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/utils/maxtext_utils.py#L53)

