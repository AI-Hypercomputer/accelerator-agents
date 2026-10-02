---
title: 'Module: src/maxtext/utils/max_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/max_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.max_utils`/
symbols:
  maybe_initialize_jax_distributed_system: maybe_initialize_jax_distributed_system().
  summarize_size_from_pytree: summarize_size_from_pytree().
  reshape_mesh_to_rings: reshape_mesh_to_rings().
  is_valid_custom_mesh: is_valid_custom_mesh().
  print_compiled_memory_stats.bytes_to_gb: print_compiled_memory_stats().bytes_to_gb().
  maybe_get_transformer_engine_context: maybe_get_transformer_engine_context().
  calculate_bytes_from_pytree: calculate_bytes_from_pytree().
  initialize_jax_for_cpu: initialize_jax_for_cpu().
  initialize_jax_for_tpu_with_emergency_checkpointing: initialize_jax_for_tpu_with_emergency_checkpointing().
  find_nans_and_infs: find_nans_and_infs().
  calculate_total_params_per_chip: calculate_total_params_per_chip().
  create_custom_device_mesh: create_custom_device_mesh().
  delete_pytree: delete_pytree().
  summarize_pytree_data: summarize_pytree_data().
  print_compiled_memory_stats: print_compiled_memory_stats().
  reorder_causal_load_balanced: reorder_causal_load_balanced().
  unscan_train_state_params: unscan_train_state_params().
  rescan_train_state_params: rescan_train_state_params().
  initialize_multi_tier_checkpointing: initialize_multi_tier_checkpointing.
  HYBRID_RING_64X4: HYBRID_RING_64X4.
  HYBRID_RING_32X8: HYBRID_RING_32X8.
  find_nans_and_infs.finder: find_nans_and_infs().finder().
  calculate_num_params_from_pytree: calculate_num_params_from_pytree().
  calculate_total_params_per_chip.calculate_leaf_params_per_chip: calculate_total_params_per_chip().calculate_leaf_params_per_chip().
  _bytes_of: _bytes_of().
  initialize_jax_for_gpu: initialize_jax_for_gpu().
  _retrieve_jax_init_info: _retrieve_jax_init_info().
  is_cpu_backend: is_cpu_backend().
  is_gpu_backend: is_gpu_backend().
  get_coordinator_ip_address: get_coordinator_ip_address().
  cross_entropy_with_logits: cross_entropy_with_logits().
  _cross_entropy_with_logits_fwd: _cross_entropy_with_logits_fwd().
  _cross_entropy_with_logits_bwd: _cross_entropy_with_logits_bwd().
  delete_pytree.delete_leaf: delete_pytree().delete_leaf().
  reorder_sequence: reorder_sequence().
  unscan_train_state_params.strip_scan_axis: unscan_train_state_params().strip_scan_axis().
  rescan_train_state_params.stack_layers: rescan_train_state_params().stack_layers().
  dummy_context_manager: dummy_context_manager().
  transformer_engine_context: transformer_engine_context().
  parse_libtpu_flags_to_dict: parse_libtpu_flags_to_dict().
  with_memory_kind: with_memory_kind().
  cast_dtype_from_to: cast_dtype_from_to().
  l2norm_pytree: l2norm_pytree().
  device_space: device_space().
  initialize_summary_writer: initialize_summary_writer().
  close_summary_writer: close_summary_writer().
  add_text_to_summary_writer: add_text_to_summary_writer().
  get_num_slices: get_num_slices().
  fill_unspecified_mesh_axes: fill_unspecified_mesh_axes().
  optimize_mesh_for_tpu_v6e: optimize_mesh_for_tpu_v6e().
  unbox_logicallypartioned: unbox_logicallypartioned().
  print_pytree_shape: print_pytree_shape().
  print_model_vars: print_model_vars().
  get_project: get_project().
  print_mem_stats: print_mem_stats().
  print_cpu_ram_stats: print_cpu_ram_stats().
  print_system_information: print_system_information().
  permute_to_match_maxtext_rope: permute_to_match_maxtext_rope().
  unpermute_from_match_maxtext_rope: unpermute_from_match_maxtext_rope().
  reorder_mask_load_balancing: reorder_mask_load_balancing().
  parse_custom_args: parse_custom_args().
  get_batch_seq_len_for_mode: get_batch_seq_len_for_mode().
  print_non_trivial_mesh_axis: print_non_trivial_mesh_axis().
  maybe_pad: maybe_pad().
---
# Module: [`src/maxtext/utils/max_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py)

## Functions
- `_bytes_of(x)` — [`L145`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L145) — Return the number of bytes used by a single leaf in a pytree.
- `_cross_entropy_with_logits_bwd(res: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,], g: tuple[jnp.ndarray, jnp.ndarray])` — [`L689`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L689) — Backward-mode of `cross_entropy_with_logits`.
- `_cross_entropy_with_logits_fwd(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0)` — [`L657`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L657) — Forward-mode of `cross_entropy_with_logits`.
- `_retrieve_jax_init_info(raw_keys)` — [`L389`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L389) — Retrieve JAX init info from a local file.
- `add_text_to_summary_writer(key, value, summary_writer)` — [`L228`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L228) — Writes given key-value pair to tensorboard as text/summary.
- `bytes_to_gb(num_bytes)` — [`L808`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L808)
- `calculate_bytes_from_pytree(params)` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L170) — Return the total memory footprint (in bytes) of all leaves in a pytree.
- `calculate_leaf_params_per_chip(arr)` — [`L136`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L136)
- `calculate_num_params_from_pytree(params)` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L117)
- `calculate_total_params_per_chip(params)` — [`L133`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L133) — Calculate total params per chip.
- `cast_dtype_from_to(nest, src, dst)` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L99) — All items in nest with dtype src are casted to dtype dst.
- `close_summary_writer(summary_writer)` — [`L223`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L223)
- `create_custom_device_mesh(mesh_shape: Sequence[int], dcn_mesh_shape: Sequence[int], devices: Sequence[Any], custom_strategy: str, process_is_granule: bool = False, should_sort_granules_by_key: bool = True)` — [`L524`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L524) — Custom device mesh for 64x4 ici parallelism
- `cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0)` — [`L624`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L624) — Computes cross entropy loss with stable custom gradient.
- `delete_leaf(leaf)` — [`L746`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L746)
- `delete_pytree(p)` — [`L745`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L745)
- `device_space()` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L124) — Version guard for jax.memory.Space.Device.
- `dummy_context_manager()` — [`L1200`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1200) — A context manager that does nothing.
- `fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type)` — [`L459`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L459) — Evaluates unspecified DCN/ICI parallelism values
- `find_nans_and_infs(pytree)` — [`L104`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L104)
- `finder(x)` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L105)
- `get_batch_seq_len_for_mode(config, model_mode)` — [`L1147`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1147) — Resolves the batch size and sequence length based on the model's operational mode.
- `get_coordinator_ip_address()` — [`L436`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L436) — Get coordinator IP Address with retries
- `get_num_slices(raw_keys, config=None)` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L409) — Calculate num_slices based on number of devices.
- `get_project()` — [`L729`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L729) — Get project
- `initialize_jax_for_cpu(raw_keys)` — [`L347`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L347) — Jax distributed initialize for CPUs. Includes retries until the coordinator is ready.
- `initialize_jax_for_gpu(raw_keys)` — [`L315`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L315) — Jax distributed initialize for GPUs.
- `initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys)` — [`L366`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L366) — Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
- `initialize_summary_writer(tensorboard_dir, run_name, enable_tensorboard=True)` — [`L185`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L185) — Return a tensorboardX SummaryWriter or a no-op stub.
- `is_cpu_backend(raw_keys)` — [`L426`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L426) — Determine whether Maxtext is intended to run on a CPU backend.
- `is_gpu_backend(raw_keys)` — [`L431`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L431) — Determine whether Maxtext is intended to run on a GPU backend.
- `is_valid_custom_mesh(ici_parallelism, strategy)` — [`L562`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L562) — Checks if the given strategy and ICI parallelism are valid.
- `l2norm_pytree(x)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L112) — L2 norm of a pytree of arrays.
- `maybe_get_transformer_engine_context(config)` — [`L1189`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1189) — Runs a transformer engine context engine manager for GPUs only.
- `maybe_initialize_jax_distributed_system(raw_keys)` — [`L234`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L234) — The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
- `maybe_pad(inputs, tile_size)` — [`L1226`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1226) — Pads the inputs leading dimension to be divisible by tile_size.
- `optimize_mesh_for_tpu_v6e(mesh, devices)` — [`L581`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L581) — Apply transformations to the mesh to optimize for TPU v6e
- `parse_custom_args(argv)` — [`L1047`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1047) — Load multiple YAML config files from command line arguments.
- `parse_libtpu_flags_to_dict(flags_str: str)` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L58) — Parses a string of XLA flags into a dictionary of compilation options.
- `permute_to_match_maxtext_rope(arr)` — [`L832`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L832) — Permutes the Huggingface Rope to match the MaxText logic.
- `print_compiled_memory_stats(compiled_stats)` — [`L803`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L803) — Prints a summary of the compiled memory statistics.
- `print_cpu_ram_stats(label: str)` — [`L788`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L788) — Print stats of CPU RAM usage/availability.
- `print_mem_stats(label: str)` — [`L776`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L776)
- `print_model_vars(print_str, model_vars)` — [`L723`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L723)
- `print_non_trivial_mesh_axis(mesh)` — [`L1181`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1181) — Print mesh axis if its axis size is larger than one.
- `print_pytree_shape(print_str, ptree)` — [`L717`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L717)
- `print_system_information()` — [`L824`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L824) — Print system information of the current environment.
- `reorder_causal_load_balanced(batch, cp_size, reorder_strategy, hardware="tpu")` — [`L918`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L918) — Reorders the example batch sequences using a hardware-appropriate backend.
- `reorder_mask_load_balancing(tensor, cp_size: int, seq_dim: int)` — [`L999`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L999) — Reorders a tensor for load balancing the compute of causal attention.
- `reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False)` — [`L854`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L854) — Reorders the sequence of the tensor. For example, with cp_size=2,
- `rescan_train_state_params(params, source_shardings, scan_axis, layer_groups)` — [`L1112`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1112) — Reconstruct scanned layers from per-layer entries using minimal HBM.
- `reshape_mesh_to_rings(a, strategy)` — [`L485`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L485) — Reshape device mesh to rings for 64x4 or 32x8 mesh shape
- `stack_layers(*layers)` — [`L1127`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1127)
- `strip_scan_axis(pspec: P)` — [`L1080`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1080) — Removes the element at `scan_axis` from a PartitionSpec tuple.
- `summarize_pytree_data(params, name="Params", raw=False)` — [`L754`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L754) — Generate basic metrics of a given Pytree.
- `summarize_size_from_pytree(params)` — [`L179`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L179)
- `transformer_engine_context()` — [`L1206`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1206) — If TransformerEngine is available, this context manager will provide
- `unbox_logicallypartioned(boxed_pytree)` — [`L605`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L605) — Unboxes the flax.LogicallyPartitioned pieces
- `unpermute_from_match_maxtext_rope(arr, model_size)` — [`L842`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L842) — Function to get the RoPE values in correct ordering
- `unscan_train_state_params(params, sharding, mesh, scan_axis, layer_groups)` — [`L1064`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L1064) — Unrolls scanned parameter groups into per-layer entries.
- `with_memory_kind(t, memory_kind)` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L95)

## Module values
- `HYBRID_RING_32X8` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L53)
- `HYBRID_RING_64X4` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L52)
- `initialize_multi_tier_checkpointing` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/utils/max_utils.py#L51)

