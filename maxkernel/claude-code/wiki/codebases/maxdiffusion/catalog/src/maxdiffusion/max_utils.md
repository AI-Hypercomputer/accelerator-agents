---
title: 'Module: src/maxdiffusion/max_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/max_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.max_utils`/
symbols:
  get_flash_block_sizes: get_flash_block_sizes().
  Profiler.start: Profiler#start().
  Profiler.stop: Profiler#stop().
  Profiler.config: Profiler#config.
  init_train_state: init_train_state().
  setup_initial_state: setup_initial_state().
  ensure_machinelearning_job_runs: ensure_machinelearning_job_runs().
  download_blobs: download_blobs().
  maybe_initialize_jax_distributed_system: maybe_initialize_jax_distributed_system().
  upload_blob: upload_blob().
  _ml_diagnostics_profiler_enabled: _ml_diagnostics_profiler_enabled().
  profiler_enabled: profiler_enabled().
  write_metrics_for_gcs: write_metrics_for_gcs().
  _jax_profiler_enabled: _jax_profiler_enabled().
  Profiler.mld_xprof: Profiler#mld_xprof.
  InferenceState.params: InferenceState#params.
  write_config_raw_keys_for_gcs: write_config_raw_keys_for_gcs().
  get_abstract_state: get_abstract_state().
  _ml_run: _ml_run.
  Profiler.session_name: Profiler#session_name.
  parse_gcs_bucket_and_prefix: parse_gcs_bucket_and_prefix().
  value_or_none: value_or_none().
  Profiler.__enter__: Profiler#__enter__().
  Profiler.__exit__: Profiler#__exit__().
  write_metrics_locally: write_metrics_locally().
  add_config_to_summary_writer: add_config_to_summary_writer().
  walk_and_upload_blobs: walk_and_upload_blobs().
  create_device_mesh: create_device_mesh().
  calculate_model_tflops: calculate_model_tflops().
  _prepare_metrics_for_json: _prepare_metrics_for_json().
  fill_unspecified_mesh_axes: fill_unspecified_mesh_axes().
  FrozenDict: FrozenDict.
  InferenceState: InferenceState#
  InferenceState.apply_fn: InferenceState#apply_fn.
  add_text_to_summary_writer: add_text_to_summary_writer().
  unbox_logicallypartioned_trainstate: unbox_logicallypartioned_trainstate().
  CustomFlashBlockSizes: CustomFlashBlockSizes#
  CustomFlashBlockSizes.block_q: CustomFlashBlockSizes#block_q.
  CustomFlashBlockSizes.block_kv: CustomFlashBlockSizes#block_kv.
  CustomFlashBlockSizes.block_kv_compute: CustomFlashBlockSizes#block_kv_compute.
  CustomFlashBlockSizes.block_kv_compute_in: CustomFlashBlockSizes#block_kv_compute_in.
  CustomFlashBlockSizes.heads_per_tile: CustomFlashBlockSizes#heads_per_tile.
  CustomFlashBlockSizes.vmem_limit_bytes: CustomFlashBlockSizes#vmem_limit_bytes.
  get_global_batch_size: get_global_batch_size().
  is_gpu_backend: is_gpu_backend().
  initialize_jax_for_gpu: initialize_jax_for_gpu().
  l2norm_pytree: l2norm_pytree().
  Profiler: Profiler#
  Profiler.__init__: Profiler#__init__().
  initialize_summary_writer: initialize_summary_writer().
  close_summary_writer: close_summary_writer().
  device_put_replicated: device_put_replicated().
  create_learning_rate_schedule: create_learning_rate_schedule().
  create_optimizer: create_optimizer().
  get_precision: get_precision().
  delete_pytree: delete_pytree().
  get_memory_allocations: get_memory_allocations().
  get_live_arrays: get_live_arrays().
  get_train_step_partial_with_signature: get_train_step_partial_with_signature().
  calculate_num_params_from_pytree: calculate_num_params_from_pytree().
  safe_getattr: safe_getattr().
---
# Module: [`src/maxdiffusion/max_utils.py`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py)

## Classes
### `CustomFlashBlockSizes`
- def: [`src/maxdiffusion/max_utils.py:617`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L617)
- doc: Hashable carrier for the custom splash kernel's block sizes.
- signature: `class CustomFlashBlockSizes:`
- members:
  - `block_kv` — [`L629`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L629)
  - `block_kv_compute` — [`L630`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L630)
  - `block_kv_compute_in` — [`L631`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L631)
  - `block_q` — [`L628`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L628)
  - `heads_per_tile` — [`L632`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L632)
  - `vmem_limit_bytes` — [`L633`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L633)
- used by: [`get_flash_block_sizes`](max_utils.md#get_flash_block_sizes)

### `InferenceState`  ·  implements/extends PyTreeNode
- def: [`src/maxdiffusion/max_utils.py:77`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L77)
- signature: `class InferenceState(struct.PyTreeNode):`
- members:
  - `apply_fn` — [`L79`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L79)
  - `params` — [`L80`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L80)
- uses (calls/refs, reference-scoped): [`FrozenDict`](max_utils.md#FrozenDict)
- used by: [`init_train_state`](max_utils.md#init_train_state)

### `Profiler`
- def: [`src/maxdiffusion/max_utils.py:117`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L117)
- signature: `class Profiler:`
- members:
  - `start(self)` — [`L124`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L124)
  - `stop(self)` — [`L148`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L148)
  - `config` — [`L121`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L121)
  - `mld_xprof` — [`L118`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L118)
  - `session_name` — [`L122`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L122)
- protocol/private: `__enter__`[`L175`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L175), `__exit__`[`L179`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L179), `__init__`[`L120`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L120)
- uses (calls/refs, reference-scoped): [`ensure_machinelearning_job_runs`](max_utils.md#ensure_machinelearning_job_runs), [`_ml_diagnostics_profiler_enabled`](max_utils.md#_ml_diagnostics_profiler_enabled), [`_jax_profiler_enabled`](max_utils.md#_jax_profiler_enabled), [`parse_gcs_bucket_and_prefix`](max_utils.md#parse_gcs_bucket_and_prefix)

## Functions
- `_jax_profiler_enabled(config)` — [`L83`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L83)
- `_ml_diagnostics_profiler_enabled(config)` — [`L87`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L87)
- `_prepare_metrics_for_json(metrics, step, run_name)` — [`L192`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L192) — Converts metric dictionary into json supported types (e.g. float)
- `add_config_to_summary_writer(config, summary_writer)` — [`L234`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L234) — Writes config params to tensorboard
- `add_text_to_summary_writer(key, value, summary_writer)` — [`L241`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L241) — Writes given key-value pair to tensorboard as text/summary
- `calculate_model_tflops(module: module_lib.Module, rngs: Union[PRNGKey, RNGSequences], train, **kwargs)` — [`L709`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L709) — Calculates model tflops by passing a module.
- `calculate_num_params_from_pytree(params)` — [`L767`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L767) — Calculates number of parameters from a pytree
- `close_summary_writer(summary_writer)` — [`L187`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L187)
- `create_device_mesh(config, devices=None, logging=True)` — [`L352`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L352) — Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas
- `create_learning_rate_schedule(learning_rate, learning_rate_schedule_steps, warmup_steps_fraction, max_train_steps)` — [`L557`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L557) — Creates a warmup to constant learning rate schedule:
- `create_optimizer(config, learning_rate_scheduler)` — [`L582`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L582)
- `delete_pytree(to_delete)` — [`L679`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L679)
- `device_put_replicated(x, sharding)` — [`L317`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L317) — Although the name indicates replication, this function can be used
- `download_blobs(source_gcs_folder, local_destination)` — [`L272`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L272) — Downloads a folder to a local location
- `ensure_machinelearning_job_runs(config)` — [`L95`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L95) — Ensures that a MachineLearningJobRun is active, and if not creates one.
- `fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type)` — [`L325`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L325) — Evaluates unspecified DCN/ICI parallelism values
- `get_abstract_state(model, tx, config, mesh, weights_init_fn, training=True)` — [`L454`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L454) — Get a shaped abstraction of the state (including optimizer)
- `get_flash_block_sizes(config)` — [`L636`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L636) — Create custom flash attention BlockSizes.
- `get_global_batch_size(per_device_batch_size)` — [`L774`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L774)
- `get_live_arrays()` — [`L694`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L694)
- `get_memory_allocations()` — [`L683`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L683)
- `get_precision(config)` — [`L598`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L598) — Get precision from config.
- `get_train_step_partial_with_signature(train_step: Callable, pipeline: object, params: Dict, config: object)` — [`L761`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L761)
- `init_train_state(model, tx, weights_init_fn, params=None, training=True, eval_only=False)` — [`L424`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L424) — We pass in "static" objects like model, tx, config, as JAX compares them by
- `initialize_jax_for_gpu()` — [`L783`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L783) — Jax distribute initialize for GPUs.
- `initialize_summary_writer(config)` — [`L183`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L183)
- `is_gpu_backend(raw_keys)` — [`L778`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L778) — Determine whether Maxdiffusion is intended to run on a GPU backend.
- `l2norm_pytree(x)` — [`L112`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L112) — L2 norm of a pytree of arrays.
- `maybe_initialize_jax_distributed_system(raw_keys)` — [`L796`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L796)
- `parse_gcs_bucket_and_prefix(destination_gcs_name)` — [`L265`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L265)
- `profiler_enabled(config)` — [`L91`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L91)
- `safe_getattr(obj: Any, name: str, default: Any)` — [`L808`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L808) — Safely reads attribute from an object, returning default if obj is None or attribute missing.
- `setup_initial_state(model, tx, config, mesh, weights_init_fn, model_params=None, checkpoint_manager=None, checkpoint_item=None, training=True)` — [`L479`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L479) — We initialize the model and optimizer state, and optionally load from a
- `unbox_logicallypartioned_trainstate(boxed_train_state: train_state.TrainState)` — [`L408`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L408) — Unboxes the flax.LogicallyPartitioned pieces in a train state.
- `upload_blob(destination_gcs_name, source_file_name)` — [`L290`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L290) — Uploads a file to a GCS location
- `value_or_none(flash_block_sizes, key)` — [`L609`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L609)
- `walk_and_upload_blobs(config, output_dir)` — [`L299`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L299)
- `write_config_raw_keys_for_gcs(raw_keys)` — [`L247`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L247) — Writes config raw keys to GCS
- `write_metrics_for_gcs(metrics, step, config, running_metrics)` — [`L214`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L214) — Writes metrics to gcs
- `write_metrics_locally(metrics, step, config, file)` — [`L202`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L202) — Writes metrics locally for testing

## Module values
- `FrozenDict` — [`L73`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L73)
- `_ml_run` — [`L74`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/max_utils.py#L74)

