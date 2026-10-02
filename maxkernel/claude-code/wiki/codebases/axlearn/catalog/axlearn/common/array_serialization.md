---
title: 'Module: axlearn/common/array_serialization.py'
type: catalog
provenance: extracted
module: axlearn/common/array_serialization.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.array_serialization`/
symbols:
  BoundedDataShardedAsyncCheckpointManager.serialize: BoundedDataShardedAsyncCheckpointManager#serialize().
  _async_serialize: _async_serialize().
  _ColocatedStateManager.load_to_cpu: _ColocatedStateManager#load_to_cpu().
  _run_colocated_deserialize: _run_colocated_deserialize().
  GlobalAsyncCheckpointManager.deserialize: GlobalAsyncCheckpointManager#deserialize().
  _ColocatedStateManager.event_loop: _ColocatedStateManager#event_loop.
  _get_shard_infos: _get_shard_infos().
  _slice_shard_and_copy_to_host: _slice_shard_and_copy_to_host().
  _slice_fn: _slice_fn().
  GlobalAsyncCheckpointManager._run_deserializer: GlobalAsyncCheckpointManager#_run_deserializer().
  GlobalAsyncCheckpointManager.serialize: GlobalAsyncCheckpointManager#serialize().
  _ShardInfo: _ShardInfo#
  _ColocatedStateManager.teardown: _ColocatedStateManager#teardown().
  _ShardInfo.data: _ShardInfo#data.
  _num_replicas_per_shard: _num_replicas_per_shard().
  _run_serializer: _run_serializer().
  GlobalAsyncCheckpointManager.stop: GlobalAsyncCheckpointManager#stop().
  _ColocatedStateManager.loop_thread: _ColocatedStateManager#loop_thread.
  _async_deserialize: _async_deserialize().
  GlobalAsyncCheckpointManager._loop_thread: GlobalAsyncCheckpointManager#_loop_thread.
  _blocking_device_put: _blocking_device_put().
  _fix_metadata: _fix_metadata().
  _async_deserialize.cb: _async_deserialize().cb().
  _run_colocated_deserialize._load_and_transfer_one: _run_colocated_deserialize()._load_and_transfer_one().
  _run_colocated_deserialize._load_and_transfer_one_rate_limited: _run_colocated_deserialize()._load_and_transfer_one_rate_limited().
  GlobalAsyncCheckpointManager: GlobalAsyncCheckpointManager#
  _CommitFuture.__init__: _CommitFuture#__init__().
  _CommitFuture.result: _CommitFuture#result().
  GlobalAsyncCheckpointManager._loop: GlobalAsyncCheckpointManager#_loop.
  _slices_to_tuple: _slices_to_tuple().
  BoundedDataShardedAsyncCheckpointManager: BoundedDataShardedAsyncCheckpointManager#
  BoundedDataShardedAsyncCheckpointManager._max_concurrent_bytes: BoundedDataShardedAsyncCheckpointManager#_max_concurrent_bytes.
  BoundedDataShardedAsyncCheckpointManager._max_data_shard_degree: BoundedDataShardedAsyncCheckpointManager#_max_data_shard_degree.
  _transfer_to_host: _transfer_to_host().
  _local_size: _local_size().
  _wait_host_array_h2d_transfer: _wait_host_array_h2d_transfer().
  _ThreadRaisingException.join: _ThreadRaisingException#join().
  BoundedDataShardedAsyncCheckpointManager._tensorstore_spec_modifier: BoundedDataShardedAsyncCheckpointManager#_tensorstore_spec_modifier().
  _ThreadRaisingException._exception: _ThreadRaisingException#_exception.
  _ColocatedStateManager.__del__: _ColocatedStateManager#__del__().
  _ShardInfo.shard_coordinate: _ShardInfo#shard_coordinate().
  TensorstoreSpecModifier.__call__: TensorstoreSpecModifier#__call__().
  _ThreadRaisingException.run: _ThreadRaisingException#run().
  GlobalAsyncCheckpointManager.__del__: GlobalAsyncCheckpointManager#__del__().
  BoundedDataShardedAsyncCheckpointManager.__init__: BoundedDataShardedAsyncCheckpointManager#__init__().
  _COLOCATED_TRANSFER_TIMEOUT_SECS: _COLOCATED_TRANSFER_TIMEOUT_SECS.
  _ColocatedStateManager.ts_context: _ColocatedStateManager#ts_context.
  _ShardInfo.index: _ShardInfo#index.
  _SliceTuple: _SliceTuple.
  TensorstoreSpecModifier: TensorstoreSpecModifier#
  _CommitFuture._t: _CommitFuture#_t.
  GlobalAsyncCheckpointManager._single_thread_pool: GlobalAsyncCheckpointManager#_single_thread_pool.
  GlobalAsyncCheckpointManager._multi_thread_pool: GlobalAsyncCheckpointManager#_multi_thread_pool.
  BoundedDataShardedAsyncCheckpointManager._logged_spec: BoundedDataShardedAsyncCheckpointManager#_logged_spec.
  _COLOCATED_H2D_CONCURRENCY: _COLOCATED_H2D_CONCURRENCY.
  _ColocatedStateManager: _ColocatedStateManager#
  _ColocatedStateManager.cpu_shardings: _ColocatedStateManager#cpu_shardings.
  _ColocatedStateManager.tensorstore_specs: _ColocatedStateManager#tensorstore_specs.
  _ColocatedStateManager.global_shapes: _ColocatedStateManager#global_shapes.
  _ColocatedStateManager.dtypes: _ColocatedStateManager#dtypes.
  _ColocatedStateManager.concurrent_bytes: _ColocatedStateManager#concurrent_bytes.
  _colocated_teardown: _colocated_teardown().
  _ShardInfo.slice_arg: _ShardInfo#slice_arg.
  _ShardInfo.replica_count: _ShardInfo#replica_count.
  use_gcs_grpc: use_gcs_grpc().
  running_on_pathways: running_on_pathways().
  _create_cpu_shardings: _create_cpu_shardings().
  _effective_bytes_per_device: _effective_bytes_per_device().
  _ThreadRaisingException: _ThreadRaisingException#
  _CommitFuture: _CommitFuture#
  _get_premapped_buffer_size: _get_premapped_buffer_size().
  GlobalAsyncCheckpointManager.__init__: GlobalAsyncCheckpointManager#__init__().
  GlobalAsyncCheckpointManager._run_serializer: GlobalAsyncCheckpointManager#_run_serializer().
  BoundedDataShardedAsyncCheckpointManager._shard_threshold_bytes: BoundedDataShardedAsyncCheckpointManager#_shard_threshold_bytes.
  BoundedDataShardedAsyncCheckpointManager._tensorstore_spec_log_fn: BoundedDataShardedAsyncCheckpointManager#_tensorstore_spec_log_fn().
  _ColocatedStateManager.__init__: _ColocatedStateManager#__init__().
---
# Module: [`axlearn/common/array_serialization.py`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py)

## Classes
### `BoundedDataShardedAsyncCheckpointManager`  ·  implements/extends GlobalAsyncCheckpointManager
- def: [`axlearn/common/array_serialization.py:1167`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1167)
- doc: Similar to GlobalAsyncCheckpointManager but with few improvements:
- signature: `class BoundedDataShardedAsyncCheckpointManager(GlobalAsyncCheckpointManager):`
- members:
  - `_tensorstore_spec_log_fn(self, specs: list[dict[str, Any]])` — [`L1221`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1221) — A function that will be called **once** after the tensorstore specs are populated.
  - `_tensorstore_spec_modifier(self, spec: dict[str, Any], *, shard_infos: list[_ShardInfo])` — [`L1213`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1213) — A function that modifies the tensorstore spec for an array in-place.
  - `serialize(self, arrays: list[Tensor], tensorstore_specs: list[dict], *, on_commit_callback: Callable[[], None], additional_futures: Optional[list[futures.Future]] = None)` — [`L1230`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1230) — See JAX `GlobalAsyncCheckpointManager` docstring.
- protocol/private: `__init__`[`L1190`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1190), `_logged_spec`[`L1199`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1199), `_max_concurrent_bytes`[`L1202`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1202), `_max_data_shard_degree`[`L1208`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1208), `_shard_threshold_bytes`[`L1211`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1211)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`_ShardInfo`](array_serialization.md#_ShardInfo), [`_run_serializer`](array_serialization.md#_run_serializer), [`GlobalAsyncCheckpointManager`](array_serialization.md#GlobalAsyncCheckpointManager), [`_local_size`](array_serialization.md#_local_size), [`_CommitFuture`](array_serialization.md#_CommitFuture), [`__init__`](array_serialization.md#GlobalAsyncCheckpointManager.__init__)
- used by: [`save_to_dir`](checkpointer.md#TensorStoreStateStorage.save_to_dir), [`_manager`](checkpointer.md#TensorStoreStateStorage._manager), [`serialize`](array_serialization.md#GlobalAsyncCheckpointManager.serialize), [`GlobalAsyncCheckpointManager`](array_serialization.md#GlobalAsyncCheckpointManager)

### `GlobalAsyncCheckpointManager`  ·  implements/extends GlobalAsyncCheckpointManager
- def: [`axlearn/common/array_serialization.py:999`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L999)
- doc: Similar to GlobalAsyncCheckpointManager but allows passing additional futures to be awaited
- signature: `class GlobalAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):`
- members:
  - `deserialize(self, shardings: Sequence[Union[jax.sharding.Sharding, Format]], tensorstore_specs: Sequence[dict[str, Any]], global_shapes: Optional[Sequence[array.Shape]] = None, dtypes: Optional[Sequence[typing.DTypeLike]] = None, concurrent_gb: int = 32)` — [`L1059`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1059) — Deserialize arrays from TensorStore.
  - `serialize(self, arrays: list[Tensor], tensorstore_specs: list[dict], *, on_commit_callback: Callable[[], None], additional_futures: Optional[list[futures.Future]] = None)` — [`L1023`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1023)
  - `stop(self)` — [`L1013`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1013) — Cleans up any internal threads.
- protocol/private: `__del__`[`L1019`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1019), `__init__`[`L1004`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1004), `_loop`[`L1006`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1006), `_loop_thread`[`L1007`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1007), `_multi_thread_pool`[`L1011`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1011), `_run_deserializer`[`L1138`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1138), `_run_serializer`[`L1037`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1037), `_single_thread_pool`[`L1009`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L1009)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`serialize`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager.serialize), `env`, [`_run_colocated_deserialize`](array_serialization.md#_run_colocated_deserialize), [`_async_deserialize`](array_serialization.md#_async_deserialize), [`BoundedDataShardedAsyncCheckpointManager`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager), [`_get_premapped_buffer_size`](array_serialization.md#_get_premapped_buffer_size), [`running_on_pathways`](array_serialization.md#running_on_pathways)
- used by: [`_restore_tensorstore_state`](checkpointer.md#TensorStoreStateStorage._restore_tensorstore_state), [`_manager`](checkpointer.md#TensorStoreStateStorage._manager), [`BoundedDataShardedAsyncCheckpointManager`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager), [`__init__`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager.__init__)

### `TensorstoreSpecModifier`
- def: [`axlearn/common/array_serialization.py:346`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L346)
- signature: `class TensorstoreSpecModifier:`
- protocol/private: `__call__`[`L347`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L347)
- uses (calls/refs, reference-scoped): [`_ShardInfo`](array_serialization.md#_ShardInfo)
- used by: [`_async_serialize`](array_serialization.md#_async_serialize), [`_run_serializer`](array_serialization.md#_run_serializer)

### `_ColocatedStateManager`
- def: [`axlearn/common/array_serialization.py:58`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L58)
- doc: Manages config and runtime resources on colocated sidecar.
- signature: `class _ColocatedStateManager:`
- members:
  - `load_to_cpu(self, idx: jax.Array)` — [`L88`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L88) — Load a single array to CPU on the sidecar.
  - `teardown(self)` — [`L108`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L108) — Release TensorStore context and event loop on colocated Python workers.
  - `concurrent_bytes` — [`L78`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L78)
  - `cpu_shardings` — [`L74`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L74)
  - `dtypes` — [`L77`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L77)
  - `event_loop` — [`L82`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L82)
  - `global_shapes` — [`L76`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L76)
  - `loop_thread` — [`L84`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L84)
  - `tensorstore_specs` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L75)
  - `ts_context` — [`L81`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L81)
- protocol/private: `__del__`[`L125`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L125), `__init__`[`L65`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L65)
- uses (calls/refs, reference-scoped): [`_async_deserialize`](array_serialization.md#_async_deserialize)
- used by: [`_run_colocated_deserialize`](array_serialization.md#_run_colocated_deserialize)

### `_CommitFuture`
- def: [`axlearn/common/array_serialization.py:980`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L980)
- doc: Represents the result of a background commit.
- signature: `class _CommitFuture:`
- members:
  - `result(self, timeout: Optional[int] = None)` — [`L987`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L987)
- protocol/private: `__init__`[`L983`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L983), `_t`[`L984`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L984)
- uses (calls/refs, reference-scoped): [`join`](array_serialization.md#_ThreadRaisingException.join), [`_ThreadRaisingException`](array_serialization.md#_ThreadRaisingException)
- used by: [`serialize`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager.serialize)

### `_ShardInfo`
- def: [`axlearn/common/array_serialization.py:137`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L137)
- doc: Stores information for a maybe sliced jax.Shard.
- signature: `class _ShardInfo:`
- members:
  - `shard_coordinate(self)` — [`L153`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L153) — Gets the shard coordinate according to the zarr format used by tensorstore.
  - `data` — [`L148`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L148)
  - `index` — [`L149`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L149)
  - `replica_count` — [`L151`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L151)
  - `slice_arg` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L150)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`_async_serialize`](array_serialization.md#_async_serialize), [`_get_shard_infos`](array_serialization.md#_get_shard_infos), [`_slice_fn`](array_serialization.md#_slice_fn), [`_slice_shard_and_copy_to_host`](array_serialization.md#_slice_shard_and_copy_to_host), [`_fix_metadata`](array_serialization.md#_fix_metadata), [`_tensorstore_spec_modifier`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager._tensorstore_spec_modifier), [`__call__`](array_serialization.md#TensorstoreSpecModifier.__call__)

### `_ThreadRaisingException`  ·  implements/extends Thread
- def: [`axlearn/common/array_serialization.py:960`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L960)
- doc: Thread that raises an exception if it encounters an error.
- signature: `class _ThreadRaisingException(threading.Thread):`
- members:
  - `join(self, timeout=None)` — [`L972`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L972)
  - `run(self)` — [`L965`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L965)
- protocol/private: `_exception`[`L963`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L963)
- used by: [`__init__`](array_serialization.md#_CommitFuture.__init__), [`result`](array_serialization.md#_CommitFuture.result)

## Functions
- `_async_deserialize(user_in_sharding: jax.sharding.Sharding | Format, tensorstore_spec: dict[str, Any], global_shape: Optional[Sequence[int]], dtype: Optional[typing.DTypeLike], *, h2d_limiter: Optional[serialization._LimitInFlightBytes], byte_limiter: serialization._LimitInFlightBytes, single_thread_pool: Optional[ThreadPoolExecutor], multi_thread_pool: Optional[ThreadPoolExecutor], ts_context: Optional[ts.Context] = None)` — [`L532`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L532) — Modified from
- `_async_serialize(arr_inp: Tensor, tensorstore_spec: dict[str, Any], d2h_future: futures.Future, *, limiter: Optional[serialization._LimitInFlightBytes], tensorstore_spec_modifier: Optional[TensorstoreSpecModifier] = None, max_data_shard_degree: int, shard_threshold_bytes: int)` — [`L350`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L350) — Similar to `serialization.ts_impl.async_serialize`, but limiting peak host memory
- `_blocking_device_put(tensor: Tensor, target: Union[Format, jax.sharding.Sharding])` — [`L498`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L498) — Device put and block until ready.
- `_colocated_teardown()` — [`L130`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L130) — Run garbage collection on the sidecar.
- `_create_cpu_shardings(cpu_devices: list, tpu_shardings: Sequence[Union[jax.sharding.Sharding, Format]], tpu_mesh: jax.sharding.Mesh)` — [`L707`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L707) — Create CPU shardings that mirror the structure of TPU shardings.
- `_effective_bytes_per_device(spec: dict[str, Any], shape: tuple, dtype: typing.DTypeLike, sharding: jax.sharding.Sharding)` — [`L740`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L740) — Estimates effective bytes read from storage per device, accounting for chunk overhead.
- `_fix_metadata(tspec: dict[str, Any], shard_infos: list[_ShardInfo])` — [`L338`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L338) — Revises the medadata of a tensorspec based on `shard_infos`.
- `_get_premapped_buffer_size()` — [`L991`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L991)
- `_get_shard_infos(arr_inp: Tensor, *, max_data_shard_degree: int, shard_threshold_bytes: int)` — [`L188`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L188) — Returns a list of _ShardInfo for addressable shards that need to be saved.
- `_load_and_transfer_one(idx: int, tpu_sharding: jax.sharding.Sharding, dispatch_pool: ThreadPoolExecutor)` — [`L850`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L850) — Load one array to CPU via colocated Python, then transfer to TPU.
- `_load_and_transfer_one_rate_limited(idx: int, tpu_sharding: jax.sharding.Sharding, dispatch_pool: ThreadPoolExecutor)` — [`L901`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L901) — Wrapper that applies global limiter to the entire load+transfer operation.
- `_local_size(arr_inp: Tensor)` — [`L333`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L333) — Calculates the size of a Tensor in bytes in the local process.
- `_num_replicas_per_shard(arr: Tensor)` — [`L178`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L178) — Gets the global replication count for each unique shard.
- `_run_colocated_deserialize(shardings: Sequence[Union[jax.sharding.Sharding, Format]], tensorstore_specs: Sequence[dict[str, Any]], global_shapes: Sequence[tuple], dtypes: Sequence[typing.DTypeLike], *, concurrent_bytes: int, tpu_mesh: jax.sharding.Mesh, pipeline_concurrent_bytes: int, multi_thread_pool: ThreadPoolExecutor)` — [`L771`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L771) — Deserialize checkpoint with pipelined load to CPU then transfer to TPU.
- `_run_serializer(arrays: list[Tensor], tensorstore_specs: list[dict[str, Any]], d2h_futures: list[futures.Future], *, max_concurrent_bytes: Optional[int] = None, tensorstore_spec_modifier: Optional[TensorstoreSpecModifier] = None, max_data_shard_degree: int, shard_threshold_bytes: int)` — [`L450`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L450) — Asynchronously serializes a list of tensors with _async_serialize.
- `_slice_fn(info: _ShardInfo)` — [`L325`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L325) — Performs slicing according to a shard_info and returns the sliced array.
- `_slice_shard_and_copy_to_host(shard_infos: list[_ShardInfo])` — [`L307`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L307) — Slices each shard according to shard info and then copy the sliced result to host.
- `_slices_to_tuple(slices: list[slice])` — [`L173`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L173) — Converts a list of slices to a hashable representation.
- `_transfer_to_host(data: Tensor)` — [`L243`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L243) — Asynchronously transfers a shard to host memory. Does not block.
- `_wait_host_array_h2d_transfer(cpu_array: Tensor, target: Union[Format, jax.sharding.Sharding])` — [`L511`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L511) — Wait for CPU array to be ready and transfer to device in a single call.
- `cb(index: array.Index, device: jax.Device)` — [`L617`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L617)
- `running_on_pathways()` — [`L298`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L298) — We use GCP only for inference with Pathways. In this setup, JAX_PLATFORMS is set to
- `use_gcs_grpc(tensorstore_spec: dict[str, Any])` — [`L261`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L261) — Switch TensorStore to the gcs_grpc driver to improve Google Cloud Storage read throughput.

## Module values
- `_COLOCATED_H2D_CONCURRENCY` — [`L49`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L49)
- `_COLOCATED_TRANSFER_TIMEOUT_SECS` — [`L54`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L54)
- `_SliceTuple` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/array_serialization.py#L170)

