---
title: 'Module: jax/experimental/multihost_utils.py'
type: catalog
provenance: extracted
module: jax/experimental/multihost_utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.multihost_utils`/
symbols:
  _handle_array_process_allgather: _handle_array_process_allgather().
  host_local_array_to_global_array_impl: host_local_array_to_global_array_impl().
  global_array_to_host_local_array_impl: global_array_to_host_local_array_impl().
  broadcast_one_to_all: broadcast_one_to_all().
  host_local_array_to_global_array: host_local_array_to_global_array().
  global_array_to_host_local_array: global_array_to_host_local_array().
  host_local_array_to_global_array_p: host_local_array_to_global_array_p.
  global_array_to_host_local_array_p: global_array_to_host_local_array_p.
  assert_equal.concat_in_tree: assert_equal().concat_in_tree().
  process_allgather: process_allgather().
  ltg_batcher: ltg_batcher().
  _LiveDevices.__call__: _LiveDevices#__call__().
  _local_to_global_aval: _local_to_global_aval().
  _global_to_local_aval: _global_to_local_aval().
  assert_equal: assert_equal().
  ltg_abstract_eval: ltg_abstract_eval().
  gtl_abstract_eval: gtl_abstract_eval().
  _LiveDevices.devices: _LiveDevices#devices.
  _flatten_pspecs: _flatten_pspecs().
  _live_devices: _live_devices().
  _psum: _psum().
  broadcast_one_to_all.pre_jit: broadcast_one_to_all().pre_jit().
  broadcast_one_to_all.post_jit: broadcast_one_to_all().post_jit().
  process_allgather._pjit: process_allgather()._pjit().
  sync_global_devices: sync_global_devices().
  live_devices: live_devices.
  _identity_fn: _identity_fn().
  _ltg_lowering: _ltg_lowering().
  _gtl_lowering: _gtl_lowering().
  ProcessFailureError: ProcessFailureError#
  _LiveDevices: _LiveDevices#
  reached_preemption_sync_point: reached_preemption_sync_point().
  ProcessFailureError.__init__: ProcessFailureError#__init__().
  ProcessFailureError.failed_devices: ProcessFailureError#failed_devices.
  _LiveDevices.__init__: _LiveDevices#__init__().
---
# Module: [`jax/experimental/multihost_utils.py`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py)

## Classes
### `ProcessFailureError`  ·  implements/extends Exception
- def: [`jax/experimental/multihost_utils.py:521`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L521)
- doc: Raised by live_devices when one or more processes have failed.
- signature: `class ProcessFailureError(Exception):`
- members:
  - `failed_devices` — [`L528`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L528)
- protocol/private: `__init__`[`L527`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L527)
- used by: [`__call__`](multihost_utils.md#_LiveDevices.__call__)

### `_LiveDevices`
- def: [`jax/experimental/multihost_utils.py:548`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L548)
- doc: A context manager for atomically running code on the set of live devices.
- signature: `class _LiveDevices:`
- members:
  - `devices` — [`L659`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L659)
- protocol/private: `__call__`[`L662`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L662), `__init__`[`L658`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L658)
- uses (calls/refs, reference-scoped): [`_live_devices`](multihost_utils.md#_live_devices), [`ProcessFailureError`](multihost_utils.md#ProcessFailureError)
- used by: [`live_devices`](multihost_utils.md#live_devices)

## Functions
- `_flatten_pspecs(name, in_tree, pspecs_thunk)` — [`L237`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L237)
- `_global_to_local_aval(global_aval, mesh, pspec)` — [`L248`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L248)
- `_gtl_lowering(ctx, x, *, global_mesh, pspec)` — [`L516`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L516)
- `_handle_array_process_allgather(inp, tiled)` — [`L97`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L97)
- `_identity_fn(x)` — [`L93`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L93)
- `_live_devices(client, devices: list[xla_client.Device])` — [`L532`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L532) — Returns the subset of the provided devices that are live and healthy.
- `_local_to_global_aval(local_aval, mesh, pspec)` — [`L242`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L242)
- `_ltg_lowering(ctx, x, *, global_mesh, pspec)` — [`L412`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L412)
- `_pjit(inp)` — [`L157`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L157)
- `_psum(xs: Any)` — [`L44`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L44)
- `assert_equal(in_tree, fail_message: str = '')` — [`L168`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L168) — Verifies that all the hosts have the same tree of values.
- `broadcast_one_to_all(in_tree: Any, is_source: bool | None = None)` — [`L48`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L48) — Broadcast data from a source host (host 0 by default) to all other hosts.
- `concat_in_tree(x)` — [`L170`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L170)
- `global_array_to_host_local_array(global_inputs: Any, global_mesh: jax.sharding.Mesh, pspecs: Any)` — [`L460`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L460) — Converts a global `jax.Array` to a host local `jax.Array`.
- `global_array_to_host_local_array_impl(arr: Any, *, global_mesh: jax.sharding.Mesh, pspec: Any)` — [`L417`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L417)
- `gtl_abstract_eval(arr, *, global_mesh, pspec)` — [`L506`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L506)
- `host_local_array_to_global_array(local_inputs: Any, global_mesh: jax.sharding.Mesh, pspecs: Any)` — [`L296`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L296) — Converts a host local value to a globally sharded jax.Array.
- `host_local_array_to_global_array_impl(arr: Any, *, global_mesh: jax.sharding.Mesh, pspec: Any)` — [`L254`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L254)
- `ltg_abstract_eval(arr, *, global_mesh, pspec)` — [`L388`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L388)
- `ltg_batcher(insert_axis, axis_data, vals_in, dims_in, global_mesh, pspec)` — [`L397`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L397)
- `post_jit(x)` — [`L81`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L81)
- `pre_jit(x)` — [`L73`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L73)
- `process_allgather(in_tree: Any, tiled: bool = False)` — [`L138`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L138) — Gather data from across processes.
- `reached_preemption_sync_point(step_id: int)` — [`L187`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L187) — Determine whether all hosts have reached a preemption sync step.
- `sync_global_devices(name: str)` — [`L162`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L162) — Creates a barrier across all hosts/devices.

## Module values
- `global_array_to_host_local_array_p` — [`L503`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L503)
- `host_local_array_to_global_array_p` — [`L385`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L385)
- `live_devices` — [`L696`](../../../../../../raw/code/jax/jax/experimental/multihost_utils.py#L696)

