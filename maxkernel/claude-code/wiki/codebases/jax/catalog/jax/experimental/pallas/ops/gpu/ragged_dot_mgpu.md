---
title: 'Module: jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.ragged_dot_mgpu`/
symbols:
  main: main().
  ragged_dot.body.mn_loop.store_scope: ragged_dot().body().mn_loop().store_scope().
  ragged_dot: ragged_dot().
  ragged_dot.body.mn_loop: ragged_dot().body().mn_loop().
  ragged_dot.body.mn_loop.acc_scope: ragged_dot().body().mn_loop().acc_scope().
  GroupInfo.create: GroupInfo#create().
  GroupInfo.actual_size: GroupInfo#actual_size.
  main.ref_ragged_dot: main().ref_ragged_dot().
  GroupInfo.start_within_block: GroupInfo#start_within_block.
  GroupInfo.group_id: GroupInfo#group_id.
  GroupInfo.block: GroupInfo#block.
  GroupInfo.block_start: GroupInfo#block_start.
  ragged_dot.body: ragged_dot().body().
  GroupInfo: GroupInfo#
  GroupInfo.actual_start: GroupInfo#actual_start.
  GroupInfo.actual_end: GroupInfo#actual_end.
---
# Module: [`jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py)

## Classes
### `GroupInfo`
- def: [`jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py:34`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L34)
- doc: Information regarding the group being processed in a block.
- signature: `class GroupInfo:`
- members:
  - `create(cls, group_lengths, tile, tid)` — [`L46`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L46) — Get the group info for the current block.
  - `actual_end` — [`L41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L41)
  - `actual_size` — [`L43`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L43)
  - `actual_start` — [`L40`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L40)
  - `block` — [`L38`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L38)
  - `block_start` — [`L39`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L39)
  - `group_id` — [`L37`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L37)
  - `start_within_block` — [`L42`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L42)
- uses (calls/refs, reference-scoped): [`Array`](../../../../_src/basearray.md#Array), [`int32`](../../../../numpy/__init__.pyi.md#int32.int32), [`select`](../../../../_src/lax/lax.md#select), [`div`](../../../../_src/lax/lax.md#div), [`array`](../../../../numpy/__init__.pyi.md#array), [`maximum`](../../../../numpy/__init__.pyi.md#maximum.maximum), [`minimum`](../../../../numpy/__init__.pyi.md#minimum.minimum)
- used by: [`do_matmul`](blackwell_ragged_dot_mgpu.md#do_matmul), [`mn_loop`](blackwell_ragged_dot_mgpu.md#ragged_dot_kernel.kernel._scoped.mn_loop), [`store_scope`](ragged_dot_mgpu.md#ragged_dot.body.mn_loop.store_scope), [`mn_loop`](ragged_dot_mgpu.md#ragged_dot.body.mn_loop), [`acc_scope`](ragged_dot_mgpu.md#ragged_dot.body.mn_loop.acc_scope)

## Functions
- `acc_scope(acc_ref)` — [`L142`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L142)
- `body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem)` — [`L127`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L127)
- `main(unused_argv)` — [`L245`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L245)
- `mn_loop(loop_info: plgpu.NDLoopInfo)` — [`L133`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L133)
- `ragged_dot(lhs, rhs, *, group_sizes, block_m: int, block_n: int, block_k: int, max_concurrent_steps: int, grid_block_n: int, transpose_rhs: bool = False, load_group_sizes_to_register: bool = True)` — [`L93`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L93)
- `ref_ragged_dot(lhs, rhs, group_sizes)` — [`L303`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L303)
- `store_scope(o_smem)` — [`L172`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py#L172)

