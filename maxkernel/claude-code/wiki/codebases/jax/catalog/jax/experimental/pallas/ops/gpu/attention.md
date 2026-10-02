---
title: 'Module: jax/experimental/pallas/ops/gpu/attention.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/attention.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.attention`/
symbols:
  _mha_backward: _mha_backward().
  mha: mha().
  mha_forward_kernel.body: mha_forward_kernel().body().
  mha_backward_kernel: mha_backward_kernel().
  mha_backward_kernel.inner_loop_dkdv: mha_backward_kernel().inner_loop_dkdv().
  mha_backward_kernel.inner_loop_dq: mha_backward_kernel().inner_loop_dq().
  mha_forward_kernel: mha_forward_kernel().
  mha_reference: mha_reference().
  _preprocess_backward: _preprocess_backward().
  segment_mask: segment_mask().
  BlockSizes.get_default: BlockSizes#get_default().
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  _preprocess_backward_kernel: _preprocess_backward_kernel().
  _mha_forward: _mha_forward().
  BlockSizes: BlockSizes#
  BlockSizes.block_q_dkv: BlockSizes#block_q_dkv.
  BlockSizes.block_kv_dkv: BlockSizes#block_kv_dkv.
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  BlockSizes.block_kv_dq: BlockSizes#block_kv_dq.
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_k: BlockSizes#block_k.
---
# Module: [`jax/experimental/pallas/ops/gpu/attention.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py)

## Classes
### `BlockSizes`
- def: [`jax/experimental/pallas/ops/gpu/attention.py:33`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L33)
- doc: Tile sizes parameterizing the attention kernel. These block sizes
- signature: `class BlockSizes:`
- members:
  - `get_default(cls)` — [`L56`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L56)
  - `has_backward_blocks(self)` — [`L67`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L67) — Returns True if all backward blocks are specified for the fused
  - `block_k` — [`L48`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L48)
  - `block_kv_dkv` — [`L51`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L51)
  - `block_kv_dq` — [`L53`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L53)
  - `block_q` — [`L47`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L47)
  - `block_q_dkv` — [`L50`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L50)
  - `block_q_dq` — [`L52`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L52)
- used by: [`_mha_backward`](attention.md#_mha_backward), [`mha`](attention.md#mha), [`_mha_forward`](attention.md#_mha_forward)

## Functions
- `_mha_backward(sm_scale: float, causal: bool, block_sizes: BlockSizes, backward_pass_impl: str, num_warps: int | None, num_stages: int, grid: Any, interpret: bool, debug: bool, return_residuals: bool, res, do)` — [`L532`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L532)
- `_mha_forward(q, k, v, segment_ids: jax.Array | None, sm_scale: float, causal: bool, block_sizes: BlockSizes, backward_pass_impl: str, num_warps: int | None, num_stages: int, grid: Any, interpret: bool, debug: bool, return_residuals: bool)` — [`L300`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L300)
- `_preprocess_backward(out, do, lse, block_q: int, debug: bool, interpret: bool)` — [`L338`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L338)
- `_preprocess_backward_kernel(out_ref, dout_ref, delta_ref, head_dim: int)` — [`L327`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L327)
- `body(start_k, carry)` — [`L119`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L119)
- `inner_loop_dkdv(start_q, carry)` — [`L411`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L411)
- `inner_loop_dq(start_k, dq)` — [`L483`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L483)
- `mha(q, k, v, segment_ids: jnp.ndarray | None, sm_scale: float = 1, causal: bool = False, block_sizes: BlockSizes = BlockSizes.get_default(), backward_pass_impl: str = "triton", num_warps: int | None = None, num_stages: int = 2, grid: tuple[int, ...] | None = None, interpret: bool = False, debug: bool = False, return_residuals: bool = False)` — [`L220`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L220)
- `mha_backward_kernel(q_ref, k_ref, v_ref, segment_ids_ref: jax.Array | None, out_ref, do_scaled_ref, lse_ref, delta_ref, dq_ref, dk_ref, dv_ref, *, sm_scale: float, causal: bool, block_q_dkv: int, block_kv_dkv: int, block_q_dq: int, block_kv_dq: int, head_dim: int)` — [`L365`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L365)
- `mha_forward_kernel(q_ref, k_ref, v_ref, segment_ids_ref: jax.Array | None, o_ref: Any, *residual_refs: Any, sm_scale: float, causal: bool, block_q: int, block_k: int, head_dim: int)` — [`L82`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L82)
- `mha_reference(q, k, v, segment_ids: jnp.ndarray | None, sm_scale=1, causal: bool = False)` — [`L655`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L655)
- `segment_mask(q_segment_ids: jax.Array, kv_segment_ids: jax.Array)` — [`L188`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L188)

## Module values
- `DEFAULT_MASK_VALUE` — [`L30`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention.py#L30)

