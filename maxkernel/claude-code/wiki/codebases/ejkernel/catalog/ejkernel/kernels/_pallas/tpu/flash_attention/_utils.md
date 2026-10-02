---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_attention._utils`/
symbols:
  BlockSizes.get_default: BlockSizes#get_default().
  MIN_BLOCK_SIZE: MIN_BLOCK_SIZE.
  BlockSizes.__post_init__: BlockSizes#__post_init__().
  BlockSizes.has_backward_blocks: BlockSizes#has_backward_blocks().
  NUM_LANES: NUM_LANES.
  _verify_block: _verify_block().
  below_or_on_diag: below_or_on_diag().
  mha_reference_no_custom_vjp: mha_reference_no_custom_vjp().
  mha_reference_bwd: mha_reference_bwd().
  _fwd_cost_estimate: _fwd_cost_estimate().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  mha_reference: mha_reference().
  NUM_SUBLANES: NUM_SUBLANES.
  TRANS_B_DIM_NUMBERS: TRANS_B_DIM_NUMBERS.
  SegmentIds: SegmentIds#
  _mha_reference: _mha_reference().
  BlockSizes.block_q_major_dkv: BlockSizes#block_q_major_dkv.
  BlockSizes.block_k_major_dkv: BlockSizes#block_k_major_dkv.
  BlockSizes.block_k_dkv: BlockSizes#block_k_dkv.
  BlockSizes.block_q_dkv: BlockSizes#block_q_dkv.
  BlockSizes.block_k_major_dq: BlockSizes#block_k_major_dq.
  BlockSizes.block_k_dq: BlockSizes#block_k_dq.
  _mha_reference_fwd: _mha_reference_fwd().
  BlockSizes: BlockSizes#
  BlockSizes.block_q_dq: BlockSizes#block_q_dq.
  BlockSizes.verify_major_minor: BlockSizes#verify_major_minor().
  _mha_reference_bwd: _mha_reference_bwd().
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  BlockSizes.block_k_major: BlockSizes#block_k_major.
  BlockSizes.block_k: BlockSizes#block_k.
  BlockSizes.block_q: BlockSizes#block_q.
  BlockSizes.block_b: BlockSizes#block_b.
  _bytes: _bytes().
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py)

## Classes
### `BlockSizes`
- def: [`ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py:86`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L86) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
- doc: Tile sizes parameterizing FlashAttention kernels.
- signature: `class BlockSizes:`
- members:
  - `__post_init__(self)` — [`L107`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L107) — Validate major/minor block-size relationships on construction. — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model)` — [`L139`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L139) — Return a default BlockSizes with all tiles set to 128. — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `has_backward_blocks(self)` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L125) — Return True if all backward-pass block fields are set (not None). — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `verify_major_minor(prefix, suffix, major, minor)` — [`L110`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L110)
  - `block_b` — [`L96`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L96) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_k` — [`L95`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L95) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_k_dkv` — [`L100`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L100) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_k_dq` — [`L104`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L104) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_k_major` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L94) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_k_major_dkv` — [`L99`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L99)
  - `block_k_major_dq` — [`L103`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L103) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_q` — [`L93`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L93) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_q_dkv` — [`L101`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L101) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_q_dq` — [`L105`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L105) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
  - `block_q_major_dkv` — [`L98`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L98) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
- used by: [`flash_attention`](_interface.md#flash_attention), [`_flash_attention_bwd`](_pallas_impl_bwd.md#_flash_attention_bwd)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py:68`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L68)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L82`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L82)
  - `q` — [`L81`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L81)
- used by: [`flash_attention`](_interface.md#flash_attention), [`mha_reference_bwd`](_utils.md#mha_reference_bwd), [`mha_reference_no_custom_vjp`](_utils.md#mha_reference_no_custom_vjp), [`_fwd_cost_estimate`](_utils.md#_fwd_cost_estimate), [`mha_reference`](_utils.md#mha_reference), [`_mha_reference`](_utils.md#_mha_reference), [`_mha_reference_fwd`](_utils.md#_mha_reference_fwd)

## Functions
- `_bytes(x: jax.Array | jax.ShapeDtypeStruct)` — [`L194`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L194) — Calculate the total memory size of an array in bytes.
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, ab: jax.Array | None, segment_ids: SegmentIds | None, *, causal: bool, softmax_scale: jax.Array | None, kernel_inputs_specs, kernel_outputs_specs)` — [`L206`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L206) — Estimate computational cost for the Flash Attention forward pass.
- `_mha_reference(q, k, v, ab, segment_ids: SegmentIds | None, causal: bool, mask_value: float, softmax_scale: float, save_residuals: bool)` — [`L359`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L359) — Internal reference MHA with custom VJP for efficient gradient computation.
- `_mha_reference_bwd(causal: bool, mask_value: float, softmax_scale: float, save_residuals: bool, residuals, do)` — [`L544`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L544) — Custom VJP backward pass for reference MHA.
- `_mha_reference_fwd(q, k, v, ab, segment_ids: SegmentIds | None, causal: bool, mask_value: float, softmax_scale: float, save_residuals: bool)` — [`L402`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L402) — Forward pass for custom VJP reference MHA.
- `_verify_block(block_name, dim_name, block, dim, should_divide=True)` — [`L172`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L172) — Verify that a block size is valid for a given dimension.
- `below_or_on_diag(r, r_blk_size, c, c_blk_size)` — [`L591`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L591) — Check if a block position is below or on the causal diagonal.
- `mha_reference(q, k, v, ab, segment_ids: SegmentIds | None = None, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, softmax_scale=1)` — [`L316`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L316) — JIT-compiled reference multi-head attention with custom VJP.
- `mha_reference_bwd(q, k, v, ab, segment_ids: SegmentIds | None, o, l, m, do, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, softmax_scale: float = 1)` — [`L462`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L462) — Compute gradients for reference MHA implementation.
- `mha_reference_no_custom_vjp(q, k, v, ab: jax.Array | None = None, segment_ids: SegmentIds | None = None, *, causal: bool = False, mask_value: float = DEFAULT_MASK_VALUE, softmax_scale: float = 1, save_residuals: bool = False)` — [`L248`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L248) — Reference multi-head attention implementation without custom VJP.

## Module values
- `DEFAULT_MASK_VALUE` — [`L52`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L52)
- `MIN_BLOCK_SIZE` — [`L61`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L61) — documented in [ejkernel-kernels-_pallas-tpu-flash_attention-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-flash_attention-_utils.md)
- `NUM_LANES` — [`L55`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L55)
- `NUM_SUBLANES` — [`L58`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L58)
- `TRANS_B_DIM_NUMBERS` — [`L64`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_attention/_utils.py#L64)

