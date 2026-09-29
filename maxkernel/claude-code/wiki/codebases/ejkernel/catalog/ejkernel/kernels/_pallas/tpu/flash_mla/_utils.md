---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_mla._utils`/
symbols:
  MIN_BLOCK_SIZE: MIN_BLOCK_SIZE.
  TRANS_B_DIM_NUMBERS: TRANS_B_DIM_NUMBERS.
  below_or_on_diag: below_or_on_diag().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _verify_block: _verify_block().
  NUM_LANES: NUM_LANES.
  NUM_SUBLANES: NUM_SUBLANES.
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py)

## Functions
- `_verify_block(block_name, dim_name, block, dim, should_divide=True)` — [`L37`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L37) — Verify that a block size is valid for a given dimension.
- `below_or_on_diag(r, r_blk_size, c, c_blk_size)` — [`L56`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L56) — Check if query block can attend to KV block under causal masking.

## Module values
- `DEFAULT_MASK_VALUE` — [`L21`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L21)
- `MIN_BLOCK_SIZE` — [`L30`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L30)
- `NUM_LANES` — [`L24`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L24)
- `NUM_SUBLANES` — [`L27`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L27)
- `TRANS_B_DIM_NUMBERS` — [`L33`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_utils.py#L33)

