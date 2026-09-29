---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._utils`/
symbols:
  align_to: align_to().
  get_dtype_packing: get_dtype_packing().
  cdiv: cdiv().
  get_tuned_block_sizes_h64: get_tuned_block_sizes_h64().
  get_tuned_block_sizes: get_tuned_block_sizes().
  next_power_of_2: next_power_of_2().
  get_lookup_keys_h64: get_lookup_keys_h64().
  get_simplified_raw_key_h64: get_simplified_raw_key_h64().
  get_lookup_keys: get_lookup_keys().
  get_simplified_raw_key: get_simplified_raw_key().
  get_dtype_bitwidth: get_dtype_bitwidth().
  get_device_name: get_device_name().
  get_tpu_version: get_tpu_version().
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  H64TUNED_BLOCK_SIZES: H64TUNED_BLOCK_SIZES.
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py)

## Functions
- `align_to(x, a)` — [`L104`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L104) — Round x up to the nearest multiple of a. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `cdiv(a, b)` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L90) — Compute ceiling division of a by b. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_device_name(num_devices: int | None = None)` — [`L168`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L168) — Detect the TPU device name with normalized variant suffix. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_dtype_bitwidth(dtype)` — [`L117`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L117) — Return the bit width of a JAX/NumPy dtype. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_dtype_packing(dtype)` — [`L135`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L135) — Compute the packing factor for a dtype relative to 32-bit words. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_lookup_keys(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len)` — [`L4694`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4694) — Build hierarchical lookup keys for the standard tuning table.
- `get_lookup_keys_h64(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len)` — [`L4528`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4528) — Build hierarchical lookup keys for the head_dim=64 tuning table. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_simplified_raw_key(page_size, q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, max_model_len)` — [`L4748`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4748) — Compute simplified, normalized key components for standard lookup. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_simplified_raw_key_h64(page_size, q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, max_model_len)` — [`L4582`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4582) — Compute simplified, normalized key components for head_dim=64 lookup.
- `get_tpu_version()` — [`L208`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L208) — Returns the numeric version of the TPU, or -1 if not on TPU. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `get_tuned_block_sizes(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, pages_per_seq)` — [`L4630`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4630) — Search tuned block sizes for standard attention configurations (head_dim >= 128).
- `get_tuned_block_sizes_h64(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, pages_per_seq)` — [`L4464`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4464) — Search tuned block sizes for head_dim=64 attention configurations. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)
- `next_power_of_2(x: int)` — [`L153`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L153) — Finds the smallest power of 2 >= x using bit manipulation.

## Module values
- `H64TUNED_BLOCK_SIZES` — [`L4226`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L4226)
- `TUNED_BLOCK_SIZES` — [`L223`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_utils.py#L223) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_utils.md)

