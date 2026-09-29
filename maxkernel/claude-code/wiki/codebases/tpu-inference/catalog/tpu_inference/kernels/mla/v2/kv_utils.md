---
title: 'Module: tpu_inference/kernels/mla/v2/kv_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/mla/v2/kv_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.mla.v2.kv_utils`/
symbols:
  pack_new_kv: pack_new_kv().
  unsigned_mod: unsigned_mod().
  unsigned_cdiv: unsigned_cdiv().
  unsigned_floor_div: unsigned_floor_div().
  unsigned_align_to: unsigned_align_to().
  align_to: align_to().
  get_dtype_bitwidth: get_dtype_bitwidth().
  get_dtype_packing: get_dtype_packing().
  pack_new_kv_reference: pack_new_kv_reference().
---
# Module: [`tpu_inference/kernels/mla/v2/kv_utils.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py)

## Functions
- `align_to(a, b)` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L58)
- `get_dtype_bitwidth(dtype)` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L62)
- `get_dtype_packing(dtype)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L66)
- `pack_new_kv(bkvc_vmem_ref, bkvpe_vmem_ref, offset, update_sz, q_end, kv_len, bkv_sz)` — [`L71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L71)
- `pack_new_kv_reference(bkvc_vmem_ref, bkvpe_vmem_ref, offset, update_sz, q_end, kv_len, bkv_sz)` — [`L244`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L244)
- `unsigned_align_to(a, b)` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L48)
- `unsigned_cdiv(a, b)` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L32)
- `unsigned_floor_div(a, b)` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L40)
- `unsigned_mod(a, b)` — [`L24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kv_utils.py#L24)

