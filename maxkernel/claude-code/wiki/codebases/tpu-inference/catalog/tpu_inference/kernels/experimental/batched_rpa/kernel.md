---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.kernel`/
symbols:
  rpa_kernel: rpa_kernel().
  strided_load_bkv: strided_load_bkv().
  calculate_and_store_out: calculate_and_store_out().
  rpa_body: rpa_body().
  create_allocs: create_allocs().
  get_kernel_name: get_kernel_name().
  get_kernel_metadata: get_kernel_metadata().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py)

## Functions
- `calculate_and_store_out(step_idx: jax.Array, schedule_ref: schedule.RpaSchedule, acc_scratch_ref: jax.Ref, l_scratch_ref: jax.Ref, o_vref: jax.Ref, *, cfgs: configs.RpaConfigs)` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L70)
- `create_allocs(kv_cache_hbm_ref: jax.Ref, o_hbm_ref: jax.Ref, cfgs: configs.RpaConfigs)` — [`L282`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L282)
- `get_kernel_metadata(cfgs: configs.RpaConfigs)` — [`L343`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L343)
- `get_kernel_name(cfgs: configs.RpaConfigs)` — [`L335`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L335)
- `rpa_body(q_vref: jax.Ref, kv_in_vref: jax.Ref, o_vref: jax.Ref, schedule_ref: schedule.RpaSchedule, m_scratch_ref: jax.Ref, l_scratch_ref: jax.Ref, acc_scratch_ref: jax.Ref, *, cu_q_lens_ref: jax.Ref, kv_lens_ref: jax.Ref, cfgs: configs.RpaConfigs)` — [`L111`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L111)
- `rpa_kernel(cu_q_lens: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, schedule_hbm: schedule.RpaSchedule, q_hbm: jax.Array, new_kv_hbm: jax.Array, kv_cache_hbm: jax.Array, *, cfgs: configs.RpaConfigs)` — [`L355`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L355) — Perform batched ragged paged attention with scheduler data.
- `strided_load_bkv(kv_in_vref: jax.Ref, b_idx: int, start: int, *, cfgs: configs.RpaConfigs)` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/kernel.py#L32)

