---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_decode_attention._pallas_impl_fwd`/
symbols:
  inner_decode_tpu: inner_decode_tpu().
  ragged_decode_mqa: ragged_decode_mqa().
  get_mha_cost_estimate: get_mha_cost_estimate().
  ragged_flash_attention_kernel: ragged_flash_attention_kernel().
  ragged_flash_attention_kernel.init: ragged_flash_attention_kernel().init().
  ragged_flash_attention_kernel.run: ragged_flash_attention_kernel().run().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py)

## Functions
- `get_mha_cost_estimate(shape_dtype)` — [`L74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L74) — Estimates the cost of MHA computation for use with Pallas.
- `init()` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L125)
- `inner_decode_tpu(query: Float[Array, "batch num_q_heads head_dim"], key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], sequence_start: Int[Array, batch], sequence_end: Int[Array, batch], softmax_scale: float | None = None, fwd_params: FwdParams | None = None, sliding_window: tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, num_sinks] | None = None)` — [`L253`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L253) — JIT-compiled core implementation of ragged MQA Flash Attention for TPU.
- `ragged_decode_mqa(query: chex.Array, key: chex.Array, value: chex.Array, sequence_start: chex.Array, sequence_end: chex.Array, softmax_scale: float | None = None, block_size: int = 256, cost_estimate: pl.CostEstimate | None = None)` — [`L164`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L164) — Run ragged multi-query attention decoding using a Flash Attention Pallas kernel.
- `ragged_flash_attention_kernel(s_ref, e_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, *, block_size: int, softmax_scale: float)` — [`L93`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L93) — Flash Attention kernel for ragged sequences on TPU via Pallas.
- `run()` — [`L135`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.py#L135)

