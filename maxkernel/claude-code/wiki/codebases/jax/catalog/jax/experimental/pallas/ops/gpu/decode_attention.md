---
title: 'Module: jax/experimental/pallas/ops/gpu/decode_attention.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/decode_attention.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.decode_attention`/
symbols:
  decode_attn_unbatched: decode_attn_unbatched().
  gqa_reference: gqa_reference().
  attn_forward_kernel: attn_forward_kernel().
  mqa_reference: mqa_reference().
  attn_forward_kernel._compute.body: attn_forward_kernel()._compute().body().
  mha_reference: mha_reference().
  attn_forward_kernel._compute: attn_forward_kernel()._compute().
  gqa: gqa().
  mqa: mqa().
  attn_forward_kernel._compute._dot: attn_forward_kernel()._compute()._dot().
---
# Module: [`jax/experimental/pallas/ops/gpu/decode_attention.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py)

## Functions
- `_compute(start_idx, kv_seq_len, o, m_i, l_i)` — [`L49`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L49)
- `_dot(a, b)` — [`L55`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L55)
- `attn_forward_kernel(q_ref, k_ref, v_ref, start_idx_ref, kv_seq_len_ref, o_ref: Any, *residual_refs: Any, sm_scale: float, block_k: int, block_h: int, num_heads: int)` — [`L28`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L28)
- `body(start_k, carry)` — [`L65`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L65)
- `decode_attn_unbatched(q, k, v, start_idx, kv_seq_len, sm_scale: float, block_h: int, block_k: int, k_splits: int, num_warps: int | None, num_stages: int, grid: tuple[int, ...] | None, interpret: bool, debug: bool, return_residuals: bool, normalize_output: bool)` — [`L131`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L131)
- `gqa(q, k, v, start_idx=None, kv_seq_len=None, sm_scale: float | None = None, block_h: int = 16, block_k: int = 128, k_splits: int = 16, num_warps: int | None = None, num_stages: int = 2, grid: tuple[int, ...] | None = None, interpret: bool = False, debug: bool = False, return_residuals: bool = False, normalize_output: bool = True)` — [`L304`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L304)
- `gqa_reference(q, k, v, start_idx=None, kv_seq_len=None, sm_scale=None, return_residuals=False, normalize_output=True)` — [`L450`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L450)
- `mha_reference(q, k, v, start_idx=None, kv_seq_len=None, sm_scale=None)` — [`L422`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L422)
- `mqa(q, k, v, start_idx=None, kv_seq_len=None, sm_scale: float | None = None, block_h: int = 16, block_k: int = 256, k_splits: int = 16, num_warps: int | None = None, num_stages: int = 2, grid: tuple[int, ...] | None = None, interpret: bool = False, debug: bool = False, return_residuals: bool = False, normalize_output: bool = True)` — [`L247`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L247)
- `mqa_reference(q, k, v, start_idx=None, kv_seq_len=None, sm_scale=None, return_residuals=False, normalize_output=True)` — [`L381`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/decode_attention.py#L381)

