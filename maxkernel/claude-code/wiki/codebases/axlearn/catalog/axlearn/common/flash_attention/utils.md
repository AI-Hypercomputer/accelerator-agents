---
title: 'Module: axlearn/common/flash_attention/utils.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/utils.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.utils`/
symbols:
  flash_attention_implementation: flash_attention_implementation().
  BACKENDS: BACKENDS.
  PAGED_ATTN_BACKENDS: PAGED_ATTN_BACKENDS.
  _interpret: _interpret().
---
# Module: [`axlearn/common/flash_attention/utils.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils.py)

## Functions
- `_interpret(backend: str)` — [`L166`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils.py#L166)
- `flash_attention_implementation(backend: Literal["cpu", "tpu", "gpu", "xla", "neuron"], *, query: Tensor, key: Tensor, value: Tensor, bias: BaseAttentionBias, logit_sink: Optional[Tensor] = None, softmax_scale: float = 1, kv_cache_type: Optional[type[BaseKVCache]] = None, tpu_block_size: int = 512, gpu_block_size: int = 128, dropout_rate: Optional[float] = 0, page_tables: Optional[Tensor] = None, backend_overrides: Optional[dict[str, Any]] = None)` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils.py#L61) — Returns a jitted "flash" multihead-attention implementation for the given backend.

## Module values
- `BACKENDS` — [`L31`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils.py#L31)
- `PAGED_ATTN_BACKENDS` — [`L54`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils.py#L54)

