---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_mla/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_mla/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_mla._interface`/
symbols:
  flash_mla: flash_mla().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_mla/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_interface.py)

## Functions
- `flash_mla(query: Float[Array, "batch seq_len q_heads q_head_dim"], key_value: Float[Array, "batch seq_len kv_lora_rank"], w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"], w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"], b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None, softmax_scale: float | None = None, causal: bool = False, cu_seqlens: Int[Array, num_seqs_plus_one] | None = None, attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None, softmax_aux: Float[Array, ...] | None = None, logits_soft_cap: float | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, dropout_prob: float = 0, sliding_window: int | tuple[int, int] | None = None, softmax_dtype: DTypeLike | None = None)` — [`L39`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_interface.py#L39) — Flash Multi-head Latent Attention on TPU using Pallas.

## Module values
- `__all__` — [`L205`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_interface.py#L205)

