---
title: 'Module: ejkernel/kernels/_pallas/tpu/gated_delta_rule/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/gated_delta_rule/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.gated_delta_rule._interface`/gated_delta_rule().
symbols:
  gated_delta_rule: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/gated_delta_rule/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_interface.py)

## Functions
- `gated_delta_rule(query: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], key: Float[Array, "batch seq_len num_qk_heads qk_head_dim"], value: Float[Array, "batch seq_len num_value_heads v_head_dim"], beta: Float[Array, "batch seq_len num_value_heads"], decay: Float[Array, "batch seq_len num_value_heads"] | None = None, *, chunk_size: int = 256, initial_state: Float[Array, "batch num_value_heads qk_head_dim v_head_dim"] | None = None, use_qk_l2norm: bool = True, use_chunked: bool = True, use_input_dtype_phase1_outputs: bool = False, use_input_dtype_state: bool = False, seg_ids: Int[Array, "batch seq_len"] | None = None)` — [`L33`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_interface.py#L33) — Gated Delta Rule linear attention using TPU Pallas kernels.

