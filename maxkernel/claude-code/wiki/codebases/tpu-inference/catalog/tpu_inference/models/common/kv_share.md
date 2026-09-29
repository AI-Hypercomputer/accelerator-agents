---
title: 'Module: tpu_inference/models/common/kv_share.py'
type: catalog
provenance: extracted
module: tpu_inference/models/common/kv_share.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.common.kv_share`/compute_
symbols:
  compute_kv_share_map: kv_share_map().
  compute_mtp_kv_share_map: mtp_kv_share_map().
---
# Module: [`tpu_inference/models/common/kv_share.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/kv_share.py)

## Functions
- `compute_kv_share_map(text_config)` — [`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/kv_share.py#L28) — Return `{shared_layer_idx: source_layer_idx}` for KV-shared layers.
- `compute_mtp_kv_share_map(draft_config, target_config)` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/kv_share.py#L69) — Return `{'draft_layer.i': 'layer.j'}` mapping speculative MTP layers

