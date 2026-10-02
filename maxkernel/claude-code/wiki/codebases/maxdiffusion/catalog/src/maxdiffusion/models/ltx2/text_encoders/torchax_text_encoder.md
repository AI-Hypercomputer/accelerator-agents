---
title: 'Module: src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.text_encoders.torchax_text_encoder`/
symbols:
  TorchaxGemma3TextEncoder.__call__: TorchaxGemma3TextEncoder#__call__().
  _patched_sliding_window_overlay: _patched_sliding_window_overlay().
  _orig_sliding_window_overlay: _orig_sliding_window_overlay.
  _patched_sliding_window_overlay.inner_mask: _patched_sliding_window_overlay().inner_mask().
  TorchaxGemma3TextEncoder._forward_inner: TorchaxGemma3TextEncoder#_forward_inner().
  TorchaxGemma3TextEncoder: TorchaxGemma3TextEncoder#
  TorchaxGemma3TextEncoder.__init__: TorchaxGemma3TextEncoder#__init__().
---
# Module: [`src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py)

## Classes
### `TorchaxGemma3TextEncoder`
- def: [`src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py:47`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L47)
- doc: A jittable Torchax module for wrapping the HuggingFace PyTorch
- signature: `class TorchaxGemma3TextEncoder(interop.JittableModule):`
- protocol/private: `__call__`[`L56`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L56), `__init__`[`L53`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L53), `_forward_inner`[`L80`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L80)
- uses (calls/refs, reference-scoped): [`_patched_sliding_window_overlay`](torchax_text_encoder.md#_patched_sliding_window_overlay), [`_orig_sliding_window_overlay`](torchax_text_encoder.md#_orig_sliding_window_overlay)

## Functions
- `_patched_sliding_window_overlay(sliding_window: int)` — [`L29`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L29)
- `inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int)` — [`L32`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L32)

## Module values
- `_orig_sliding_window_overlay` — [`L26`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/torchax_text_encoder.py#L26)

