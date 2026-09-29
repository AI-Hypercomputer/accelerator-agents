---
title: 'Module: src/maxdiffusion/models/lora_nnx.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/lora_nnx.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.lora_nnx`/
symbols:
  merge_lora_for_scanned: merge_lora_for_scanned().
  _merge_lora_layer: _merge_lora_layer().
  merge_lora: merge_lora().
  parse_lora_dict: parse_lora_dict().
  _to_jax_array: _to_jax_array().
  _compute_and_add_single_jit: _compute_and_add_single_jit().
  _compute_and_add_scanned_jit: _compute_and_add_scanned_jit().
---
# Module: [`src/maxdiffusion/models/lora_nnx.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py)

## Functions
- `_compute_and_add_scanned_jit(kernel, downs, ups, alphas, global_scale, w_diffs=None, b_diffs=None, bias=None)` — [`L55`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L55) — Applies scanned LoRA + Diffs.
- `_compute_and_add_single_jit(kernel, bias, down, up, scale, w_diff, b_diff)` — [`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L32) — Applies LoRA + Weight Diff + Bias Diff on device.
- `_merge_lora_layer(module, weights, scale)` — [`L150`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L150) — Merges LoRA weights into a single non-scanned layer.
- `_to_jax_array(v, dtype)` — [`L83`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L83)
- `merge_lora(model: nnx.Module, state_dict: dict, rank: int, scale: float, translate_fn=None, dtype: str = "float32")` — [`L248`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L248) — Merges weights for non-scanned layers (Embeddings, singular Dense, etc).
- `merge_lora_for_scanned(model: nnx.Module, state_dict: dict, rank: int, scale: float, translate_fn=None, dtype: str = "float32")` — [`L279`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L279) — Device-Side Optimized Merge for Scanned Layers.
- `parse_lora_dict(state_dict, dtype)` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora_nnx.py#L90) — Helper to parse state_dict into structured params including diffs.

