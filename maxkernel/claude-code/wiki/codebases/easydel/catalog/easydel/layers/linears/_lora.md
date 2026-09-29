---
title: 'Module: easydel/layers/linears/_lora.py'
type: catalog
provenance: extracted
module: easydel/layers/linears/_lora.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.linears._lora`/eLoRA#
symbols:
  eLoRA.__call__: __call__().
  eLoRA.native_forward: native_forward().
  eLoRA: ''
---
# Module: [`easydel/layers/linears/_lora.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_lora.py)

## Classes
### `eLoRA`  ·  implements/extends LoRA
- def: [`easydel/layers/linears/_lora.py:38`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_lora.py#L38)
- doc: LoRA wrapper that behaves like the wrapped EasyDeL linear module.
- signature: `class eLoRA(nn.LoRA):`
- members:
  - `__call__(self, x: jax.Array, *args, **kwargs)` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_lora.py#L52) — Apply the LoRA update and delegate extra call arguments to the base module.
  - `native_forward(self, inputs: jax.Array, *, w: jax.Array | None = None)` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_lora.py#L69) — Project through LoRA using EasyDeL's trace-safe linear-layer contract.
- uses (calls/refs, reference-scoped): `base_module`, `lora_a`, `lora_b`, `dtype`, `promote_dtype`
- used by: [`apply_lora_to_layers`](../../infra/utils.md#apply_lora_to_layers), [`_rebuild_lora_modules_from_checkpoint`](../../infra/mixins/bridge.md#_rebuild_lora_modules_from_checkpoint)

