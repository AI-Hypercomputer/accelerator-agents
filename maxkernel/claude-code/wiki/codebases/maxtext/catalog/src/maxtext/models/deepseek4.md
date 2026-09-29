---
title: 'Module: src/maxtext/models/deepseek4.py'
type: catalog
provenance: extracted
module: src/maxtext/models/deepseek4.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.deepseek4`/DeepSeek4
symbols:
  DeepSeek4ScannableBlock.layers_0: ScannableBlock#layers_0.
  DeepSeek4ScannableBlock.layers_1: ScannableBlock#layers_1.
  DeepSeek4ScannableBlock.__call__: ScannableBlock#__call__().
  DeepSeek4DecoderLayer.mlp_op: DecoderLayer#mlp_op().
  DeepSeek4DecoderLayer: DecoderLayer#
  DeepSeek4DecoderLayer.__call__: DecoderLayer#__call__().
  DeepSeek4LayerToLinen: LayerToLinen.
  DeepSeek4ScannableBlockToLinen: ScannableBlockToLinen.
  DeepSeek4ScannableBlock.config: ScannableBlock#config.
  DeepSeek4ScannableBlock.mesh: ScannableBlock#mesh.
  DeepSeek4ScannableBlock.model_mode: ScannableBlock#model_mode.
  DeepSeek4ScannableBlock.quant: ScannableBlock#quant.
  DeepSeek4ScannableBlock.rngs: ScannableBlock#rngs.
  DeepSeek4DecoderLayer.mlp: DecoderLayer#mlp.
  DeepSeek4ScannableBlock: ScannableBlock#
  DeepSeek4DecoderLayer.__init__: DecoderLayer#__init__().
  DeepSeek4DecoderLayer.self_attention: DecoderLayer#self_attention.
  DeepSeek4ScannableBlock.__init__: ScannableBlock#__init__().
---
# Module: [`src/maxtext/models/deepseek4.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py)

## Classes
### `DeepSeek4DecoderLayer`  ·  implements/extends DeepSeekGenericLayer
- def: [`src/maxtext/models/deepseek4.py:34`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L34)
- doc: DeepSeek-V4 specific decoder layer.
- signature: `class DeepSeek4DecoderLayer(deepseek.DeepSeekGenericLayer):`
- members:
  - `mlp_op(self, inputs, deterministic, *args, **kwargs)` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L122)
  - `mlp` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L81)
  - `self_attention` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L96)
- protocol/private: `__call__`[`L130`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L130), `__init__`[`L53`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L53)
- used by: [`layers_0`](deepseek4.md#DeepSeek4ScannableBlock.layers_0), [`layers_1`](deepseek4.md#DeepSeek4ScannableBlock.layers_1), [`DeepSeek4LayerToLinen`](deepseek4.md#DeepSeek4LayerToLinen)

### `DeepSeek4ScannableBlock`  ·  implements/extends Module
- def: [`src/maxtext/models/deepseek4.py:173`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L173)
- doc: A scannable block containing exactly two DeepSeek V4 layers (HCA and CSA).
- signature: `class DeepSeek4ScannableBlock(nnx.Module):`
- members:
  - `config` — [`L189`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L189)
  - `layers_0` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L196)
  - `layers_1` — [`L207`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L207)
  - `mesh` — [`L190`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L190)
  - `model_mode` — [`L191`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L191)
  - `quant` — [`L192`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L192)
  - `rngs` — [`L193`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L193)
- protocol/private: `__call__`[`L217`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L217), `__init__`[`L181`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L181)
- uses (calls/refs, reference-scoped): [`DeepSeek4DecoderLayer`](deepseek4.md#DeepSeek4DecoderLayer)
- used by: [`DeepSeek4ScannableBlockToLinen`](deepseek4.md#DeepSeek4ScannableBlockToLinen)

## Module values
- `DeepSeek4LayerToLinen` — [`L263`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L263)
- `DeepSeek4ScannableBlockToLinen` — [`L268`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek4.py#L268)

