---
title: 'Module: src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.utils.skip_layer_strategy`/SkipLayerStrategy#
symbols:
  SkipLayerStrategy: ''
  SkipLayerStrategy.Residual: Residual.
  SkipLayerStrategy.AttentionSkip: AttentionSkip.
  SkipLayerStrategy.AttentionValues: AttentionValues.
  SkipLayerStrategy.TransformerBlock: TransformerBlock.
---
# Module: [`src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py)

## Classes
### `SkipLayerStrategy`  ·  implements/extends Enum
- def: [`src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py:20`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py#L20)
- signature: `class SkipLayerStrategy(Enum):`
- members:
  - `AttentionSkip` — [`L21`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py#L21)
  - `AttentionValues` — [`L22`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py#L22)
  - `Residual` — [`L23`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py#L23)
  - `TransformerBlock` — [`L24`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/skip_layer_strategy.py#L24)
- used by: [`__call__`](../transformers_pytorch/attention.md#AttnProcessor2_0.__call__), [`forward`](../transformers_pytorch/transformer3d.md#Transformer3DModel.forward), [`forward`](../transformers_pytorch/attention.md#BasicTransformerBlock.forward), [`forward`](../transformers_pytorch/attention.md#Attention.forward)

