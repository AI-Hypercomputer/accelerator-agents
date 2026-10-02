---
title: 'Module: axlearn/common/poolings.py'
type: catalog
provenance: extracted
module: axlearn/common/poolings.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.poolings`/
symbols:
  BasePoolingLayer: BasePoolingLayer#
  BasePoolingLayer.Config: BasePoolingLayer#Config#
  BasePoolingLayer.Config.input_dim: BasePoolingLayer#Config#input_dim.
  BasePoolingLayer.Config.output_dim: BasePoolingLayer#Config#output_dim.
  BasePoolingLayer.Config.num_outputs: BasePoolingLayer#Config#num_outputs.
  BasePoolingLayer.forward: BasePoolingLayer#forward().
  AttentionPooling: AttentionPooling#
  AttentionPooling.Config: AttentionPooling#Config#
  AttentionPooling.Config.cross_attention: AttentionPooling#Config#cross_attention.
  AttentionPooling.Config.feed_forward: AttentionPooling#Config#feed_forward.
  AttentionPooling.default_config: AttentionPooling#default_config().
  AttentionPooling.__init__: AttentionPooling#__init__().
  AttentionPooling.forward: AttentionPooling#forward().
  AttentionPooling._create_layer_parameter_specs: AttentionPooling#_create_layer_parameter_specs().
  AveragePooling: AveragePooling#
  AveragePooling.Config: AveragePooling#Config#
  AveragePooling.Config.eps: AveragePooling#Config#eps.
  AveragePooling.forward: AveragePooling#forward().
  MaxPooling: MaxPooling#
  MaxPooling.forward: MaxPooling#forward().
  FirstNTokenPooling: FirstNTokenPooling#
  FirstNTokenPooling.forward: FirstNTokenPooling#forward().
  LastNTokenPooling: LastNTokenPooling#
  LastNTokenPooling.forward: LastNTokenPooling#forward().
  PoolingWithProjection: PoolingWithProjection#
  PoolingWithProjection.Config: PoolingWithProjection#Config#
  PoolingWithProjection.Config.pooler: PoolingWithProjection#Config#pooler.
  PoolingWithProjection.Config.proj: PoolingWithProjection#Config#proj.
  PoolingWithProjection.__init__: PoolingWithProjection#__init__().
  PoolingWithProjection.forward: PoolingWithProjection#forward().
---
# Module: [`axlearn/common/poolings.py`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py)

## Classes
### `AttentionPooling`
- def: [`axlearn/common/poolings.py:55`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L55)
- doc: Attention-based pooling.
- signature: `class AttentionPooling(BasePoolingLayer):`
- members:
  - `default_config(cls)` — [`L69`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L69)
  - `forward(self, tokens: Tensor, paddings: Tensor=None)` — [`L84`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L84) — Args:
- protocol/private: `__init__`[`L77`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L77), `_create_layer_parameter_specs`[`L113`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L113)

### `AveragePooling`
- def: [`axlearn/common/poolings.py:125`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L125)
- doc: Average pooling layer.
- signature: `class AveragePooling(BasePoolingLayer):`
- members:
  - `forward(self, tokens: Tensor, paddings: Optional[Tensor]=None)` — [`L133`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L133) — Args:

### `BasePoolingLayer`
- def: [`axlearn/common/poolings.py:30`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L30)
- doc: The base class of a pooling layer.
- signature: `class BasePoolingLayer(BaseLayer):`
- members:
  - `forward(self, tokens: Tensor, paddings: Tensor=None)` — [`L42`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L42) — Args:

### `Config`
- def: [`axlearn/common/poolings.py:287`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L287)
- signature: `class Config(BasePoolingLayer.Config):`
- members:
  - `cross_attention` — [`L65`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L65)
  - `eps` — [`L131`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L131)
  - `feed_forward` — [`L66`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L66)
  - `input_dim` — [`L36`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L36)
  - `num_outputs` — [`L40`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L40)
  - `output_dim` — [`L37`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L37)
  - `pooler` — [`L290`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L290)
  - `proj` — [`L292`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L292)

### `FirstNTokenPooling`
- def: [`axlearn/common/poolings.py:195`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L195)
- doc: Take the first N tokens as the pooler output.
- signature: `class FirstNTokenPooling(BasePoolingLayer):`
- members:
  - `forward(self, tokens: Tensor, paddings: Optional[Tensor]=None)` — [`L198`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L198) — Computes pooling from first N tokens.

### `LastNTokenPooling`
- def: [`axlearn/common/poolings.py:232`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L232)
- doc: Take the last N tokens as the pooler output.
- signature: `class LastNTokenPooling(BasePoolingLayer):`
- members:
  - `forward(self, tokens: Tensor, paddings: Optional[Tensor]=None)` — [`L235`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L235) — Computes pooling from last N tokens.

### `MaxPooling`
- def: [`axlearn/common/poolings.py:164`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L164)
- doc: Max pooling layer.
- signature: `class MaxPooling(BasePoolingLayer):`
- members:
  - `forward(self, tokens: Tensor, paddings: Optional[Tensor]=None)` — [`L167`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L167) — Args:

### `PoolingWithProjection`
- def: [`axlearn/common/poolings.py:283`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L283)
- doc: Composite pooler containing a regular pooler followed by a projection.
- signature: `class PoolingWithProjection(BasePoolingLayer):`
- members:
  - `forward(self, tokens: Tensor, paddings: Tensor=None)` — [`L312`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L312) — See BasePoolingLayer.forward docstring for details.
- protocol/private: `__init__`[`L294`](../../../../../../raw/code/axlearn/axlearn/common/poolings.py#L294)

