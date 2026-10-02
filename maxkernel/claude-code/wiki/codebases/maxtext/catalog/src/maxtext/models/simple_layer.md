---
title: 'Module: src/maxtext/models/simple_layer.py'
type: catalog
provenance: extracted
module: src/maxtext/models/simple_layer.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.simple_layer`/Simple
symbols:
  SimpleMlpDecoderLayer.__call__: MlpDecoderLayer#__call__().
  SimpleMlpDecoderLayer.ff_1: MlpDecoderLayer#ff_1.
  SimpleDecoderLayer.__call__: DecoderLayer#__call__().
  SimpleDecoderLayer.weights: DecoderLayer#weights.
  SimpleMlpDecoderLayer.ff_2: MlpDecoderLayer#ff_2.
  SimpleMlpDecoderLayer.config: MlpDecoderLayer#config.
  SimpleDecoderLayer.out_sharding: DecoderLayer#out_sharding.
  SimpleDecoderLayer.config: DecoderLayer#config.
  SimpleDecoderLayer.quant: DecoderLayer#quant.
  SimpleDecoderLayerToLinen: DecoderLayerToLinen.
  SimpleMlpDecoderLayer.quant: MlpDecoderLayer#quant.
  SimpleMlpDecoderLayerToLinen: MlpDecoderLayerToLinen.
  SimpleDecoderLayer.mesh: DecoderLayer#mesh.
  SimpleMlpDecoderLayer.mesh: MlpDecoderLayer#mesh.
  SimpleMlpDecoderLayer.rngs: MlpDecoderLayer#rngs.
  SimpleDecoderLayer: DecoderLayer#
  SimpleDecoderLayer.rngs: DecoderLayer#rngs.
  SimpleMlpDecoderLayer: MlpDecoderLayer#
  SimpleMlpDecoderLayer.activation_sharding: MlpDecoderLayer#activation_sharding.
  SimpleMlpDecoderLayer.mlp_sharding: MlpDecoderLayer#mlp_sharding.
  SimpleDecoderLayer.__init__: DecoderLayer#__init__().
  SimpleDecoderLayer.model_mode: DecoderLayer#model_mode.
  SimpleMlpDecoderLayer.__init__: MlpDecoderLayer#__init__().
  SimpleMlpDecoderLayer.model_mode: MlpDecoderLayer#model_mode.
---
# Module: [`src/maxtext/models/simple_layer.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py)

## Classes
### `SimpleDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/simple_layer.py:31`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L31)
- doc: Decoder layer consisting of a single \[embed, embed\] weight matrix.
- signature: `class SimpleDecoderLayer(nnx.Module):`
- members:
  - `config` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L43)
  - `mesh` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L44)
  - `model_mode` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L45)
  - `out_sharding` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L56)
  - `quant` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L47)
  - `rngs` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L46)
  - `weights` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L51)
- protocol/private: `__call__`[`L60`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L60), `__init__`[`L34`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L34)
- used by: [`SimpleDecoderLayerToLinen`](simple_layer.md#SimpleDecoderLayerToLinen)

### `SimpleMlpDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/simple_layer.py:87`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L87)
- doc: Decoder layer consisting of \[embed,mlp\] followed by an \[mlp,embed\] matmul.
- signature: `class SimpleMlpDecoderLayer(nnx.Module):`
- members:
  - `activation_sharding` — [`L118`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L118)
  - `config` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L99)
  - `ff_1` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L107)
  - `ff_2` — [`L113`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L113)
  - `mesh` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L100)
  - `mlp_sharding` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L122)
  - `model_mode` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L101)
  - `quant` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L103)
  - `rngs` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L102)
- protocol/private: `__call__`[`L124`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L124), `__init__`[`L90`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L90)
- used by: [`SimpleMlpDecoderLayerToLinen`](simple_layer.md#SimpleMlpDecoderLayerToLinen)

## Module values
- `SimpleDecoderLayerToLinen` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L81)
- `SimpleMlpDecoderLayerToLinen` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/models/simple_layer.py#L147)

