---
title: 'Module: src/maxdiffusion/models/resnet_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/resnet_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.resnet_flax`/
symbols:
  FlaxResnetBlock2D.conv_shortcut: FlaxResnetBlock2D#conv_shortcut.
  FlaxResnetBlock2D.__call__: FlaxResnetBlock2D#__call__().
  NdInitializer: NdInitializer.
  FlaxResnetBlock2D.dtype: FlaxResnetBlock2D#dtype.
  FlaxResnetBlock2D.weights_dtype: FlaxResnetBlock2D#weights_dtype.
  Initializer: Initializer.
  FlaxUpsample2D.setup: FlaxUpsample2D#setup().
  FlaxDownsample2D.setup: FlaxDownsample2D#setup().
  FlaxResnetBlock2D.out_channels: FlaxResnetBlock2D#out_channels.
  FlaxResnetBlock2D.precision: FlaxResnetBlock2D#precision.
  FlaxResnetBlock2D.in_channels: FlaxResnetBlock2D#in_channels.
  FlaxResnetBlock2D.norm1: FlaxResnetBlock2D#norm1.
  FlaxResnetBlock2D.norm2: FlaxResnetBlock2D#norm2.
  FlaxResnetBlock2D.dropout: FlaxResnetBlock2D#dropout.
  FlaxResnetBlock2D.conv1: FlaxResnetBlock2D#conv1.
  FlaxResnetBlock2D.time_emb_proj: FlaxResnetBlock2D#time_emb_proj.
  FlaxResnetBlock2D.conv2: FlaxResnetBlock2D#conv2.
  FlaxResnetBlock2D.norm_num_groups: FlaxResnetBlock2D#norm_num_groups.
  FlaxResnetBlock2D: FlaxResnetBlock2D#
  FlaxResnetBlock2D.dropout_prob: FlaxResnetBlock2D#dropout_prob.
  FlaxUpsample2D.__call__: FlaxUpsample2D#__call__().
  FlaxResnetBlock2D.setup: FlaxResnetBlock2D#setup().
  Array: Array.
  FlaxUpsample2D: FlaxUpsample2D#
  FlaxUpsample2D.dtype: FlaxUpsample2D#dtype.
  FlaxDownsample2D: FlaxDownsample2D#
  FlaxDownsample2D.dtype: FlaxDownsample2D#dtype.
  FlaxDownsample2D.weights_dtype: FlaxDownsample2D#weights_dtype.
  Activation: Activation.
  FlaxDownsample2D.__call__: FlaxDownsample2D#__call__().
  DType: DType.
  PRNGKey: PRNGKey.
  Shape: Shape.
  InitializerAxis: InitializerAxis.
  FlaxUpsample2D.weights_dtype: FlaxUpsample2D#weights_dtype.
  FlaxUpsample2D.precision: FlaxUpsample2D#precision.
  FlaxResnetBlock2D.use_nin_shortcut: FlaxResnetBlock2D#use_nin_shortcut.
  FlaxUpsample2D.out_channels: FlaxUpsample2D#out_channels.
  FlaxUpsample2D.conv: FlaxUpsample2D#conv.
  FlaxDownsample2D.out_channels: FlaxDownsample2D#out_channels.
  FlaxDownsample2D.precision: FlaxDownsample2D#precision.
  FlaxDownsample2D.conv: FlaxDownsample2D#conv.
  Dtype: Dtype.
---
# Module: [`src/maxdiffusion/models/resnet_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py)

## Classes
### `FlaxDownsample2D`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/resnet_flax.py:71`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L71)
- signature: `class FlaxDownsample2D(nn.Module):`
- members:
  - `setup(self)` — [`L77`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L77)
  - `conv` — [`L78`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L78)
  - `dtype` — [`L73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L73)
  - `out_channels` — [`L72`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L72)
  - `precision` — [`L75`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L75)
  - `weights_dtype` — [`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L74)
- protocol/private: `__call__`[`L92`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L92)
- used by: [`downsamplers_0`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.downsamplers_0), [`downsamplers_0`](unet_2d_blocks_flax.md#FlaxDownBlock2D.downsamplers_0)

### `FlaxResnetBlock2D`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/resnet_flax.py:98`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L98)
- signature: `class FlaxResnetBlock2D(nn.Module):`
- members:
  - `setup(self)` — [`L108`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L108)
  - `conv1` — [`L137`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L137) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `conv2` — [`L152`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L152) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `conv_shortcut` — [`L122`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L122) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `dropout` — [`L118`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L118) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `dropout_prob` — [`L101`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L101) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `dtype` — [`L103`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L103) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `in_channels` — [`L99`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L99) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `norm1` — [`L111`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L111) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `norm2` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L115) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `norm_num_groups` — [`L105`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L105) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `out_channels` — [`L100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L100) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `precision` — [`L106`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L106) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `time_emb_proj` — [`L150`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L150) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `use_nin_shortcut` — [`L102`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L102) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
  - `weights_dtype` — [`L104`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L104) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
- protocol/private: `__call__`[`L165`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L165)
- used by: [`setup`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.setup), [`setup`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.setup), [`setup`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.setup), [`setup`](unet_2d_blocks_flax.md#FlaxUpBlock2D.setup), [`setup`](unet_2d_blocks_flax.md#FlaxDownBlock2D.setup)

### `FlaxUpsample2D`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/resnet_flax.py:37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L37)
- signature: `class FlaxUpsample2D(nn.Module):`
- members:
  - `setup(self)` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L43)
  - `conv` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L44)
  - `dtype` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L39)
  - `out_channels` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L38)
  - `precision` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L41)
  - `weights_dtype` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L40)
- protocol/private: `__call__`[`L58`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L58)
- used by: [`upsamplers_0`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.upsamplers_0), [`upsamplers_0`](unet_2d_blocks_flax.md#FlaxUpBlock2D.upsamplers_0)

## Module values
- `Activation` — [`L30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L30)
- `Array` — [`L25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L25) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
- `DType` — [`L26`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L26) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
- `Dtype` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L27)
- `Initializer` — [`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L32)
- `InitializerAxis` — [`L33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L33) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
- `NdInitializer` — [`L34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L34) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
- `PRNGKey` — [`L28`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L28) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)
- `Shape` — [`L29`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/resnet_flax.py#L29) — documented in [maxdiffusion-models-resnet_flax](../../../../concepts/maxdiffusion-models-resnet_flax.md)

