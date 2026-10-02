---
title: 'Module: axlearn/common/convolution.py'
type: catalog
provenance: extracted
module: axlearn/common/convolution.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.convolution`/
symbols:
  Conv2D.default_config: Conv2D#default_config().
  ConvPaddingType: ConvPaddingType.
  Conv1D: Conv1D#
  Conv2D: Conv2D#
  Conv2DWith1DPadding: Conv2DWith1DPadding#
  Conv3D: Conv3D#
  Conv2DTranspose: Conv2DTranspose#
  SUPPORT_CONV_PADDING: SUPPORT_CONV_PADDING.
  _check_conv_cfg: _check_conv_cfg().
  BaseConv: BaseConv#
  BaseConv.Config: BaseConv#Config#
  BaseConv.Config.input_dim: BaseConv#Config#input_dim.
  BaseConv.Config.num_input_dim_groups: BaseConv#Config#num_input_dim_groups.
  BaseConv._compute_fan_axes: BaseConv#_compute_fan_axes().
  conv_dilate_window: conv_dilate_window().
  conv_explicit_padding: conv_explicit_padding().
  conv_output_shape: conv_output_shape().
  compute_conv_paddings: compute_conv_paddings().
  Conv1D.Config: Conv1D#Config#
  Conv1D.Config.window: Conv1D#Config#window.
  Conv1D.Config.strides: Conv1D#Config#strides.
  Conv1D.Config.padding: Conv1D#Config#padding.
  Conv1D.Config.output_dim: Conv1D#Config#output_dim.
  Conv1D.Config.bias: Conv1D#Config#bias.
  Conv1D.Config.dilation: Conv1D#Config#dilation.
  Conv1D.Config.input_partition_spec: Conv1D#Config#input_partition_spec.
  Conv1D.Config.output_partition_spec: Conv1D#Config#output_partition_spec.
  Conv1D.default_config: Conv1D#default_config().
  Conv1D._create_layer_parameter_specs: Conv1D#_create_layer_parameter_specs().
  Conv1D.forward: Conv1D#forward().
  Conv1D._conv: Conv1D#_conv().
  Conv1D.output_shape: Conv1D#output_shape().
  Conv1DWithPadding: Conv1DWithPadding#
  Conv1DWithPadding.Config: Conv1DWithPadding#Config#
  Conv1DWithPadding.Config.anchor: Conv1DWithPadding#Config#anchor.
  Conv1DWithPadding.forward: Conv1DWithPadding#forward().
  Conv1DWithPadding.conv_paddings: Conv1DWithPadding#conv_paddings().
  Conv2D.Config: Conv2D#Config#
  Conv2D.Config.window: Conv2D#Config#window.
  Conv2D.Config.strides: Conv2D#Config#strides.
  Conv2D.Config.padding: Conv2D#Config#padding.
  Conv2D.Config.dilation: Conv2D#Config#dilation.
  Conv2D.Config.output_dim: Conv2D#Config#output_dim.
  Conv2D.Config.bias: Conv2D#Config#bias.
  Conv2D._create_layer_parameter_specs: Conv2D#_create_layer_parameter_specs().
  Conv2D.forward: Conv2D#forward().
  Conv2D._conv: Conv2D#_conv().
  Conv2D.output_shape: Conv2D#output_shape().
  Conv2DWith1DPadding.Config: Conv2DWith1DPadding#Config#
  Conv2DWith1DPadding.Config.anchor: Conv2DWith1DPadding#Config#anchor.
  Conv2DWith1DPadding.forward: Conv2DWith1DPadding#forward().
  Conv2DWith1DPadding.conv_paddings: Conv2DWith1DPadding#conv_paddings().
  Conv3D.Config: Conv3D#Config#
  Conv3D.Config.window: Conv3D#Config#window.
  Conv3D.Config.strides: Conv3D#Config#strides.
  Conv3D.Config.padding: Conv3D#Config#padding.
  Conv3D.Config.dilation: Conv3D#Config#dilation.
  Conv3D.Config.output_dim: Conv3D#Config#output_dim.
  Conv3D.Config.bias: Conv3D#Config#bias.
  Conv3D.default_config: Conv3D#default_config().
  Conv3D._create_layer_parameter_specs: Conv3D#_create_layer_parameter_specs().
  Conv3D.forward: Conv3D#forward().
  Conv3D._conv: Conv3D#_conv().
  Conv3D.output_shape: Conv3D#output_shape().
  conv_transpose_explicit_padding: conv_transpose_explicit_padding().
  conv_transpose_output_shape: conv_transpose_output_shape().
  compute_conv_transpose_paddings: compute_conv_transpose_paddings().
  Conv1DTranspose: Conv1DTranspose#
  Conv1DTranspose.Config: Conv1DTranspose#Config#
  Conv1DTranspose.Config.window: Conv1DTranspose#Config#window.
  Conv1DTranspose.Config.strides: Conv1DTranspose#Config#strides.
  Conv1DTranspose.Config.padding: Conv1DTranspose#Config#padding.
  Conv1DTranspose.Config.dilation: Conv1DTranspose#Config#dilation.
  Conv1DTranspose.Config.output_dim: Conv1DTranspose#Config#output_dim.
  Conv1DTranspose.Config.bias: Conv1DTranspose#Config#bias.
  Conv1DTranspose.Config.anchor: Conv1DTranspose#Config#anchor.
  Conv1DTranspose.default_config: Conv1DTranspose#default_config().
  Conv1DTranspose._create_layer_parameter_specs: Conv1DTranspose#_create_layer_parameter_specs().
  Conv1DTranspose.forward: Conv1DTranspose#forward().
  Conv1DTranspose._conv: Conv1DTranspose#_conv().
  Conv1DTranspose.conv_paddings: Conv1DTranspose#conv_paddings().
  Conv1DTranspose.output_shape: Conv1DTranspose#output_shape().
  Conv2DTranspose.Config: Conv2DTranspose#Config#
  Conv2DTranspose.Config.window: Conv2DTranspose#Config#window.
  Conv2DTranspose.Config.strides: Conv2DTranspose#Config#strides.
  Conv2DTranspose.Config.padding: Conv2DTranspose#Config#padding.
  Conv2DTranspose.Config.dilation: Conv2DTranspose#Config#dilation.
  Conv2DTranspose.Config.output_dim: Conv2DTranspose#Config#output_dim.
  Conv2DTranspose.Config.bias: Conv2DTranspose#Config#bias.
  Conv2DTranspose.Config.transpose_kernel: Conv2DTranspose#Config#transpose_kernel.
  Conv2DTranspose.default_config: Conv2DTranspose#default_config().
  Conv2DTranspose._create_layer_parameter_specs: Conv2DTranspose#_create_layer_parameter_specs().
  Conv2DTranspose.forward: Conv2DTranspose#forward().
  Conv2DTranspose._conv: Conv2DTranspose#_conv().
  Conv2DTranspose.output_shape: Conv2DTranspose#output_shape().
  Conv2DTransposeWith1DPadding: Conv2DTransposeWith1DPadding#
  Conv2DTransposeWith1DPadding.Config: Conv2DTransposeWith1DPadding#Config#
  Conv2DTransposeWith1DPadding.Config.transpose_kernel: Conv2DTransposeWith1DPadding#Config#transpose_kernel.
  Conv2DTransposeWith1DPadding.Config.anchor: Conv2DTransposeWith1DPadding#Config#anchor.
  Conv2DTransposeWith1DPadding.default_config: Conv2DTransposeWith1DPadding#default_config().
  Conv2DTransposeWith1DPadding.forward: Conv2DTransposeWith1DPadding#forward().
  Conv2DTransposeWith1DPadding.conv_paddings: Conv2DTransposeWith1DPadding#conv_paddings().
  Conv3DTranspose: Conv3DTranspose#
  Conv3DTranspose.Config: Conv3DTranspose#Config#
  Conv3DTranspose.Config.window: Conv3DTranspose#Config#window.
  Conv3DTranspose.Config.strides: Conv3DTranspose#Config#strides.
  Conv3DTranspose.Config.padding: Conv3DTranspose#Config#padding.
  Conv3DTranspose.Config.dilation: Conv3DTranspose#Config#dilation.
  Conv3DTranspose.Config.output_dim: Conv3DTranspose#Config#output_dim.
  Conv3DTranspose.Config.bias: Conv3DTranspose#Config#bias.
  Conv3DTranspose.default_config: Conv3DTranspose#default_config().
  Conv3DTranspose._create_layer_parameter_specs: Conv3DTranspose#_create_layer_parameter_specs().
  Conv3DTranspose.forward: Conv3DTranspose#forward().
  Conv3DTranspose._conv: Conv3DTranspose#_conv().
  Conv3DTranspose.output_shape: Conv3DTranspose#output_shape().
  StackOverTime: StackOverTime#
  StackOverTime.Config: StackOverTime#Config#
  StackOverTime.Config.stride: StackOverTime#Config#stride.
  StackOverTime.Config.padding: StackOverTime#Config#padding.
  StackOverTime.forward: StackOverTime#forward().
  StackOverTime.conv_paddings: StackOverTime#conv_paddings().
  StackOverTime.output_shape: StackOverTime#output_shape().
---
# Module: [`axlearn/common/convolution.py`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py)

## Classes
### `BaseConv`
- def: [`axlearn/common/convolution.py:68`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L68)
- doc: Base class for convolution layers.
- signature: `class BaseConv(BaseLayer):`
- protocol/private: `_compute_fan_axes`[`L87`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L87)

### `Config`
- def: [`axlearn/common/convolution.py:1795`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1795)
- doc: Configures StackOverTime.
- signature: `class Config(BaseLayer.Config):`
- members:
  - `anchor` — [`L467`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L467)
  - `anchor` — [`L698`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L698)
  - `anchor` — [`L1390`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1390)
  - `anchor` — [`L1637`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1637)
  - `bias` — [`L353`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L353)
  - `bias` — [`L533`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L533)
  - `bias` — [`L765`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L765)
  - `bias` — [`L1384`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1384)
  - `bias` — [`L1519`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1519)
  - `bias` — [`L1699`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1699)
  - `dilation` — [`L356`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L356)
  - `dilation` — [`L531`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L531)
  - `dilation` — [`L762`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L762)
  - `dilation` — [`L1382`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1382)
  - `dilation` — [`L1517`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1517)
  - `dilation` — [`L1696`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1696)
  - `input_dim` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L75)
  - `input_partition_spec` — [`L358`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L358)
  - `num_input_dim_groups` — [`L84`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L84)
  - `output_dim` — [`L352`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L352)
  - `output_dim` — [`L532`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L532)
  - `output_dim` — [`L764`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L764)
  - `output_dim` — [`L1383`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1383)
  - `output_dim` — [`L1518`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1518)
  - `output_dim` — [`L1698`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1698)
  - `output_partition_spec` — [`L360`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L360)
  - `padding` — [`L351`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L351)
  - `padding` — [`L529`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L529)
  - `padding` — [`L756`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L756)
  - `padding` — [`L1381`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1381)
  - `padding` — [`L1516`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1516)
  - `padding` — [`L1695`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1695)
  - `padding` — [`L1805`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1805)
  - `stride` — [`L1798`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1798)
  - `strides` — [`L348`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L348)
  - `strides` — [`L526`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L526)
  - `strides` — [`L753`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L753)
  - `strides` — [`L1380`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1380)
  - `strides` — [`L1515`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1515)
  - `strides` — [`L1693`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1693)
  - `transpose_kernel` — [`L1523`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1523)
  - `transpose_kernel` — [`L1631`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1631)
  - `window` — [`L347`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L347)
  - `window` — [`L525`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L525)
  - `window` — [`L752`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L752)
  - `window` — [`L1379`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1379)
  - `window` — [`L1514`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1514)
  - `window` — [`L1692`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1692)

### `Conv1D`
- def: [`axlearn/common/convolution.py:336`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L336)
- doc: The 1D convolution layer.
- signature: `class Conv1D(BaseConv):`
- members:
  - `default_config(cls)` — [`L363`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L363)
  - `forward(self, x: Tensor)` — [`L396`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L396)
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L433`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L433)
- protocol/private: `_conv`[`L410`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L410), `_create_layer_parameter_specs`[`L368`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L368)
- used by: [`Conv1D`](layers.md#Conv1D)

### `Conv1DTranspose`
- def: [`axlearn/common/convolution.py:1372`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1372)
- doc: The 1D transposed convolution layer.
- signature: `class Conv1DTranspose(BaseConv):`
- members:
  - `conv_paddings(self, paddings: Tensor)` — [`L1474`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1474)
  - `default_config(cls)` — [`L1393`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1393)
  - `forward(self, x: Tensor, *, paddings: Optional[Tensor]=None)` — [`L1422`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1422)
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L1485`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1485)
- protocol/private: `_conv`[`L1450`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1450), `_create_layer_parameter_specs`[`L1398`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1398)

### `Conv1DWithPadding`
- def: [`axlearn/common/convolution.py:455`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L455)
- doc: The 1-D convolution with 1-D padding on the time axis.
- signature: `class Conv1DWithPadding(Conv1D):`
- members:
  - `conv_paddings(self, paddings: Tensor)` — [`L500`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L500)
  - `forward(self, x: Tensor, *, paddings: Tensor)` — [`L471`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L471) — Computes convolution outputs and paddings.

### `Conv2D`
- def: [`axlearn/common/convolution.py:514`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L514)
- doc: The 2-D convolution layer.
- signature: `class Conv2D(BaseConv):`
- members:
  - `default_config(cls)` — [`L536`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L536)
  - `forward(self, x: Tensor)` — [`L566`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L566)
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L596`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L596)
- protocol/private: `_conv`[`L573`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L573), `_create_layer_parameter_specs`[`L541`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L541)
- used by: [`__init__`](layers.md#SqueezeExcitation.__init__), [`Conv2D`](layers.md#Conv2D)

### `Conv2DTranspose`
- def: [`axlearn/common/convolution.py:1507`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1507)
- doc: The 2-D transposed convolution layer.
- signature: `class Conv2DTranspose(BaseConv):`
- members:
  - `default_config(cls)` — [`L1526`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1526)
  - `forward(self, x: Tensor)` — [`L1562`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1562)
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L1603`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1603)
- protocol/private: `_conv`[`L1569`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1569), `_create_layer_parameter_specs`[`L1534`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1534)
- used by: [`Conv2DTranspose`](layers.md#Conv2DTranspose)

### `Conv2DTransposeWith1DPadding`
- def: [`axlearn/common/convolution.py:1624`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1624)
- doc: The 2-D convolution transpose with 1-D padding on the time axis.
- signature: `class Conv2DTransposeWith1DPadding(Conv2DTranspose):`
- members:
  - `conv_paddings(self, paddings: Tensor)` — [`L1673`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1673)
  - `default_config(cls)` — [`L1640`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1640)
  - `forward(self, x: Tensor, *, paddings: Tensor)` — [`L1647`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1647) — Computes convolution outputs and paddings.

### `Conv2DWith1DPadding`
- def: [`axlearn/common/convolution.py:617`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L617)
- doc: The 2-D convolution with 1-D padding on the time axis.
- signature: `class Conv2DWith1DPadding(Conv2D):`
- members:
  - `conv_paddings(self, paddings: Tensor)` — [`L728`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L728)
  - `forward(self, x: Tensor, *, paddings: Tensor)` — [`L702`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L702) — Computes convolution outputs and paddings.
- used by: [`Conv2DWith1DPadding`](layers.md#Conv2DWith1DPadding)

### `Conv3D`
- def: [`axlearn/common/convolution.py:741`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L741)
- doc: The 3-D convolution layer.
- signature: `class Conv3D(BaseConv):`
- members:
  - `default_config(cls)` — [`L768`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L768)
  - `forward(self, x: Tensor)` — [`L798`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L798)
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L828`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L828)
- protocol/private: `_conv`[`L805`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L805), `_create_layer_parameter_specs`[`L773`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L773)
- used by: [`Conv3D`](layers.md#Conv3D)

### `Conv3DTranspose`
- def: [`axlearn/common/convolution.py:1685`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1685)
- doc: The 3-D convolution transpose layer.
- signature: `class Conv3DTranspose(BaseConv):`
- members:
  - `default_config(cls)` — [`L1702`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1702)
  - `forward(self, x: Tensor)` — [`L1731`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1731)
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L1762`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1762)
- protocol/private: `_conv`[`L1738`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1738), `_create_layer_parameter_specs`[`L1707`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1707)

### `StackOverTime`
- def: [`axlearn/common/convolution.py:1786`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1786)
- doc: Stack inputs along the time axis.
- signature: `class StackOverTime(BaseLayer):`
- members:
  - `conv_paddings(self, paddings: Tensor)` — [`L1847`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1847)
  - `forward(self, inputs: Tensor, *, paddings: Tensor)` — [`L1807`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1807) — Stacks stride number of frames into one frame along the time axis.
  - `output_shape(self, *, input_shape: Sequence[Optional[int]])` — [`L1855`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1855) — Computes stacked output shape.

## Functions
- `_check_conv_cfg(*, window: Sequence[int], strides: Sequence[int], padding: ConvPaddingType, dilation: Optional[Sequence[int]], input_dim: int, output_dim: int, num_input_dim_groups: int)` — [`L28`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L28)
- `compute_conv_paddings(in_paddings: Tensor, *, window: int, stride: int, conv_padding: ConvPaddingType, dilation: Optional[int]=None, anchor: Optional[int]=None)` — [`L269`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L269) — Compute output paddings w.r.t. conv_padding.
- `compute_conv_transpose_paddings(in_paddings: Tensor, *, window: int, stride: int, conv_padding: ConvPaddingType, dilation: int=1, anchor: Optional[int]=None)` — [`L1141`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1141) — Compute output paddings w.r.t. conv_padding for conv transpose.
- `conv_dilate_window(*, window: Sequence[int], dilation: Optional[Sequence[int]])` — [`L101`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L101) — Returns dilated effective window size.
- `conv_explicit_padding(*, window: Sequence[int], strides: Sequence[int], padding: ConvPaddingType, dilation: Optional[Sequence[int]]=None)` — [`L119`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L119) — Returns the explicit padding for "SAME", "VALID", and "CAUSAL" modes.
- `conv_output_shape(in_shape: Sequence[Optional[int]], *, window: Sequence[int], strides: Sequence[int], padding: ConvPaddingType, dilation: Optional[Sequence[int]]=None)` — [`L220`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L220) — Returns output size for convolution.
- `conv_transpose_explicit_padding(*, window: Sequence[int], strides: Sequence[int], padding: ConvPaddingType, dilation: Sequence[int])` — [`L853`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L853) — Convert str padding to tuple padding for conv_transpose.
- `conv_transpose_output_shape(in_shape: Sequence[Optional[int]], *, window: Sequence[int], strides: Sequence[int], padding: ConvPaddingType, dilation: Sequence[int])` — [`L1077`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L1077) — Returns output size for conv transpose.

## Module values
- `ConvPaddingType` — [`L22`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L22)
- `SUPPORT_CONV_PADDING` — [`L24`](../../../../../../raw/code/axlearn/axlearn/common/convolution.py#L24)

