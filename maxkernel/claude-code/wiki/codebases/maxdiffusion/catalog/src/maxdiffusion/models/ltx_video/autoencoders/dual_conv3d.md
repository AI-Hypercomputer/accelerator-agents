---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.dual_conv3d`/
symbols:
  DualConv3d.forward_with_3d: DualConv3d#forward_with_3d().
  DualConv3d.forward_with_2d: DualConv3d#forward_with_2d().
  DualConv3d.reset_parameters: DualConv3d#reset_parameters().
  DualConv3d.bias2: DualConv3d#bias2.
  DualConv3d.weight2: DualConv3d#weight2.
  DualConv3d.forward: DualConv3d#forward().
  DualConv3d.padding_mode: DualConv3d#padding_mode.
  DualConv3d.groups: DualConv3d#groups.
  DualConv3d.weight1: DualConv3d#weight1.
  DualConv3d.stride1: DualConv3d#stride1.
  DualConv3d.padding1: DualConv3d#padding1.
  DualConv3d.dilation1: DualConv3d#dilation1.
  DualConv3d.bias1: DualConv3d#bias1.
  DualConv3d.__init__: DualConv3d#__init__().
  DualConv3d.weight: DualConv3d#weight().
  test_dual_conv3d_consistency: test_dual_conv3d_consistency().
  DualConv3d: DualConv3d#
  DualConv3d.stride2: DualConv3d#stride2.
  DualConv3d.padding2: DualConv3d#padding2.
  DualConv3d.dilation2: DualConv3d#dilation2.
  DualConv3d.bias: DualConv3d#bias.
  DualConv3d.in_channels: DualConv3d#in_channels.
  DualConv3d.out_channels: DualConv3d#out_channels.
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py)

## Classes
### `DualConv3d`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py:26`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L26)
- signature: `class DualConv3d(nn.Module):`
- members:
  - `forward(self, x, use_conv3d=False, skip_time_conv=False)` — [`L106`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L106)
  - `forward_with_2d(self, x, skip_time_conv)` — [`L142`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L142)
  - `forward_with_3d(self, x, skip_time_conv)` — [`L112`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L112)
  - `reset_parameters(self)` — [`L95`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L95)
  - `weight(self)` — [`L194`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L194)
  - `bias` — [`L59`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L59)
  - `bias1` — [`L78`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L78)
  - `bias2` — [`L88`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L88)
  - `dilation1` — [`L76`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L76)
  - `dilation2` — [`L86`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L86)
  - `groups` — [`L58`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L58)
  - `in_channels` — [`L42`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L42)
  - `out_channels` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L43)
  - `padding1` — [`L75`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L75)
  - `padding2` — [`L85`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L85)
  - `padding_mode` — [`L44`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L44)
  - `stride1` — [`L74`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L74)
  - `stride2` — [`L84`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L84)
  - `weight1` — [`L65`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L65)
  - `weight2` — [`L83`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L83)
- protocol/private: `__init__`[`L28`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L28)
- used by: [`test_dual_conv3d_consistency`](dual_conv3d.md#test_dual_conv3d_consistency)

## Functions
- `test_dual_conv3d_consistency()` — [`L198`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/dual_conv3d.py#L198)

