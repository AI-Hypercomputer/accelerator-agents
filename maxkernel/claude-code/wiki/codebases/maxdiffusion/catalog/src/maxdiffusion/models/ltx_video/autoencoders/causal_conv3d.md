---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.causal_conv3d`/CausalConv3d#
symbols:
  CausalConv3d.forward: forward().
  CausalConv3d.time_kernel_size: time_kernel_size.
  CausalConv3d.weight: weight().
  CausalConv3d.conv: conv.
  CausalConv3d: ''
  CausalConv3d.__init__: __init__().
  CausalConv3d.in_channels: in_channels.
  CausalConv3d.out_channels: out_channels.
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py)

## Classes
### `CausalConv3d`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py:23`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L23)
- signature: `class CausalConv3d(nn.Module):`
- members:
  - `forward(self, x, causal: bool = True)` — [`L61`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L61)
  - `weight(self)` — [`L73`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L73)
  - `conv` — [`L50`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L50)
  - `in_channels` — [`L38`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L38)
  - `out_channels` — [`L39`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L39)
  - `time_kernel_size` — [`L42`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L42)
- protocol/private: `__init__`[`L25`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_conv3d.py#L25)

