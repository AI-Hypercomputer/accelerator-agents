---
title: 'Module: src/maxdiffusion/models/ltx_video/utils/torch_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/utils/torch_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.utils.torch_utils`/
symbols:
  Identity: Identity#
  append_dims: append_dims().
  Identity.__init__: Identity#__init__().
  Identity.forward: Identity#forward().
---
# Module: [`src/maxdiffusion/models/ltx_video/utils/torch_utils.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_utils.py)

## Classes
### `Identity`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/utils/torch_utils.py:31`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_utils.py#L31)
- doc: A placeholder identity operator that is argument-insensitive.
- signature: `class Identity(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, *args, **kwargs)` — [`L38`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_utils.py#L38)
- protocol/private: `__init__`[`L34`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_utils.py#L34)
- used by: [`downsample`](../autoencoders/video_autoencoder.md#DownEncoderBlock3D.downsample), [`upsample`](../autoencoders/video_autoencoder.md#UpDecoderBlock3D.upsample)

## Functions
- `append_dims(x: torch.Tensor, target_dims: int)` — [`L21`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_utils.py#L21) — Appends dimensions to the end of a tensor until it has target_dims dimensions.

