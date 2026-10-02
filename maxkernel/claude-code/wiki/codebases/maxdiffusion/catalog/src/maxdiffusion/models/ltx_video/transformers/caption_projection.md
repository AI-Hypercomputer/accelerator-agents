---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers/caption_projection.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers/caption_projection.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers.caption_projection`/CaptionProjection#
symbols:
  CaptionProjection.__call__: __call__().
  CaptionProjection.hidden_size: hidden_size.
  CaptionProjection.dtype: dtype.
  CaptionProjection.weight_dtype: weight_dtype.
  CaptionProjection.matmul_precision: matmul_precision.
  CaptionProjection: ''
  CaptionProjection.in_features: in_features.
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers/caption_projection.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py)

## Classes
### `CaptionProjection`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/caption_projection.py:24`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L24)
- doc: Projects caption embeddings. Also handles dropout for classifier-free guidance.
- signature: `class CaptionProjection(nn.Module):`
- members:
  - `dtype` — [`L31`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L31)
  - `hidden_size` — [`L30`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L30)
  - `in_features` — [`L29`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L29)
  - `matmul_precision` — [`L33`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L33)
  - `weight_dtype` — [`L32`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L32)
- protocol/private: `__call__`[`L36`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/caption_projection.py#L36)

