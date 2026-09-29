---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers_pytorch.symmetric_patchifier`/
symbols:
  Patchifier._patch_size: Patchifier#_patch_size.
  SymmetricPatchifier.patchify: SymmetricPatchifier#patchify().
  Patchifier: Patchifier#
  Patchifier.get_latent_coords: Patchifier#get_latent_coords().
  Patchifier.patchify: Patchifier#patchify().
  Patchifier.unpatchify: Patchifier#unpatchify().
  Patchifier.patch_size: Patchifier#patch_size().
  SymmetricPatchifier: SymmetricPatchifier#
  SymmetricPatchifier.unpatchify: SymmetricPatchifier#unpatchify().
  Patchifier.__init__: Patchifier#__init__().
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py)

## Classes
### `Patchifier`  ·  implements/extends ABC
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py:26`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L26)
- signature: `class Patchifier(ConfigMixin, ABC):`
- members:
  - `get_latent_coords(self, latent_num_frames, latent_height, latent_width, batch_size, device)` — [`L50`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L50) — Return a tensor of shape [batch_size, 3, num_patches] containing the
  - `patch_size(self)` — [`L47`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L47)
  - `patchify(self, latents: Tensor)` — [`L33`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L33)
  - `unpatchify(self, latents: Tensor, output_height: int, output_width: int, out_channels: int)` — [`L37`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L37)
- protocol/private: `__init__`[`L28`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L28), `_patch_size`[`L30`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L30)
- uses (calls/refs, reference-scoped): [`patchify`](symmetric_patchifier.md#SymmetricPatchifier.patchify), [`SymmetricPatchifier`](symmetric_patchifier.md#SymmetricPatchifier), [`unpatchify`](symmetric_patchifier.md#SymmetricPatchifier.unpatchify)
- used by: [`patchify`](symmetric_patchifier.md#SymmetricPatchifier.patchify), [`SymmetricPatchifier`](symmetric_patchifier.md#SymmetricPatchifier), [`unpatchify`](symmetric_patchifier.md#SymmetricPatchifier.unpatchify)

### `SymmetricPatchifier`  ·  implements/extends Patchifier
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py:67`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L67)
- signature: `class SymmetricPatchifier(Patchifier):`
- members:
  - `patchify(self, latents: Tensor)` — [`L69`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L69)
  - `unpatchify(self, latents: Tensor, output_height: int, output_width: int, out_channels: int)` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/symmetric_patchifier.py#L81)
- uses (calls/refs, reference-scoped): [`_patch_size`](symmetric_patchifier.md#Patchifier._patch_size), [`Patchifier`](symmetric_patchifier.md#Patchifier), [`get_latent_coords`](symmetric_patchifier.md#Patchifier.get_latent_coords)
- used by: [`Patchifier`](symmetric_patchifier.md#Patchifier), [`patchify`](symmetric_patchifier.md#Patchifier.patchify), [`unpatchify`](symmetric_patchifier.md#Patchifier.unpatchify)

