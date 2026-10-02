---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers_pytorch.embeddings`/
symbols:
  get_3d_sincos_pos_embed_from_grid: get_3d_sincos_pos_embed_from_grid().
  get_1d_sincos_pos_embed_from_grid: get_1d_sincos_pos_embed_from_grid().
  get_3d_sincos_pos_embed: get_3d_sincos_pos_embed().
  get_timestep_embedding: get_timestep_embedding().
  SinusoidalPositionalEmbedding: SinusoidalPositionalEmbedding#
  SinusoidalPositionalEmbedding.__init__: SinusoidalPositionalEmbedding#__init__().
  SinusoidalPositionalEmbedding.forward: SinusoidalPositionalEmbedding#forward().
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py)

## Classes
### `SinusoidalPositionalEmbedding`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py:117`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L117)
- doc: Apply positional information to a sequence of embeddings.
- signature: `class SinusoidalPositionalEmbedding(nn.Module):`
- members:
  - `forward(self, x)` — [`L138`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L138)
- protocol/private: `__init__`[`L129`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L129)

## Functions
- `get_1d_sincos_pos_embed_from_grid(embed_dim, pos)` — [`L93`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L93) — embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
- `get_3d_sincos_pos_embed(embed_dim, grid, w, h, f)` — [`L67`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L67) — grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
- `get_3d_sincos_pos_embed_from_grid(embed_dim, grid)` — [`L80`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L80)
- `get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 1, scale: float = 1, max_period: int = 10000)` — [`L26`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/embeddings.py#L26) — This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

