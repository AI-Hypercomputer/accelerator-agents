---
title: 'Module: graphcast/sparse_transformer.py'
type: catalog
provenance: extracted
module: graphcast/sparse_transformer.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.sparse_transformer`/
symbols:
  splash_mha: splash_mha().
  triblockdiag_mha: triblockdiag_mha().
  Transformer._cfg: Transformer#_cfg.
  mha: mha().
  multihead_linear: multihead_linear().
  ffw: ffw().
  Block.attn: Block#attn().
  Transformer.__call__: Transformer#__call__().
  Transformer.mask: Transformer#mask.
  _ModelConfig.__post_init__: _ModelConfig#__post_init__().
  Block.__call__: Block#__call__().
  _ModelConfig.num_heads: _ModelConfig#num_heads.
  _ModelConfig.d_model: _ModelConfig#d_model.
  Block._cfg: Block#_cfg.
  _ModelConfig.key_size: _ModelConfig#key_size.
  _ModelConfig.num_layers: _ModelConfig#num_layers.
  _ModelConfig: _ModelConfig#
  _ModelConfig.value_size: _ModelConfig#value_size.
  WeatherMeshMask.__init__: WeatherMeshMask#__init__().
  Transformer.num_padding_nodes: Transformer#num_padding_nodes.
  Block.norm_conditioning_layer: Block#norm_conditioning_layer().
  Transformer.norm_conditioning_layer: Transformer#norm_conditioning_layer().
  _ModelConfig.attn_winit_final_mult: _ModelConfig#attn_winit_final_mult.
  _ModelConfig.ffw_hidden: _ModelConfig#ffw_hidden.
  triblockdiag_mha.qk_prod: triblockdiag_mha().qk_prod().
  triblockdiag_mha.av_prod: triblockdiag_mha().av_prod().
  layernorm: layernorm().
  Block.mask: Block#mask.
  Block.num_nodes: Block#num_nodes.
  Block.num_padding_nodes: Block#num_padding_nodes.
  WeatherMeshMask.shape: WeatherMeshMask#shape().
  WeatherMeshMask.__getitem__: WeatherMeshMask#__getitem__().
  Transformer.__init__: Transformer#__init__().
  _ModelConfig.block_q: _ModelConfig#block_q.
  _ModelConfig.block_kv: _ModelConfig#block_kv.
  _ModelConfig.block_kv_compute: _ModelConfig#block_kv_compute.
  _ModelConfig.block_q_dkv: _ModelConfig#block_q_dkv.
  _ModelConfig.block_kv_dkv: _ModelConfig#block_kv_dkv.
  _ModelConfig.block_kv_dkv_compute: _ModelConfig#block_kv_dkv_compute.
  _ModelConfig.mask_type: _ModelConfig#mask_type.
  WeatherMeshMask._shape: WeatherMeshMask#_shape.
  WeatherMeshMask.mask: WeatherMeshMask#mask.
  _ModelConfig.mask_block_size: _ModelConfig#mask_block_size.
  _ModelConfig.attention_type: _ModelConfig#attention_type.
  _ModelConfig.activation: _ModelConfig#activation.
  _ModelConfig.ffw_winit_mult: _ModelConfig#ffw_winit_mult.
  _ModelConfig.ffw_winit_final_mult: _ModelConfig#ffw_winit_final_mult.
  _ModelConfig.attn_winit_mult: _ModelConfig#attn_winit_mult.
  get_mask_block_size: get_mask_block_size().
  triblockdiag_softmax: triblockdiag_softmax().
  mha.apply_mask: mha().apply_mask().
  _make_splash_mha: _make_splash_mha().
  mask_block_diags: mask_block_diags().
  _pad_mask: _pad_mask().
  WeatherMeshMask: WeatherMeshMask#
  Block: Block#
  Transformer: Transformer#
  Block.__init__: Block#__init__().
---
# Module: [`graphcast/sparse_transformer.py`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py)

## Classes
### `Block`
- def: [`graphcast/sparse_transformer.py:411`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L411)
- doc: Transformer block (mha and ffw).
- signature: `class Block(hk.Module):`
- members:
  - `attn(x)` — [`L423`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L423) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `norm_conditioning_layer(x)` — [`L459`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L459)
  - `mask` — [`L417`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L417)
  - `num_nodes` — [`L418`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L418)
  - `num_padding_nodes` — [`L419`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L419)
- protocol/private: `__call__`[`L421`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L421), `__init__`[`L414`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L414), `_cfg`[`L416`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L416)
- uses (calls/refs, reference-scoped): [`splash_mha`](sparse_transformer.md#splash_mha), [`triblockdiag_mha`](sparse_transformer.md#triblockdiag_mha), [`mha`](sparse_transformer.md#mha), [`ffw`](sparse_transformer.md#ffw), [`LinearNormConditioning`](mlp.md#LinearNormConditioning), [`layernorm`](sparse_transformer.md#layernorm)
- used by: [`__call__`](sparse_transformer.md#Transformer.__call__)

### `Transformer`
- def: [`graphcast/sparse_transformer.py:480`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L480)
- doc: Main transformer module that processes embeddings.
- signature: `class Transformer(hk.Module):`
- members:
  - `norm_conditioning_layer(x)` — [`L567`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L567)
  - `mask` — [`L515`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L515)
  - `num_padding_nodes` — [`L512`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L512)
- protocol/private: `__call__`[`L557`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L557), `__init__`[`L488`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L488), `_cfg`[`L544`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L544)
- uses (calls/refs, reference-scoped): [`num_heads`](sparse_transformer.md#_ModelConfig.num_heads), [`num_layers`](sparse_transformer.md#_ModelConfig.num_layers), [`_ModelConfig`](sparse_transformer.md#_ModelConfig), [`LinearNormConditioning`](mlp.md#LinearNormConditioning), [`layernorm`](sparse_transformer.md#layernorm), [`block_kv`](sparse_transformer.md#_ModelConfig.block_kv), [`block_kv_compute`](sparse_transformer.md#_ModelConfig.block_kv_compute), [`block_kv_dkv`](sparse_transformer.md#_ModelConfig.block_kv_dkv), [`block_kv_dkv_compute`](sparse_transformer.md#_ModelConfig.block_kv_dkv_compute), [`block_q`](sparse_transformer.md#_ModelConfig.block_q), [`block_q_dkv`](sparse_transformer.md#_ModelConfig.block_q_dkv), [`mask_type`](sparse_transformer.md#_ModelConfig.mask_type), [`Block`](sparse_transformer.md#Block), [`WeatherMeshMask`](sparse_transformer.md#WeatherMeshMask), [`_pad_mask`](sparse_transformer.md#_pad_mask), [`attention_type`](sparse_transformer.md#_ModelConfig.attention_type), [`get_mask_block_size`](sparse_transformer.md#get_mask_block_size), [`mask_block_diags`](sparse_transformer.md#mask_block_diags), [`mask_block_size`](sparse_transformer.md#_ModelConfig.mask_block_size)
- used by: [`_mesh_gnn`](denoiser.md#_DenoiserArchitecture._mesh_gnn)

### `WeatherMeshMask`
- def: [`graphcast/sparse_transformer.py:380`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L380)
- doc: Lazy local mask, prevent attention to embeddings outside window.
- signature: `class WeatherMeshMask(splash_attention.splash_attention_mask.Mask):`
- members:
  - `shape(self)` — [`L398`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L398)
  - `mask` — [`L388`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L388)
- protocol/private: `__getitem__`[`L401`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L401), `__init__`[`L390`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L390), `_shape`[`L387`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L387)
- used by: [`mask`](sparse_transformer.md#Transformer.mask)

### `_ModelConfig`
- def: [`graphcast/sparse_transformer.py:41`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L41) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- doc: Transformer config.
- signature: `class _ModelConfig:`
- members:
  - `activation` — [`L66`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L66)
  - `attention_type` — [`L52`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L52)
  - `attn_winit_final_mult` — [`L74`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L74) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `attn_winit_mult` — [`L72`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L72)
  - `block_kv` — [`L54`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L54) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `block_kv_compute` — [`L55`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L55) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `block_kv_dkv` — [`L57`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L57)
  - `block_kv_dkv_compute` — [`L58`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L58)
  - `block_q` — [`L53`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L53)
  - `block_q_dkv` — [`L56`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L56)
  - `d_model` — [`L46`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L46) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `ffw_hidden` — [`L76`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L76)
  - `ffw_winit_final_mult` — [`L70`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L70)
  - `ffw_winit_mult` — [`L68`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L68)
  - `key_size` — [`L62`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L62) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `mask_block_size` — [`L50`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L50)
  - `mask_type` — [`L60`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L60)
  - `num_heads` — [`L48`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L48) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `num_layers` — [`L44`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L44) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
  - `value_size` — [`L64`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L64) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- protocol/private: `__post_init__`[`L78`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L78)
- used by: [`splash_mha`](sparse_transformer.md#splash_mha), [`triblockdiag_mha`](sparse_transformer.md#triblockdiag_mha), [`_cfg`](sparse_transformer.md#Transformer._cfg), [`mha`](sparse_transformer.md#mha), [`multihead_linear`](sparse_transformer.md#multihead_linear), [`ffw`](sparse_transformer.md#ffw), [`__call__`](sparse_transformer.md#Transformer.__call__)

## Functions
- `_make_splash_mha(mask, mask_type: str, num_heads: int, block_q: Optional[int] = None, block_kv: Optional[int] = None, block_kv_compute: Optional[int] = None, block_q_dkv: Optional[int] = None, block_kv_dkv: Optional[int] = None, block_kv_dkv_compute: Optional[int] = None, tanh_soft_cap: Optional[float] = None)` — [`L244`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L244) — Construct attention kernel.
- `_pad_mask(mask, num_padding_nodes: Tuple[int, int])` — [`L369`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L369)
- `apply_mask(m, l)` — [`L223`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L223)
- `av_prod(attn_weights, values)` — [`L176`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L176) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `ffw(x: jnp.ndarray, cfg: _ModelConfig)` — [`L105`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L105) — Feed-forward block.
- `get_mask_block_size(mask: sp.sparse.csr_matrix)` — [`L92`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L92) — Get blocksize of the adjacency matrix (attn mask) for the permuted mesh.
- `layernorm(x: jnp.ndarray, create_scale: bool, create_offset: bool)` — [`L324`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L324)
- `mask_block_diags(mask: sp.sparse.csr_matrix, num_padding_nodes: int, block_size: int)` — [`L332`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L332) — Pad and reshape mask diag, super-siag and sub-diag blocks.
- `mha(q_input: jnp.ndarray, kv_input: jnp.ndarray, mask: jnp.ndarray, cfg: _ModelConfig, normalize_logits: bool = True)` — [`L209`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L209) — Multi head attention. — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `multihead_linear(x: jnp.ndarray, qkv: str, cfg: _ModelConfig)` — [`L192`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L192) — Linearly project `x` to have `head_size` dimensions per head. — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `qk_prod(queries, keys)` — [`L157`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L157) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `splash_mha(q_input: jnp.ndarray, kv_input: jnp.ndarray, mask: jnp.ndarray | splash_attention.splash_attention_mask.Mask, cfg: _ModelConfig, tanh_soft_cap: Optional[float] = None, normalize_q: bool = True)` — [`L278`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L278) — Splash attention. — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `triblockdiag_mha(q_input: jnp.ndarray, kv_input: jnp.ndarray, mask: jnp.ndarray, cfg: _ModelConfig)` — [`L144`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L144) — Triblockdiag multihead attention. — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `triblockdiag_softmax(logits: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray])` — [`L116`](../../../../../raw/code/graphcast/graphcast/sparse_transformer.py#L116) — Softmax given the diag, upper diag, and lower diag logit blocks.

