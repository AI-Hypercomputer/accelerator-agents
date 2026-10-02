---
title: 'Module: src/maxdiffusion/models/flux/util.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/flux/util.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.flux.util`/
symbols:
  configs: configs.
  load_flow_model: load_flow_model().
  ModelSpec.params: ModelSpec#params.
  ModelSpec.repo_id: ModelSpec#repo_id.
  ModelSpec.repo_flow: ModelSpec#repo_flow.
  FluxParams: FluxParams#
  ModelSpec.ckpt_path: ModelSpec#ckpt_path.
  FluxParams.in_channels: FluxParams#in_channels.
  FluxParams.vec_in_dim: FluxParams#vec_in_dim.
  FluxParams.context_in_dim: FluxParams#context_in_dim.
  FluxParams.hidden_size: FluxParams#hidden_size.
  FluxParams.mlp_ratio: FluxParams#mlp_ratio.
  FluxParams.num_heads: FluxParams#num_heads.
  FluxParams.depth: FluxParams#depth.
  FluxParams.depth_single_blocks: FluxParams#depth_single_blocks.
  FluxParams.axes_dim: FluxParams#axes_dim.
  FluxParams.theta: FluxParams#theta.
  FluxParams.qkv_bias: FluxParams#qkv_bias.
  FluxParams.guidance_embed: FluxParams#guidance_embed.
  FluxParams.rngs: FluxParams#rngs.
  FluxParams.param_dtype: FluxParams#param_dtype.
  ModelSpec: ModelSpec#
  validate_flax_state_dict: validate_flax_state_dict().
  print_load_warning: print_load_warning().
---
# Module: [`src/maxdiffusion/models/flux/util.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py)

## Classes
### `FluxParams`
- def: [`src/maxdiffusion/models/flux/util.py:34`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L34)
- signature: `class FluxParams:`
- members:
  - `axes_dim` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L43)
  - `context_in_dim` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L37)
  - `depth` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L41)
  - `depth_single_blocks` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L42)
  - `guidance_embed` — [`L46`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L46)
  - `hidden_size` — [`L38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L38)
  - `in_channels` — [`L35`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L35)
  - `mlp_ratio` — [`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L39)
  - `num_heads` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L40)
  - `param_dtype` — [`L48`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L48)
  - `qkv_bias` — [`L45`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L45)
  - `rngs` — [`L47`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L47)
  - `theta` — [`L44`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L44)
  - `vec_in_dim` — [`L36`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L36)
- used by: [`configs`](util.md#configs), [`params`](util.md#ModelSpec.params)

### `ModelSpec`
- def: [`src/maxdiffusion/models/flux/util.py:52`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L52)
- signature: `class ModelSpec:`
- members:
  - `ckpt_path` — [`L54`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L54)
  - `params` — [`L53`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L53)
  - `repo_flow` — [`L56`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L56)
  - `repo_id` — [`L55`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L55)
- uses (calls/refs, reference-scoped): [`FluxParams`](util.md#FluxParams)
- used by: [`configs`](util.md#configs), [`load_flow_model`](util.md#load_flow_model)

## Functions
- `load_flow_model(name: str, eval_shapes: dict, device: str, hf_download: bool = True)` — [`L141`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L141)
- `print_load_warning(missing: list[str], unexpected: list[str])` — [`L105`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L105)
- `validate_flax_state_dict(expected_pytree: dict, new_pytree: dict)` — [`L116`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L116) — expected_pytree: dict - a pytree that comes from initializing the model.

## Module values
- `configs` — [`L59`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/flux/util.py#L59)

