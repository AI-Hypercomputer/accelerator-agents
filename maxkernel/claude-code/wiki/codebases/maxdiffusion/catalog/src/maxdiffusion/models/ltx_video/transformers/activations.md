---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers/activations.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers/activations.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers.activations`/
symbols:
  GELU.__call__: GELU#__call__().
  GEGLU.__call__: GEGLU#__call__().
  ApproximateGELU.__call__: ApproximateGELU#__call__().
  GELU.gelu: GELU#gelu().
  GELU.approximate: GELU#approximate.
  get_activation: get_activation().
  ACTIVATION_FUNCTIONS: ACTIVATION_FUNCTIONS.
  approximate_gelu: approximate_gelu().
  GELU.dim_out: GELU#dim_out.
  GELU.bias: GELU#bias.
  GELU.kernel_axes: GELU#kernel_axes.
  GELU.kernel_init: GELU#kernel_init.
  GELU.dtype: GELU#dtype.
  GELU.weight_dtype: GELU#weight_dtype.
  GELU.matmul_precision: GELU#matmul_precision.
  GEGLU.dim_out: GEGLU#dim_out.
  GEGLU.bias: GEGLU#bias.
  GEGLU.kernel_axes: GEGLU#kernel_axes.
  GEGLU.kernel_init: GEGLU#kernel_init.
  GEGLU.dtype: GEGLU#dtype.
  GEGLU.weight_dtype: GEGLU#weight_dtype.
  GEGLU.matmul_precision: GEGLU#matmul_precision.
  ApproximateGELU.dim_out: ApproximateGELU#dim_out.
  ApproximateGELU.bias: ApproximateGELU#bias.
  ApproximateGELU.kernel_axes: ApproximateGELU#kernel_axes.
  ApproximateGELU.kernel_init: ApproximateGELU#kernel_init.
  ApproximateGELU.dtype: ApproximateGELU#dtype.
  ApproximateGELU.weight_dtype: ApproximateGELU#weight_dtype.
  ApproximateGELU.matmul_precision: ApproximateGELU#matmul_precision.
  GELU: GELU#
  GELU.dim_in: GELU#dim_in.
  GEGLU: GEGLU#
  GEGLU.dim_in: GEGLU#dim_in.
  ApproximateGELU: ApproximateGELU#
  ApproximateGELU.dim_in: ApproximateGELU#dim_in.
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers/activations.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py)

## Classes
### `ApproximateGELU`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/activations.py:154`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L154)
- doc: The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
- signature: `class ApproximateGELU(nn.Module):`
- members:
  - `bias` — [`L167`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L167)
  - `dim_in` — [`L165`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L165)
  - `dim_out` — [`L166`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L166)
  - `dtype` — [`L172`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L172)
  - `kernel_axes` — [`L169`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L169)
  - `kernel_init` — [`L170`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L170)
  - `matmul_precision` — [`L174`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L174)
  - `weight_dtype` — [`L173`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L173)
- protocol/private: `__call__`[`L177`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L177)

### `GEGLU`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/activations.py:111`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L111)
- doc: A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.
- signature: `class GEGLU(nn.Module):`
- members:
  - `bias` — [`L123`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L123)
  - `dim_in` — [`L121`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L121)
  - `dim_out` — [`L122`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L122)
  - `dtype` — [`L128`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L128)
  - `kernel_axes` — [`L125`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L125)
  - `kernel_init` — [`L126`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L126)
  - `matmul_precision` — [`L130`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L130)
  - `weight_dtype` — [`L129`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L129)
- protocol/private: `__call__`[`L133`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L133)

### `GELU`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/activations.py:62`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L62)
- doc: GELU activation function with tanh approximation support with `approximate="tanh"`.
- signature: `class GELU(nn.Module):`
- members:
  - `gelu(self, gate: jax.Array)` — [`L85`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L85)
  - `approximate` — [`L75`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L75)
  - `bias` — [`L76`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L76)
  - `dim_in` — [`L73`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L73)
  - `dim_out` — [`L74`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L74)
  - `dtype` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L81)
  - `kernel_axes` — [`L78`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L78)
  - `kernel_init` — [`L79`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L79)
  - `matmul_precision` — [`L83`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L83)
  - `weight_dtype` — [`L82`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L82)
- protocol/private: `__call__`[`L93`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L93)
- uses (calls/refs, reference-scoped): [`approximate_gelu`](activations.md#approximate_gelu)

## Functions
- `approximate_gelu(x: jax.Array)` — [`L38`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L38) — Computes Gaussian Error Linear Unit (GELU) activation function
- `get_activation(act_fn: str)` — [`L54`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L54) — Returns the activation function from string.

## Module values
- `ACTIVATION_FUNCTIONS` — [`L27`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/activations.py#L27)

